"""MODEL 2, evaluated: a learned correction that lets a lattice twice as coarse run a burning front like the fine one.

The question: on initial conditions the model never saw, is the coarse lattice + learned flux closer to the fine lattice
than (a) the plain coarse lattice and (b) the coarse lattice with its physical coefficients re-fitted to the fine one
(the obvious classical fix: same data, same loss, same optimizer, five numbers instead of a network)? And what does it
cost against simply running the fine lattice?

Data. The engine's CPU reference runs front.law on the fine lattice (256 x 16, h = 0.5 m) for 600 steps of 0.05 s from
initial conditions drawn per seed: fuel level, fuel noise (amplitude, pattern), width of the ignition strip. Training
uses 6 of them; the test set is 6 others from the same distribution, plus 2 out of distribution (sparser, patchier
fuel; a full, uniform load). Targets are the fine states averaged 2 x 2.

Methods, all on the coarse lattice (128 x 8, h = 1 m):
  plain        front.law as it is;
  calibrated   k_heat, ignite, burn, heat, cool fitted (as log-factors) through the unrolled solver;
  learned      + front_fix.law: a face flux from a 4-32-32-1 net, antisymmetric (conserves T to the bit) and limited;
               3 training seeds (init and batches), reported as mean ± std over seeds.
Training: jax.grad through 10 unrolled coarse steps from states of the training trajectories (tools/lawjax).
Metrics on full 600-step rollouts from the start: T RMSE (mean over time) and the front position at the end.

    python bench/model2_front.py [--iterations 400] [--seeds 3]

Writes assets/nets/front_fix.net (the first seed's net), bench/results/model2.json and .md.
"""

import argparse
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common import BUILD, ROOT, lawc, lawc_json, mean_std, read_state, save, write_scene
from lawjax.ir import load_law
from lawjax.nets import init_net, write_net

STEPS, DT, UNROLL = 600, 0.05, 10
TRAIN = list(range(6))
TEST = list(range(100, 106))
OOD = {"sparse fuel": dict(fuel=0.45, amp=0.35, pattern=901, width=6), "full uniform load": dict(fuel=1.0, amp=0.0, pattern=902, width=6)}
CAL_PARAMS = ["k_heat", "ignite", "burn", "heat", "cool"]


def ic(seed):
    r = np.random.default_rng(seed)
    return dict(fuel=float(r.uniform(0.6, 0.95)), amp=float(r.uniform(0.05, 0.25)), pattern=int(r.integers(1, 10 ** 6)),
                width=int(r.choice([4, 6, 8])))


def fine_run(spec, work, name):
    scene = write_scene(os.path.join(work, name + ".scene"), (256, 16, 1, 0.5), ["assets/law/bench/front.law"], [
        f"fill fuel {spec['fuel'] - spec['amp'] / 2:.6f}", f"noise fuel {spec['amp']:.6f} {spec['pattern']}",
        f"box T 1500 0 0 0 {spec['width']} 16 1"])
    out = os.path.join(work, name)
    lawc("--scene", scene, "--steps", str(STEPS), "--dt", str(DT), "--dump", out, "--every", "1")
    T, F = [], []
    for s in range(STEPS + 1):
        st, _, _ = read_state(os.path.join(out, f"step_{s}"))
        T.append(coarsen(st["T"]))
        F.append(coarsen(st["fuel"]))
    return jnp.asarray(np.stack(T)), jnp.asarray(np.stack(F))


def coarsen(f):
    z, y, x = f.shape
    return f.reshape(z, y // 2, 2, x // 2, 2).mean(axis=(2, 4))


def front_position(T, h, line=600.0):
    hot = np.asarray(T)[0] > line
    return float(np.mean([(np.nonzero(r)[0].max() + 1) * h if r.any() else 0.0 for r in hot]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    work = tempfile.mkdtemp(prefix="bench_front_")
    t0 = time.time()

    print("fine reference trajectories on the engine's CPU solver ...")
    data = {}
    for s in TRAIN + TEST:
        data[s] = fine_run(ic(s), work, f"ic{s}")
    for name, spec in OOD.items():
        data[name] = fine_run(spec, work, name.replace(" ", "_"))
    print(f"  {len(data)} trajectories in {time.time() - t0:.0f} s")

    ir = os.path.join(work, "coarse.json")
    coarse_scene = write_scene(os.path.join(work, "coarse.scene"), (128, 8, 1, 1.0), [])
    lawc("--scene", coarse_scene, "assets/law/bench/front.law", "assets/law/bench/front_fix.law", "--ir-json", ir)
    law = load_law(ir)
    h = 1.0
    base_params = dict(law.param_values)
    net = law.nets["front_fix"]
    counts = law.substeps({"T": data[TRAIN[0]][0][0], "fuel": data[TRAIN[0]][1][0]}, DT, h)
    print(f"coarse substeps per term {counts}")

    def with_net(n):
        n.in_shift = np.asarray([800.0, 0.5, 800.0, 0.5], np.float32)
        n.in_scale = np.asarray([1 / 600.0, 2.0, 1 / 600.0, 2.0], np.float32)
        n.out_scale = 200.0
        return n

    with_net(net)  # the scales the net sees, fixed (the trainer of tools/lawjax uses the same)
    zero = jax.tree_util.tree_map(jnp.zeros_like, init_net("front_fix", net.sizes, net.act).params())
    train_T = jnp.stack([data[s][0] for s in TRAIN])
    train_F = jnp.stack([data[s][1] for s in TRAIN])

    def params_of(logf):
        out = dict(base_params)
        for i, name in enumerate(CAL_PARAMS):
            out[name] = base_params[name] * jnp.exp(jnp.clip(logf[i], -1.4, 0.9))
        return out

    def unroll_loss(net_params, logf, traj, idx):
        state = {"T": train_T[traj, idx], "fuel": train_F[traj, idx]}
        total = 0.0
        prm = params_of(logf)
        for k in range(UNROLL):
            state = law.step(state, DT, h, counts=counts, params=prm, nets={"front_fix": net_params})
            total = total + jnp.mean(((state["T"] - train_T[traj, idx + k + 1]) / 500.0) ** 2) + jnp.mean((state["fuel"] - train_F[traj, idx + k + 1]) ** 2)
        return total / UNROLL

    def batch_loss(theta, trajs, idxs, which):
        net_params, logf = theta
        if which == "net":
            logf = jax.lax.stop_gradient(logf)
        return jnp.mean(jax.vmap(lambda t, i: unroll_loss(net_params, logf, t, i))(trajs, idxs))

    def train(which, seed):
        rng = np.random.default_rng(seed)
        net_params = with_net(init_net("front_fix", net.sizes, net.act, seed=seed)).params() if which == "net" else zero
        logf = jnp.zeros(len(CAL_PARAMS), jnp.float32)
        theta = (net_params, logf)
        opt = optax.adam(2e-3 if which == "net" else 2e-2)
        state = opt.init(theta)
        grad = jax.jit(jax.value_and_grad(lambda th, t, i: batch_loss(th, t, i, which)))
        for it in range(a.iterations):
            trajs = jnp.asarray(rng.integers(0, len(TRAIN), a.batch))
            idxs = jnp.asarray(rng.integers(0, STEPS - UNROLL, a.batch))
            value, g = grad(theta, trajs, idxs)
            if which == "cal":
                g = (jax.tree_util.tree_map(jnp.zeros_like, g[0]), g[1])
            updates, state = opt.update(g, state)
            theta = optax.apply_updates(theta, updates)
            if it % 100 == 0 or it + 1 == a.iterations:
                print(f"    {which} seed {seed} it {it:4d} loss {float(value):.5f} ({time.time() - t0:.0f} s)")
        return theta

    step_fn = jax.jit(lambda st, net_params, logf: law.step(st, DT, h, counts=counts, params=params_of(logf), nets={"front_fix": net_params}))

    def evaluate(theta, key):
        net_params, logf = theta
        T_ref, F_ref = data[key]
        state = {"T": T_ref[0], "fuel": F_ref[0]}
        errs = []
        total0 = float(jnp.sum(T_ref[0]))
        for s in range(1, STEPS + 1):
            state = step_fn(state, net_params, logf)
            errs.append(float(jnp.sqrt(jnp.mean((state["T"] - T_ref[s]) ** 2))))
        return {"rmse": float(np.mean(errs)), "front_err": front_position(state["T"], h) - front_position(T_ref[STEPS], h),
                "finite": bool(np.all(np.isfinite(errs))), "total_T_start": total0}

    keys = TEST + list(OOD)
    results = {"plain": {}, "calibrated": {}, "learned": {}}
    plain_theta = (zero, jnp.zeros(len(CAL_PARAMS), jnp.float32))
    for k in keys:
        results["plain"][str(k)] = evaluate(plain_theta, k)
    print("calibrating the classical coefficients ...")
    cal = train("cal", 0)
    fitted = {n: float(base_params[n] * np.exp(np.clip(float(cal[1][i]), -1.4, 0.9))) for i, n in enumerate(CAL_PARAMS)}
    print(f"  fitted {fitted}")
    for k in keys:
        results["calibrated"][str(k)] = evaluate(cal, k)
    learned_thetas = []
    for seed in range(a.seeds):
        print(f"learned flux, seed {seed} ...")
        th = train("net", seed)
        learned_thetas.append(th)
        results["learned"][seed] = {str(k): evaluate(th, k) for k in keys}
    trained = with_net(init_net("front_fix", net.sizes, net.act)).with_params(learned_thetas[0][0])
    for path in (os.path.join(BUILD, "assets", "nets", "front_fix.net"), os.path.join(ROOT, "assets", "nets", "front_fix.net")):
        write_net(trained, path)

    # ---- cost on the engine's CUDA backend, at a size where the card is busy ---------------------------------------
    cost = {}
    try:
        fine_s = write_scene(os.path.join(work, "fine_big.scene"), (2048, 2048, 1, 0.5), ["assets/law/bench/front.law"],
                             ["fill fuel 0.8", "noise fuel 0.15 3", "box T 1500 0 0 0 6 2048 1"])
        coarse_s = write_scene(os.path.join(work, "coarse_big.scene"), (1024, 1024, 1, 1.0), ["assets/law/bench/front.law"],
                               ["fill fuel 0.8", "noise fuel 0.15 3", "box T 1500 0 0 0 3 1024 1"])
        for name, args in (("fine", [fine_s]), ("coarse plain", [coarse_s]),
                           ("coarse + learned flux", [coarse_s, "assets/law/bench/front.law", "assets/law/bench/front_fix.law"])):
            j = lawc_json("--scene", *args, "--steps", "40", "--warmup", "5", "--dt", str(DT), "--backend", "cuda")
            cost[name] = {"card_ms": sum(j["terms"].values()), "step_ms": j["median_ms"], "substeps": j["substeps"], "cells": j["cells"]}
        print(f"cost: {cost}")
    except RuntimeError as e:
        print(f"no CUDA cost measurement: {e}")

    # ---- the report --------------------------------------------------------------------------------------------------
    def summary(sel):
        out = {}
        for m in ("plain", "calibrated"):
            r = [results[m][str(k)] for k in sel]
            out[m] = {"rmse": [x["rmse"] for x in r], "front": [abs(x["front_err"]) for x in r]}
        per_seed_rmse = [np.mean([results["learned"][s][str(k)]["rmse"] for k in sel]) for s in results["learned"]]
        per_seed_front = [np.mean([abs(results["learned"][s][str(k)]["front_err"]) for k in sel]) for s in results["learned"]]
        out["learned"] = {"rmse_seed_means": per_seed_rmse, "front_seed_means": per_seed_front}
        return out

    test_sum = summary(TEST)
    md = ["# Model 2: learned correction for a coarse lattice (burning front)\n\n",
          f"Coarse 128 x 8 (h = 1 m) against the fine 256 x 16 (h = 0.5 m) averaged 2 x 2; 600 steps of {DT} s from the start. "
          f"Trained on {len(TRAIN)} initial conditions, tested on {len(TEST)} unseen ones from the same distribution and 2 out of it. "
          f"Learned: {a.seeds} training seeds, mean ± std over seeds of the mean over test ICs. Others: mean ± std over test ICs.\n\n",
          "| method | T RMSE, unseen ICs (K) | front position error at 600 steps (m) |\n|---|---|---|\n",
          f"| plain coarse | {mean_std(test_sum['plain']['rmse'])} | {mean_std(test_sum['plain']['front'])} |\n",
          f"| calibrated coarse (5 coefficients fitted) | {mean_std(test_sum['calibrated']['rmse'])} | {mean_std(test_sum['calibrated']['front'])} |\n",
          f"| coarse + learned flux | {mean_std(test_sum['learned']['rmse_seed_means'])} | {mean_std(test_sum['learned']['front_seed_means'])} |\n\n",
          "Out of distribution (learned: mean over seeds):\n\n| IC | plain RMSE / front | calibrated | learned |\n|---|---|---|---|\n"]
    for name in OOD:
        pl, ca = results["plain"][name], results["calibrated"][name]
        le_r = np.mean([results["learned"][s][name]["rmse"] for s in results["learned"]])
        le_f = np.mean([abs(results["learned"][s][name]["front_err"]) for s in results["learned"]])
        md.append(f"| {name} | {pl['rmse']:.1f} K / {abs(pl['front_err']):.1f} m | {ca['rmse']:.1f} K / {abs(ca['front_err']):.1f} m | "
                  f"{le_r:.1f} K / {le_f:.1f} m |\n")
    md.append(f"\nFitted coefficients (factor on the law's value): " + ", ".join(f"{n} x{fitted[n] / base_params[n]:.2f}" for n in CAL_PARAMS) + "\n")
    if cost:
        md.append("\nCost on the engine's CUDA backend (one step of 0.05 s; fine 2048², coarse 1024²; card time of the terms), taken at "
                  "the end of this script on a machine that may be busy; the clean measurement is in [perf.md](perf.md):\n\n"
                  "| run | card ms / step | substeps per term |\n|---|---|---|\n")
        for name, c in cost.items():
            md.append(f"| {name} | {c['card_ms']:.3f} | {c['substeps']} |\n")
    save("model2", {"results": results, "fitted": fitted, "cost": cost, "train": TRAIN, "test": TEST, "ood": OOD,
                    "iterations": a.iterations, "summary_test": test_sum}, "".join(md))
    print("".join(md))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

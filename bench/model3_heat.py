"""MODEL 3, evaluated: a learned discretization of heat conduction that takes twice the classical step, under guard.

heat.law's explicit scheme is stable for dt <= 0.15 h^2/k (the law's declared limit; 0.25 is the true 2D limit), so
at dt = 0.3 h^2/k the engine takes it in 2 substeps. The learned scheme (ldiffuse: a 4-cell face flux whose weights a
net gives, antisymmetric and consistent by construction) takes it in 1 pass. Questions:

  accuracy     on fields it was not trained on (a held-out bank from the training generator, the heat scene, and two
               out-of-distribution families: smooth low modes, sharp high-contrast boxes), after 100 steps, against a
               reference of 8 classical substeps a step: is 1 learned pass as good as 2 classical substeps?
  stability    does it keep the maximum principle (no new extremes)? the explicit scheme in 1 pass does not;
  guard        in the engine (CUDA), how many learned steps break a declared contract and are undone (trained and
               untrained net)?
  cost         the engine's card time a step: 1 learned pass against 2 classical substeps, on a 1024^2 lattice.

Trained like tools/lawjax/train_heat.py (jax.grad through 8 unrolled steps, against 2 classical substeps), with 3
training seeds; mean ± std over seeds.

    python bench/model3_heat.py [--iterations 600] [--seeds 3]

Writes assets/nets/heat_ld.net (the first seed's net), bench/results/model3.json and .md.
"""

import argparse
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common import BUILD, ROOT, lawc, lawc_json, mean_std, read_state, save, write_scene, write_state
from lawjax.ir import load_law
from lawjax.nets import init_net, write_net

N = 48
STEPS = 100


def bank_training(rng, count, n):
    """The training generator: hot spots and slabs on 290 K (as tools/lawjax/train_heat.py)."""
    out = []
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    for _ in range(count):
        T = np.full((n, n), 290.0, np.float32)
        for _ in range(rng.integers(1, 5)):
            cx, cy, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(3, 14)
            amp = rng.uniform(200, 1300)
            if rng.random() < 0.5:
                T += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
            else:
                T = np.where((np.abs(x - cx) < r) & (np.abs(y - cy) < r * 0.6), np.maximum(T, 290 + amp), T)
        out.append(T[None])
    return np.stack(out)


def bank_smooth(rng, count, n):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    out = []
    for _ in range(count):
        T = np.full((n, n), 700.0, np.float32)
        for _ in range(6):
            kx, ky = rng.integers(0, 4, 2)
            T += rng.uniform(-150, 150) * np.cos(np.pi * kx * x + rng.uniform(0, 6.28)) * np.cos(np.pi * ky * y + rng.uniform(0, 6.28))
        out.append(np.maximum(T, 1.0)[None])
    return np.stack(out)


def bank_sharp(rng, count, n):
    out = []
    y, x = np.mgrid[0:n, 0:n]
    for _ in range(count):
        T = np.full((n, n), 290.0, np.float32)
        for _ in range(rng.integers(8, 16)):
            x0, y0 = rng.integers(0, n - 2, 2)
            w, hgt = rng.integers(1, 6, 2)
            T = np.where((x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + hgt), rng.uniform(1500, 3000), T)
        out.append(T.astype(np.float32)[None])
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--unroll", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="bench_heat_")
    ir = os.path.join(work, "heat.json")
    lawc("--scene", "assets/scenes/heat.scene", "--ir-json", ir, "--steps", "1", "--dt", "0.3", "--dump", os.path.join(work, "scene"))
    law = load_law(ir)
    net = law.nets["heat_ld"]
    h, k = 1.0, law.param_values["k_heat"]
    dt = 0.3 * h * h / k

    train_bank = jnp.asarray(bank_training(np.random.default_rng(1), 64, N))
    scene0, _, _ = read_state(os.path.join(work, "scene", "step_0"))
    tests = {
        "held-out (training generator)": bank_training(np.random.default_rng(2), 24, N),
        "heat scene (96 x 96)": np.asarray(scene0["T"])[None],
        "OOD smooth low modes": bank_smooth(np.random.default_rng(3), 24, N),
        "OOD sharp boxes": bank_sharp(np.random.default_rng(4), 24, N),
    }

    def classical(T, n):
        """n classical substeps of dt / n: the engine's substeps (heat.law has no derive), as a loop XLA compiles quickly."""
        return jax.lax.fori_loop(0, n, lambda i, x: law.step({"T": x}, dt / n, h, counts=[1])["T"], T)

    def train(seed):
        rng = np.random.default_rng(seed)
        params = init_net("heat_ld", net.sizes, net.act, seed=seed).params()

        def single(prm, T0):
            ref, got, total = {"T": T0}, {"T": T0}, 0.0
            for _ in range(a.unroll):
                ref = {"T": classical(ref["T"], 2)}
                got = law.step(got, dt, h, counts=[1], learned=True, nets={"heat_ld": prm})
                total = total + jnp.mean(((got["T"] - jax.lax.stop_gradient(ref["T"])) / 500.0) ** 2)
            return total / a.unroll

        loss = lambda prm, batch: jnp.mean(jax.vmap(lambda T: single(prm, T))(batch))  # noqa: E731
        opt = optax.adam(2e-3)
        state = opt.init(params)
        grad = jax.jit(jax.value_and_grad(loss))
        for it in range(a.iterations):
            value, g = grad(params, train_bank[rng.integers(0, train_bank.shape[0], a.batch)])
            updates, state = opt.update(g, state)
            params = optax.apply_updates(params, updates)
            if it % 100 == 0 or it + 1 == a.iterations:
                print(f"  seed {seed} it {it:4d} loss {float(value):.2e} ({time.time() - t0:.0f} s)")
        return params

    zero = jax.tree_util.tree_map(jnp.zeros_like, init_net("heat_ld", net.sizes, net.act).params())
    schemes = {
        "classical, 2 substeps": lambda T, prm: classical(T, 2),
        "classical, 1 pass": lambda T, prm: classical(T, 1),
        "learned, 1 pass": lambda T, prm: law.step({"T": T}, dt, h, counts=[1], learned=True, nets={"heat_ld": prm})["T"],
    }

    def rollout(scheme):
        """All fields of a set at once (vmap), STEPS steps (scan) of the scheme and of the reference (8 substeps)."""
        step = schemes[scheme]

        def one(T0, prm):
            def body(c, _):
                s, r, worst = c
                s = step(s, prm)
                r = classical(r, 8)
                worst = jnp.maximum(worst, jnp.maximum(jnp.max(s) - jnp.max(T0), jnp.min(T0) - jnp.min(s)))
                return (s, r, worst), None
            (s, r, worst), _ = jax.lax.scan(body, (T0, T0, jnp.float32(0.0)), None, length=STEPS)
            return jnp.sqrt(jnp.mean((s - r) ** 2)), worst, jnp.max(jnp.abs(s))

        return jax.jit(jax.vmap(one, in_axes=(0, None)))

    runners = {name: rollout(name) for name in schemes}

    def evaluate(scheme, fields, prm):
        rmse, worst, peak = (np.asarray(v, np.float64) for v in runners[scheme](jnp.asarray(fields, jnp.float32), prm))
        bad = ~np.isfinite(rmse) | ~np.isfinite(peak) | (peak > 1e7)
        return {"rmse": np.where(bad, np.inf, rmse).tolist(), "overshoot": np.where(bad, np.inf, worst).tolist(), "blowup": int(bad.sum())}

    results = {"classical, 2 substeps": {}, "classical, 1 pass": {}, "learned, 1 pass": {}}
    for name, fields in tests.items():
        results["classical, 2 substeps"][name] = evaluate("classical, 2 substeps", fields, zero)
        results["classical, 1 pass"][name] = evaluate("classical, 1 pass", fields, zero)
        print(f"  {name}: classical 2 substeps RMSE {np.mean(results['classical, 2 substeps'][name]['rmse']):.3g} K ({time.time() - t0:.0f} s)")
    seeds = []
    for seed in range(a.seeds):
        print(f"training seed {seed} ...")
        prm = train(seed)
        seeds.append(prm)
        results["learned, 1 pass"][seed] = {name: evaluate("learned, 1 pass", f, prm) for name, f in tests.items()}
        print(f"  seed {seed}: " + ", ".join(f"{n} {np.mean(r['rmse']):.3g} K" for n, r in results["learned, 1 pass"][seed].items()))
    trained = init_net("heat_ld", net.sizes, net.act).with_params(seeds[0])
    for path in (os.path.join(BUILD, "assets", "nets", "heat_ld.net"), os.path.join(ROOT, "assets", "nets", "heat_ld.net")):
        write_net(trained, path)

    # ---- in the engine, under guard (CUDA): steps undone, trained and untrained ------------------------------------
    guard = {}
    untrained_law = os.path.join(work, "heat_untrained.law")
    with open(os.path.join(ROOT, "assets", "law", "bench", "heat.law")) as f:
        text = f.read().replace("act tanh", "act tanh file none")
    with open(untrained_law, "w") as f:
        f.write(text)
    try:
        for which, lawfile in (("trained", "assets/law/bench/heat.law"), ("untrained", untrained_law)):
            guard[which] = {}
            for name in ("held-out (training generator)", "OOD sharp boxes"):
                undone, violations, steps = 0, 0, 0
                for i, T0 in enumerate(tests[name][:8]):
                    scene = write_scene(os.path.join(work, "g.scene"), (N, N, 1, h), [])
                    init = write_state(os.path.join(work, f"g_{i}"), {"T": T0.reshape(1, N, N)}, h)
                    j = lawc_json("--scene", scene, lawfile, "--init", init, "--steps", "40", "--dt", str(dt), "--learned", "--backend", "cuda")
                    undone += j["fallbacks"]
                    violations += j["violations"]
                    steps += 40
                guard[which][name] = {"undone": undone, "steps": steps, "violations_reaching_world": violations}
        print(f"guard: {guard}")
    except RuntimeError as e:
        print(f"no guard measurement on CUDA: {e}")

    # ---- cost: card time a step on 1024^2 -------------------------------------------------------------------------
    cost = {}
    try:
        big = write_scene(os.path.join(work, "big.scene"), (1024, 1024, 1, h), ["assets/law/bench/heat.law"], ["fill T 290", "noise T 1000 5"])
        for name, extra in (("classical, 2 substeps", []), ("learned, 1 pass", ["--learned"])):
            j = lawc_json("--scene", big, "--steps", "30", "--warmup", "5", "--dt", str(dt), "--backend", "cuda", *extra)
            cost[name] = {"card_ms": sum(j["terms"].values()), "step_ms": j["median_ms"], "substeps": j["substeps"], "fallbacks": j["fallbacks"]}
        print(f"cost: {cost}")
    except RuntimeError as e:
        print(f"no cost measurement on CUDA: {e}")

    # ---- report ----------------------------------------------------------------------------------------------------
    def fmt(vals):
        v = np.asarray(vals, np.float64)
        return "diverges" if not np.all(np.isfinite(v)) else f"{v.mean():.3g}"

    md = ["# Model 3: learned discretization at twice the step, under guard\n\n",
          f"Heat, dt = 0.3 h²/k (twice the law's limit), {STEPS} steps; error = RMSE (K) at the end against 8 classical substeps a step; "
          "overshoot = how far past the initial [min, max] the field went (K; the maximum principle says 0). Learned: mean ± std over "
          f"{a.seeds} training seeds of the mean over the set.\n\n",
          "| set | classical, 2 substeps | classical, 1 pass | learned, 1 pass |\n|---|---|---|---|\n"]
    for name in tests:
        c2 = results["classical, 2 substeps"][name]
        c1 = results["classical, 1 pass"][name]
        le_rmse = [np.mean(results["learned, 1 pass"][s][name]["rmse"]) for s in results["learned, 1 pass"]]
        le_over = [np.mean(results["learned, 1 pass"][s][name]["overshoot"]) for s in results["learned, 1 pass"]]
        c1_text = f"diverges ({c1['blowup']}/{len(tests[name])})" if c1["blowup"] else f"{fmt(c1['rmse'])} K, overshoot {fmt(c1['overshoot'])}"
        md.append(f"| {name} | {fmt(c2['rmse'])} K, overshoot {fmt(c2['overshoot'])} | {c1_text} | "
                  f"{mean_std(le_rmse) if np.all(np.isfinite(le_rmse)) else 'diverges'} K, overshoot {mean_std(le_over) if np.all(np.isfinite(le_over)) else 'inf'} |\n")
    if guard:
        md.append("\nIn the engine (CUDA, `lawc --learned`), 40 steps on each of 8 fields: learned steps undone by the guard; contract "
                  "violations that reached the world:\n\n| net | set | undone / steps | violations |\n|---|---|---|---|\n")
        for which, sets in guard.items():
            for name, g in sets.items():
                md.append(f"| {which} | {name} | {g['undone']} / {g['steps']} | {g['violations_reaching_world']} |\n")
    if cost:
        md.append("\nCost on the engine's CUDA backend, 1024² lattice, one step of dt (card time of the term), taken at the end of this "
                  "script on a machine that may be busy; the clean measurement is in [perf.md](perf.md):\n\n| scheme | card ms | substeps |\n|---|---|---|\n")
        for name, c in cost.items():
            md.append(f"| {name} | {c['card_ms']:.3f} | {c['substeps']} |\n")
    save("model3", {"results": results, "guard": guard, "cost": cost, "iterations": a.iterations, "steps": STEPS}, "".join(md))
    print("".join(md))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

"""SUPERSEDED by bench/model2_front.py, which trains the same way and also evaluates on held-out initial conditions,
against a calibrated classical baseline, over several seeds. Kept as the minimal trainer.

MODEL 2: a learned correction that lets a coarse lattice run what a fine one runs (LOD), trained with the solver in
the loop.

The engine's CPU reference runs the burning front of front.law on the fine lattice (h = 0.5 m) and dumps every state;
those, averaged two by two, are what the coarse lattice (h = 1 m) should show. The coarse step is front.law plus
front_fix.law: a face flux of T from a net that sees T and fuel on both sides of each face. It is antisymmetric by
construction (half the difference of both ways round) and limited to T's range, so `conserves T` and the range hold
whatever the weights are; what is learned is only how heat should move. The loss is the error of K coarse steps run from
a state of the fine trajectory, through the solver (jax.grad through the unrolled steps).

    cd build && python ../tools/lawjax/train_front.py

writes assets/nets/front_fix.net (and ../assets/nets/front_fix.net), and docs/model2_front.md with the measurements.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawjax.ir import load_law, read_state  # noqa: E402
from lawjax.nets import write_net  # noqa: E402


def coarsen(field):
    """(1, ny, nx) -> (1, ny/2, nx/2): the mean of each 2 x 2."""
    z, y, x = field.shape
    return field.reshape(z, y // 2, 2, x // 2, 2).mean(axis=(2, 4))


def front_position(T, h, line=600.0):
    """Where the front is: the furthest x the heat has reached, averaged over the rows."""
    hot = np.asarray(T)[0] > line
    pos = []
    for row in hot:
        idx = np.nonzero(row)[0]
        pos.append((idx.max() + 1) * h if idx.size else 0.0)
    return float(np.mean(pos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lawc", default=os.path.join("Release", "lawc.exe"))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--unroll", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-3)
    a = p.parse_args()

    work = tempfile.mkdtemp(prefix="lawjax_front_")
    fine_ir = os.path.join(work, "fine.json")
    coarse_ir = os.path.join(work, "coarse.json")
    print("the fine reference, on the engine's CPU solver ...")
    subprocess.run([a.lawc, "--scene", "assets/scenes/front_fine.scene", "--ir-json", fine_ir, "--steps", str(a.steps), "--dt", str(a.dt),
                    "--dump", os.path.join(work, "fine"), "--every", "1"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([a.lawc, "--scene", "assets/scenes/front_coarse.scene", "assets/law/bench/front.law", "assets/law/bench/front_fix.law",
                    "--ir-json", coarse_ir, "--steps", "1", "--dt", str(a.dt), "--dump", os.path.join(work, "coarse")], check=True,
                   stdout=subprocess.DEVNULL)
    targets_T, targets_fuel = [], []
    for s in range(a.steps + 1):
        st, _, _ = read_state(os.path.join(work, "fine", f"step_{s}"))
        targets_T.append(coarsen(np.asarray(st["T"])))
        targets_fuel.append(coarsen(np.asarray(st["fuel"])))
    targets_T = jnp.asarray(np.stack(targets_T))
    targets_fuel = jnp.asarray(np.stack(targets_fuel))

    law = load_law(coarse_ir)
    plain = load_law(coarse_ir)
    net = law.nets["front_fix"]
    # What the net sees, scaled to about one: T on both sides, fuel on both sides; its flux is in K m/s.
    net.in_shift = np.asarray([800.0, 0.5, 800.0, 0.5], np.float32)
    net.in_scale = np.asarray([1 / 600.0, 2.0, 1 / 600.0, 2.0], np.float32)
    net.out_scale = 200.0
    _, shape, h = read_state(os.path.join(work, "coarse", "step_0"))
    counts = law.substeps({"T": targets_T[0], "fuel": targets_fuel[0]}, a.dt, h)
    print(f"coarse lattice {shape}, h = {h} m, substeps {counts}")

    def single(params, idx):
        state = {"T": targets_T[idx], "fuel": targets_fuel[idx]}
        total = 0.0
        for k in range(a.unroll):
            state = law.step(state, a.dt, h, counts=counts, nets={"front_fix": params})
            total = total + jnp.mean(((state["T"] - targets_T[idx + k + 1]) / 500.0) ** 2) + jnp.mean((state["fuel"] - targets_fuel[idx + k + 1]) ** 2)
        return total / a.unroll

    def loss(params, starts):
        return jnp.mean(jax.vmap(lambda i: single(params, i))(starts))

    params = net.params()
    opt = optax.adam(a.lr)
    opt_state = opt.init(params)
    grad_fn = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(0)
    started = time.time()
    for it in range(a.iterations):
        starts = jnp.asarray(rng.integers(0, a.steps - a.unroll, a.batch))
        value, grads = grad_fn(params, starts)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if it % 50 == 0 or it + 1 == a.iterations:
            print(f"  iteration {it:4d}  loss {float(value):.5f}  ({time.time() - started:.0f} s)")

    # ---- measure: the whole trajectory from the start, plain coarse against corrected against the fine one ----------
    def run(with_fix):
        state = {"T": targets_T[0], "fuel": targets_fuel[0]}
        zero = jax.tree_util.tree_map(jnp.zeros_like, params)
        errs, fronts = [], []
        step = jax.jit(lambda st: law.step(st, a.dt, h, counts=counts, nets={"front_fix": params if with_fix else zero}))
        for s in range(1, a.steps + 1):
            state = step(state)
            errs.append(float(jnp.sqrt(jnp.mean((state["T"] - targets_T[s]) ** 2))))
            fronts.append(front_position(state["T"], h) - front_position(targets_T[s], h))
        total = float(jnp.sum(state["T"]))
        return np.asarray(errs), np.asarray(fronts), total

    e_plain, f_plain, tot_plain = run(False)
    e_fix, f_fix, tot_fix = run(True)
    tot_ref = float(jnp.sum(targets_T[0]))
    print(f"T RMSE over {a.steps} steps: plain coarse {e_plain.mean():.1f} K, corrected {e_fix.mean():.1f} K")
    print(f"front position at the end: plain {f_plain[-1]:+.2f} m, corrected {f_fix[-1]:+.2f} m (against the fine lattice)")

    trained = net.with_params(params)
    for path in ("assets/nets/front_fix.net", "../assets/nets/front_fix.net"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_net(trained, path)
    os.makedirs("../docs", exist_ok=True)
    with open("../docs/model2_front.md", "w") as out:
        out.write("# Model 2: a learned LOD correction for a burning front\n\n")
        out.write("front.law on a lattice twice as coarse (h = 1 m), against the engine's CPU run of the fine lattice (h = 0.5 m) "
                  "averaged 2 x 2. The correction is front_fix.law: a learned face flux of T, antisymmetric by construction and "
                  "limited, trained by tools/lawjax/train_front.py with the solver in the loop "
                  f"({a.iterations} iterations, unrolls of {a.unroll} steps, dt = {a.dt} s).\n\n")
        out.write("| | plain coarse | corrected coarse |\n|---|---|---|\n")
        out.write(f"| mean T RMSE over {a.steps} steps (K) | {e_plain.mean():.1f} | {e_fix.mean():.1f} |\n")
        out.write(f"| T RMSE at the end (K) | {e_plain[-1]:.1f} | {e_fix[-1]:.1f} |\n")
        out.write(f"| front ahead of the fine one at the end (m) | {f_plain[-1]:+.2f} | {f_fix[-1]:+.2f} |\n")
        out.write(f"| total T change by the correction term | - | exactly 0 by construction |\n\n")
        out.write(json.dumps({"rmse_plain": e_plain.tolist()[::20], "rmse_fixed": e_fix.tolist()[::20]}) + "\n")
    print("wrote assets/nets/front_fix.net and docs/model2_front.md")


if __name__ == "__main__":
    main()

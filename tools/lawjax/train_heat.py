"""SUPERSEDED by bench/model3_heat.py, which trains the same way and also evaluates on held-out and out-of-distribution
fields, the guard and the cost, over several seeds. Kept as the minimal trainer.

MODEL 3: a learned discretization in the law's compiler, run under guard.

heat.law's classical conduction is explicit, stable under a step of 0.15 h^2/k; its learned scheme (ldiffuse) takes
twice that step in one pass. The face flux is interpolated from four cells with weights a net gives from the local
shape and from k dt / h^2; the weights sum to zero and start at the classical ones, and the flux is half the difference
of both ways round, so the total is kept to the bit whatever the net learns. Trained here so one learned step matches
the classical scheme's two substeps over long unrolls (stability is what the unroll teaches); in the engine every
learned step is checked against the term's contracts and undone (classical substeps instead) when it breaks one.

    cd build && python ../tools/lawjax/train_heat.py

writes assets/nets/heat_ld.net and docs/model3_heat.md.
"""

import argparse
import os
import re
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


def random_fields(rng, count, n):
    """Hot spots and slabs on 290 K: smooth and sharp at once (what fire leaves behind)."""
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
    return jnp.asarray(np.stack(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lawc", default=os.path.join("Release", "lawc.exe"))
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--unroll", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-3)
    a = p.parse_args()

    work = tempfile.mkdtemp(prefix="lawjax_heat_")
    ir = os.path.join(work, "heat.json")
    subprocess.run([a.lawc, "--scene", "assets/scenes/heat.scene", "--ir-json", ir, "--steps", "1", "--dt", "0.3", "--dump",
                    os.path.join(work, "scene")], check=True, stdout=subprocess.DEVNULL)
    law = load_law(ir)
    net = law.nets["heat_ld"]
    h, k = 1.0, law.param_values["k_heat"]
    dt = 0.3 * h * h / k                  # the learned step
    classical = [2]                       # the classical scheme takes it in two substeps
    learned = [1]

    rng = np.random.default_rng(1)
    bank = random_fields(rng, 64, a.n)

    def single(params, T0):
        ref = {"T": T0}
        got = {"T": T0}
        total = 0.0
        for _ in range(a.unroll):
            ref = law.step(ref, dt, h, counts=classical)
            got = law.step(got, dt, h, counts=learned, learned=True, nets={"heat_ld": params})
            total = total + jnp.mean(((got["T"] - jax.lax.stop_gradient(ref["T"])) / 500.0) ** 2)
        return total / a.unroll

    def loss(params, batch):
        return jnp.mean(jax.vmap(lambda T: single(params, T))(batch))

    params = net.params()
    opt = optax.adam(a.lr)
    opt_state = opt.init(params)
    grad_fn = jax.jit(jax.value_and_grad(loss))
    started = time.time()
    for it in range(a.iterations):
        batch = bank[rng.integers(0, bank.shape[0], a.batch)]
        value, grads = grad_fn(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if it % 50 == 0 or it + 1 == a.iterations:
            print(f"  iteration {it:4d}  loss {float(value):.6f}  ({time.time() - started:.0f} s)")

    trained = net.with_params(params)
    for path in ("assets/nets/heat_ld.net", "../assets/nets/heat_ld.net"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_net(trained, path)

    # ---- measure on the scene: against a fine-in-time reference (8 classical substeps a step) ----------------------
    start, _, _ = read_state(os.path.join(work, "scene", "step_0"))
    steps = 200
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    runs = {
        "classical, 2 substeps": jax.jit(lambda s: law.step(s, dt, h, counts=classical)),
        "learned, 1 pass": jax.jit(lambda s: law.step(s, dt, h, counts=learned, learned=True, nets={"heat_ld": params})),
        "untrained learned, 1 pass": jax.jit(lambda s: law.step(s, dt, h, counts=learned, learned=True, nets={"heat_ld": zero})),
    }
    reference = jax.jit(lambda s: law.step(s, dt, h, counts=[8]))
    ref = {"T": start["T"]}
    states = {name: {"T": start["T"]} for name in runs}
    err = {name: [] for name in runs}
    for _ in range(steps):
        ref = reference(ref)
        for name, step in runs.items():
            states[name] = step(states[name])
            T = states[name]["T"]
            err[name].append(float(jnp.sqrt(jnp.mean((T - ref["T"]) ** 2))) if bool(jnp.all(jnp.isfinite(T))) else float("inf"))
    for name in runs:
        print(f"  {name:28s} RMSE at the end {err[name][-1]:.3f} K, worst {max(err[name]):.3f} K")

    # ---- and in the engine, under guard: how often a learned step is undone --------------------------------------
    guard = subprocess.run([a.lawc, "--scene", "assets/scenes/heat.scene", "--steps", "100", "--dt", f"{dt}", "--learned", "--backend", "cuda",
                            "--profile"], capture_output=True, text=True)
    undone = sum(int(m) for m in re.findall(r"(\d+) learned step\(s\) undone", guard.stdout))
    print(guard.stdout[-800:])

    os.makedirs("../docs", exist_ok=True)
    with open("../docs/model3_heat.md", "w") as out:
        out.write("# Model 3: a learned discretization under guard\n\n")
        out.write(f"heat.law on the scene assets/scenes/heat.scene, {steps} steps of {dt} s (twice the classical scheme's stable step). "
                  "The reference takes each step in 8 classical substeps.\n\n")
        out.write("| scheme | passes a step | T RMSE at the end (K) | worst (K) |\n|---|---|---|---|\n")
        for name, passes in (("classical, 2 substeps", 2), ("learned, 1 pass", 1), ("untrained learned, 1 pass", 1)):
            out.write(f"| {name} | {passes} | {err[name][-1]:.3f} | {max(err[name]):.3f} |\n")
        out.write(f"\nIn the engine (CUDA, `lawc --learned`), learned steps undone by the guard in the profiled steps: {undone}.\n")
    print("wrote assets/nets/heat_ld.net and docs/model3_heat.md")


if __name__ == "__main__":
    main()

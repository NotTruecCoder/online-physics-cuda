"""CALIBRATION: Heston's five numbers fitted to a price surface by gradient, through the PDE.

The law is heston_adi.law (the ADI scheme, implicit through tridiag_x / tridiag_y). Its IR's JAX twin (tools/lawjax) runs
the same steps as the engine, and JAX differentiates them: tridiag's solve has JAX's own rules, the rest is jnp. One
z-plane a strike, the maturities as states on the way, so one run prices the whole surface, and jax.value_and_grad gives
the loss and its gradient in kappa, theta, sig_v, rho and v0 together (the variance lattice is sig_v times y, so sig_v
moves the lattice too, and the read-out at v0 is a cubic in v0: both differentiable).

  market   Heston prices from the characteristic function at known numbers (finance.heston_put), puts, 9 strikes x 3
           maturities. Then the same with noise of a tenth of a cent on every price.
  fit      L-BFGS-B (scipy) on the summed squared price error, the gradient by jax.grad through every step.
  checks   the gradient against central finite differences of the same loss; the fit against the true numbers and
           against a fit of the characteristic function itself (the best a perfect pricer could do), whose gap is the
           lattice's error.

Writes bench/results/calibration.{json,md}.

    python bench/calibrate.py [--lattices 128:32,256:64]
"""

import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import minimize

from common import BUILD, HERE, lawc, save, write_scene
from finance import K, cell_payoff, heston_put

import jax
import jax.numpy as jnp

LAW = os.path.join(HERE, "laws", "heston_adi.law")
NAMES = ("kappa", "theta", "sig_v", "rho", "v0")
TRUE = {"kappa": 1.5, "theta": 0.04, "sig_v": 0.3, "rho": -0.7, "v0": 0.05}
START = {"kappa": 2.5, "theta": 0.06, "sig_v": 0.45, "rho": -0.3, "v0": 0.03}
BOUNDS = [(0.2, 6.0), (0.01, 0.2), (0.12, 1.0), (-0.98, 0.5), (0.01, 0.2)]
R = 0.025
STRIKES = np.linspace(80.0, 120.0, 9)
MATURITIES = (0.25, 0.5, 1.0)
XW = 3.0


def market(p):
    return np.array([[heston_put(100.0, p["v0"], p["kappa"], p["theta"], p["sig_v"], p["rho"], R, k=k, t=t) for k in STRIKES]
                     for t in MATURITIES])


def pricer(nx, steps, work):
    """The surface as a function of the five numbers, by the law's JAX twin: (maturities, strikes)."""
    from lawjax.ir import load_law

    h = 2 * XW / nx
    ny = int(math.ceil(0.9 / (START["sig_v"] * h)))  # vmax 0.9 at the start; sig_v moves it
    x = math.log(K) - XW + (np.arange(nx) + 0.5) * h
    nz = len(STRIKES)
    scene = write_scene(os.path.join(work, "cal.scene"), (nx, ny, nz, h), [LAW])
    ir = os.path.join(work, "cal_ir.json")
    lawc("--scene", scene, "--ir-json", ir, "--steps", "0")
    law = load_law(ir, BUILD)
    payoff = jnp.asarray(cell_payoff(x, h, STRIKES)[:, None, :] * np.ones((1, ny, 1)), jnp.float32)
    dt = 1.0 / steps
    marks = [int(round(t / dt)) for t in MATURITIES]
    assert all(abs(m * dt - t) < 1e-9 for m, t in zip(marks, MATURITIES)), "maturities must fall on steps"
    j = jnp.arange(ny, dtype=jnp.float32)
    mid = nx // 2  # ln 100 lies on the face between mid - 1 and mid: a cubic through four cells, weights at the midpoint
    wx = jnp.asarray([-1.0, 9.0, 9.0, -1.0], jnp.float32) / 16.0

    def surface(theta):
        kappa, th, sig_v, rho, v0 = (theta[i] for i in range(5))
        shape = (nz, ny, nx)
        state = {"V": payoff, "var": jnp.broadcast_to((sig_v * (j + 0.5) * h)[None, :, None], shape),
                 "kappa": jnp.full(shape, kappa), "theta": jnp.full(shape, th), "sig_v": jnp.full(shape, sig_v),
                 "rho": jnp.full(shape, rho), "r": jnp.full(shape, R, jnp.float32)}

        def body(s, _):
            s = law.step(s, dt, h, counts=[1])
            return s, s["V"]

        _, trail = jax.lax.scan(body, state, None, length=marks[-1])
        V = trail[jnp.asarray(marks) - 1]  # (maturities, strikes, ny, nx)
        row = jnp.tensordot(V[..., mid - 2:mid + 2], wx, axes=([3], [0]))  # at ln 100: (maturities, strikes, ny)
        t = v0 / (sig_v * h) - 0.5  # v0 in rows
        j0 = jnp.clip(jnp.floor(t).astype(jnp.int32), 1, ny - 3)
        f = t - j0
        lag = jnp.stack([-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2, -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6])
        rows = jax.lax.dynamic_slice_in_dim(row, j0 - 1, 4, axis=2)
        return jnp.tensordot(rows, lag, axes=([2], [0]))

    return surface, {"nx": nx, "ny": ny, "nz": nz, "steps_per_year": steps, "cells": nx * ny * nz}


def fit(loss_and_grad, x0):
    history = []

    def fun(x):
        value, grad = loss_and_grad(x)
        history.append(float(value))
        return float(value), np.asarray(grad, np.float64)

    t0 = time.perf_counter()
    res = minimize(fun, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B", bounds=BOUNDS,
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    return res, history, time.perf_counter() - t0


def closed_form_fit(target, start):
    """The yardstick: the characteristic function fitted to the same prices (no lattice), by finite differences."""
    ref = minimize(lambda th: float(np.sum((market(dict(zip(NAMES, th))) - target) ** 2)), start, method="L-BFGS-B",
                   bounds=BOUNDS, options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10})
    return dict(zip(NAMES, ref.x.tolist())), float(np.sqrt(ref.fun / target.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattices", default="128:32,256:64", help="cells along x : steps a year, comma separated")
    args = ap.parse_args()
    lattices = [tuple(int(v) for v in item.split(":")) for item in args.lattices.split(",")]
    start = np.array([START[n] for n in NAMES])
    truth = np.array([TRUE[n] for n in NAMES])
    mkt = market(TRUE)
    rng = np.random.default_rng(3)
    noisy = mkt + rng.normal(0.0, 1e-3, mkt.shape)
    targets = {"exact": mkt, "noisy": noisy}
    out = {"true": TRUE, "start": START, "r": R, "strikes": STRIKES.tolist(), "maturities": list(MATURITIES), "lattices": []}
    out["closed_form"] = {}
    for tag, target in targets.items():
        params, rms = closed_form_fit(target, start)
        out["closed_form"][tag] = {"params": params, "rms_price": rms}
        print(f"closed form fit, {tag}: RMS {rms:.1e}; " + ", ".join(f"{n} {v:.4f}" for n, v in params.items()))
    with tempfile.TemporaryDirectory() as work:
        for nx, steps in lattices:
            surface, lattice = pricer(nx, steps, work)
            entry = {"lattice": lattice}

            def make_loss(target):
                tgt = jnp.asarray(target, jnp.float32)
                return jax.jit(jax.value_and_grad(lambda th: jnp.sum((surface(th) - tgt) ** 2)))

            loss = make_loss(mkt)
            x = jnp.asarray(start, jnp.float32)
            t0 = time.perf_counter()
            jax.block_until_ready(loss(x))
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, grad = jax.block_until_ready(loss(x))
            grad_s = time.perf_counter() - t0
            price_only = jax.jit(lambda th: jnp.sum((surface(th) - jnp.asarray(mkt, jnp.float32)) ** 2))
            jax.block_until_ready(price_only(x))
            t0 = time.perf_counter()
            jax.block_until_ready(price_only(x))
            fwd_s = time.perf_counter() - t0
            # The gradient against central differences of the same float32 loss (steps of 1e-3 of each number's scale).
            fd = []
            for i in range(5):
                e = np.zeros(5, np.float32)
                e[i] = 1e-3 * max(abs(start[i]), 0.05)
                fd.append(float((price_only(x + e) - price_only(x - e)) / (2 * e[i])))
            rel = [abs(float(g) - f) / max(abs(f), 1e-12) for g, f in zip(np.asarray(grad), fd)]
            entry["gradient_check"] = {"jax": np.asarray(grad).tolist(), "central_differences": fd, "relative_difference": rel,
                                       "seconds_value_and_grad": grad_s, "seconds_value": fwd_s, "seconds_compile": compile_s}
            print(f"{nx}: gradient jax {np.asarray(grad)} fd {np.asarray(fd)}; value+grad {grad_s:.2f} s, value {fwd_s:.2f} s")
            at_truth = np.asarray(surface(jnp.asarray(truth, jnp.float32)))
            entry["lattice_error_at_truth"] = {"rms": float(np.sqrt(np.mean((at_truth - mkt) ** 2))), "max": float(np.max(np.abs(at_truth - mkt)))}
            entry["fits"] = {}
            for tag, target in targets.items():
                lg = make_loss(target)
                res, hist, secs = fit(lambda th: lg(jnp.asarray(th, jnp.float32)), start)
                fitted = np.asarray(surface(jnp.asarray(res.x, jnp.float32)))
                entry["fits"][tag] = {"params": dict(zip(NAMES, res.x.tolist())), "iterations": int(res.nit), "evaluations": int(res.nfev),
                                      "seconds": secs, "loss": float(res.fun), "rms_price": float(np.sqrt(np.mean((fitted - target) ** 2))),
                                      "history": hist, "message": str(res.message)}
                print(f"{nx} {tag}: {res.nit} iterations, {secs:.1f} s, RMS {entry['fits'][tag]['rms_price']:.2e}; "
                      + ", ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, res.x)))
            out["lattices"].append(entry)

    md = ["# Calibration: Heston fitted through the PDE by jax.grad\n\n",
          f"Market: puts on S = 100, r = {R}, strikes {STRIKES[0]:g}-{STRIKES[-1]:g} ({len(STRIKES)}), maturities "
          f"{', '.join(f'{t:g}' for t in MATURITIES)}: {mkt.size} prices from the characteristic function at the true numbers, "
          "and the same with noise of a tenth of a cent on each. The pricer: `heston_adi.law` through its JAX twin, the "
          "strikes as z-planes, the maturities read on the way; L-BFGS-B on the summed squared error with the gradient by "
          "`jax.value_and_grad` through every step. Beside it, the characteristic function itself fitted to the same prices "
          "(no lattice: what a perfect pricer would find).\n\n## The gradient\n\nAgainst central differences of the same "
          "float32 loss, at the start:\n\n| lattice | number | jax.grad | central difference | relative difference |\n|---|---|---|---|---|\n"]
    for e in out["lattices"]:
        lat, g = e["lattice"], e["gradient_check"]
        for n, a, b, r in zip(NAMES, g["jax"], g["central_differences"], g["relative_difference"]):
            md.append(f"| {lat['nx']} x {lat['ny']} | {n} | {a:.5g} | {b:.5g} | {r:.1e} |\n")
    md.append("\n| lattice | cells (all strikes) | steps a year | loss | loss and gradient | compile | lattice's RMS error at the true numbers |\n|---|---|---|---|---|---|---|\n")
    for e in out["lattices"]:
        lat, g = e["lattice"], e["gradient_check"]
        md.append(f"| {lat['nx']} x {lat['ny']} | {lat['cells']} | {lat['steps_per_year']} | {g['seconds_value']:.2f} s | "
                  f"{g['seconds_value_and_grad']:.2f} s ({g['seconds_value_and_grad'] / g['seconds_value']:.1f} losses) | "
                  f"{g['seconds_compile']:.0f} s | {e['lattice_error_at_truth']['rms']:.1e} |\n")
    md.append("\nCentral differences would cost 10 losses for the five derivatives; reverse mode costs what the table says, "
              "whatever the number of parameters (CPU, JAX).\n\n## The fit\n\n")
    for tag in targets:
        cf = out["closed_form"][tag]
        md.append(f"{'Exact prices' if tag == 'exact' else 'Prices with noise'}:\n\n| number | true | start | "
                  + " | ".join(f"PDE {e['lattice']['nx']}" for e in out["lattices"]) + " | closed form |\n|---|---|---|"
                  + "---|" * len(out["lattices"]) + "---|\n")
        for n in NAMES:
            md.append(f"| {n} | {TRUE[n]:g} | {START[n]:g} | " + " | ".join(f"{e['fits'][tag]['params'][n]:.4f}" for e in out["lattices"])
                      + f" | {cf['params'][n]:.4f} |\n")
        md.append("| RMS price error | | | " + " | ".join(f"{e['fits'][tag]['rms_price']:.1e}" for e in out["lattices"])
                  + f" | {cf['rms_price']:.1e} |\n")
        md.append("| iterations (seconds) | | | " + " | ".join(f"{e['fits'][tag]['iterations']} ({e['fits'][tag]['seconds']:.0f} s)" for e in out["lattices"])
                  + " | |\n\n")
    md.append("What separates a PDE fit from the true numbers is the lattice's pricing error turned into the numbers "
              "(rho and kappa move most: the surface is least sensitive to them); the table shows how it shrinks with the lattice.\n")
    save("calibration", out, "".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

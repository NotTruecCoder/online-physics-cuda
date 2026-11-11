"""FINANCE: option pricing PDEs written as laws, solved by the same compiler, checked against closed forms.

  black_scholes.law  1-D, x = ln S. A European put against the Black-Scholes formula, refined 64 -> 1024 cells
                     (explicit Euler with dt ~ h^2: expected order 2), on the CPU reference and on CUDA. An American
                     put (the early-exercise projection as a penalty term) against a 20 000-step binomial tree.
                     Greeks: delta and gamma from the lattice; vega and rho by jax.grad through the unrolled solver
                     (tools/lawjax, the IR's JAX twin), against the formula's.
  heston.law         2-D, x = ln S and the variance, with the mixed derivative. A European put against the
                     semi-analytic Heston price (characteristic function, numerical integration), refined.

Also the card time of a step, from lawc --json. Writes bench/results/finance.{json,md}.

    python bench/finance.py
"""

import math
import os
import tempfile

import numpy as np
from scipy import integrate
from scipy.stats import norm

from common import BUILD, HERE, lawc, lawc_json, read_state, save, write_scene, write_state

BS_LAW = os.path.join(HERE, "laws", "black_scholes.law")
HESTON_LAW = os.path.join(HERE, "laws", "heston.law")

K, T = 100.0, 1.0
SIG, R = 0.2, 0.05
HALF_WIDTH = 4.0  # the domain is ln K +- this (S from 1.8 to 5460)
QUOTES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


# ---- closed forms ---------------------------------------------------------------------------------------------------

def bs_put(S, k=K, t=T, sig=SIG, r=R):
    d1 = (np.log(S / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return k * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_greeks(S, k=K, t=T, sig=SIG, r=R):
    d1 = (math.log(S / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return {"delta": norm.cdf(d1) - 1.0, "gamma": norm.pdf(d1) / (S * sig * math.sqrt(t)),
            "vega": S * norm.pdf(d1) * math.sqrt(t), "rho": -k * t * math.exp(-r * t) * norm.cdf(-d2)}


def binomial_american_put(S, n, k=K, t=T, sig=SIG, r=R):
    dt = t / n
    u = math.exp(sig * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    prices = S * u ** (n - 2 * j).astype(np.float64)
    v = np.maximum(k - prices, 0.0)
    for step in range(n, 0, -1):
        prices = prices[:-1] / u
        v = np.maximum(disc * (p * v[:-1] + (1 - p) * v[1:]), k - prices)
    return float(v[0])


def heston_put(S, v0, kappa, theta, sig_v, rho, r, k=K, t=T):
    """Heston (1993) through the characteristic function in the form that does not wrap (Albrecher et al. 2007)."""
    x = math.log(S)

    def phi(u):
        iu = 1j * u
        d = np.sqrt((rho * sig_v * iu - kappa) ** 2 + sig_v ** 2 * (iu + u * u))
        g = (kappa - rho * sig_v * iu - d) / (kappa - rho * sig_v * iu + d)
        e = np.exp(-d * t)
        c = kappa * theta / sig_v ** 2 * ((kappa - rho * sig_v * iu - d) * t - 2.0 * np.log((1 - g * e) / (1 - g)))
        dd = v0 / sig_v ** 2 * (kappa - rho * sig_v * iu - d) * (1 - e) / (1 - g * e)
        return np.exp(iu * (x + r * t) + c + dd)

    lk = math.log(k)
    forward = S * math.exp(r * t)
    p1 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * lk) * phi(u - 1j) / (1j * u * forward)).real, 1e-9, 200, limit=400)[0] / math.pi
    p2 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * lk) * phi(u) / (1j * u)).real, 1e-9, 200, limit=400)[0] / math.pi
    call = S * p1 - k * math.exp(-r * t) * p2
    return call - S + k * math.exp(-r * t)


# ---- the engine -----------------------------------------------------------------------------------------------------

def bs_grid(n):
    h = 2 * HALF_WIDTH / n
    x = math.log(K) - HALF_WIDTH + (np.arange(n) + 0.5) * h
    return x, h


def bs_steps(h, sig=SIG):
    return int(math.ceil(T / (0.4 * h * h / (sig * sig))))  # one substep a step: the projection runs every substep


def run_bs(n, backend, work, american=False):
    x, h = bs_grid(n)
    payoff = np.maximum(K - np.exp(x), 0.0)
    scene = write_scene(os.path.join(work, f"bs_{n}.scene"), (n, 1, 1, h), [BS_LAW])
    start = write_state(os.path.join(work, f"bs_init_{n}"), {"V": payoff.reshape(1, 1, n), "P": payoff.reshape(1, 1, n)}, h)
    steps = bs_steps(h)
    out = os.path.join(work, f"bs_out_{backend}_{n}_{int(american)}")
    lam = steps / T if american else 0.0
    lawc("--scene", scene, "--init", start, "--steps", str(steps), "--dt", repr(T / steps), "--backend", backend,
         "--param", "lam", repr(lam), "--dump", out, "--every", str(steps))
    end, _, _ = read_state(os.path.join(out, f"step_{steps}"))
    return x, end["V"].reshape(-1).astype(np.float64), steps


def at(x, values, S):
    return np.interp(np.log(S), x, values)


def bs_greeks_from_lattice(x, v, S):
    """delta = V_x / S, gamma = (V_xx - V_x) / S^2, central differences on the lattice, interpolated to ln S."""
    h = x[1] - x[0]
    vx = np.gradient(v, h)
    vxx = np.gradient(vx, h)
    return float(at(x, vx, S) / S), float((at(x, vxx, S) - at(x, vx, S)) / (S * S))


def jax_greeks(n, work):
    """Price, vega and rho by jax.grad through the IR's JAX twin, unrolled over every step."""
    import jax
    import jax.numpy as jnp
    from lawjax.ir import load_law

    x, h = bs_grid(n)
    payoff = np.maximum(K - np.exp(x), 0.0).astype(np.float32)
    scene = write_scene(os.path.join(work, f"bsj_{n}.scene"), (n, 1, 1, h), [BS_LAW])
    ir = os.path.join(work, "bs_ir.json")
    lawc("--scene", scene, "--ir-json", ir, "--steps", "0")
    law = load_law(ir, BUILD)
    steps = bs_steps(h)
    dt = T / steps
    state0 = {"V": jnp.asarray(payoff.reshape(1, 1, n)), "P": jnp.asarray(payoff.reshape(1, 1, n))}
    xs = jnp.asarray(x)

    def price(sig, r):
        params = {"sig": sig, "r": r, "lam": jnp.float32(0.0)}

        def body(state, _):
            return law.step(state, dt, h, counts=[1, 1], params=params), None

        final, _ = jax.lax.scan(body, state0, None, length=steps)
        return jnp.interp(jnp.log(100.0), xs, final["V"].reshape(-1))

    value, (vega, rho) = jax.value_and_grad(price, argnums=(0, 1))(jnp.float32(SIG), jnp.float32(R))
    return float(value), float(vega), float(rho)


def heston_grid(h, vmax, sig_v):
    nx = int(round(2 * 3.0 / h))
    ny = int(round(vmax / sig_v / h))
    x = math.log(K) - 3.0 + (np.arange(nx) + 0.5) * h
    v = sig_v * (np.arange(ny) + 0.5) * h
    return x, v, nx, ny


def run_heston(nx_target, backend, work, p):
    h = 6.0 / nx_target
    x, v, nx, ny = heston_grid(h, p["vmax"], p["sig_v"])
    payoff = np.maximum(K - np.exp(x), 0.0)
    V0 = np.broadcast_to(payoff[None, :], (ny, nx))
    var = np.broadcast_to(v[:, None], (ny, nx))
    scene = write_scene(os.path.join(work, f"h_{nx}.scene"), (nx, ny, 1, h), [HESTON_LAW])
    start = write_state(os.path.join(work, f"h_init_{nx}"), {"V": V0.reshape(1, ny, nx), "var": var.reshape(1, ny, nx)}, h)
    steps = int(math.ceil(T / (0.2 * h * h / v[-1])))
    out = os.path.join(work, f"h_out_{backend}_{nx}")
    args = ["--scene", scene, "--init", start, "--steps", str(steps), "--dt", repr(T / steps), "--backend", backend]
    for name in ("sig_v", "kappa", "theta", "rho", "r"):
        args += ["--param", name, repr(p[name])]
    args += ["--param", "var_max", repr(float(v[-1]))]
    lawc(*args, "--dump", out, "--every", str(steps))
    end, _, _ = read_state(os.path.join(out, f"step_{steps}"))
    V = end["V"].reshape(ny, nx).astype(np.float64)
    # Bilinear at (ln 100, v0).
    col = np.array([np.interp(math.log(100.0), x, V[j]) for j in range(ny)])
    price = float(np.interp(p["v0"], v, col))
    timing = lawc_json(*args, "--steps", "50", "--warmup", "5") if backend == "cuda" else None
    return price, steps, nx * ny, timing


def orders(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors[:-1], errors[1:])]


def main():
    results = {}
    md = ["# Finance: option pricing PDEs as laws\n\n",
          f"Put, K = {K:g}, T = {T:g}, sigma = {SIG:g}, r = {R:g}; x = ln S on ln K +- {HALF_WIDTH:g}. "
          "Explicit Euler, one substep a step at dt = 0.4 h^2 / sigma^2.\n\n"]
    exact = bs_put(QUOTES)
    with tempfile.TemporaryDirectory() as work:
        # ---- European put: convergence ---------------------------------------------------------------------------
        ns = [64, 128, 256, 512, 1024]
        md.append("## European put against the Black-Scholes formula\n\n| backend | cells | steps | V(100) | error at 100 | RMS error, S = 80..120 | order |\n|---|---|---|---|---|---|---|\n")
        for backend in ("cpu", "cuda"):
            errs, rows = [], []
            for n in ns:
                x, v, steps = run_bs(n, backend, work)
                got = at(x, v, QUOTES)
                rms = float(np.sqrt(np.mean((got - exact) ** 2)))
                errs.append(rms)
                rows.append({"n": n, "steps": steps, "v100": float(at(x, v, 100.0)), "err100": float(at(x, v, 100.0) - bs_put(100.0)), "rms": rms})
            o = orders(errs)
            results[f"european_{backend}"] = {"rows": rows, "order": o}
            for i, row in enumerate(rows):
                md.append(f"| {backend} | {row['n']} | {row['steps']} | {row['v100']:.5f} | {row['err100']:.2e} | {row['rms']:.2e} | {o[i - 1]:.2f} |\n" if i else
                          f"| {backend} | {row['n']} | {row['steps']} | {row['v100']:.5f} | {row['err100']:.2e} | {row['rms']:.2e} | |\n")
            print(f"european {backend}: RMS {['%.2e' % e for e in errs]} orders {['%.2f' % q for q in o]}")
        md.append(f"\nFormula: V(100) = {bs_put(100.0):.5f}.\n\n")

        # ---- American put --------------------------------------------------------------------------------------
        tree = 0.5 * (binomial_american_put(100.0, 20000) + binomial_american_put(100.0, 20001))
        md.append("## American put against a binomial tree\n\nReference: CRR tree, mean of 20 000 and 20 001 steps "
                  f"(the tree's odd-even wobble), V(100) = {tree:.5f}; the European value is {bs_put(100.0):.5f}.\n\n"
                  "| backend | cells | V(100) | error |\n|---|---|---|---|\n")
        results["american_tree"] = tree
        for backend in ("cpu", "cuda"):
            rows = []
            for n in (256, 512, 1024):
                x, v, steps = run_bs(n, backend, work, american=True)
                got = float(at(x, v, 100.0))
                rows.append({"n": n, "v100": got, "err": got - tree})
                md.append(f"| {backend} | {n} | {got:.5f} | {got - tree:+.2e} |\n")
            results[f"american_{backend}"] = rows
            print(f"american {backend}: " + ", ".join(f"{r['n']}: {r['v100']:.5f}" for r in rows) + f" (tree {tree:.5f})")

        # ---- Greeks ----------------------------------------------------------------------------------------------
        n = 512
        x, v, _ = run_bs(n, "cpu", work)
        delta, gamma = bs_greeks_from_lattice(x, v, 100.0)
        value, vega, rho = jax_greeks(n, work)
        ref = bs_put_greeks(100.0)
        results["greeks"] = {"n": n, "delta": delta, "gamma": gamma, "vega": vega, "rho": rho, "jax_price": value,
                             "engine_price": float(at(x, v, 100.0)), "exact": ref}
        md.append(f"\n## Greeks at S = 100 ({n} cells)\n\nDelta and gamma from the lattice (central differences in x = ln S); "
                  "vega and rho by `jax.grad` through every step of the IR's JAX twin (tools/lawjax).\n\n"
                  "| greek | engine | formula | relative error |\n|---|---|---|---|\n")
        for name, got in (("delta", delta), ("gamma", gamma), ("vega", vega), ("rho", rho)):
            md.append(f"| {name} | {got:.5f} | {ref[name]:.5f} | {abs(got - ref[name]) / abs(ref[name]):.1e} |\n")
        md.append(f"\nThe JAX twin's price {value:.5f} against the engine's {float(at(x, v, 100.0)):.5f}.\n")
        print(f"greeks: delta {delta:.5f} ({ref['delta']:.5f}) gamma {gamma:.5f} ({ref['gamma']:.5f}) "
              f"vega {vega:.4f} ({ref['vega']:.4f}) rho {rho:.4f} ({ref['rho']:.4f}); jax price {value:.5f}")

        # ---- Heston ----------------------------------------------------------------------------------------------
        p = {"sig_v": 0.3, "kappa": 1.5, "theta": 0.04, "rho": -0.9, "r": 0.025, "v0": 0.04, "vmax": 0.6}
        check = heston_put(100.0, 0.04, 1.5, 0.04, 1e-4, 0.0, R)
        assert abs(check - bs_put(100.0)) < 1e-4, (check, bs_put(100.0))  # vol of vol -> 0: Black-Scholes at sqrt(theta)
        ref_h = heston_put(100.0, p["v0"], p["kappa"], p["theta"], p["sig_v"], p["rho"], p["r"])
        md.append(f"\n## Heston, European put against the semi-analytic price\n\nkappa {p['kappa']}, theta {p['theta']}, "
                  f"sigma_v {p['sig_v']}, rho {p['rho']}, r {p['r']}, v0 {p['v0']}; x on ln K +- 3, v on [0, {p['vmax']}]. "
                  f"Reference (characteristic function) V(100, v0) = {ref_h:.5f}.\n\n"
                  "| backend | cells | steps | V(100, v0) | error | card ms a step |\n|---|---|---|---|---|---|\n")
        rows = []
        for nx in (64, 128, 256, 512):
            price, steps, cells, timing = run_heston(nx, "cuda", work, p)
            card = timing["gpu_ms"] if timing else None
            rows.append({"nx": nx, "cells": cells, "steps": steps, "price": price, "err": price - ref_h, "timing": timing})
            md.append(f"| cuda | {cells} | {steps} | {price:.5f} | {price - ref_h:+.2e} | {card if card is None else f'{card:.3f}'} |\n")
            print(f"heston {nx}: {price:.5f} (ref {ref_h:.5f}) in {steps} steps")
        price_cpu, _, _, _ = run_heston(128, "cpu", work, p)
        results["heston"] = {"params": p, "reference": ref_h, "rows": rows, "cpu_128": price_cpu}
        md.append(f"\nCPU reference at 128: {price_cpu:.5f} (CUDA {rows[1]['price']:.5f}).\n")
    save("finance", results, "".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

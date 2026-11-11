"""FINANCE: option pricing PDEs written as laws, solved by the same compiler, checked against closed forms.

  black_scholes.law  1-D, x = ln S. A European put against the Black-Scholes formula, refined 64 -> 1024 cells
                     (explicit Euler with dt ~ h^2: expected order 2), on the CPU reference and on CUDA. An American
                     put (the early-exercise projection as a penalty term) against a 20 000-step binomial tree.
                     Greeks: delta and gamma from the lattice; vega and rho by jax.grad through the unrolled solver
                     (tools/lawjax, the IR's JAX twin), against the formula's.
  heston.law         2-D, x = ln S and the variance, with the mixed derivative, every derivative second order. A
                     European put against the semi-analytic Heston price (characteristic function, numerical
                     integration), refined 64 -> 2048 cells along x, with the observed order; heston_first_order.law
                     (the scheme it replaced: upwind drift, harmonic face mean) beside it.

Also the card time of a step, from lawc --json. Writes bench/results/finance.{json,md}.

    python bench/finance.py
    python bench/finance.py --cpu     # no card (CI): the CPU reference on smaller lattices, orders asserted;
                                      # writes bench/results/finance_cpu.{json,md}
"""

import math
import os
import sys
import tempfile

import numpy as np
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from common import BUILD, HERE, lawc, lawc_json, read_state, save, write_scene, write_state

BS_LAW = os.path.join(HERE, "laws", "black_scholes.law")
HESTON_LAW = os.path.join(HERE, "laws", "heston.law")
HESTON_FIRST_LAW = os.path.join(HERE, "laws", "heston_first_order.law")

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
    """delta = V_x / S, gamma = (V_xx - V_x) / S^2, interpolated to ln S. V_x central (2h wide), V_xx the three-point
    stencil (not np.gradient twice: that is a 4h-wide stencil, four times the error)."""
    h = x[1] - x[0]
    vx = np.gradient(v, h)
    vxx = np.zeros_like(v)
    vxx[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (h * h)
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


def heston_grid(h, vmax, sig_v, xw=3.0):
    nx = int(round(2 * xw / h))
    ny = int(round(vmax / sig_v / h))
    x = math.log(K) - xw + (np.arange(nx) + 0.5) * h
    v = sig_v * (np.arange(ny) + 0.5) * h
    return x, v, nx, ny


def heston_max_dt(h, p, var_max):
    """The law's max_dt (heston.law): the diffusion's limit or the drift's CFL, the smaller."""
    drift = abs(p["r"]) + 0.5 * var_max + (p["kappa"] * max(p["theta"], var_max - p["theta"]) + 0.5 * p["sig_v"] ** 2) / p["sig_v"]
    return min(0.2 * h * h / var_max, 0.5 * h / drift)


def run_heston(nx_target, backend, work, p, law=HESTON_LAW):
    h = 6.0 / nx_target  # nx_target cells on ln K +- 3, whatever the domain's width
    x, v, nx, ny = heston_grid(h, p["vmax"], p["sig_v"], p.get("xw", 3.0))
    payoff = np.maximum(K - np.exp(x), 0.0)
    V0 = np.broadcast_to(payoff[None, :], (ny, nx))
    var = np.broadcast_to(v[:, None], (ny, nx))
    tag = os.path.splitext(os.path.basename(law))[0]
    scene = write_scene(os.path.join(work, f"h_{tag}_{nx}.scene"), (nx, ny, 1, h), [law])
    start = write_state(os.path.join(work, f"h_init_{nx}"), {"V": V0.reshape(1, ny, nx), "var": var.reshape(1, ny, nx)}, h)
    steps = int(math.ceil(T / heston_max_dt(h, p, float(v[-1]))))  # one substep a step
    out = os.path.join(work, f"h_out_{tag}_{backend}_{nx}")
    args = ["--scene", scene, "--init", start, "--steps", str(steps), "--dt", repr(T / steps), "--backend", backend]
    for name in ("sig_v", "kappa", "theta", "rho", "r"):
        args += ["--param", name, repr(p[name])]
    args += ["--param", "var_max", repr(float(v[-1]))]
    lawc(*args, "--dump", out, "--every", str(steps))
    end, _, _ = read_state(os.path.join(out, f"step_{steps}"))
    V = end["V"].reshape(ny, nx).astype(np.float64)
    # Bicubic at (ln 100, v0): its error is O(h^4). A bilinear one is O(h^2) with a coefficient that jumps with where v0
    # falls in its cell (V_vv is about -1250 at v = 0.04): as large as the scheme's error, and it hid the order.
    price = float(RectBivariateSpline(v, x, V, kx=3, ky=3)(p["v0"], math.log(100.0))[0, 0])
    timing = lawc_json(*args, "--steps", "50", "--warmup", "5") if backend == "cuda" else None
    return price, steps, nx * ny, timing


def heston_float64(nx_target, p):
    """heston.law's scheme written again in NumPy, in float64: the same stencils, edges and steps as the engine (it
    agrees with the engine's float32 to 1e-5 where rounding is small). It shows the scheme's own order where the
    engine's float32 state stops showing it: over 10^4+ steps an increment of a slowly moving cell is rounded the same
    way step after step, and that adds up."""
    h = 6.0 / nx_target
    x, v, nx, ny = heston_grid(h, p["vmax"], p["sig_v"])
    V = np.broadcast_to(np.maximum(K - np.exp(x), 0.0)[None, :], (ny, nx)).copy()
    var = np.broadcast_to(v[:, None], (ny, nx))
    steps = int(math.ceil(T / heston_max_dt(h, p, float(v[-1]))))
    dt = T / steps
    s, kap, th, rho, r = (p[n] for n in ("sig_v", "kappa", "theta", "rho", "r"))
    iy, ix = np.arange(ny), np.arange(nx)

    def sh(a, dy, dx):  # a at (y + dy, x + dx), the edge cell repeating
        return a[np.ix_(np.clip(iy + dy, 0, ny - 1), np.clip(ix + dx, 0, nx - 1))]

    wx = np.full(nx, 2.0 * h); wx[[0, -1]] = h  # gx/gy: one-sided at an edge, as the engine's
    wy = np.full(ny, 2.0 * h); wy[[0, -1]] = h

    def gx(a):
        return (sh(a, 0, 1) - sh(a, 0, -1)) / wx[None, :]

    def gy(a):
        return (sh(a, 1, 0) - sh(a, -1, 0)) / wy[:, None]

    b = kap * (th - var) / s - 0.5 * s
    bp, bm = np.maximum(b, 0.0), np.maximum(-b, 0.0)
    for _ in range(steps):
        rate = (0.5 * var * (sh(V, 0, 1) - 2 * V + sh(V, 0, -1)) / (h * h)
                + 0.5 * ((var + 0.5 * s * h) * (sh(V, 1, 0) - V) - (var - 0.5 * s * h) * (V - sh(V, -1, 0))) / (h * h)
                + rho * var * gx(gy(V)) + (r - 0.5 * var) * gx(V)
                + bp * (4 * sh(V, 1, 0) - sh(V, 2, 0) - 3 * V) / (2 * h) - bm * (3 * V - 4 * sh(V, -1, 0) + sh(V, -2, 0)) / (2 * h)
                - r * V)
        V = V + dt * rate
    return float(RectBivariateSpline(v, x, V, kx=3, ky=3)(p["v0"], math.log(100.0))[0, 0])


def heston_surface(nx_target, strikes, quarters, work, p):
    """A price surface in one run: strikes along the lattice's z (heston.law couples nothing along z, so each plane is
    its own option: the payoff is the plane's), maturities as the states dumped on the way (tau = time to maturity).
    Returns the prices (strike x maturity), the maturities, the step count and the card time of a step."""
    h = 6.0 / nx_target
    x, v, nx, ny = heston_grid(h, p["vmax"], p["sig_v"])
    nz = len(strikes)
    # The payoff averaged over each cell, not taken at its centre: a strike's kink falls inside a cell (only K = 100 is
    # on a face), and sampling it there is a first-order error.
    ks = np.asarray(strikes)[:, None]
    lo = np.minimum(x[None, :] - 0.5 * h, np.log(ks))
    hi = np.minimum(x[None, :] + 0.5 * h, np.log(ks))
    V0 = ((ks * (hi - lo) - (np.exp(hi) - np.exp(lo))) / h)[:, None, :] * np.ones((1, ny, 1))
    var = np.broadcast_to(v[None, :, None], (nz, ny, nx))
    scene = write_scene(os.path.join(work, f"hs_{nx}_{nz}.scene"), (nx, ny, nz, h), [HESTON_LAW])
    start = write_state(os.path.join(work, f"hs_init_{nx}_{nz}"), {"V": V0, "var": var}, h)
    every = int(math.ceil(T / quarters / heston_max_dt(h, p, float(v[-1]))))
    steps = every * quarters
    out = os.path.join(work, f"hs_out_{nx}_{nz}")
    args = ["--scene", scene, "--init", start, "--steps", str(steps), "--dt", repr(T / steps), "--backend", "cuda"]
    for name in ("sig_v", "kappa", "theta", "rho", "r"):
        args += ["--param", name, repr(p[name])]
    args += ["--param", "var_max", repr(float(v[-1]))]
    lawc(*args, "--dump", out, "--every", str(every))
    prices = np.zeros((nz, quarters))
    for q in range(quarters):
        end, _, _ = read_state(os.path.join(out, f"step_{every * (q + 1)}"))
        V = end["V"].astype(np.float64)
        for z in range(nz):
            prices[z, q] = RectBivariateSpline(v, x, V[z], kx=3, ky=3)(p["v0"], math.log(100.0))[0, 0]
    timing = lawc_json(*args, "--steps", "50", "--warmup", "5")
    return prices, [T * (q + 1) / quarters for q in range(quarters)], steps, timing["gpu_ms"]


def orders(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors[:-1], errors[1:])]


def order_cells(o, i):
    return f"{o[i - 1]:.2f}" if i else ""


def main(cpu_only=False):
    """cpu_only: no card (CI). The CPU reference alone, on lattices it runs in seconds, and the orders asserted."""
    results = {}
    md = ["# Finance: option pricing PDEs as laws\n\n",
          f"Put, K = {K:g}, T = {T:g}, sigma = {SIG:g}, r = {R:g}; x = ln S on ln K +- {HALF_WIDTH:g}. "
          "Explicit Euler, one substep a step at dt = 0.4 h^2 / sigma^2.\n\n"]
    if cpu_only:
        md.append("CPU reference only (no card): `python bench/finance.py --cpu`.\n\n")
    exact = bs_put(QUOTES)
    backends = ("cpu",) if cpu_only else ("cpu", "cuda")
    with tempfile.TemporaryDirectory() as work:
        # ---- European put: convergence ---------------------------------------------------------------------------
        ns = [64, 128, 256, 512] if cpu_only else [64, 128, 256, 512, 1024]
        md.append("## European put against the Black-Scholes formula\n\n"
                  "| backend | cells | steps | V(100) | error at 100 | order | RMS error, S = 80..120 | order |\n"
                  "|---|---|---|---|---|---|---|---|\n")
        for backend in backends:
            errs, errs100, rows = [], [], []
            for n in ns:
                x, v, steps = run_bs(n, backend, work)
                got = at(x, v, QUOTES)
                rms = float(np.sqrt(np.mean((got - exact) ** 2)))
                err100 = float(at(x, v, 100.0) - bs_put(100.0))
                errs.append(rms)
                errs100.append(abs(err100))
                rows.append({"n": n, "steps": steps, "v100": float(at(x, v, 100.0)), "err100": err100, "rms": rms})
            o, o100 = orders(errs), orders(errs100)
            results[f"european_{backend}"] = {"rows": rows, "order": o, "order100": o100}
            for i, row in enumerate(rows):
                md.append(f"| {backend} | {row['n']} | {row['steps']} | {row['v100']:.5f} | {row['err100']:.2e} | "
                          f"{order_cells(o100, i)} | {row['rms']:.2e} | {order_cells(o, i)} |\n")
            print(f"european {backend}: RMS {['%.2e' % e for e in errs]} orders {['%.2f' % q for q in o]}")
            if cpu_only:
                assert min(o) > 1.7, f"Black-Scholes: order {o}, expected 2"
        md.append(f"\nFormula: V(100) = {bs_put(100.0):.5f}.\n\n")

        # ---- American put --------------------------------------------------------------------------------------
        tree = 0.5 * (binomial_american_put(100.0, 20000) + binomial_american_put(100.0, 20001))
        md.append("## American put against a binomial tree\n\nReference: CRR tree, mean of 20 000 and 20 001 steps "
                  f"(the tree's odd-even wobble), V(100) = {tree:.5f}; the European value is {bs_put(100.0):.5f}.\n\n"
                  "| backend | cells | V(100) | error |\n|---|---|---|---|\n")
        results["american_tree"] = tree
        for backend in backends:
            rows = []
            for n in ((256, 512) if cpu_only else (256, 512, 1024)):
                x, v, steps = run_bs(n, backend, work, american=True)
                got = float(at(x, v, 100.0))
                rows.append({"n": n, "v100": got, "err": got - tree})
                md.append(f"| {backend} | {n} | {got:.5f} | {got - tree:+.2e} |\n")
            results[f"american_{backend}"] = rows
            print(f"american {backend}: " + ", ".join(f"{r['n']}: {r['v100']:.5f}" for r in rows) + f" (tree {tree:.5f})")
            if cpu_only:
                assert abs(rows[-1]["err"]) < 1e-2, f"American put {rows[-1]['v100']} against the tree's {tree}"

        # ---- Greeks ----------------------------------------------------------------------------------------------
        n = 256 if cpu_only else 512
        x, v, _ = run_bs(n, "cpu", work)
        delta, gamma = bs_greeks_from_lattice(x, v, 100.0)
        value, vega, rho = jax_greeks(n, work)
        ref = bs_put_greeks(100.0)
        results["greeks"] = {"n": n, "delta": delta, "gamma": gamma, "vega": vega, "rho": rho, "jax_price": value,
                             "engine_price": float(at(x, v, 100.0)), "exact": ref}
        md.append(f"\n## Greeks at S = 100 ({n} cells)\n\nDelta and gamma from the lattice (V_x central, V_xx the "
                  "three-point stencil, in x = ln S); vega and rho by `jax.grad` through every step of the IR's JAX twin "
                  "(tools/lawjax).\n\n| greek | engine | formula | relative error |\n|---|---|---|---|\n")
        greeks = (("delta", delta), ("gamma", gamma), ("vega", vega), ("rho", rho))
        for name, got in greeks:
            md.append(f"| {name} | {got:.5f} | {ref[name]:.5f} | {abs(got - ref[name]) / abs(ref[name]):.1e} |\n")
        md.append(f"\nThe JAX twin's price {value:.5f} against the engine's {float(at(x, v, 100.0)):.5f}.\n")
        print(f"greeks: delta {delta:.5f} ({ref['delta']:.5f}) gamma {gamma:.5f} ({ref['gamma']:.5f}) "
              f"vega {vega:.4f} ({ref['vega']:.4f}) rho {rho:.4f} ({ref['rho']:.4f}); jax price {value:.5f}")
        if cpu_only:
            worst = max(abs(got - ref[name]) / abs(ref[name]) for name, got in greeks)
            assert worst < 2e-2, f"a greek {worst:.1e} off"

        # ---- Heston ----------------------------------------------------------------------------------------------
        p = {"sig_v": 0.3, "kappa": 1.5, "theta": 0.04, "rho": -0.9, "r": 0.025, "v0": 0.04, "vmax": 0.6}
        check = heston_put(100.0, 0.04, 1.5, 0.04, 1e-4, 0.0, R)
        assert abs(check - bs_put(100.0)) < 1e-4, (check, bs_put(100.0))  # vol of vol -> 0: Black-Scholes at sqrt(theta)
        ref_h = heston_put(100.0, p["v0"], p["kappa"], p["theta"], p["sig_v"], p["rho"], p["r"])
        hb = "cpu" if cpu_only else "cuda"
        md.append(f"\n## Heston, European put against the semi-analytic price\n\nkappa {p['kappa']}, theta {p['theta']}, "
                  f"sigma_v {p['sig_v']}, rho {p['rho']}, r {p['r']}, v0 {p['v0']}; x on ln K +- 3, v on [0, {p['vmax']}]. "
                  f"Reference (characteristic function) V(100, v0) = {ref_h:.5f}. `heston.law`: every derivative second "
                  "order (three-point V_xx, the y flux with exact face variances, second-order upwind drift); "
                  "`heston_first_order.law`: the scheme it replaced (first-order upwind drift, harmonic face mean). "
                  "Order: log2 of the ratio of successive errors against the reference. Self-convergence order: "
                  "log2((V_n - V_2n) / (V_2n - V_4n)), the lattice against itself, which the truncated domain (the same at "
                  "every lattice) does not enter.\n\n"
                  "| law | backend | cells along x | cells | steps | V(100, v0) | error | order | self-convergence order | card ms a step |\n"
                  "|---|---|---|---|---|---|---|---|---|---|\n")
        heston = {"params": p, "reference": ref_h}
        sizes = {"heston": (32, 64, 128) if cpu_only else (64, 128, 256, 512, 1024, 2048),
                 "heston_first_order": (32, 64, 128) if cpu_only else (64, 128, 256, 512)}
        for law, name in ((HESTON_LAW, "heston"), (HESTON_FIRST_LAW, "heston_first_order")):
            rows = []
            for nx in sizes[name]:
                price, steps, cells, timing = run_heston(nx, hb, work, p, law)
                rows.append({"nx": nx, "cells": cells, "steps": steps, "price": price, "err": price - ref_h, "timing": timing})
                print(f"{name} {nx}: {price:.5f} (ref {ref_h:.5f}, error {price - ref_h:+.2e}) in {steps} steps")
            o = orders([abs(r["err"]) for r in rows])
            diffs = [abs(a["price"] - b["price"]) for a, b in zip(rows[:-1], rows[1:])]
            self_o = orders(diffs)  # self_o[i] belongs to rows[i + 2]
            for i, row in enumerate(rows):
                card = f"{row['timing']['gpu_ms']:.3f}" if row["timing"] else ""
                so = f"{self_o[i - 2]:.2f}" if i >= 2 else ""
                md.append(f"| {name} | {hb} | {row['nx']} | {row['cells']} | {row['steps']} | {row['price']:.5f} | "
                          f"{row['err']:+.2e} | {order_cells(o, i)} | {so} | {card} |\n")
            heston[name] = {"rows": rows, "order": o, "self_order": self_o}
            print(f"{name}: orders {['%.2f' % q for q in o]}, self-convergence {['%.2f' % q for q in self_o]}")
        # The same scheme in float64 (NumPy): the scheme's order without float32's rounding.
        f64 = []
        for nx in ((32, 64, 128) if cpu_only else (64, 128, 256, 512)):
            price = heston_float64(nx, p)
            f64.append({"nx": nx, "price": price, "err": price - ref_h})
            print(f"heston.law in float64 {nx}: {price:.6f} (error {price - ref_h:+.2e})")
        o = orders([abs(r["err"]) for r in f64])
        for i, row in enumerate(f64):
            md.append(f"| heston, NumPy float64 | cpu | {row['nx']} | | | {row['price']:.5f} | {row['err']:+.2e} | {order_cells(o, i)} | | |\n")
        heston["heston_float64"] = {"rows": f64, "order": o}
        if not cpu_only:
            # Away from var = 0 (v0 = 0.2, 7+ cells up at 64) both schemes are in their asymptotic range: the first
            # scheme's errors halve (order 1), the second's fall four times (order 2) until float32's rounding floor.
            p2 = dict(p, v0=0.2)
            ref2 = heston_put(100.0, p2["v0"], p2["kappa"], p2["theta"], p2["sig_v"], p2["rho"], p2["r"])
            md.append(f"\nThe same at v0 = 0.2, away from var = 0 (reference {ref2:.5f}):\n\n"
                      "| law | cells along x | V(100, 0.2) | error | order |\n|---|---|---|---|---|\n")
            away = {}
            for law, name in ((HESTON_LAW, "heston"), (HESTON_FIRST_LAW, "heston_first_order"), (None, "heston, NumPy float64")):
                rows = []
                for nx in ((128, 256, 512) if law is None else (64, 128, 256, 512)):
                    price = heston_float64(nx, p2) if law is None else run_heston(nx, "cuda", work, p2, law)[0]
                    rows.append({"nx": nx, "price": price, "err": price - ref2})
                o2 = orders([abs(r["err"]) for r in rows])
                for i, row in enumerate(rows):
                    md.append(f"| {name} | {row['nx']} | {row['price']:.5f} | {row['err']:+.2e} | {order_cells(o2, i)} |\n")
                away[name] = {"rows": rows, "order": o2}
                print(f"{name} at v0 = 0.2: errors {['%.2e' % r['err'] for r in rows]} orders {['%.2f' % q for q in o2]}")
            heston["v0_0.2"] = {"reference": ref2, **away}
        if cpu_only:
            # Coarse lattices are not yet second order (v0 lies a few cells from var = 0): what holds there is that the
            # engine is the scheme (to float32 rounding, small at a few thousand steps) and that the error falls.
            for a, b in zip(heston["heston"]["rows"], f64):
                assert abs(a["price"] - b["price"]) < 1e-4, f"Heston at {a['nx']}: engine {a['price']} against the float64 scheme {b['price']}"
            errs = [abs(r["err"]) for r in heston["heston"]["rows"]]
            assert errs[-1] < errs[0] / 3, f"Heston: errors {errs}"
        else:
            price_cpu, _, _, _ = run_heston(128, "cpu", work, p)
            heston["cpu_128"] = price_cpu
            md.append(f"\nCPU reference at 128: {price_cpu:.5f} (CUDA {heston['heston']['rows'][1]['price']:.5f}).\n")

            # ---- a surface in one run: strikes along z, maturities on the way --------------------------------------
            nx, quarters = 512, 4
            strikes = list(np.linspace(80.0, 120.0, 64))
            prices, taus, steps, card = heston_surface(nx, strikes, quarters, work, p)
            _, _, _, one = heston_surface(nx, [100.0], quarters, work, p)
            exact_s = np.array([[heston_put(100.0, p["v0"], p["kappa"], p["theta"], p["sig_v"], p["rho"], p["r"], k=k, t=t)
                                 for t in taus] for k in strikes])
            err = prices - exact_s
            worst_k, worst_t = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
            heston["surface"] = {"nx": nx, "strikes": strikes, "maturities": taus, "steps": steps, "card_ms": card,
                                 "card_ms_one": one, "max_abs_err": float(np.max(np.abs(err))),
                                 "rms_err": float(np.sqrt(np.mean(err ** 2))), "errors": err.tolist(),
                                 "rms_by_maturity": [float(np.sqrt(np.mean(err[:, q] ** 2))) for q in range(quarters)]}
            md.append(f"\n### A price surface in one run\n\n{len(strikes)} strikes (80 to 120) along the lattice's z, "
                      f"{quarters} maturities ({', '.join(f'{t:g}' for t in taus)}) as states dumped on the way: "
                      f"{len(strikes) * quarters} Heston puts from one run of {steps} steps at {nx} cells along x. Against "
                      f"the characteristic function: RMS error {heston['surface']['rms_err']:.1e} (by maturity "
                      f"{', '.join(f'{e:.1e}' for e in heston['surface']['rms_by_maturity'])}), worst "
                      f"{heston['surface']['max_abs_err']:.1e} at K = {strikes[worst_k]:.1f}, T = {taus[worst_t]:g} (the "
                      "in-the-money strikes, K > 100, at the shorter maturities are the worst; their error falls about 3x "
                      "from 256 to 512 cells). Payoffs are averaged over each cell (a kink inside a cell sampled at its "
                      f"centre is a first-order error). Card time a step {card:.3f} ms for the {len(strikes)} strikes "
                      f"against {one:.3f} ms for one: {card / len(strikes) * steps:.1f} ms of card time a strike (its "
                      f"{quarters} maturities) in the batch, against {one * steps:.1f} ms for a strike run alone.\n")
            print(f"surface: {len(strikes)} x {quarters} puts, RMS {heston['surface']['rms_err']:.1e}, card {card:.3f} ms a step "
                  f"(one strike {one:.3f})")
        results["heston"] = heston
    save("finance_cpu" if cpu_only else "finance", results, "".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main(cpu_only="--cpu" in sys.argv[1:])

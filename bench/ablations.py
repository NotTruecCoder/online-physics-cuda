"""ABLATIONS: are the guarantees in the architecture, or in the training?

The two learned operators of the language are built so that some properties hold for ANY weights. Which piece gives
which property:

  flux form          F changes only by the difference of face fluxes (what leaves a cell enters its neighbour):
                     the total of F is conserved (up to float rounding);
  antisymmetry       a face flux is half the difference of the net both ways round, r(a, b) = -r(b, a): the scheme does
                     not care which way the lattice is laid out (mirror equivariance);
  limiter            (flux) the same per-face limit on both sides keeps every cell inside F's declared range in a step;
  mean removal       (ldiffuse) the 4 stencil weights sum to zero, so the flux sees only differences: adding a
                     constant to the field adds it to the result (offset invariance; consistency of the stencil).

Each piece is taken out in turn, with random untrained weights (20 draws, outputs scaled so they matter), in the JAX
twin of the engine (tools/lawjax, parity 1e-7 with the C++ reference). flux: 200 steps from a burning-front state;
ldiffuse: 20 steps at twice the explicit step from a hot spot. Worst draw reported.

    python bench/ablations.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from common import BUILD, save
from lawjax.ir import divergence, limit_face, sh
from lawjax.nets import forward, init_net

DRAWS = 20
LO, HI = 0.0, 6000.0
ARRAY_AXIS = {0: 2, 1: 1, 2: 0}


def random_net(name, sizes, seed, scale):
    net = init_net(name, sizes, "tanh", seed=seed)
    rng = np.random.default_rng(seed + 1000)
    w, b = net.weights[-1]
    net.weights[-1] = (rng.normal(0, 1, w.shape).astype(np.float32), rng.normal(0, 1, b.shape).astype(np.float32))
    net.out_scale = scale
    return net


def open_faces(shape, ax):
    n = shape[ARRAY_AXIS[ax]]
    view = [1, 1, 1]
    view[ARRAY_AXIS[ax]] = n
    return jnp.arange(n).reshape(view) + 1 < n


def flux_rate(net, F, aux, dt, h, form="flux", antisymmetric=True, limited=True):
    shape = F.shape
    if form == "cellwise":  # a learned source in each cell: d F = sum over axes of g(cell, neighbour)
        total = 0.0
        for ax in range(3):
            total = total + forward(net, net.params(), jnp.stack([F, aux, sh(F, ax, 1), sh(aux, ax, 1)], -1))[..., 0]
        return total / h
    faces = []
    for ax in range(3):
        Fb, Ab = sh(F, ax, 1), sh(aux, ax, 1)
        gij = forward(net, net.params(), jnp.stack([F, aux, Fb, Ab], -1))[..., 0]
        r = 0.5 * (gij - forward(net, net.params(), jnp.stack([Fb, Ab, F, aux], -1))[..., 0]) if antisymmetric else gij
        if limited:
            r = limit_face(r, F, Fb, LO, HI, dt, h)
        faces.append(jnp.where(open_faces(shape, ax), r, 0.0))
    return divergence(faces, shape, h)


def ldiffuse_rate(net, k, F, dt, h, antisymmetric=True, consistent=True):
    shape = F.shape
    base = jnp.asarray([0.0, -1.0, 1.0, 0.0], jnp.float32)
    faces = []
    for ax in range(3):
        w = jnp.stack([sh(F, ax, -1), F, sh(F, ax, 1), sh(F, ax, 2)], -1)

        def raw(window):
            centre = 0.5 * (window[..., 1] + window[..., 2])
            spread = jnp.maximum(1e-6, jnp.max(jnp.abs(window - centre[..., None]), -1))
            x = jnp.concatenate([(window - centre[..., None]) / spread[..., None], jnp.full(window.shape[:-1] + (1,), k * dt / (h * h))], -1)
            y = forward(net, net.params(), x)
            if consistent:
                y = y - jnp.mean(y, -1, keepdims=True)
            return -k / h * jnp.sum((base + y) * window, -1)

        face = 0.5 * (raw(w) - raw(w[..., ::-1])) if antisymmetric else raw(w)
        faces.append(jnp.where(open_faces(shape, ax), face, 0.0))
    return divergence(faces, shape, h)


def total(T):
    return float(np.sum(np.asarray(T, np.float64)))


def main():
    h, dt = 1.0, 0.05
    x = np.arange(128, dtype=np.float32)
    T0 = jnp.asarray(np.broadcast_to(290.0 + 1500.0 * np.exp(-((x - 20) / 6.0) ** 2), (1, 8, 128)).astype(np.float32))
    fuel = jnp.asarray(np.broadcast_to(0.8 * (x > 25) + 0.1 * np.sin(x), (1, 8, 128)).astype(np.float32))
    results, rows = {}, []
    for name, kw in (("full: flux form, antisymmetric, limited", {}),
                     ("no limiter", dict(limited=False)),
                     ("no antisymmetry", dict(antisymmetric=False)),
                     ("no flux form (learned source per cell)", dict(form="cellwise", limited=False))):
        drift, outside, mirror = [], [], []
        for d in range(DRAWS):
            net = random_net("f", [4, 32, 32, 1], d, 400.0)
            net.in_shift = np.asarray([800.0, 0.5, 800.0, 0.5], np.float32)
            net.in_scale = np.asarray([1 / 600.0, 2.0, 1 / 600.0, 2.0], np.float32)
            step = jax.jit(lambda T, A: T + dt * flux_rate(net, T, A, dt, h, **kw))
            one = step(T0, fuel)
            flipped = step(T0[..., ::-1], fuel[..., ::-1])
            mirror.append(float(jnp.max(jnp.abs(flipped[..., ::-1] - one))))
            T, bad = T0, 0
            for _ in range(200):
                T = step(T, fuel)
                bad = max(bad, int(jnp.sum((T < LO) | (T > HI) | ~jnp.isfinite(T))))
            finite = bool(np.all(np.isfinite(np.asarray(T))))
            drift.append(abs(total(T) - total(T0)) / total(T0) if finite else float("inf"))
            outside.append(bad)
        r = {"total_drift": float(np.max(drift)), "draws_out_of_range": int(np.sum(np.asarray(outside) > 0)), "mirror_error_K": float(np.max(mirror))}
        results["flux / " + name] = r
        rows.append(("flux", name, r["total_drift"], f"{r['draws_out_of_range']} / {DRAWS}", r["mirror_error_K"], "", ""))

    k, dt2 = 1.0, 0.3
    g = np.arange(48, dtype=np.float32)
    Th = jnp.asarray((290.0 + 900.0 * np.exp(-((g[None, :] - 20) ** 2 + (g[:, None] - 24) ** 2) / 50.0))[None].astype(np.float32))
    U0 = jnp.full((1, 48, 48), 500.0, jnp.float32)
    for name, kw in (("full: flux form, antisymmetric, mean removed", {}),
                     ("no mean removal", dict(consistent=False)),
                     ("no antisymmetry", dict(antisymmetric=False))):
        drift, uniform, mirror, offset, peak = [], [], [], [], []
        for d in range(DRAWS):
            net = random_net("l", [5, 32, 32, 4], d, 0.3)
            step = jax.jit(lambda T: T + dt2 * ldiffuse_rate(net, k, T, dt2, h, **kw))
            mirror.append(float(jnp.max(jnp.abs(step(Th[..., ::-1])[..., ::-1] - step(Th)))))
            offset.append(float(jnp.max(jnp.abs(step(Th + 1000.0) - 1000.0 - step(Th)))))
            T, U = Th, U0
            for _ in range(20):
                T, U = step(T), step(U)
            finite = bool(np.all(np.isfinite(np.asarray(T))))
            peak.append(float(jnp.max(jnp.abs(T))) if finite else float("inf"))
            drift.append(abs(total(T) - total(Th)) / total(Th) if finite else float("inf"))
            uniform.append(float(jnp.max(jnp.abs(U - 500.0))) if bool(np.all(np.isfinite(np.asarray(U)))) else float("inf"))
        r = {"total_drift": float(np.max(drift)), "uniform_drift_K": float(np.max(uniform)), "mirror_error_K": float(np.max(mirror)),
             "offset_error_K": float(np.max(offset)), "peak_abs_T": float(np.max(peak))}
        results["ldiffuse / " + name] = r
        rows.append(("ldiffuse", name, r["total_drift"], "", r["mirror_error_K"], f"{r['offset_error_K']:.2e}", f"{r['peak_abs_T']:.3g}"))

    md = [f"# Ablations: which piece gives which guarantee\n\nRandom untrained weights, {DRAWS} draws, worst draw. "
          "Mirror error: one step of the x-mirrored state, mirrored back, against one step of the state (K). Offset error: one "
          "step of T + 1000 K, minus 1000, against one step of T (K). Peak: the largest |T| reached (random weights at twice "
          "the explicit step can go unstable; catching that is the guard's job in model 3, not these properties').\n\n",
          "| operator | variant | drift of the total (relative) | draws leaving the range | mirror error (K) | offset error (K) | peak abs T |\n"
          "|---|---|---|---|---|---|---|\n"]
    for op, name, drift, out, mirror, off, peak in rows:
        md.append(f"| {op} | {name} | {drift:.1e} | {out} | {mirror:.2e} | {off} | {peak} |\n")
    save("ablations", results, "".join(md))
    print("".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

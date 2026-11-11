"""VERIFICATION: does the engine solve the equations it is given, at the order its schemes promise?

Runs the engine itself (lawc: the CPU reference, and the generated CUDA) on problems with exact solutions and refines
the lattice:

  diffusion  dT/dt = k lap T on [0, 1] with zero-flux ends, T0 = 1 + cos(4 pi x): T = 1 + cos(4 pi x) exp(-k 16 pi^2 t).
             Flux-form second-order stencil, explicit Euler with the law's max_dt (dt ~ h^2): expected order 2.
  advection  dF/dt + u dF/dx = 0, a Gaussian carried 0.4 to the right: first-order upwind, expected order 1.

The state is float32: with a lower mode (cos(pi x)) the error reaches a round-off floor near 1e-5 by n = 128, which is
why the mode here is high enough to keep the truncation error above it. Also the conservation drift of the diffusion runs (the total of T, in float32, over thousands of substeps).

    python bench/verify.py
"""

import os
import tempfile

import numpy as np

from common import BUILD, HERE, cuda_available, lawc, read_state, save, write_scene, write_state

DIFF_LAW = os.path.join(HERE, "laws", "diffusion.law")
ADV_LAW = os.path.join(HERE, "laws", "advection.law")


def cell_centres(n):
    return (np.arange(n) + 0.5) / n


def run(law, field, init, n, t_end, steps, backend, work):
    h = 1.0 / n
    scene = write_scene(os.path.join(work, "s.scene"), (n, 1, 1, h), [law])
    start = write_state(os.path.join(work, f"init_{n}"), {field: init.reshape(1, 1, n)}, h)
    out = os.path.join(work, f"out_{backend}_{n}")
    lawc("--scene", scene, "--init", start, "--steps", str(steps), "--dt", str(t_end / steps), "--backend", backend, "--dump", out)
    end, _, _ = read_state(os.path.join(out, f"step_{steps}"))
    return end[field].reshape(-1).astype(np.float64)


def orders(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors[:-1], errors[1:])]


def main():
    results = {}
    md = ["# Verification: order of convergence\n"]
    backends = ("cpu", "cuda") if cuda_available() else ("cpu",)  # no card: the reference only
    with tempfile.TemporaryDirectory() as work:
        for backend in backends:
            # ---- diffusion --------------------------------------------------------------------------------------
            k, t_end, m = 0.01, 0.5, 4
            ns = [16, 32, 64, 128, 256]
            errs, drift = [], []
            for n in ns:
                x = cell_centres(n)
                init = 1.0 + np.cos(m * np.pi * x)
                got = run(DIFF_LAW, "T", init, n, t_end, 10, backend, work)
                exact = 1.0 + np.cos(m * np.pi * x) * np.exp(-k * (m * np.pi) ** 2 * t_end)
                errs.append(float(np.sqrt(np.mean((got - exact) ** 2))))
                drift.append(float(abs(got.sum() - init.astype(np.float32).astype(np.float64).sum()) / init.sum()))
            results[f"diffusion_{backend}"] = {"n": ns, "l2": errs, "order": orders(errs), "relative_total_drift": drift}

            # ---- advection --------------------------------------------------------------------------------------
            u, t_end = 1.0, 0.4
            ns_a = [64, 128, 256, 512, 1024]
            errs_a = []
            for n in ns_a:
                x = cell_centres(n)
                init = np.exp(-((x - 0.3) / 0.05) ** 2)
                got = run(ADV_LAW, "F", init, n, t_end, 4, backend, work)
                exact = np.exp(-((x - 0.3 - u * t_end) / 0.05) ** 2)
                errs_a.append(float(np.sqrt(np.mean((got - exact) ** 2))))
            results[f"advection_{backend}"] = {"n": ns_a, "l2": errs_a, "order": orders(errs_a)}

            d, a = results[f"diffusion_{backend}"], results[f"advection_{backend}"]
            print(f"{backend}: diffusion L2 {['%.2e' % e for e in d['l2']]} orders {['%.2f' % o for o in d['order']]}; "
                  f"drift of the total {max(d['relative_total_drift']):.1e}")
            print(f"{backend}: advection L2 {['%.2e' % e for e in a['l2']]} orders {['%.2f' % o for o in a['order']]}")
            # The schemes' promises, at the finest refinement: flux-form diffusion order 2, upwind order 1.
            assert d["order"][-1] > 1.8, f"{backend}: diffusion order {d['order']}"
            assert a["order"][-1] > 0.8, f"{backend}: advection order {a['order']}"

    md.append("| problem | backend | lattice | L2 error | observed order |\n|---|---|---|---|---|\n")
    for key, r in results.items():
        problem, backend = key.split("_")
        for i, n in enumerate(r["n"]):
            order = f"{r['order'][i - 1]:.2f}" if i > 0 else ""
            md.append(f"| {problem} | {backend} | {n} | {r['l2'][i]:.3e} | {order} |\n")
    md.append(f"\nLargest relative drift of the conserved total of T (float32, up to "
              f"{int(np.ceil(0.05 / (0.15 * (1 / 256) ** 2 / 0.01))) * 10} substeps): "
              f"{max(max(results[f'diffusion_{b}']['relative_total_drift']) for b in backends):.1e}\n")
    # Without a card (a container, CI) the results go to verify_cpu: the shipped verify.json keeps the CUDA rows.
    save("verify" if "cuda" in backends else "verify_cpu", results, "".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

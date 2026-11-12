# A standard benchmark: PDEBench's 2-D diffusion-reaction

FitzHugh-Nagumo on [-1, 1]², 128 x 128 cells, t in [0, 5] at 101 frames, Du = 0.001, Dv = 0.005, k = 0.005, zero-flux walls, u0, v0 ~ N(0, 1) per cell: PDEBench's `configs/diff-react.yaml`. The law (`bench/laws/diff_react.law`) has PDEBench's spatial operator exactly, the classical RK4 in time, a sample a z-plane.

## Accuracy

8 samples (seeds 0-7) against the same method of lines at tight tolerances (DOP853, rtol 1e-10, float64): the same spatial operator, so what is left is time stepping and float32. nRMSE = ||pred - ref|| / ||ref|| over frames 1-100 and both fields.

| solver | nRMSE, mean | worst | order in time |
|---|---|---|---|
| engine, RK4, 4 substeps a frame (cfl 0.3) | 1.19e-03 | 1.25e-03 |  |
| engine, RK4, 7 substeps a frame (cfl 0.15) | 3.49e-05 | 3.63e-05 | 6.30 |
| engine, RK4, 14 substeps a frame (cfl 0.075) | 1.57e-06 | 1.63e-06 | 4.47 |
| PDEBench's recipe: solve_ivp, RK45, rtol 1e-3, atol 1e-6 | 3.04e-05 | 4.36e-05 | |

At cfl 0.3 the stiffest (checkerboard) diffusion modes sit near the edge of RK4's stability region, so the first ratio is not yet the asymptotic order. The throughput below is at cfl 0.15, the recipe's accuracy.

## Throughput

A step is a frame (0.05, its substeps by the law's max_dt). Card time against the batch; PDEBench's recipe takes 2.4 s a sample on this CPU (one core, scipy).

| samples in one run | cells | card ms a frame | card s a sample (100 frames) | samples a second |
|---|---|---|---|---|
| 1 | 16384 | 3.702 | 0.3702 | 3 |
| 16 | 262144 | 4.185 | 0.0262 | 38 |
| 256 | 4194304 | 53.136 | 0.0208 | 48 |
| 1024 | 16777216 | 256.688 | 0.0251 | 40 |

PDEBench's 1000 samples of this set: 20.8 s of card time on this laptop GPU, against 0.7 h of the recipe on one CPU core, with an error 1.1x the recipe's own (against the tight reference). The recipe's time is one core's; scipy does not use more.

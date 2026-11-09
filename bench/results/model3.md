# Model 3: learned discretization at twice the step, under guard

Heat, dt = 0.3 h²/k (twice the law's limit), 100 steps; error = RMSE (K) at the end against 8 classical substeps a step; overshoot = how far past the initial [min, max] the field went (K; the maximum principle says 0). Learned: mean ± std over 3 training seeds of the mean over the set.

| set | classical, 2 substeps | classical, 1 pass | learned, 1 pass |
|---|---|---|---|
| held-out (training generator) | 0.12 K, overshoot 0 | diverges (24/24) | 0.259 ± 0.069 K, overshoot 0 ± 0 |
| heat scene (96 x 96) | 0.16 K, overshoot 0 | diverges (1/1) | 0.318 ± 0.081 K, overshoot 0 ± 0 |
| OOD smooth low modes | 0.0638 K, overshoot 0 | diverges (24/24) | 0.139 ± 0.033 K, overshoot 0 ± 0 |
| OOD sharp boxes | 0.122 K, overshoot 0 | diverges (24/24) | 0.193 ± 0.038 K, overshoot 0 ± 0 |

In the engine (CUDA, `lawc --learned`), 40 steps on each of 8 fields: learned steps undone by the guard; contract violations that reached the world:

| net | set | undone / steps | violations |
|---|---|---|---|
| trained | held-out (training generator) | 0 / 320 | 0 |
| trained | OOD sharp boxes | 0 / 320 | 0 |
| untrained | held-out (training generator) | 20 / 320 | 0 |
| untrained | OOD sharp boxes | 24 / 320 | 0 |

Cost on the engine's CUDA backend, 1024² lattice, one step of dt (card time of the term), taken at the end of this script on a machine that was busy, and before the fix of the net kernels' unrolling (`src/field/cuda/CudaNet.cpp`); the clean measurement after the fix is in [perf.md](perf.md):

| scheme | card ms | substeps |
|---|---|---|
| classical, 2 substeps | 0.396 | [2] |
| learned, 1 pass | 319.387 | [1] |

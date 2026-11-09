# Model 2: learned correction for a coarse lattice (burning front)

Coarse 128 x 8 (h = 1 m) against the fine 256 x 16 (h = 0.5 m) averaged 2 x 2; 600 steps of 0.05 s from the start. Trained on 6 initial conditions, tested on 6 unseen ones from the same distribution and 2 out of it. Learned: 3 training seeds, mean ± std over seeds of the mean over test ICs. Others: mean ± std over test ICs.

| method | T RMSE, unseen ICs (K) | front position error at 600 steps (m) |
|---|---|---|
| plain coarse | 104 ± 51 | 39.2 ± 24 |
| calibrated coarse (5 coefficients fitted) | 107 ± 50 | 7.12 ± 3.7 |
| coarse + learned flux | 28 ± 7.2 | 3.08 ± 3.6 |

Out of distribution (learned: mean over seeds):

| IC | plain RMSE / front | calibrated | learned |
|---|---|---|---|
| sparse fuel | 2.2 K / 0.0 m | 3.6 K / 0.0 m | 4.5 K / 0.0 m |
| full uniform load | 159.7 K / 14.0 m | 187.7 K / 15.0 m | 11.6 K / 0.0 m |

Fitted coefficients (factor on the law's value): k_heat x1.61, ignite x1.00, burn x0.88, heat x1.12, cool x0.87

Cost on the engine's CUDA backend (one step of 0.05 s; fine 2048², coarse 1024²; card time of the terms):

| run | card ms / step | substeps per term |
|---|---|---|
| fine | 1.328 | [2, 1] |
| coarse plain | 0.325 | [1, 1] |
| coarse + learned flux | 2.439 | [1, 1, 1] |

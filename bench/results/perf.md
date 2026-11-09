# Performance and parity

Card: NVIDIA GeForce RTX 3060 Laptop GPU (sm_86). Medians over 100 steps.

One conduction term (`d T = diffuse(k, T)`, conserves T, a range) on N³, times on the card:

| N³ | copy GB/s | handwritten ms (GB/s at 8 B/cell) | engine ms, contracts off | engine ms, contracts on | engine step incl. readback ms | CPU ms | max abs diff engine vs handwritten |
|---|---|---|---|---|---|---|---|
| 64³ | 186 | 0.033 (64) | 0.034 | 0.103 | 0.56 | 2.8 | 0.0e+00 K |
| 128³ | 256 | 0.096 (174) | 0.206 | 0.369 | 3.34 | 21.3 | 0.0e+00 K |
| 256³ | 297 | 0.500 (269) | 1.525 | 2.196 | 28.18 | 236.0 | 0.0e+00 K |

The shipped world law (43 terms, 39 fields) on 144 × 144 × 48 (1 M cells), median step: CPU (OpenMP) 306.5 ms, CUDA 11.8 ms (incl. readback of the fields that changed) - 26x.

What the learned operators cost on the card a step (sum of the terms' card time), against what they replace:

| run | card ms a step |
|---|---|
| heat: classical, 2 substeps | 0.310 |
| heat: learned ldiffuse, 1 pass | 7.707 |
| front: fine 2048² | 1.124 |
| front: coarse 1024² | 0.289 |
| front: coarse 1024² + learned flux (tensor cores) | 2.000 |

Parity: CUDA against the CPU reference after 6 steps of the whole law, worst relative difference 1.2e-07; the JAX twin against the CPU reference, 1.8e-07.

# Performance and parity

Card: NVIDIA GeForce RTX 3060 Laptop GPU (sm_86). Medians over 100 steps.

One conduction term (`d T = diffuse(k, T)`, conserves T, a range) on N³, times on the card:

| N³ | copy GB/s | handwritten ms (GB/s at 8 B/cell) | engine ms, contracts off | engine ms, contracts on | step, every step read back | step, lazy readback: p50 / p99 / p99.9 | CPU ms | max abs diff engine vs handwritten |
|---|---|---|---|---|---|---|---|---|
| 64³ | 103 | 0.037 (57) | 0.068 | 0.138 | 0.82 | 0.21 / 0.54 / 1.10 | 2.9 | 0.0e+00 K |
| 128³ | 256 | 0.098 (171) | 0.154 | 0.310 | 4.55 | 0.35 / 0.89 / 1.55 | 36.8 | 0.0e+00 K |
| 256³ | 293 | 0.510 (263) | 0.937 | 1.690 | 33.00 | 1.08 / 2.27 / 4.35 | 253.0 | 0.0e+00 K |

The first three are card time (CUDA events); the steps are wall time, the lazy ones over 1000+ steps with contracts off (the latency tail of a solver loop).

The shipped world law (43 terms, 39 fields) on 144 × 144 × 48 (1 M cells), median step: CPU (OpenMP) 403.9 ms, CUDA 7.8 ms (lawc: lazy readback, contracts checked every step) - 52x.

What the learned operators cost on the card a step (sum of the terms' card time), against what they replace:

| run | card ms a step |
|---|---|
| heat: classical, 2 substeps | 0.243 |
| heat: learned ldiffuse, 1 pass | 8.367 |
| front: fine 2048² | 0.998 |
| front: coarse 1024² | 0.265 |
| front: coarse 1024² + learned flux (tensor cores) | 2.547 |

Parity: CUDA against the CPU reference after 6 steps of the whole law, worst relative difference 1.2e-07; the JAX twin against the CPU reference, 1.8e-07.

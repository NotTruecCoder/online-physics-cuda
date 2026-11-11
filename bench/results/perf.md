# Performance and parity

Card: NVIDIA GeForce RTX 3060 Laptop GPU (sm_86). Medians over 100 steps.

One conduction term (`d T = diffuse(k, T)`, conserves T, a range) on N³, times on the card:

| N³ | copy GB/s | handwritten ms (GB/s at 8 B/cell) | engine ms, contracts off | engine ms, contracts on | step, every step read back | step, lazy readback: p50 / p99 / p99.9 | CPU ms | max abs diff engine vs handwritten |
|---|---|---|---|---|---|---|---|---|
| 64³ | 128 | 0.035 (60) | 0.043 | 0.132 | 0.65 | 0.23 / 0.50 / 1.47 | 3.6 | 0.0e+00 K |
| 128³ | 224 | 0.101 (166) | 0.133 | 0.296 | 3.24 | 0.27 / 0.66 / 2.27 | 29.0 | 0.0e+00 K |
| 256³ | 251 | 0.570 (235) | 0.921 | 1.545 | 22.89 | 1.32 / 2.16 / 4.57 | 183.9 | 0.0e+00 K |

The first three are card time (CUDA events); the steps are wall time, the lazy ones over 1000+ steps with contracts off (the latency tail of a solver loop).

The shipped world law (43 terms, 39 fields) on 144 × 144 × 48 (1 M cells), median step: CPU (OpenMP) 413.9 ms, CUDA 7.8 ms (lawc: lazy readback, contracts checked every step) - 53x.

What the learned operators cost on the card a step (sum of the terms' card time), against what they replace:

| run | card ms a step |
|---|---|
| heat: classical, 2 substeps | 0.250 |
| heat: learned ldiffuse, 1 pass | 11.567 |
| front: fine 2048² | 1.069 |
| front: coarse 1024² | 0.293 |
| front: coarse 1024² + learned flux (tensor cores) | 2.653 |

Parity: CUDA against the CPU reference after 6 steps of the whole law, worst relative difference 1.2e-07; the JAX twin against the CPU reference, 1.8e-07.

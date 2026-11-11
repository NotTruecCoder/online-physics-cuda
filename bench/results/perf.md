# Performance and parity

Card: NVIDIA GeForce RTX 3060 Laptop GPU (sm_86). Medians over 100 steps.

One conduction term (`d T = diffuse(k, T)`, conserves T, a range) on N³, times on the card:

| N³ | copy GB/s | handwritten ms (GB/s at 8 B/cell) | engine ms, contracts off | engine ms, contracts on | step, every step read back | step, lazy readback: p50 / p99 / p99.9 | CPU ms | max abs diff engine vs handwritten |
|---|---|---|---|---|---|---|---|---|
| 64³ | 89 | 0.039 (54) | 0.086 | 0.202 | 0.79 | 0.28 / 0.50 / 0.87 | 3.6 | 0.0e+00 K |
| 128³ | 218 | 0.104 (161) | 0.203 | 0.416 | 3.43 | 0.37 / 0.85 / 3.39 | 28.9 | 0.0e+00 K |
| 256³ | 255 | 0.569 (236) | 1.000 | 1.893 | 24.55 | 1.40 / 2.24 / 3.63 | 207.9 | 0.0e+00 K |

The first three are card time (CUDA events); the steps are wall time, the lazy ones over 1000+ steps with contracts off (the latency tail of a solver loop).

The shipped world law (43 terms, 39 fields) on 144 × 144 × 48 (1 M cells), median step, contracts checked every step: CPU (OpenMP) 348.0 ms; CUDA 9.6 ms with lazy readback (the cells stay on the card: 36x), 13.4 ms with every changed field read back every step, as the sandbox runs it (26x). One run of each; the CPU's step moves with the laptop's clocks (300-410 ms seen across runs).

What the learned operators cost on the card a step (sum of the terms' card time), against what they replace:

| run | card ms a step |
|---|---|
| heat: classical, 2 substeps | 0.436 |
| heat: learned ldiffuse, 1 pass | 12.044 |
| front: fine 2048² | 1.392 |
| front: coarse 1024² | 0.408 |
| front: coarse 1024² + learned flux (tensor cores) | 2.974 |

Parity: CUDA against the CPU reference after 6 steps of the whole law, worst relative difference 1.2e-07; the JAX twin against the CPU reference, 1.8e-07.

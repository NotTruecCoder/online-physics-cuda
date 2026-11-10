# The same stencil in the engine and in other tools

Card: NVIDIA GeForce RTX 3060 Laptop GPU. 7-point explicit heat on N³, float32, 50 steps; median wall time of a step with the device synchronised after it (launch overheads count). Effective bandwidth at the minimum 8 bytes a cell.

## 128³ (copy kernel: 260 GB/s)

| implementation | ms a step | GB/s at 8 B/cell | max abs diff vs NumPy | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.094 | 178 | - | bench_stencil |
| Warp (CUDA) | 0.209 | 80 | 2.4e-04 K | warp 1.19.0 |
| engine, lazy readback, contracts off | 0.245 | 68 | - | this repo |
| Taichi (CUDA) | 0.398 | 42 | 3.1e-04 K | taichi (1, 7, 4) |
| PyTorch eager (CUDA) | 0.878 | 19 | 0.0e+00 K | torch 2.11.0+cu128 |
| Numba parallel (CPU) | 1.925 | 9 | 2.4e-04 K | numba 0.61.2, 12 threads |
| engine, contracts on, every step read back | 3.260 | 5 | - | this repo |
| NumPy (CPU, 1 thread) | 44.736 | 0 | 0.0e+00 K | numpy 1.26.4 |

The engine against the handwritten kernel after the same steps: 0.0e+00 K. The handwritten row is card time (CUDA events); the others are wall time.

## 256³ (copy kernel: 299 GB/s)

| implementation | ms a step | GB/s at 8 B/cell | max abs diff vs NumPy | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.477 | 281 | - | bench_stencil |
| Warp (CUDA) | 0.825 | 163 | 3.1e-04 K | warp 1.19.0 |
| engine, lazy readback, contracts off | 1.008 | 133 | - | this repo |
| Taichi (CUDA) | 3.143 | 43 | 3.1e-04 K | taichi (1, 7, 4) |
| PyTorch eager (CUDA) | 5.872 | 23 | 0.0e+00 K | torch 2.11.0+cu128 |
| Numba parallel (CPU) | 14.468 | 9 | 3.1e-04 K | numba 0.61.2, 12 threads |
| engine, contracts on, every step read back | 25.599 | 5 | - | this repo |
| NumPy (CPU, 1 thread) | 328.862 | 0 | 0.0e+00 K | numpy 1.26.4 |

The engine against the handwritten kernel after the same steps: 0.0e+00 K. The handwritten row is card time (CUDA events); the others are wall time.


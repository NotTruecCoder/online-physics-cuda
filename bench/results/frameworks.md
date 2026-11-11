# The same stencil in the engine and in other tools

Card: NVIDIA GeForce RTX 3060 Laptop GPU. 7-point explicit heat on N³, float32, 50 steps; median wall time of a step with the device synchronised after it (launch overheads count). Effective bandwidth at the minimum 8 bytes a cell.

## 128³ (copy kernel: 221 GB/s)

| implementation | wall ms a step | GB/s at 8 B/cell | max abs diff | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.112 | 150 | - | bench_stencil |
| Warp (CUDA) | 0.191 | 88 | 2.4e-04 K vs NumPy | warp 1.19.0 |
| engine, lazy readback, contracts off | 0.413 | 41 | 0.0e+00 K vs handwritten | this repo |
| Taichi (CUDA) | 0.535 | 31 | 3.1e-04 K vs NumPy | taichi (1, 7, 4) |
| PyTorch eager (CUDA) | 1.031 | 16 | 0.0e+00 K vs NumPy | torch 2.11.0+cu128 |
| Numba parallel (CPU) | 1.887 | 9 | 2.4e-04 K vs NumPy | numba 0.61.2, 12 threads |
| engine, contracts on, every step read back | 3.415 | 5 | 0.0e+00 K vs handwritten | this repo |
| NumPy (CPU, 1 thread) | 48.846 | 0 | 0.0e+00 K vs NumPy | numpy 1.26.4 |

Every row is wall time. The handwritten kernel's card time (CUDA events) is 0.101 ms; the difference is what a launch and a synchronisation cost the host.

## 256³ (copy kernel: 257 GB/s)

| implementation | wall ms a step | GB/s at 8 B/cell | max abs diff | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.575 | 234 | - | bench_stencil |
| Warp (CUDA) | 1.154 | 116 | 3.1e-04 K vs NumPy | warp 1.19.0 |
| engine, lazy readback, contracts off | 1.417 | 95 | 0.0e+00 K vs handwritten | this repo |
| Taichi (CUDA) | 3.902 | 34 | 3.1e-04 K vs NumPy | taichi (1, 7, 4) |
| PyTorch eager (CUDA) | 7.181 | 19 | 0.0e+00 K vs NumPy | torch 2.11.0+cu128 |
| Numba parallel (CPU) | 10.221 | 13 | 3.1e-04 K vs NumPy | numba 0.61.2, 12 threads |
| engine, contracts on, every step read back | 24.275 | 6 | 0.0e+00 K vs handwritten | this repo |
| NumPy (CPU, 1 thread) | 338.158 | 0 | 0.0e+00 K vs NumPy | numpy 1.26.4 |

Every row is wall time. The handwritten kernel's card time (CUDA events) is 0.564 ms; the difference is what a launch and a synchronisation cost the host.


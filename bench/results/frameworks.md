# The same stencil in the engine and in other tools

Card: NVIDIA GeForce RTX 3060 Laptop GPU. 7-point explicit heat on N³, float32, 200 steps; median wall time of a step with the device synchronised after it (launch overheads count). Effective bandwidth at the minimum 8 bytes a cell.

## 128³ (copy kernel: 264 GB/s)

| implementation | wall ms a step | GB/s at 8 B/cell | max abs diff | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.102 | 164 | - | bench_stencil |
| Warp (CUDA) | 0.204 | 82 | 3.7e-04 K vs NumPy | warp 1.19.0 |
| engine, lazy readback, contracts off | 0.275 | 61 | 0.0e+00 K vs handwritten | this repo |
| Taichi (CUDA) | 0.335 | 50 | 3.7e-04 K vs NumPy | taichi (1, 7, 4) |
| PyTorch eager (CUDA) | 0.924 | 18 | 0.0e+00 K vs NumPy | torch 2.11.0+cu128 |
| Numba parallel (CPU) | 2.153 | 8 | 2.4e-04 K vs NumPy | numba 0.61.2, 12 threads |
| engine, contracts on, every step read back | 3.086 | 5 | 0.0e+00 K vs handwritten | this repo |
| NumPy (CPU, 1 thread) | 63.420 | 0 | 0.0e+00 K vs NumPy | numpy 1.26.4 |

Every row is wall time. The handwritten kernel's card time (CUDA events) is 0.093 ms; the difference is what a launch and a synchronisation cost the host.

## 256³ (copy kernel: 294 GB/s)

| implementation | wall ms a step | GB/s at 8 B/cell | max abs diff | version |
|---|---|---|---|---|
| handwritten CUDA (ping-pong) | 0.503 | 267 | - | bench_stencil |
| engine, lazy readback, contracts off | 0.749 | 179 | 0.0e+00 K vs handwritten | this repo |
| Warp (CUDA) | 2.067 | 65 | 3.7e-04 K vs NumPy | warp 1.19.0 |
| Taichi (CUDA) | 3.848 | 35 | 4.3e-04 K vs NumPy | taichi (1, 7, 4) |
| Numba parallel (CPU) | 11.491 | 12 | 3.1e-04 K vs NumPy | numba 0.61.2, 12 threads |
| PyTorch eager (CUDA) | 12.468 | 11 | 0.0e+00 K vs NumPy | torch 2.11.0+cu128 |
| engine, contracts on, every step read back | 24.153 | 6 | 0.0e+00 K vs handwritten | this repo |
| NumPy (CPU, 1 thread) | 314.537 | 0 | 0.0e+00 K vs NumPy | numpy 1.26.4 |

Every row is wall time. The handwritten kernel's card time (CUDA events) is 0.486 ms; the difference is what a launch and a synchronisation cost the host.


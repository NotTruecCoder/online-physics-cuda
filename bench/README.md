# Benchmarks

Every number in the top-level README comes from a script here. Each one runs the engine itself (`lawc`, or a bench
executable linked to the engine), writes `results/<name>.json` (everything measured) and `results/<name>.md` (the
table the README quotes), and says in its docstring what it measures and how.

Build first (`cmake --preset dev && cmake --build --preset dev`, or the `core` preset without vcpkg), then from the
repository root:

| script | what it answers | time on an RTX 3060 Laptop |
|---|---|---|
| `python bench/verify.py` | do the schemes converge at their order against exact solutions (CPU and CUDA)? | ~1 min |
| `python bench/perf.py` | generated code against a handwritten kernel and the copy bandwidth; the world law CPU vs CUDA; parity CPU/CUDA/JAX | ~3 min |
| `python bench/ablations.py` | which piece of the learned operators gives which guarantee, for random weights | ~1 min |
| `python bench/model2_front.py` | learned coarse-lattice correction on unseen initial conditions, against plain and calibrated coarse, and its cost | ~40 min (CPU JAX) |
| `python bench/model3_heat.py` | learned discretization at 2x the step: accuracy on held-out and OOD fields, guard, cost | ~45 min (CPU JAX) |
| `python bench/mind.py` | world model + MPPI against still, random, a hand-written rule and MPPI with the true dynamics | ~30 min |
| `python bench/finance.py` | Black-Scholes (European, American) and Heston as laws, against closed forms; greeks by `jax.grad`; a Heston surface (64 strikes x 4 maturities) in one run | ~30 min (Heston to 2048 cells, 350 000 steps; its float64 replica in NumPy) |
| `python bench/finance.py --cpu` | the same on the CPU reference, smaller lattices, orders asserted (CI) | ~3 min |
| `python bench/verify.py` (no card) | the CPU rows only, orders asserted, to `results/verify_cpu.*` (CI) | ~1 min |
| `python bench/frameworks.py` | the same stencil in the engine, Taichi, Warp, PyTorch, Numba and NumPy (`pip install taichi warp-lang torch numba`) | ~3 min |
| `python bench/plots.py` | the SVG figures in `docs/figures/` from the results | seconds |

`model2_front.py` and `model3_heat.py` also write the trained nets the engine ships (`assets/nets/*.net`). Python
dependencies: `pip install -r requirements.txt` (CPU JAX is enough).

C++ benchmark executables (built with the engine): `bench_stencil` (`stencil_bench.cpp`), `bench_mind`
(`mind_bench.cpp`). Laws used only for verification: `laws/`.

Run the CUDA timings on an otherwise idle machine; the GPU of a laptop changes clocks with load and temperature, so
every timing is a median over many steps.

# Field Engine

**A compiler from physical laws written as text to deterministic CUDA kernels. Neural networks are operators of the
language, structure-preserving and run under runtime contracts. Every learned model is evaluated against classical
baselines on data it was not trained on.**

C++20, ~42k lines (engine 27k, of which the language, reference solver, IR, CUDA backend, nets and world model 9k;
benchmarks and tools 2.7k), no CUDA SDK needed to build (the
driver API and NVRTC are loaded at run time), a differentiable JAX twin of every IR instruction, a reproducible
benchmark suite (`bench/`).

![A town in the sandbox: people about their work, balls of stone, wood and ice, a flood of water](docs/sandbox_town.png)

## Report

All numbers: one RTX 3060 Laptop GPU (6 GB), 8 CPU threads; reproduce with the script named in each row
([bench/README.md](bench/README.md)). Full tables in [bench/results/](bench/results/).

### Is the solver right?

| check | result | script |
| --- | --- | --- |
| diffusion against the exact solution, refined 16 → 256 | observed order **2.0** (2.09, 2.94, 2.21, 2.04), CPU and CUDA identical to 3 digits | [verify](bench/results/verify.md) |
| upwind advection, refined 64 → 1024 | order **0.52 → 0.89**, approaching the scheme's 1 | [verify](bench/results/verify.md) |
| conserved total over 2190 float32 substeps | relative drift **1.5·10⁻⁶** | [verify](bench/results/verify.md) |
| generated CUDA against a handwritten kernel, same step, 256³ | **bit-identical** (max difference 0) | [perf](bench/results/perf.md) |
| CUDA against the CPU reference, the whole 43-term law | **1.2·10⁻⁷** relative | [perf](bench/results/perf.md) |
| JAX twin against the CPU reference, whole law | **1.8·10⁻⁷** relative | [perf](bench/results/perf.md) |
| two runs on one card | bit-identical (no float atomics; fixed-order compensated reductions) | gate `cuda_parity` |

### What does the generated code cost?

One conduction term on N³, card time a step ([perf](bench/results/perf.md)):

| N³ | copy kernel | handwritten fused kernel | engine, contracts off | engine, contracts on | engine step incl. readback |
| --- | --- | --- | --- | --- | --- |
| 128³ | 256 GB/s | 0.096 ms | 0.21 ms | 0.37 ms | 3.3 ms |
| 256³ | 297 GB/s | 0.50 ms (269 GB/s, 91% of copy) | 1.53 ms | 2.20 ms | 28 ms |

- Generated code is **~3x the handwritten kernel** at 256³: a term that reads its own neighbours is gathered into a
  delta buffer and stored in a second pass (4 field transfers instead of 2), plus ring addressing.
- Contract checks (range, conserved total, every step) cost **+44%**. Before this round they cost +290%: the total was
  a thread a row (uncoalesced) and every changed cell wrote one shared flag. Now it is a warp a row with a fixed shuffle
  tree and a ballot, still bit-reproducible.
- **The bottleneck is the readback**, not the kernels: the engine copies changed fields to the host every step (the
  mesher and bodies read them), 64 MB at 256³.
- The shipped world law (43 terms, 39 fields, 1 M cells): CPU (OpenMP, 8 threads) 307 ms → CUDA **11.8 ms** a step, 26x.

### Do the learned models help, out of sample?

**Model 2, a learned correction for a lattice twice as coarse** (burning front; trained on 6 initial conditions, tested
on 6 unseen ones and 2 out of distribution; 3 training seeds; [model2](bench/results/model2.md)):

| method | T RMSE, unseen ICs | front position error at 30 s |
| --- | --- | --- |
| plain coarse | 104 ± 51 K | 39 ± 24 m |
| coarse with 5 coefficients fitted (same data, loss, optimizer) | 107 ± 50 K | 7.1 ± 3.7 m |
| **coarse + learned flux** | **28 ± 7 K** | **3.1 ± 3.6 m** |

Out of distribution: a full uniform fuel load, 12 K against 160 K (plain) and 188 K (fitted); sparse fuel where the
front dies, all three within 5 K. Card time: coarse + learned flux **2.0 ms**, while the fine lattice costs **1.1 ms**.
It is more accurate than the classical options at the same resolution, but here it costs more than running the fine
lattice.

**Model 3, a learned discretization at twice the explicit step, under guard** (heat; 3 seeds;
[model3](bench/results/model3.md)):

| set (100 steps) | classical, 2 substeps | classical, 1 pass | learned, 1 pass |
| --- | --- | --- | --- |
| held-out fields | 0.12 K | diverges 24/24 | 0.26 ± 0.07 K |
| OOD smooth modes | 0.06 K | diverges 24/24 | 0.14 ± 0.03 K |
| OOD sharp boxes | 0.12 K | diverges 24/24 | 0.19 ± 0.04 K |

It is stable where the explicit scheme at that step blows up, with no new extremes (overshoot 0). The guard undoes
0/640 trained steps and 44/640 untrained ones, and no violation reaches the world. But it is **2x less accurate** than
the 2 classical substeps it replaces and **25x more expensive** (7.7 ms against 0.31 ms on 1024², after an
unrolling fix in the net code generator that made it 12x faster). This is a negative result for the scheme and a
working demonstration of the guard.

**Model 1, a world model learned online + MPPI** (64 foragers, 5 world layouts, 30 episodes;
[mind](bench/results/mind.md)). Cost a decision, lower is better:

| still | random | hand-written rule (reads the true gradient) | **MPPI, learned model** | MPPI, true dynamics (ceiling) |
| --- | --- | --- | --- | --- |
| 0.95 ± 0.12 | 0.92 ± 0.12 | 0.49 ± 0.07 | **0.50 ± 0.04** | 0.32 ± 0.07 |

Planning through the learned model matches the hand-written rule without being told the gradient, and it closes 70%
of the gap from random to the true-dynamics ceiling. As a simulator it is weak: one-step R² is 0.36 on its own world
and ≈0 on an unseen layout. It predicts fear worse than a constant, because the planner keeps away from danger and the
replay holds few such transitions. Its 12-step cost forecast is worse than persistence. Plan 12 ms for 64 agents ×
256 futures × 12 steps on the card.

### Which piece of the architecture gives which guarantee?

Random, untrained weights (20 draws, worst shown), each piece removed in turn ([ablations](bench/results/ablations.md)):

| operator | variant | total drift | draws leaving the range | mirror error | offset error |
| --- | --- | --- | --- | --- | --- |
| flux | full | 6·10⁻⁸ | 0/20 | 1·10⁻⁴ K | |
| flux | no limiter | 2·10⁻⁷ | **13/20** | 1·10⁻⁴ K | |
| flux | no antisymmetry | 5·10⁻⁷ | 0/20 | **142 K** | |
| flux | no flux form (learned source) | **160** | 14/20 | 95 K | |
| ldiffuse | full | 5·10⁻⁶ | | 0 | 6·10⁻⁴ K |
| ldiffuse | no mean removal | 2·10⁻³ | | 0 | **926 K** |
| ldiffuse | no antisymmetry | 2·10⁻⁶ | | **43 K** | 6·10⁻⁴ K |

Conservation comes from the flux form, range from the limiter, layout symmetry from antisymmetry, and offset
invariance from mean removal. Each holds for any weights; no training is involved.

### What this adds up to

- The compiler is correct: the right orders of convergence, bit-identical to a handwritten kernel, 1e-7 against the
  CPU and JAX references. It is not yet fast in the bandwidth sense: kernels run at ~3x the minimum traffic, and the
  per-step readback dominates large lattices.
- Learned operators with built-in guarantees work as designed, and the guard catches what the law declares (only
  that: an oscillation inside the declared range passes it; see gate `guarded`).
- Of the three models, only the coarse-lattice correction clearly beats its classical alternatives out of sample, and
  even it costs more than running finer at this scale. The learned discretization loses to plain substeps. The world
  model plans well but does not generalise as a simulator.
- Not done: multi-GPU; lazy readback (it needs the per-field bounds that let idle terms be skipped computed on the
  card, since today they are read off the downloaded cells); flowing rivers; a learned simulator over the engine's own
  fields with the guard as its safety net (the natural next model).
- Known issue: one check of gate `voxel_map` fails. Water mana poured every tick while falling, with the law on its own
  thread and the window moving up and down, ends up more than 3 m under where you land. The horizontal case passes
  (`voxel_gpu`). 19 other gates pass; `voxel_beyond` runs for over 40 minutes and was not run to its end in this round.

## What it is

```
law text (assets/law/*.law) ──► parser ──► expression trees ──┬─► CPU solver (OpenMP, the reference)
                                                              └─► SSA IR (CSE, folding, stencil radii, fusion plan)
                                                                    ├─► CUDA codegen ──► NVRTC ──► kernels (main backend)
                                                                    ├─► JSON ──► tools/lawjax (jnp, jax.grad, training)
                                                                    └─  (HLSL codegen ──► D3D11: fallback, no nets)
```

```
field T     init 290  range 0 8000  unit K       # a field and its contracts (checked on the card)
param k_rock = 0.6
term conduction
  d T = diffuse(k_rock * solid + k_air * (1 - occ), T)
  conserves T
  max_dt = 0.15 * h * h / k_rock                 # the author's stability limit, turned into substeps
derive sdf = distance(occ)                       # exact signed distance transform

net front_fix in 4 hidden 32 32 out 1 act tanh half       # `half`: tensor cores
term front_fix
  d T = flux(front_fix, T, fuel)                 # learned face flux: conservative, antisymmetric, range-limited

term heat
  d T = diffuse(k, T)
  max_dt = 0.15 * h * h / k
  learned d T = ldiffuse(heat_ld, k, T)          # learned discretization at a larger step, under guard
  learned max_dt = 0.3 * h * h / k
```

- **A compiler, not an interpreter.** One immutable SSA graph for the whole law (on the shipped law 340 of 622
  instructions are found twice and kept once), constant folding with the reference's semantics, stencil radii, a cost
  model deciding what to recompute and what to store, in-place fusion when there is no read-after-write hazard (34 of
  43 terms run as one kernel). CUDA C → NVRTC → CUBIN, cached by content hash, `--fmad=false` so the arithmetic is
  the reference's.
- **Deterministic GPU numerics.** No floating-point atomics. Totals and reductions are two-level, Neumaier-compensated,
  in a fixed order (a warp a row, a fixed shuffle tree); contract checks count with integers and find the first bad
  cell with `atomicMax(~i)`.
- **Hand-written tensor-core code.** Nets in the law run through `mma.sync.aligned.m16n8k16` in inline PTX (no CUDA
  headers): weights pre-packed into fragment order on the host; one layer's fp32 accumulators are, register for
  register, the next layer's f16x2 A fragments. Fused forward+backward training kernels without atomics (backward
  checked against finite differences, 6·10⁻⁵).
- **Learned operators with guarantees by construction.** `flux`: a face flux (conservative for any weights),
  antisymmetrised (mirror-equivariant), limited per face (stays in the declared range). `ldiffuse`: a 4-cell face
  stencil whose weights a net gives, mean-removed (sees only differences) and antisymmetrised; an untrained net is
  `diffuse` exactly.
- **Guarded execution.** A learned scheme runs at its own larger step; its term's contracts (range, finiteness,
  conservation) are checked on the card; a breach restores a device-side snapshot and reruns the classical scheme.
- **Differentiable twin.** `tools/lawjax` rebuilds every IR instruction in `jax.numpy` with the engine's semantics;
  `jax.grad` through unrolled solver steps trains models against data the C++ reference produced.
- **World model + MPPI** (`src/field/mind`): a creature's `plan = ...` and `observe ...` lines; a neural ODE on its
  observations learned online (replay, GPU training), MPPI with a block an agent and a thread a future, the law's plan
  line compiled into the kernel as the cost, noise a hash of (agent, tick, sample, step).
- **Systems.** Driver API and NVRTC loaded at run time; ring-addressed lattice so a moving window uploads only arriving
  chunks; pinned readback of changed fields only; CUDA-event profiler per term; a law hot-reloaded while the world runs.
- **Testing as falsification.** 21 gates (`src/tests`) state claims about the world, each naming the mutation of the
  engine that makes it fail; those mutations were run.

`src/field/FIELD.md` is the design document (the language, contracts, the IR, the backends, the open world, NPCs).

## The world it runs

The engine drives a voxel world (`src/field/Playground`, `WorldMap`, `Terrain`, `People`, the sandbox): heat,
combustion, phase changes, water, lava, structural support and light as 43 terms over 39 fields; rigid voxel pieces;
a 262 × 262 km generated continent in layers (plates, climate, drainage, a 16-age history, towns built by culture and
changed by events); creatures and townspeople whose minds are lines of the same language. No line of C++ says
"hunter" or "prey": it all goes through fields. This is the demonstration and the stress test of the compiler, not a
validated physical model: its laws are game physics.

![The same world from far away: level of detail follows pixels per voxel](docs/sandbox_far.png)

## Build and reproduce

The engine, `lawc`, the benchmarks and the gates need only a C++20 compiler and OpenMP; the sandbox also needs vcpkg
(SDL3, Dear ImGui, bgfx, glm). CUDA needs only the NVIDIA driver and an NVRTC DLL (from `CUDA_PATH`, next to the
executable, a PyTorch install, or `FIELD_NVRTC`). `FIELD_BACKEND=cuda|hlsl|cpu` chooses the backend.

```bash
cmake --preset core && cmake --build --preset core && ctest --preset core   # Windows, engine only
cmake --preset linux-core && cmake --build --preset linux-core               # Linux (CPU; CUDA if a card is there)
cmake --preset dev && cmake --build --preset dev                             # everything, with VCPKG_ROOT set
```

```bash
pip install -r requirements.txt
python bench/verify.py && python bench/perf.py && python bench/ablations.py
python bench/model2_front.py && python bench/model3_heat.py && python bench/mind.py
```

`lawc` is the compiler on the command line (from `build/`): `--ir` (the IR listing), `--cuda FILE` (the generated
translation unit), `--scene S --steps N --backend cuda --profile` (per-term profile), `--compare` (CPU against CUDA),
`--dump DIR` / `--init DIR` (states as float32), `--json FILE` (timings).

Sandbox (`build/Release/sandbox.exe`): WASD/wheel/right-drag to move and zoom (to the whole continent), 1–0 to throw
balls of stone, wood, ice, pour fire, water, lava; F1 the law (every param live, every equation as the solver runs
it), F2 the profiler (per-term card time, learned steps undone).

## Related work

The ideas are not new one by one; the contribution is putting them in one compiler with one IR, checked against one
reference. Learned discretizations: Bar-Sinai et al. 2019 (model 3 is in that line); solver-in-the-loop training and
learned corrections for coarse grids: Um et al. 2020, Kochkov et al. 2021 (model 2); structure-preserving learned
fluxes: conservative finite-volume networks. DSLs to GPU for simulation: Taichi, NVIDIA Warp, Devito, Halide, Firedrake
(these are more general and faster; this engine's distinctive parts are declared contracts checked on the card, the
guarded learned schemes and the bit-reproducible kernels). Learned dynamics + MPPI: PETS (Chua et al. 2018), Williams
et al. 2017.

## Layout

| path | what |
| --- | --- |
| `src/field/` | the engine: law language, CPU reference solver, surface/bodies/rays, world, creatures, society |
| `src/field/ir/`, `src/field/cuda/` | the IR; CUDA codegen, NVRTC loading, the CUDA backend, network kernels |
| `src/field/nn/`, `src/field/mind/` | networks of the law (CPU reference); the world model and MPPI planner |
| `tools/lawc/`, `tools/lawjax/` | the compiler CLI; the JAX twin and parity check |
| `bench/` | verification, performance, ablations, the three models' evaluations, and their results |
| `src/tests/` | the gates |
| `src/render/`, `src/sandbox/` | bgfx renderer and the interactive sandbox |
| `assets/law/` | the laws (world, magic, life, people, map, pours; bench laws) |

## License

MIT, see [LICENSE](LICENSE).

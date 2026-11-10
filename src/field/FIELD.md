# The field engine (`src/field/`)

The world is a set of scalar fields on one voxel lattice. Physics, magic, light, collision and
(next) perception all read the same cells. The law that couples them is **text**
(`assets/law/*.law`), compiled at load and integrated deterministically.

The shape of the world is kept in one place, so what you see, what stops you and what burns cannot disagree.

## Files

| File | What it is |
| --- | --- |
| `Grid.hpp/.cpp` | `GridShape`, `FieldSet` (named float fields, trilinear `sample`, `digest`) |
| `Law.hpp/.cpp` | the law language: tokenizer, parser, expression tree, whole-lattice evaluator |
| `LawNode.hpp` | the compiled expression tree, shared by the two things that run it |
| `Solver.hpp/.cpp` | integrates a compiled law; checks its contracts and reports violations by name; the reference |
| `GpuSolver.hpp/.cpp` | the same law as compute shaders on Direct3D 11 |
| `Body.hpp/.cpp` | `surface_distance` / `surface_normal`, `resolve_sphere`, `sweep_sphere` |
| `Ray.hpp/.cpp` | a ray through the fields: picking, and curved light along grad n |
| `Parallel.hpp` | fixed-block parallel loops: the bits do not depend on the thread count |
| `WorldMap.hpp/.cpp` | the layers of the map: continent, climate, water, history, plans, buildings; a column at any coarseness |
| `Terrain.hpp/.cpp` | a chunk as a function of (seed, chunk) from those layers: strata, sea, rivers, trees, buildings, herds; `ChunkStore` |
| `FarLife.cpp` | what goes on beyond the window: busy chunks, far creatures, the land's law |
| `VoxelFigure.hpp/.cpp` | a figure of cubes from text, posed and meshed; picture only |
| `Playground.hpp/.cpp` | the world: proving ground or open world (simulated window, chunk store), bodies (walk, jump, step up, pushed by `push_*`, burned by `harm`), creatures, pours of mana, water or stone into voxels, the observer the window follows, `reload_law` |
| `VoxelMesh.hpp/.cpp` | fields to `VoxelCell`s to cubes: per-voxel shade, corner occlusion, turf on the top face, indexed quads, coarser voxels, skirts |
| `WorldMesher.hpp/.cpp` | what the camera sees as meshes, built on worker threads, at the detail the screen can show |

The library depends only on OpenMP (and Direct3D on Windows), so it is tested alone, without a window.

## The law language

```
field T     init 290  range 0 8000  unit K      # a field and its contracts
field solid init 0    range 0 1     static       # no term may write it
field b     init 0    range 0 1     clamp        # out-of-range is clamped, not reported
param k = 0.6                                    # params may use earlier params

term conduction                                  # equations read the same pre-state
  d T = diffuse(k, T)
  conserves T                                    # checked after every step
  max_dt = 0.15 * h * h / k                      # the author's stability limit -> substeps

derive sdf = distance(solid)                     # recomputed after every step
derive n   = 1 - 2 * phi / c2
derive n  += 0.001 * mana                        # another file can ADD to a derived field
```

Expressions: `+ - * / ^`, unary `-`, numbers, `h` (voxel edge), `pi`, params, fields.
Functions: `abs sqrt exp log step pos min max clamp`; stencils `lap gx gy gz diffuse(k,F)
adv(F,u,v,w)`, `shift(F,dx,dy,dz)` (F that many whole cells away, so `shift(solid,0,0,-1)` is what lies
below); global `distance(F)` (signed, metres, beyond the lattice is air) and `poisson(S[,iters])`
(warm-started); reductions `maxof minof sumof` (the only way `max_dt` may read a field).

What a term reads and writes is **derived from its equations**, never declared by hand.
Refused at compile time, with `source:line`: unknown names, arity, writing a `static` or derived
field, `conserves X` on a term that never writes X, a `max_dt` that depends on a cell.

## Contracts at run time

After every term: declared `range` (or `clamp`), non-finite values, and `conserves` totals. A broken
contract is a `Violation` naming the term, the field and the first bad cell. `max_dt` sets the
substeps; `Solver::set_respect_max_dt(false)` exists only so a gate can show it is load-bearing.

Determinism: fixed order, gather-then-apply, no clock or RNG. Same law + same start = same `digest()`,
on any number of threads: a cell's value depends only on the pre-state, and a reduction sums fixed
blocks in a fixed order (`Parallel.hpp`; checked at 1, 4 and 7 threads by `voxel_life_smoke`). At most 4
threads by default, `OMP_NUM_THREADS` overrides.

Cost of what does nothing: an array that comes out all zero is the scalar 0 and zero propagates through
products (a product evaluates its cheaper factor first); differences of a uniform field are exactly 0
without computing them; a subexpression written twice in a term is one shared node, computed once per
gather; conservation is counted only for a term that actually changed something.

## One surface

`surface_distance(fields, Surface{sdf, solid}, p)` is what rays stop at and bodies collide with:
the law's coarse `distance(solid)` far away (trimmed to stay a lower bound), the exact distance to the
voxel cubes within a cell (flat faces, sharp edges), and a box rule beyond the lattice. Ray hits and
body stops share `kSurfaceSkin`. A sweep marches the same function, so nothing tunnels.

## Gates

- `voxel_gpu_smoke` — every op of the language, `distance`, `poisson` and the whole shipped law with every
  element cast and magma underfoot, against the CPU reference; every contract reported by name; two runs on one
  card agree to the bit; what is cast during a window move lands where it was cast; a window move costs the law no
  time; the picture is the state. Without a hardware GPU it says so and passes.
- `voxel_lod_smoke` — level follows the pixels a voxel covers, with hysteresis; what the view wants is covered
  exactly once; about the same number of meshes at every zoom; a frame never waits for a build; no holes while
  zooming; the last state is what is drawn, even if it changed during a build; an edited region stays exact when
  far; far columns stand where the chunks would; meshing changes nothing in the world.
- `voxel_map_smoke` — several lands and many islands with climates, rivers, roads, peoples and ruins; the same seed makes the same bits,
  alone or in a chunk; every level of water is held in, and a river stays in its bed under the law; every whole
  building can be walked to every room and floor; no building or wall is buried; nearly every settlement builds its
  own way, for the land its people began in, and what happened to it stands in it and is told; bad styles are refused,
  and every style reads back as its text; houses stand and timber burns by the law;
  each layer has its own far (thin rivers go, big ones stay, canopy, roofs, town patches) and stands where the near
  one does; `height_bounds` bounds; the window follows you up and down with no jump; a fire left behind burns about
  as much as one you stay by (controls: stayed by, far life off); creatures walk on and come back; the land's law
  stays in bounds; all of it deterministic, in line and on threads.
- `voxel_open_world_smoke` — the generator is a function; the world has soil, turf, trees, herds; you
  stand on it and the window follows you; a dug hole survives leaving and returning; a chunk looks the same
  from the lattice and the store; levels of detail get cheaper and keep trunks; skirts and sea level at seams;
  force breaks the ground and fire burns a tree down; the law on its own thread gives the same bits however busy
  the machine, and a frame pays less for it.
- `voxel_life_smoke` — grazers eat (with a control), flee your scent (with a control), fear fire, starve; the
  language refuses lattice ops in creatures; the law changes in flight and bad text changes nothing; the
  printed equations are the running law; the same bits on 1, 4 and 7 threads; fire walks through turf and stops
  at bare ground (control: too little heat); a hunter kills and lives on it (controls: no hunter, no prey); fed
  grazers breed up to the limit; `show` lines compile.
- `voxel_law_smoke` — shipped laws compile; reads/writes derived; broken laws refused by line;
  conservation checked (falsified with a leaky term); `max_dt` keeps the maximum principle (falsified
  by ignoring it); range reported or clamped; determinism + its one-kelvin anti-vacuity twin; magic
  needs no C++ (mana kindles fuel and adds to gravity's source); fire walks down a fuse; mass pulls light.
- `voxel_ball_smoke` — a thrown ball of voxels comes to rest as what it is made of (stone, ice, wood), where it fell and
  not in the floor; at 60 m/s it stops on its own side of a wall; it hurts and throws the body it hits; the same throws
  give the same bits.
- `voxel_world_smoke` — the SDF is a distance; a dropped ball rests on the floor; nothing tunnels (and
  the naive teleport does, measured); camera ray and swept body stop at the same point; digging opens
  the hole for light and body alike; hot rock glows only while hot; mass bends light toward itself, hot air
  away from itself.

## The voxel world (assets/law)

`world.law`: stone (`solid`), `rubble`, `lava`, `water`, `ice`, `wood`, `leaf` are fractions of a voxel; `occ =
solid + rubble + ice + wood + leaf` is what bodies stand on and rays stop at. Stone melts above 1400 K (with latent heat),
lava flows and sets back into stone; rubble falls and heaps; water falls, levels, freezes and thaws
(latent heat), boils, and quenches lava into stone. `harm` is what heat does to a body.

Trees are `wood` and `leaf`: solid to stand on, poor conductors, and above `ignite` they burn at a steady
rate until they are gone (a burn rate proportional to what remains starves its own flame halfway, and a
trunk stands charred forever). Fire also walks through turf and stops at bare ground. `life.law` grows turf on soil and burns it; `grass` lives in the air cell on
top of the soil and is drawn as the soil's top face.

`magic.law`: one mana field per element. Fire mana heats; earth mana sets into stone where there is
room; force mana breaks stone into rubble and its gradient throws bodies (`push_*`); water mana
condenses; ice mana draws heat out; air mana's gradient throws bodies.

`magic.law` also has storm (heats, hurts bodies, more in water), life (bodies mend in it, and `life.law` grows turf
with it) and pull (bodies are drawn up its gradient, the one mana that draws in).

`pours.law`: a named pour = a shape (ball, wall, beam, ring, pillar, cut) of some field of the law, with a second field in the
same shape with `also`. Nothing about it costs, waits or flies: `Playground::pour_at` puts it where it is asked, and what it
does to the world after that is the law. Bodies read `PlayBody::surge` (a blow's step holds its velocity) and `stagger`
(thrown, the feet do not catch it yet). Everything that happened is told as `PlayEvent`s, for whoever draws it; nothing in the
simulation reads them back.

`VoxelFigure` is a figure made of cubes, read from text (`assets/world/skins`, `beasts`, `clothes`), parts turning about their pivots;
pure picture. `demo/sandbox/Effects` is what comes off a pour or a landing (cubes, the view shaking): none of it
reaches the simulation.

Matter-moving ops, all conservative: `fall(F,B,speed)` drops F into the free space below (`1 - B - F`);
`spread(F,B,k)` levels F sideways into free neighbours (liquids); `topple(F,B,rate)` slides what cannot
fall straight off a ledge, diagonally down (a 45-degree heap). `conserves a + b + c` checks a group's
total.

Cost: idle terms are free (an all-zero field evaluates as the scalar 0 and zero propagates through
products), so the 48x48x20 proving ground steps in ~3-9 ms on the CPU. Gate: `voxel_playground_smoke`.

## The open world

`Config::open_world`: a world of 262 x 262 km and 512 m of height, continents and island chains in a sea that goes on
beyond the map, a pure function of
`(seed, chunk)` (`generate_chunk`). A chunk is a cube of 16 voxels (8 m). Only a **window** of chunks around you is
simulated (3 x 3 x 3 by default; in the client 9 x 9 x 3 on the GPU, 72 m across and 24 m tall, or 5 x 5 x 3 on the
CPU): that is the lattice the solver runs on. Everything keeps lattice coordinates; `origin()` turns them into
world metres.

When you walk a metre past the middle chunk the window moves a chunk, up and down too (with more slack, so a jump is
not a move): chunks that leave go to the `ChunkStore`, chunks that stay slide across the lattice, chunks that arrive
are read. A stored chunk keeps only the fields that differ from what the generator makes, so an untouched chunk
costs nothing and a dug hole is still there when you return. A body left behind (the target) is parked.

The window moving depends on your position alone, so it is as deterministic as the rest.

**The law on its own thread** (`Config::async_fields`, the client's mode). A field step starts at a fixed tick
and is taken up at the next one; bodies meanwhile read the last finished state, and what they write (pours,
eat and emit lines) shows at once in that state and lands on the solver's at the same fixed tick. When a step
is taken up never depends on how long it took: a slow step makes the frame wait, not the world differ. Gate:
two runs on a differently busy main thread give the same bits. Gates and experiments step in line by default.
The client runs fields at 15 Hz (`field_every = 4`); `max_dt` keeps the law as fine as it asks.

A window move runs on the worker too, in place of two field steps (the next step makes up the time): the lattice
slides, leaving chunks are captured, arriving ones read. Two field ticks later the main thread takes it up:
bodies shift, creatures leave and arrive, and what was written meanwhile lands in the cells it now lives in.

## The layers of the map

Everything a chunk holds comes down through layers (`WorldMap`), each a pure function of the seed and the layers
above it. The map is finite, so drainage and history can be worked out whole when the world is made (about 2.5 s),
and a chunk only reads the answers.

| Layer | What it is | How far away it looks |
|---|---|---|
| continent | continents and island chains, plates, ranges where they meet; 2048 x 2048 nodes, 128 m apart | octaves finer than a voxel are not computed |
| climate | wind carries the sea inland and ranges wring it out; temperature by latitude and height; biomes | the biome colours the surface |
| water | every pit fills to its spill, every drop has a way to the sea; rivers, lakes | a river too thin to show goes, a big one stays a voxel wide |
| history | peoples, foundings, plague, floods, war, ruin (`chronicle()`); 16 ages | what it left: ruins are low and charred |
| plan | levelled ground, streets by the town's layout, lots along and between them, squares, temples, towers, town walls, monuments, graves, roads between places | roads go first, then a town is a patch of its roofs |
| building | walls, rooms, stairs, roofs: a function of a cell, by style | a roof at the height of the roof |
| voxels | what a chunk holds (`Terrain.cpp`) | trees become canopy |

**Water stays where it is put.** The law moves water freely, so every level is held in by what stands next to it: a
water column beside anything lower is itself a bank, as high as its water would stand. So rivers are chains of still
pools, each spilling over a sill as drawn but not running: the law leaves them as they are. (A river that really
flows needs sources and sinks at the window's edge. It does not have them yet.)

**No two towns alike.** Three steps, each with a reason you can read in the result:

1. *A people's way of building comes from the land it began in* (`WorldMap::architect`): what grows and what can be
   quarried there decides walls; rain and cold decide roofs (steep where snow must slide off, flat or domed where it is
   hot and dry); wet ground means piles; hot places have high rooms and walled yards, cold ones low rooms and few
   windows. Within that a people has its own mind: layout (grid, winding, rings round the square, one long street,
   scattered farmsteads), temple (hall, dome, stepped, spire), jetties, wings, towers.
2. *Each town takes that and changes it by what happened to it* (`make_local_styles`). A town that burned no longer
   builds in what burns; one that flooded stands higher; one that has been fought over has a wall; a rich one builds
   high. Details wander from town to town.
3. *What happened stays standing* (`plan_settlement`). The burnt quarter is left charred, the quarter emptied by plague
   is left to fall, with a field of graves; a conqueror builds a quarter in its own style; a war won raises a stone in
   the square. `chronicle()` says why, and the HUD reads it to you.

**Styles are text.** `style_text` and `parse_styles` go both ways, and every style in the world, the worked-out ones
too, reads back as what it is: this is the surface a model can write to, with the same refusals (a storey too low for
a door, floors with no room for a stair, an unknown biome or material). Nothing calls a model yet. Now and then a people
takes a style from `assets/world/styles.world` instead of working one out. Walls and roofs of wood and leaf burn, by
the law, as trees do.

**Nothing is buried.** A town levels its ground, over a longer way the steeper the slope; a lot is checked for level and
dry ground under all of it, not at its corners; a town wall stands inside the levelled ground and goes over any bank it
meets. A town stands near a river, not in it.

## Beyond the window (`FarLife.cpp`)

Three grains, each cheaper than the last, all taken up at fixed ticks, so the world stays a function of its inputs.

- **Busy chunks.** A chunk that leaves hot, burning, molten or full of mana goes on under the same law: on the CPU, up
  to six at a time and none beside another, each with a ring of its neighbours. What arrives in the ring goes to the
  neighbour, which wakes if it matters. A fire left behind burned 103 where one stayed by burned 112 in 40 s. Too many at
  once and they fall behind real time (`far_lag`, shown on the HUD); past 90 s behind, that time is simply missed.
  Water and rubble are not what wakes a chunk, and across a chunk face matter is passed without an exact account,
  so nothing is promised to be conserved out here.
- **Far creatures.** A creature that leaves walks over the map's columns once a second: its amble, the ground it can
  stand on, nothing else: it does not eat, breed or get hurt. Within 1.5 km of you it is kept; beyond that it is gone,
  and only the land's numbers remain.
- **The land.** `assets/law/map.law`, the same language and solver on one cell per node, every ten seconds: how much
  life each node can feed, herds and hunters on it. A chunk seen for the first time is peopled from these numbers.

## The law as a program: the IR, CUDA, JAX

`ir/LawIR`: the parser's trees carry the CPU interpreter's buffers and caches; the IR is the same law with none of that.
One SSA graph for all terms, learned schemes and derives, built bottom-up with structural hashing (a value written in
four equations is one instruction) and folding with the CPU's own semantics (`0 * x` is 0 without looking at x, exactly
as the interpreter does). Each instruction knows its shape (a number or a field), how far from its cell it reads, and
what one point of it costs with everything inlined. Then per program:

- stages: `distance`, `poisson`, reductions, networks, and any argument of a stencil whose point costs more than
  `kInlineBudget` times the stencil's points: those are computed over the lattice first and read from memory;
- in place: if no field the program writes is read off its own cell through what is inlined, the store happens in the
  same kernel (each thread computes all rates of its cell, then writes); otherwise gather, then store.

`cuda/`: the IR becomes one translation unit (a `v<id>(x, y, z)` device function per instruction, so stencils nest as
in the HLSL backend; stage kernels; one gather kernel per program) plus the generic kernels (EDT axis passes, Jacobi,
contracts, ring scatter and linearise). NVRTC compiles it for the card's `sm_XY`, with `--fmad=false` so the arithmetic
is the reference's; the CUBIN is cached by text. The driver API and NVRTC are loaded at run time (`CudaApi`), only the
entry points used are declared.

- The lattice is a ring: `I(x, y, z)` adds the origin modulo the size. `Solver::shift_window` records a move and the
  arriving boxes; the backend moves the origin and scatters only those boxes. What is kept on the card for distance and
  poisson moves with the ring, so a moved window does not reseed them. Downloads linearise first.
- No float atomics: range checks count with integers and keep the least bad index with `atomicMax(~i)`; totals and
  reductions sum rows, then planes, in order. Two runs on one card agree to the bit.
- Reductions stay on the card (`U[]`), read back only for a `max_dt`. Re-run flags let distance and poisson kernels
  return at once when their input repeated.
- Learned terms under guard: snapshot what the term writes (device copies), run the learned program with its own
  `learned max_dt`, read back only this term's contract records, and on a breach restore the snapshot, free the records
  and run the classical program. `StepProfile::fallbacks` counts it.
- Profile: a CUDA event before and after each term, the whole step, the derives, uploads and downloads.

Decided against: Direct3D–CUDA interop. The picture is meshed on the CPU from the fields the mesher already receives;
what costs is the readback, and that goes through pinned memory, only for fields whose change flag is set.

`tools/lawjax`: every instruction has a `jnp` twin with the engine's semantics, a step is the engine's (Lie splitting,
substeps, gather-then-apply, derives). With fixed substep counts it is a pure function: `jax.grad` through unrolled
steps is how models 2 and 3 are trained against data the engine's CPU reference produced (`lawc --dump`).

## Networks in the law

`net NAME in N hidden H... out M act tanh|relu|silu [half] [file PATH]`, weights in `assets/nets/NAME.net` (text; written
by the trainers), the last layer zero when there is no file, so an untrained net outputs exactly 0.

- `mlp(NET, a, b, ...)`: the first output, cell by cell; also at a point (creatures).
- `flux(NET, F, a, ...)`: across each face the net sees both cells both ways round; the flux is half the difference, so
  it is antisymmetric to the bit (the scheme does not depend on which way the lattice is laid out). F changes only by
  differences of face fluxes, so `conserves F` holds for any weights (up to float rounding). Limited per face (the same
  limit on both sides) so that in one step no cell goes past F's declared range: positivity and maximum by construction.
  Which piece gives which property is measured in `bench/ablations.py`.
- `ldiffuse(NET, k, F)`: a face flux interpolated from four cells, F[i-1] F[i] | F[j] F[j+1], with weights the net
  gives from the window scaled to its spread and from k dt / h^2; weights summing to zero, starting at [0 -1 1 0]
  (`diffuse` exactly when untrained), half the difference of both ways round.
- `learned d F = ...` / `learned max_dt = ...` in a term: the scheme run instead when learned schemes are on, under guard.

On the card a net is generated code with its sizes as constants: fp32 in the reference's order, or (`half`) a warp of 32
rows on tensor cores, `mma.sync.aligned.m16n8k16` in inline PTX, the fp32 accumulators of one layer becoming the next
layer's f16x2 operand fragments in registers; weights are packed on the host into fragment order. Networks are stages:
they run warp-wide, never inside another stencil.

## Minds that plan

`plan = COST` and `observe F... [horizon N] [samples N] [dt S]` in a creature block (`mind/WorldModel`): the creature no
longer climbs a seek gradient. Its observation is its locals and each observed field at its cell and on a ring of eight
points 3 `sense` out; a 2-hidden-layer MLP gives ds/dt from (s, a). Every `dt` each creature of the kind adds a
transition (what it saw, what it did, what it sees now) to its kind's replay; every few plans the kind trains on a batch
(CUDA: fused forward/backward rows, gradients one thread a weight, Adam). Planning is MPPI, all creatures at once: a
block a creature, a thread a future, the learned ODE integrated in registers with the weights in shared memory, the cost
the plan line compiled from the law into the kernel; the futures weighed by a softmin with a temperature from their
spread. The noise is a hash of (creature, tick, future, step). What it does is the plan's first step, plus exploration
that fades as it learns. No CUDA: the same on the CPU with fewer futures.

## The law on the GPU (Direct3D 11, the fallback)

`Config::gpu_fields`: when the law compiles (or changes while the world runs) every term and derive becomes HLSL:
one function per expression node, a function of the cell coordinate, so a stencil is the same function at another
coordinate and stencils nest in anything. Compute shaders run on the high-performance adapter through its own
Direct3D 11 device; compiled shaders are cached by source text in the temp directory, so only what changed
recompiles. No Direct3D (or a law that does not compile for it) and the law stays on the CPU, with a line in the log.

- The lattice is one buffer. A term is a gather pass into a second buffer and a store pass, per substep.
- `distance`: seeds are compared with the last ones and the passes are dispatched indirectly, only if one changed.
  Each axis scans outward until no farther cell can win, so it is exact, not a jump flood.
- `poisson`: Jacobi sweeps, skipped while the source repeats. Reductions read one value back, only when a term
  that is about to run asks for it.
- Contracts: range over every cell with atomics only on a bad cell; totals in two levels (rows, then planes), so
  all of the card works on them. All are read back once, at the end of the step, and named as on the CPU.
- What is idle is known without looking at a cell: a term whose right-hand sides are zero given the fields known
  to be zero is never dispatched.
- Stores flag which fields really changed. Only those come back, get copied to the picture and are compared by
  the mesher; edits name their z-plane and only those planes go up. In a quiet 9 x 9 window that is 4 of 37 fields.

The GPU computes in single precision and the CPU reference does not agree with it to the bit, only to the
rounding: over four seconds of everything at once they stay within about 1e-6 of each other. Two runs on one card
agree to the bit. Across different cards or drivers the bits are not promised.

## Drawing the open world

The camera is orthographic, so every voxel on the screen is the same size: level of detail depends on the pixels
a voxel covers at this zoom, not on distance (`WorldMesher`, shown by `demo/sandbox/WorldView`). Each level doubles the
voxel's width and the region it is drawn in, so a region is always 16 voxels wide and the screen holds about the
same number of meshes at any zoom, from single cubes to the whole world (twelve levels; a voxel of 1 km at the
last). Height coarsens more slowly than width, so hills keep their shape. The mouse wheel sets how far you see; the
camera's eye and far plane move back with it.

Only what the camera sees (plus a margin) is built, nearest first, on worker threads; a frame takes up what is ready
within a budget and an old mesh stays until its replacement has come. Live chunks are classified where the field
step runs. The two nearest levels are made from real chunks, as thick as the ground and what stands on it asks. From
level 2 on a voxel is one column of the map asked for at that coarseness (`far_column`), with no chunk generated; a
column whose chunk the law has touched, or that is in the window, is read from what is really there. The view is
looked through from the top of a tree down, with `height_bounds` for each box, so a tall world costs what is in sight.

## Creatures

A creature is a body whose mind is law, read at the point where it stands:

```
creature grazer
  radius 0.3  speed 2.4  hp 40  wander 0.35  sense 2.5  heed 0.05
  state hunger init 0.3 range 0 1                # its own numbers...
  d hunger = hunger_rate - eat_gain * eaten      # ...with their own equations
  seek = (0.2 + hunger) * grass - 0.3 * scent    # dx/dt = speed * grad(seek) / (|grad(seek)| + heed)
  eat grass = graze * hunger * grass             # taken from its cell (never more than is there)
  emit scent = 1                                 # added to its cell
  hurt = starve * step(hunger - 0.98)            # HP/s, on top of what `harm` does to any body
  breed = litter * step(0.25 - hunger)           # newborns/s: adds up; each whole one is born beside it
  colour 0.63 0.49 0.31
```

A field name reads its trilinear sample at the body; `hp`, `eaten` (what its eat lines took last tick, per
second) and `age` are always there. `sense` is how far either side it compares `seek`. What the pull leaves
of its stride goes to ambling on a heading that changes every two seconds (by its id and the tick). Whole-
lattice ops are refused in creature expressions: derive a field with the op and read that.

`creature player` is the body you steer: `seek` is your keyboard, the other lines hold (you leave scent).

Everything between creatures goes through fields. A grazer emits `musk`; a hunter climbs it, and where it is
thick the hunter emits `claw`, which a grazer's `hurt` line reads and the hunter feeds on the musk it takes. No
line in the engine says "hunter" or "prey". `Config::max_creatures` caps a window, so a law that breeds cannot
swamp the frame.

## What shows in the air

The picture draws matter by itself; everything else is invisible until the law says how to show it:

```
show T         colour 1.0 0.55 0.12  above 480  full 1300  alpha 0.8    # flame is hot air
show smoke     colour 0.27 0.27 0.30 above 0.08 full 1.5   alpha 0.6
show mana_fire colour 1.0 0.45 0.10  above 0.3  full 8     alpha 0.6
```

A cell not full of matter, with the field above `above`, holds a translucent cube, growing to a whole voxel by
`full`. A new element needs no renderer code.

## The law, changed while the world runs

`Playground::reload_law` compiles new text. If anything is broken, the running law stays and the errors name
the line. Otherwise every field keeps its values by name, new fields start at their init, params take the
new values, creatures keep their bodies and the states the new law still has.

`to_math` / `law_equations` print the running expression trees as the equations they are: `diffuse(k, T)` as
div(k grad T), `step` as the Heaviside H, `pos` as [x]+. The sandbox's law panel (F1, `demo/sandbox/Panels.cpp`) shows
them, every param on a live slider, and the contracts broken so far.

## Next

Rivers that flow; a smoother step
between levels of detail; batches of near-identical creatures keyed by their field samples, with automata written by
a model; styles written by a model, from biome and history, through the same checks.

## What a step costs

The frame does not wait for the law (`field_never_waits`, the client): a step that is not done when it is due is left to finish, and the next one makes up the time, up to `field_most_owed`; past that the law runs slower than you. Gates keep the deterministic mode, where a slow step makes the frame wait instead.

On the GPU a term is dispatched only if something can come of it. `Bounds` carries each field's lowest and highest value (read off the cells at the end of every step) through the expression: `step(T - melt_point)` is exactly [0, 0] while nothing is near melting. Terms keep the bounds of what they write up to date for the terms after them; if that is ever short, a term starts a step late. A derive runs only if something it reads moved. Contracts are looked at in one step of `law_checks_every` (8 in the client, every step in the gates).

## Fire, what holds, what grows

`alight` is flame that takes hold. Burning asks for `T > ignite` and `alight > alight_min`, and keeps itself alight; the flame goes out by itself and reaches just past what burns. `mana_fire` is hot and only scorches where it stands; `mana_ember` (`kindle`), lava and lightning light things.

`held` is what is left of the support on the way from the ground: 1 at the footing (`footing`: rock with rock all round it that nobody built, marked `built` by the generator, or `rim`, the edge of the lattice, past which the law takes the world on trust). Resting on the voxel below costs `hold_up`; through anything else, `hold_stone` through stone (it has no tension: 8 m is as far as it spans) and `hold_fibre` through wood. It is looked at only where `shaken` is: a wave from whatever gave way, crossing a little air, so a cut in a trunk is crossed and a valley is not. What nothing leads back to the footing comes down. A tree cut through, a house with its base taken out and a span too long for stone are one rule. Trees are generated joined face to face, because a support goes through faces.

Trees are worked out from the numbers in `assets/world/trees.world`; the formulas are at the top of that file.

The picture: `fs_basic.sc` thins a tube toward the camera above your feet (dithered, nothing to sort). Under a roof it opens the house as wide as it is and takes off what is overhead. Trees are marked in the alpha of their vertices (254) and always step aside in a tube of their own, seen or not: lines of sight cannot find what is above the lattice. The walls are still there for the law.

## People

A town is `creature town` in `assets/law/people.law`: a body-less creature, one step of its states a month. `q_G` are shares of the work, kept adding up to 1, so that is a replicator and no list of jobs exists: a good is a `q_`, `p_`, `res_`, `cult_`, `trade_` set of states and a line in `assets/world/people.world`. An `event NAME STATE above|below X weight W [sets STATE V]` line makes a state crossing a line into a fact.

Facts are one number a town (`FactRow`), a telegraph equation per fact over the graph of the roads (`rumour_*` params). A person is a town and an index, and everything about it comes from a hash of that: what it knows is `hash(person, fact) < share`. Only what a real body lived is stored (`Record`). `src/field/Society.cpp` is all of this; it never touches a voxel.

In the window, a town's people out of doors are bodies (`config.people`, `src/field/People.cpp`): `creature villager` and `creature watchman`. Their minds are lines of the law: anger past `fight_at`, it comes for whoever it is set on with the moves of the book; fear past `flee_at`, it runs. Struck, it answers or runs, and the watch comes for whoever struck; a death is written down and told.

Beyond the window they are where their day has them, with no body (`people_beyond`, `day_place`): from the last place to this one at a walking pace (work, the market, home, a watch's round), indoors when home. Who is out of doors today is a hash of who they are and the day (`out_today`, `Config::out_share`), the same for both: a person the window comes to becomes a body where its day had it. Gate: `voxel_beyond_smoke`.

## What people wear, what falls

`assets/world/items.world` is what people wear and hold: clothes by the weather and their work, and the weapon their moves are (`outfit`). Nothing is carried, bought or used up. A blow that breaks things leaves force mana where it strikes the world, and what the law breaks is gone.

With `config.falling` wood and leaves that nothing holds up leave the fields as a rigid body (Falling.cpp): its own mass and inertia, gravity, contacts with the surface bodies stand on, and voxels again when it rests. Bodies collide with it in the air as they do with the ground (`out_of_pieces`). A leaf goes with the nearest wood through other leaves, and one further than 8 cells from any wood falls. A blow opens the cut on its own side, so a trunk goes over toward where it was struck from. `fibre_falls = 0` takes wood and leaves out of the law's own collapse; off, they come down as rubble.

`throw_ball` makes the same kind of piece out of nothing: a ball of stone, wood or ice cubes with its own inertia, thrown with a velocity. A piece that would cross more than a third of a voxel in a tick moves in substeps, so a fast one does not pass through a wall between two looks at the surface. Where it rests it is stone, wood or ice in the fields, and from then on the law's.

A body is only what someone is born with (`Looks`, `body_model`). Clothes are figures in `assets/world/clothes`, on the same bones, with `f:` and `m:` lines for the two shapes. A person's looks come from their parents' genes and how warm the place is (`looks_of_person`), what they wear from the weather and their work (`dress_person`).

A door is a line of a style (`door none|plank|double|slide|lift stone|wood MATERIAL`); the generator shuts it.

## Beyond the window, for what leaves it

The law runs only in the window, but the world is there past it. `ground_at` is the top of what stands at a world column as the map makes it (ground, roof, crown); a piece that leaves the window sideways lands on that, and where it rests out there it is written into the chunk it lies in (`set_down_beyond`), kept as any changed chunk is. `pick_world` is a ray through the window's surface and then the map's heights. Zoomed out, `WorldMesher` draws the live window at full detail over the coarser regions around it (`live_overlay`, `window_box`; the renderer clips the coarse ones there), so what is seen in it is what bodies stand on.

`Config::observer`: nobody plays. The body the window follows is a point of interest: it does not move by itself, nothing sees it, hits it or smells it, and it cannot be hurt.

## What breaks, and what is left

`crumble` (world.law) is how much of what force or a collapse breaks lies there after as rubble; the rest is dust, gone. So `solid + wood + rubble` is not conserved by those two terms. `mana_cut` (magic.law) takes what it passes through and leaves nothing.

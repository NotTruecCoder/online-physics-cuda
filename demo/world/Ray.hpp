#pragma once

#include "world/Body.hpp"
#include "field/Grid.hpp"

// A RAY THROUGH THE SAME FIELDS THE WORLD IS MADE OF. It sphere-traces the surface bodies stand on, so what
// the cursor picks is what stops you. If the law derives an optical index `n`, a ray bends along grad n
// (d/ds (n dx/ds) = grad n): heat haze and gravitational lensing are terms in the law that write into `n`.
// Bending is safe with sphere tracing: any path shorter than the SDF value from a point cannot reach a
// surface, curved or not, so arc-length steps bounded by the SDF never tunnel.
namespace fieldengine::field {

struct RayBindings {
    Surface surface;  // what the ray stops at
    int index = -1;   // optional: optical index n (rays bend toward higher n)
};

struct RayHit {
    bool hit = false;
    Vec3 position{}; // the hit, or where a missing ray left the volume
    Vec3 normal{};
    Vec3 direction{}; // the ray's direction when it stopped (bent rays end turned)
    double length = 0.0;
    int steps = 0;
};

// Marches one ray from `origin` along `dir` through the lattice volume.
RayHit march_ray(const FieldSet& fields, const RayBindings& bindings, Vec3 origin, Vec3 dir, double max_length,
    bool bend = true);

// Clips a ray to the lattice box [0, n*cell]^3: the stretch [t0, t1] of it inside; false if it misses the box.
bool clip_to_box(const GridShape& shape, Vec3 origin, Vec3 dir, double& t0, double& t1);

} // namespace fieldengine::field

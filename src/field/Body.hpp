#pragma once

#include "field/Grid.hpp"

namespace fieldengine::field {

// A BODY MEETS THE WORLD WHERE A RAY DOES. Collision and rendering both ask surface_distance(), so
// "it looks like a wall" and "it stops you like a wall" cannot disagree: there is one surface.

// Where a ray counts as "on the surface" and where a body stops short of it, as a fraction of a voxel.
// ONE number for both: two thresholds would be two surfaces.
inline constexpr double kSurfaceSkin = 0.02;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator*(Vec3 a, double s);
double dot(Vec3 a, Vec3 b);
double length(Vec3 a);
Vec3 normalize(Vec3 a);

// Unit gradient of a field at a point (central differences over half a cell).

// What the world's surface is made of. `sdf` (the law's `distance(solid)`) answers far from the
// surface, where only a safe step size matters; `solid` (optional) answers near it, exactly, as the
// distance to the voxel CUBES — so faces are flat and edges sharp for the eye and the body alike. The
// coarse field alone rounds every concave corner into a fillet a cell wide.
struct Surface {
    int sdf = -1;
    int solid = -1;
};

// THE distance to the world's surface (negative inside). Beyond the lattice is air: the answer there
// is built from the distance to the lattice box — never the clamped edge value, which would extend
// the edge voxels to infinity and let a body stand on a floor the eye sees end. Always a lower bound,
// so marching by it never steps through anything.
double surface_distance(const FieldSet& fields, const Surface& surface, Vec3 p);
// Outward normal of that surface (gradient of surface_distance).
Vec3 surface_normal(const FieldSet& fields, const Surface& surface, Vec3 p);

struct Contact {
    bool touching = false;
    Vec3 normal{};   // outward, from the surface toward the body
    double gap = 0.0; // distance from the sphere's surface to the world's (negative = was inside)
};

// Pushes a sphere out of the solid and removes the velocity component heading into it. Returns the
// deepest contact found.
Contact resolve_sphere(const FieldSet& fields, const Surface& surface, Vec3& position, Vec3& velocity, double radius);

// Moves a sphere from `from` toward `to` and stops it at first contact, marching the surface exactly
// like a ray: no step can be longer than the free space around the sphere, so nothing tunnels through
// a wall however fast it moves. Returns the position reached.
Vec3 sweep_sphere(const FieldSet& fields, const Surface& surface, Vec3 from, Vec3 to, double radius,
    Contact* contact = nullptr);

} // namespace fieldengine::field

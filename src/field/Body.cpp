#include "field/Body.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fieldengine::field {

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double length(Vec3 a) { return std::sqrt(dot(a, a)); }

Vec3 normalize(Vec3 a)
{
    const double len = length(a);
    return len > 1e-12 ? a * (1.0 / len) : Vec3{0.0, 0.0, 1.0};
}

namespace {

// Coarse answer from the law's distance field, made safe beyond the lattice box.
double coarse_distance(const FieldSet& fields, int sdf, Vec3 p)
{
    const GridShape& shape = fields.shape();
    const Vec3 hi{shape.nx * shape.cell, shape.ny * shape.cell, shape.nz * shape.cell};
    const Vec3 q{std::clamp(p.x, 0.0, hi.x), std::clamp(p.y, 0.0, hi.y), std::clamp(p.z, 0.0, hi.z)};
    const double inside = fields.sample(sdf, q.x, q.y, q.z);
    const double outside = length(p - q);
    if (outside <= 0.0) {
        return inside;
    }
    // Every solid voxel is inside the box, so the box distance bounds the answer from below; and the
    // surface seen from q is at most `outside` closer from p.
    return std::max(outside, inside - outside);
}

bool solid_at(const FieldSet& fields, int solid, int x, int y, int z)
{
    const GridShape& shape = fields.shape();
    return shape.inside(x, y, z) && fields.values(solid)[shape.index(x, y, z)] >= 0.5f;
}

// Distance to the nearest voxel cube of the OTHER kind among the 27 around p (out-of-lattice is air),
// capped at one cell. The cap is what makes 27 enough: any cube two or more cells over is at least a
// cell away, so below one cell the answer is exact and at one cell it is a true lower bound.
double exact_voxel_distance(const FieldSet& fields, int solid, Vec3 p, bool& inside)
{
    const GridShape& shape = fields.shape();
    const double c = shape.cell;
    const int ix = static_cast<int>(std::floor(p.x / c));
    const int iy = static_cast<int>(std::floor(p.y / c));
    const int iz = static_cast<int>(std::floor(p.z / c));
    inside = solid_at(fields, solid, ix, iy, iz);
    const int ry = shape.ny > 1 ? 1 : 0;
    const int rz = shape.nz > 1 ? 1 : 0;
    double best = c;
    for (int dz = -rz; dz <= rz; ++dz) {
        for (int dy = -ry; dy <= ry; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = ix + dx;
                const int y = iy + dy;
                const int z = iz + dz;
                if (solid_at(fields, solid, x, y, z) == inside) {
                    continue;
                }
                // Distance from p to the cube [x, x+1] * c (zero if p is in it, which it is not).
                const double ox = std::max({x * c - p.x, 0.0, p.x - (x + 1) * c});
                const double oy = shape.ny > 1 ? std::max({y * c - p.y, 0.0, p.y - (y + 1) * c}) : 0.0;
                const double oz = shape.nz > 1 ? std::max({z * c - p.z, 0.0, p.z - (z + 1) * c}) : 0.0;
                best = std::min(best, std::sqrt(ox * ox + oy * oy + oz * oz));
            }
        }
    }
    return best;
}

} // namespace

double surface_distance(const FieldSet& fields, const Surface& surface, Vec3 p)
{
    const double c = fields.shape().cell;
    const double coarse = coarse_distance(fields, surface.sdf, p);
    if (surface.solid < 0) {
        return coarse;
    }
    // The coarse field measures from voxel CENTRES; to a cube it can overshoot by up to (sqrt(3)-1)/2
    // of a cell on a diagonal, so far from the surface it is trimmed to stay a lower bound...
    if (coarse > 2.5 * c) {
        return coarse - 0.6 * c;
    }
    if (coarse < -2.5 * c) {
        return coarse;
    }
    // ...and near it the cubes answer exactly (within a cell; a cell is a safe step beyond that).
    bool inside = false;
    const double exact = exact_voxel_distance(fields, surface.solid, p, inside);
    return inside ? -exact : std::max(exact, coarse - 0.6 * c);
}

Vec3 surface_normal(const FieldSet& fields, const Surface& surface, Vec3 p)
{
    const GridShape& shape = fields.shape();
    // Small enough that a face's normal is its own, not its neighbour's.
    const double e = 0.05 * shape.cell;
    auto s = [&](Vec3 at) { return surface_distance(fields, surface, at); };
    return normalize(Vec3{
        s(p + Vec3{e, 0, 0}) - s(p - Vec3{e, 0, 0}),
        shape.ny > 1 ? s(p + Vec3{0, e, 0}) - s(p - Vec3{0, e, 0}) : 0.0,
        shape.nz > 1 ? s(p + Vec3{0, 0, e}) - s(p - Vec3{0, 0, e}) : 0.0,
    });
}

Contact resolve_sphere(const FieldSet& fields, const Surface& surface, Vec3& position, Vec3& velocity, double radius)
{
    Contact deepest;
    deepest.gap = 1e30;
    // A few passes: leaving one surface can press into another in a corner.
    for (int pass = 0; pass < 4; ++pass) {
        const double distance = surface_distance(fields, surface, position);
        const double gap = distance - radius;
        if (gap >= 0.0) {
            if (pass == 0) {
                deepest.gap = gap;
                deepest.normal = surface_normal(fields, surface, position);
            }
            break;
        }
        const Vec3 normal = surface_normal(fields, surface, position);
        if (gap < deepest.gap || !deepest.touching) {
            deepest = Contact{.touching = true, .normal = normal, .gap = gap};
        }
        position = position + normal * (-gap);
        const double into = dot(velocity, normal);
        if (into < 0.0) {
            velocity = velocity - normal * into;
        }
    }
    return deepest;
}

Vec3 sweep_sphere(const FieldSet& fields, const Surface& surface, Vec3 from, Vec3 to, double radius, Contact* contact)
{
    const Vec3 path = to - from;
    const double total = length(path);
    if (contact != nullptr) {
        *contact = Contact{.gap = surface_distance(fields, surface, to) - radius};
    }
    if (total <= 1e-12) {
        return from;
    }
    const Vec3 dir = path * (1.0 / total);
    const double min_step = 0.02 * fields.shape().cell;
    const double skin = kSurfaceSkin * fields.shape().cell;
    double travelled = 0.0;
    Vec3 p = from;
    for (int i = 0; i < 4096 && travelled < total; ++i) {
        const double free = surface_distance(fields, surface, p) - radius;
        if (free <= skin) {
            // Touching. Only motion INTO the surface stops here; a body resting on the floor still
            // slides along it and lifts off it, or nothing standing could ever walk.
            const Vec3 normal = surface_normal(fields, surface, p);
            if (dot(dir, normal) < -1e-6) {
                if (contact != nullptr) {
                    *contact = Contact{.touching = true, .normal = normal, .gap = free};
                }
                return p;
            }
        }
        const double step = std::min(total - travelled, std::max(free * 0.9, min_step));
        travelled += step;
        p = from + dir * travelled;
    }
    return p;
}

} // namespace fieldengine::field

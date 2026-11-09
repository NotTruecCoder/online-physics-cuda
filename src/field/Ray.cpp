#include "field/Ray.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::field {

bool clip_to_box(const GridShape& shape, Vec3 origin, Vec3 dir, double& t0, double& t1)
{
    const double size[3] = {shape.nx * shape.cell, shape.ny * shape.cell, shape.nz * shape.cell};
    const double o[3] = {origin.x, origin.y, origin.z};
    const double d[3] = {dir.x, dir.y, dir.z};
    t0 = 0.0;
    t1 = 1e30;
    for (int a = 0; a < 3; ++a) {
        if (std::abs(d[a]) < 1e-12) {
            if (o[a] < 0.0 || o[a] > size[a]) {
                return false;
            }
            continue;
        }
        double near = (0.0 - o[a]) / d[a];
        double far = (size[a] - o[a]) / d[a];
        if (near > far) {
            std::swap(near, far);
        }
        t0 = std::max(t0, near);
        t1 = std::min(t1, far);
    }
    return t1 > t0;
}

namespace {

bool in_box(const GridShape& shape, Vec3 p, double margin)
{
    return p.x >= -margin && p.y >= -margin && p.z >= -margin && p.x <= shape.nx * shape.cell + margin &&
        p.y <= shape.ny * shape.cell + margin && p.z <= shape.nz * shape.cell + margin;
}

Vec3 raw_gradient(const FieldSet& fields, int id, Vec3 p)
{
    const GridShape& shape = fields.shape();
    const double e = 0.5 * shape.cell;
    auto s = [&](double x, double y, double z) { return static_cast<double>(fields.sample(id, x, y, z)); };
    return Vec3{
        (s(p.x + e, p.y, p.z) - s(p.x - e, p.y, p.z)) / (2.0 * e),
        shape.ny > 1 ? (s(p.x, p.y + e, p.z) - s(p.x, p.y - e, p.z)) / (2.0 * e) : 0.0,
        shape.nz > 1 ? (s(p.x, p.y, p.z + e) - s(p.x, p.y, p.z - e)) / (2.0 * e) : 0.0,
    };
}

} // namespace

RayHit march_ray(const FieldSet& fields, const RayBindings& bindings, Vec3 origin, Vec3 dir, double max_length,
    bool bend)
{
    RayHit result;
    const GridShape& shape = fields.shape();
    const double cell = shape.cell;
    dir = normalize(dir);
    double t0 = 0.0;
    double t1 = 0.0;
    if (!clip_to_box(shape, origin, dir, t0, t1)) {
        result.direction = dir;
        return result;
    }
    Vec3 p = origin + dir * t0;
    double travelled = t0;
    const bool bending = bend && bindings.index >= 0;
    const double hit_epsilon = kSurfaceSkin * cell;
    const double min_step = 0.05 * cell;
    for (int i = 0; i < 2048; ++i) {
        result.steps = i + 1;
        if (travelled > max_length || !in_box(shape, p, 0.5 * cell)) {
            break;
        }
        const double distance = surface_distance(fields, bindings.surface, p);
        if (distance < hit_epsilon) {
            result.hit = true;
            result.position = p;
            result.normal = surface_normal(fields, bindings.surface, p);
            result.direction = dir;
            result.length = travelled;
            return result;
        }
        double step = std::max(distance * 0.9, min_step);
        if (bending) {
            const Vec3 g = raw_gradient(fields, bindings.index, p);
            const double gl = length(g);
            if (gl > 1e-9) {
                const double n = std::max(1e-3, static_cast<double>(fields.sample(bindings.index, p.x, p.y, p.z)));
                // A bent ray is integrated with short steps where the index changes.
                step = std::min(step, cell);
                const Vec3 transverse = g - dir * dot(g, dir);
                dir = normalize(dir + transverse * (step / n));
            }
        }
        p = p + dir * step;
        travelled += step;
    }
    result.position = p; // where the ray left the volume (or gave up)
    result.direction = dir;
    result.length = travelled;
    return result;
}

} // namespace fieldengine::field

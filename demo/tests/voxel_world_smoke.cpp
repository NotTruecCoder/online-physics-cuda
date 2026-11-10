// GATE: ONE SURFACE. WHAT YOU SEE IS WHAT STOPS YOU, AND LIGHT OBEYS THE SAME LAW AS EVERYTHING ELSE.
//
// Properties:
//   1. The distance field is a distance: above a flat floor it reads the height above it, inside rock
//      it is negative, and in free space its slope is ~1.
//   2. A dropped body comes to rest ON the floor, never in it.
//   3. Nothing tunnels: a body crossing a one-voxel wall in a single frame stops at it. The naive way
//      (teleport, then push out) ends on the far side — that is measured too, so the sweep's reason to
//      exist is part of the gate.
//   4. The ray and the body stop at the same place: march a camera ray, sweep a sphere down it, the two
//      points agree to a tenth of a voxel.
//   5. Digging a hole opens it for light and body alike — they read the same field, so they cannot
//      disagree before the edit or after it.
//   6. What the mesher draws comes from the fields: hot rock glows only while hot.
//   7. Mass bends light toward itself (a ray passing either side of a dense voxel turns toward it);
//      without the mass the same ray goes straight.
//   8. Hot air bends light away from itself (heat haze): same mechanism, a different term writing `n`.
//
// Falsify by: removing the `- 0.5` half-cell offset in distance() (1 fails); dropping the velocity
// projection in resolve_sphere (2 drifts); replacing sweep_sphere with a teleport (3 fails); marching
// rays with a different epsilon than bodies stop at (4 fails); forgetting refresh_derived after the
// dig (5 fails — and in the game, so would the player); flipping the bend sign in march_ray (7, 8 fail).

#include "world/Body.hpp"
#include "field/Law.hpp"
#include "world/Ray.hpp"
#include "field/Solver.hpp"
#include "world/VoxelMesh.hpp"
#include "tests/Gate.hpp"

#include <cmath>
#include <iostream>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

Law load_world()
{
    std::vector<LawSource> sources;
    if (const std::optional<LawSource> source = read_law_file("assets/law/world.law")) {
        sources.push_back(*source);
    }
    CompileResult result = compile_law(sources);
    for (const std::string& error : result.errors) {
        std::cerr << "  world.law: " << error << "\n";
    }
    check(result.law.has_value(), "world.law compiles");
    return result.law ? *result.law : Law{};
}

void fill(Solver& solver, const std::string& field, int x0, int x1, int y0, int y1, int z0, int z1, float value)
{
    const GridShape& shape = solver.fields().shape();
    std::vector<float>& values = solver.fields().values(solver.field(field));
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (shape.inside(x, y, z)) {
                    values[shape.index(x, y, z)] = value;
                }
            }
        }
    }
}

// 16 m x 16 m x 6 m: a floor (top at z = 1 m), a pillar, a thin wall and a pool of lava.
void build_scene(Solver& solver)
{
    fill(solver, "solid", 0, 31, 0, 31, 0, 1, 1.0f);    // floor
    fill(solver, "material", 0, 31, 0, 31, 0, 1, 1.0f);
    fill(solver, "solid", 14, 17, 14, 17, 2, 7, 1.0f);  // pillar: x,y in [7, 9] m, top at 4 m
    fill(solver, "material", 14, 17, 14, 17, 2, 7, 2.0f);
    fill(solver, "solid", 24, 24, 4, 27, 2, 9, 1.0f);   // wall: x in [12, 12.5] m
    fill(solver, "material", 24, 24, 4, 27, 2, 9, 3.0f);
    fill(solver, "T", 4, 8, 20, 26, 1, 1, 1800.0f);     // lava in the floor's top layer
    solver.refresh_derived();
}

RayBindings bindings_for(const Solver& solver)
{
    RayBindings b;
    b.surface = Surface{.sdf = solver.field("sdf"), .solid = solver.field("solid")};
    b.index = solver.field("n");
    return b;
}

} // namespace

int main()
{
    const Law world = load_world();
    const GridShape shape{.nx = 32, .ny = 32, .nz = 12, .cell = 0.5};
    Solver solver(world, shape);
    build_scene(solver);
    const FieldSet& f = solver.fields();
    const int sdf = solver.field("sdf");
    const Surface surface{.sdf = sdf, .solid = solver.field("solid")};

    // ---- 1. the distance field is a distance --------------------------------------------------------
    {
        bool heights = true;
        for (double z = 1.5; z <= 3.0; z += 0.25) {
            const double d = f.sample(sdf, 3.0, 3.0, z);
            heights = heights && std::abs(d - (z - 1.0)) < 0.05;
        }
        check(heights, "above the floor, the field reads the height above it");
        check(f.sample(sdf, 3.0, 3.0, 0.25) < 0.0, "inside rock the field is negative");
        bool unit_slope = true;
        for (const Vec3 p : {Vec3{3.0, 3.0, 2.5}, Vec3{5.0, 10.0, 3.0}, Vec3{10.0, 4.0, 4.0}}) {
            const double e = 0.25;
            const Vec3 g{(f.sample(sdf, p.x + e, p.y, p.z) - f.sample(sdf, p.x - e, p.y, p.z)) / (2 * e),
                (f.sample(sdf, p.x, p.y + e, p.z) - f.sample(sdf, p.x, p.y - e, p.z)) / (2 * e),
                (f.sample(sdf, p.x, p.y, p.z + e) - f.sample(sdf, p.x, p.y, p.z - e)) / (2 * e)};
            unit_slope = unit_slope && length(g) > 0.8 && length(g) < 1.2;
        }
        check(unit_slope, "in free space the field's slope is ~1 (it is a distance, not a density)");
    }

    // ---- 2. a dropped body rests on the floor ---------------------------------------------------------
    {
        const double radius = 0.4;
        Vec3 p{3.0, 3.0, 4.0};
        Vec3 v{0.0, 0.0, 0.0};
        double worst_gap = 1e9;
        for (int i = 0; i < 180; ++i) {
            const double dt = 1.0 / 60.0;
            v.z -= 9.8 * dt;
            Contact contact;
            p = sweep_sphere(f, surface, p, p + v * dt, radius, &contact);
            if (contact.touching && dot(v, contact.normal) < 0.0) {
                v = v - contact.normal * dot(v, contact.normal);
            }
            resolve_sphere(f, surface, p, v, radius);
            worst_gap = std::min(worst_gap, surface_distance(f, surface, p) - radius);
        }
        check(std::abs(p.z - 1.4) < 0.06, "a dropped ball comes to rest on the floor (centre at " +
                std::to_string(p.z) + " m, expected 1.4)");
        check(worst_gap > -0.025, "...and was never inside it (deepest " + std::to_string(worst_gap) + " m)");
        check(std::abs(v.z) < 0.2, "...and stays put instead of jittering");
    }

    // ---- 3. nothing tunnels ----------------------------------------------------------------------------
    {
        const double radius = 0.2;
        const Vec3 start{10.0, 8.0, 3.0};
        const Vec3 end{15.5, 8.0, 3.0}; // 5.5 m in one frame, through the wall at x in [12, 12.5]
        Contact contact;
        const Vec3 swept = sweep_sphere(f, surface, start, end, radius, &contact);
        check(swept.x < 12.0 - radius + 0.05 && contact.touching, "a body crossing a wall in one frame stops at it (x = " +
                std::to_string(swept.x) + ")");
        Vec3 teleported = end;
        Vec3 dummy{};
        resolve_sphere(f, surface, teleported, dummy, radius);
        check(teleported.x > 12.5, "the naive teleport-then-push-out ends on the far side (why the sweep exists)");
    }

    // ---- 4. the ray and the body stop at the same place ----------------------------------------------
    {
        // Parallel rays down the isometric view, as the camera shoots them.
        const Vec3 forward = normalize(Vec3{-0.577, -0.577, -0.577});
        const Vec3 right = normalize(Vec3{forward.y, -forward.x, 0.0});
        const Vec3 up{right.y * forward.z - right.z * forward.y, right.z * forward.x - right.x * forward.z,
            right.x * forward.y - right.y * forward.x};
        const RayBindings b = bindings_for(solver);
        bool agree = true;
        int compared = 0;
        for (double v = -8.0; v <= 8.0; v += 3.0) {
            for (double u = -10.0; u <= 10.0; u += 3.0) {
                const Vec3 origin = Vec3{8.0, 8.0, 2.0} + right * u + up * v - forward * 40.0;
                const RayHit hit = march_ray(f, b, origin, forward, 200.0, false);
                if (!hit.hit) {
                    continue;
                }
                const Vec3 from = hit.position - forward * 4.0;
                const Vec3 stop = sweep_sphere(f, surface, from, hit.position + forward * 2.0, 0.0);
                agree = agree && length(stop - hit.position) < 0.1 * shape.cell;
                ++compared;
            }
        }
        check(compared >= 10, "enough camera rays hit the scene to compare (" + std::to_string(compared) + ")");
        check(agree, "every camera ray and a body swept down it stop at the same point");
    }

    // ---- 5. digging opens the hole for light and body alike -------------------------------------------
    {
        Solver dig(world, shape);
        build_scene(dig);
        const RayBindings b = bindings_for(dig);
        const Vec3 from{10.0, 8.25, 2.75};
        const Vec3 dir{1.0, 0.0, 0.0};
        auto state = [&]() {
            const RayHit ray = march_ray(dig.fields(), b, from, dir, 4.0, false);
            const Vec3 body = sweep_sphere(dig.fields(), Surface{.sdf = dig.field("sdf"), .solid = dig.field("solid")}, from, from + dir * 4.0, 0.2);
            return std::pair{ray.hit && ray.position.x < 12.6, body.x < 12.5};
        };
        const auto [ray_blocked, body_blocked] = state();
        check(ray_blocked && body_blocked, "before the dig, the wall stops both the ray and the body");
        fill(dig, "solid", 24, 24, 15, 17, 4, 6, 0.0f);
        dig.refresh_derived();
        const auto [ray_after, body_after] = state();
        check(!ray_after && !body_after, "after the dig, both pass through the hole");
        check(ray_blocked == body_blocked && ray_after == body_after, "light and body never disagree about the wall");
    }

    // ---- 6. what is drawn comes from the fields ---------------------------------------------------------
    {
        const VoxelPalette palette;
        auto red_of = [&](const FieldSet& fields) {
            return classify(MatterView::of(fields), 6, 23, 1, palette).r;
        };
        Solver cold(world, shape);
        build_scene(cold);
        fill(cold, "T", 0, 31, 0, 31, 0, 11, 290.0f);
        cold.refresh_derived();
        check(red_of(f) > red_of(cold.fields()) + 60, "hot rock glows, and the same rock at room temperature does not");
    }

    // ---- 7. mass bends light toward itself -------------------------------------------------------------
    {
        Solver lens(world, shape);
        build_scene(lens);
        fill(lens, "mass", 16, 16, 16, 16, 9, 9, 5000.0f); // centre (8.25, 8.25, 4.75) m
        for (int i = 0; i < 12; ++i) {
            lens.refresh_derived();
        }
        const RayBindings b = bindings_for(lens);
        const RayHit above = march_ray(lens.fields(), b, Vec3{1.0, 9.5, 4.75}, Vec3{1.0, 0.0, 0.0}, 10.0);
        const RayHit below = march_ray(lens.fields(), b, Vec3{1.0, 7.0, 4.75}, Vec3{1.0, 0.0, 0.0}, 10.0);
        check(!above.hit && !below.hit, "the probe rays pass the mass in open air");
        check(above.direction.y < -1e-4, "a ray passing on the +y side turns toward the mass (" +
                std::to_string(above.direction.y) + ")");
        check(below.direction.y > 1e-4, "a ray passing on the -y side turns toward it too (" +
                std::to_string(below.direction.y) + ")");
        const RayHit straight = march_ray(f, bindings_for(solver), Vec3{1.0, 9.5, 4.75}, Vec3{1.0, 0.0, 0.0}, 10.0);
        check(std::abs(straight.direction.y) < 1e-6, "without the mass the same ray goes straight");
    }

    // ---- 8. hot air bends light away from itself --------------------------------------------------------
    {
        Solver haze(world, shape);
        build_scene(haze);
        fill(haze, "T", 10, 13, 0, 31, 8, 11, 1500.0f); // a slab of hot air at x in [5, 7] m, z in [4, 6] m
        haze.refresh_derived();
        const RayHit ray = march_ray(haze.fields(), bindings_for(haze), Vec3{7.0, 0.5, 5.0}, Vec3{0.0, 1.0, 0.0}, 14.0);
        // Both: it TURNED away (direction) and it STAYED out (a ray bent the wrong way crosses the slab
        // and turns back on the far side, so the final direction alone does not prove anything).
        check(ray.direction.x > 1e-3 && ray.position.x > 7.0, "a ray grazing hot air bends away from it (dir.x " +
                std::to_string(ray.direction.x) + ", left at x " + std::to_string(ray.position.x) +
                ") — heat haze is a term in the law, not a shader");
    }

    return fieldengine::gate::report("voxel_world");
}

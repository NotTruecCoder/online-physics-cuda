// GATE: THE WORLD IS ENDLESS, GENERATED, AND REMEMBERS WHAT THE LAW DID TO IT.
//
// Claims about the open world a player walks through:
//   1. The generator is a function: the same chunk comes out bit for bit the same, in any order, and
//      its ground is where terrain_top says (no seam can open between two chunks of one column).
//   2. The world has what was promised: stone in layers, soil with turf on it, sea, trees (trunk and
//      crown, wood and leaves), and herds of creatures somewhere near home.
//   3. You stand on generated ground; walking far moves the simulated window with you, and the world
//      you walk into is solid under your feet.
//   4. A chunk the law never changed is not stored (only "seen"); a chunk you dug a hole in keeps the
//      hole after it leaves the window, and the hole is still there when you come back.
//   5. A chunk looks the same drawn from the live lattice and from the store.
//   6. Far chunks are coarser: fewer faces at each level of detail, and a tree keeps its trunk.
//   7. Two walks with the same inputs give the same world, bit for bit.
//   8. Where two levels of detail meet there is no slit: a skirt keeps the side faces near the surface.
//      The sea stands at one height at every level.
//  10. The law on its own thread: the world is as deterministic as stepping in line, whatever the machine
//      is doing; what bodies write reaches the law; a frame does not pay for the solver.
//   9. The surface breaks and the trees burn by the law alone: force turns ground to rubble, and fire set
//      in a crown burns the tree away, wood and leaves both.
//
// Falsify by: salting the generator with a global counter (1 fails); dropping the `leaving`
// branch's store write in save_chunk (4 fails); not shifting bodies in center_window_on (3 fails:
// you fall into the void); counting a trunk once in downsample (6 fails); ignoring `skirt` in
// mesh_box, or filling every coarse liquid voxel to the top (8 fails); zeroing `shatter`, `char_rate` or
// `leaf_rate` in the law (9 fails).

#include "field/Playground.hpp"
#include "field/VoxelMesh.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

Playground open_world()
{
    std::vector<std::string> errors;
    std::optional<Playground> world = Playground::load_default(errors, true);
    for (const std::string& e : errors) {
        std::cerr << "  " << e << "\n";
    }
    check(world.has_value(), "the open world loads");
    if (!world) {
        std::exit(fieldengine::gate::report("voxel_open_world"));
    }
    return std::move(*world);
}

double sum(const ChunkData& data, const char* field)
{
    const std::vector<float>* v = data.find(field);
    double s = 0.0;
    if (v != nullptr) {
        for (const float x : *v) {
            s += x;
        }
    }
    return s;
}

// Walks east (world +x) for `ticks`, the way the keyboard would.
void walk(Playground& world, int ticks, double dx = 1.0, double dy = 0.0)
{
    for (int i = 0; i < ticks; ++i) {
        // Hop now and then: the terrain has ledges higher than a step.
        world.step(Control{.move_x = dx, .move_y = dy, .jump = (i % 40) == 0});
    }
}

std::vector<VoxelCell> classify_chunk(const MatterView& view, int n, int nz)
{
    std::vector<VoxelCell> cells;
    const VoxelPalette palette;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                cells.push_back(classify(view, x, y, z, palette));
            }
        }
    }
    return cells;
}

// (Not its top row: the turf on it is drawn from the chunk above, which only the lattice has here.)
bool same_cells(const std::vector<VoxelCell>& a, const std::vector<VoxelCell>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size() * 15 / 16; ++i) {
        if (a[i].kind != b[i].kind || a[i].r != b[i].r || a[i].g != b[i].g || a[i].b != b[i].b ||
            a[i].tr != b[i].tr || a[i].fill != b[i].fill) {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    const TerrainConfig t;
    const int n = t.chunk;
    const std::shared_ptr<const WorldMap> map = WorldMap::of(t);
    int home_x = 0;
    int home_y = 0;
    map->home(home_x, home_y);
    const int home_cx = home_x / n;
    const int home_cy = home_y / n;
    const int home_cz = map->column(home_x, home_y).top / n;

    // 1. A function of (seed, chunk).
    {
        ChunkData a;
        ChunkData b;
        ChunkData other;
        generate_chunk(t, home_cx + 3, home_cy - 2, home_cz, a, nullptr);
        generate_chunk(t, home_cx + 7, home_cy + 9, home_cz, other, nullptr);
        generate_chunk(t, home_cx + 3, home_cy - 2, home_cz, b, nullptr);
        bool same = a.fields.size() == b.fields.size();
        for (std::size_t i = 0; same && i < a.fields.size(); ++i) {
            same = a.fields[i].first == b.fields[i].first && a.fields[i].second == b.fields[i].second;
        }
        check(same, "1: a chunk generates the same, bit for bit, whatever was generated in between");
        bool tops = true;
        for (int y = 0; y < n && tops; ++y) {
            for (int x = 0; x < n && tops; ++x) {
                const int wx = (home_cx + 3) * n + x;
                const int wy = (home_cy - 2) * n + y;
                const int top = terrain_top(t, wx, wy);
                ChunkData here;
                generate_chunk(t, home_cx + 3, home_cy - 2, top / n, here, nullptr);
                const std::size_t cell = (static_cast<std::size_t>(top % n) * n + y) * n + x;
                const std::shared_ptr<const ColumnTile> tile = map->tile(home_cx + 3, home_cy - 2);
                // The ground is there, unless something was built into it.
                tops = (*here.find("solid"))[cell] == 1.0f || !tile->buildings.empty();
            }
        }
        check(tops, "1: every column's ground is at terrain_top (so neighbouring chunks meet without a seam)");
    }

    // 2. What the world is made of, over a 12x12-chunk area around home (and a few layers of it).
    {
        double soil = 0.0;
        double turf = 0.0;
        double wood = 0.0;
        double leaves = 0.0;
        std::size_t creatures = 0;
        for (int cy = home_cy - 6; cy < home_cy + 6; ++cy) {
            for (int cx = home_cx - 6; cx < home_cx + 6; ++cx) {
                for (int cz = home_cz - 1; cz <= home_cz + 1; ++cz) {
                    ChunkData data;
                    std::vector<CreatureSpawn> spawns;
                    generate_chunk(t, cx, cy, cz, data, &spawns);
                    const std::vector<float>& m = *data.find("material");
                    const std::vector<float>& s = *data.find("solid");
                    for (std::size_t i = 0; i < m.size(); ++i) {
                        soil += s[i] > 0.5f && m[i] == static_cast<float>(material::soil) ? 1.0 : 0.0;
                    }
                    turf += sum(data, "grass");
                    wood += sum(data, "wood");
                    leaves += sum(data, "leaf");
                    creatures += spawns.size();
                }
            }
        }
        std::cout << "  12x12 chunks: soil " << soil << ", turf " << turf << ", wood " << wood
                  << ", leaves " << leaves << ", creatures " << creatures << "\n";
        check(soil > 1000.0 && turf > 500.0, "2: soil with turf on it");
        check(wood > 20.0, "2: wood, in trunks or walls");
        check(creatures >= 2, "2: creatures can start somewhere near home");
    }

    Playground world = open_world();
    const GridShape g = world.fields().shape();
    check(g.nx == n * world.window_chunks() && g.nz == n * world.window_layers(), "3: the simulated window is window_chunks chunks wide");
    const int start_cx = world.window_cx();
    const int start_cy = world.window_cy();
    const int start_cz = world.window_cz();

    // 3. Standing on home ground, then walking far.
    for (int i = 0; i < 90; ++i) {
        world.step(Control{});
    }
    {
        const PlayBody& me = world.player();
        const Vec3 w = me.position + world.origin();
        std::cout << "  home: (" << w.x << ", " << w.y << ", " << w.z << ") ground top cell "
                  << terrain_top(t, static_cast<int>(std::floor(w.x / t.cell)), static_cast<int>(std::floor(w.y / t.cell)))
                  << ", velocity z " << me.velocity.z << ", hp " << me.hp << "\n";
    }
    check(world.player().grounded, "3: you stand on the generated ground");
    const double start_z = world.player().position.z + world.origin().z;
    check(start_z > (t.sea_level + 1) * t.cell, "3: home is dry land, above the sea");

    // 4. Dig a hole (three cells of ground gone) in the window's first chunk, then leave it behind.
    const int hole_cx = start_cx;
    const int hole_cy = start_cy;
    const int hx = 5;
    const int hy = 6;
    const int hole_top = terrain_top(t, hole_cx * n + hx, hole_cy * n + hy);
    const int hole_cz = (hole_top - 2) / n;
    check(hole_cz >= start_cz && hole_cz < start_cz + world.window_layers(), "4: the ground to dig is in the window");
    {
        FieldSet& f = world.solver().fields();
        const int solid = f.find("solid");
        const int lz = start_cz * n;
        for (int z = hole_cz * n; z <= hole_top; ++z) {
            if (z >= hole_top - 2) {
                f.values(solid)[g.index(hx, hy, z - lz)] = 0.0f;
            }
        }
        world.solver().refresh_derived();
    }
    std::vector<VoxelCell> live;
    {
        const MatterView view = MatterView::of(world.fields()).offset(0, 0, (hole_cz - start_cz) * n);
        live = classify_chunk(view, n, n);
    }
    const std::uint64_t version_before = world.window_version();
    // Twelve seconds east: the window moves several chunks. World position is what you experience: it
    // must never jump when the window does, and the walk must add up to a walk.
    const double start_world_x = (world.player().position + world.origin()).x;
    double worst_jump = 0.0;
    for (int i = 0; i < 60 * 12; ++i) {
        const Vec3 before = world.player().position + world.origin();
        // (Weaving a little: a trunk in the way is not what is being tested.)
        world.step(Control{.move_x = 1.0, .move_y = (i / 60) % 2 == 0 ? 0.7 : -0.7, .jump = (i % 40) == 0});
        const Vec3 after = world.player().position + world.origin();
        worst_jump = std::max(worst_jump, length(Vec3{after.x - before.x, after.y - before.y, 0.0}));
    }
    const double travelled = (world.player().position + world.origin()).x - start_world_x;
    std::cout << "  walked " << travelled << " m, worst step " << worst_jump << " m\n";
    check(worst_jump < 0.5, "3: your world position never jumps when the window moves");
    check(travelled > 20.0 && travelled < 90.0, "3: twelve seconds of walking is a walk's length in the world");
    check(world.window_version() > version_before && world.window_cx() > start_cx + 2,
        "3: the window followed you east (it moved " + std::to_string(world.window_cx() - start_cx) + " chunks)");
    check(world.player().position.z > 0.5 && world.player().hp > 0.0,
        "3: the ground you walked into held you (z " + std::to_string(world.player().position.z) + ")");
    check(!world.in_window(hole_cx, hole_cy, hole_cz), "4: the dug chunk has left the window");
    const StoredChunk* dug = world.store().find(hole_cx, hole_cy, hole_cz);
    bool has_solid = false;
    if (dug != nullptr) {
        for (const StoredField& f : dug->fields) {
            has_solid = has_solid || f.name == "solid";
        }
    }
    check(dug != nullptr && has_solid, "4: the dug chunk is stored, with its solid field");
    {
        ChunkData data;
        world.read_chunk(hole_cx, hole_cy, hole_cz, data);
        const std::vector<float>& solid = *data.find("solid");
        const std::size_t cell = (static_cast<std::size_t>(hole_top - hole_cz * n) * n + hy) * n + hx;
        check((hole_top - hole_cz * n) >= n || solid[cell] == 0.0f, "4: read from the store, the hole is there");
        // 5. The same chunk drawn live and drawn from the store.
        const std::vector<VoxelCell> stored = classify_chunk(MatterView::of(data, n, n), n, n);
        check(same_cells(live, stored), "5: a chunk looks the same from the lattice and from the store");
    }
    {
        // A chunk the window passed over and nobody touched is remembered only as seen.
        std::size_t untouched = 0;
        std::size_t seen = 0;
        for (int cy = start_cy; cy < start_cy + 3; ++cy) {
            for (int cx = start_cx; cx < world.window_cx(); ++cx) {
                for (int cz = start_cz; cz < start_cz + world.window_layers(); ++cz) {
                    if (const StoredChunk* c = world.store().find(cx, cy, cz)) {
                        ++seen;
                        untouched += c->fields.size() <= 3 ? 1 : 0; // grass regrowth, scent and what beasts leave may differ
                    }
                }
            }
        }
        check(seen > 3 && untouched * 2 >= seen,
            "4: most chunks the window left keep little or nothing (" + std::to_string(untouched) + " of " +
                std::to_string(seen) + ")");
    }
    // ...and back home: the hole is in the live lattice again.
    world.center_window_on(hole_cx + 1, hole_cy + 1, hole_cz);
    {
        const FieldSet& f = world.fields();
        const int solid = f.find("solid");
        const int lx = (hole_cx - world.window_cx()) * n + hx;
        const int ly = (hole_cy - world.window_cy()) * n + hy;
        check(f.values(solid)[g.index(lx, ly, hole_top - world.window_cz() * n)] == 0.0f, "4: back in the window, the hole is still a hole");
    }

    // 6. Level of detail.
    {
        // Find a trunk column in some chunk near home.
        ChunkData data;
        int tree_cx = 0;
        int tree_cy = 0;
        bool found = false;
        for (int cy = home_cy - 200; cy < home_cy + 200 && !found; ++cy) {
            for (int cx = home_cx - 200; cx < home_cx + 200 && !found; ++cx) {
                const std::shared_ptr<const ColumnTile> tile = map->tile(cx, cy);
                if (tile->trees.empty() || !tile->buildings.empty()) {
                    continue;
                }
                for (int cz = tile->min_top / n; cz <= tile->max_top / n + 1 && !found; ++cz) {
                    ChunkData d;
                    generate_chunk(t, cx, cy, cz, d, nullptr);
                    if (sum(d, "wood") > 60.0 && sum(d, "leaf") < 60.0) { // a trunk, and not only branches
                        tree_cx = cx;
                        tree_cy = cy;
                        data = d;
                        found = true;
                    }
                }
            }
        }
        check(found, "6: a chunk with a tree to look at");
        VoxelBox box;
        box.resize(n, n, n);
        box.wx = tree_cx * n;
        box.wy = tree_cy * n;
        const MatterView view = MatterView::of(data, n, n);
        const VoxelPalette palette;
        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    box.at(x, y, z) = classify(view, x, y, z, palette);
                }
            }
        }
        ChunkMesh fine;
        ChunkMesh half;
        ChunkMesh quarter;
        VoxelBox coarse;
        mesh_box(box, 0, t.cell, fine);
        downsample(box, 2, coarse);
        mesh_box(coarse, 0, t.cell * 2, half);
        VoxelBox coarser;
        downsample(box, 4, coarser);
        mesh_box(coarser, 0, t.cell * 4, quarter);
        std::cout << "  faces: full " << fine.faces << ", half " << half.faces << ", quarter " << quarter.faces << "\n";
        check(fine.faces > half.faces && half.faces > quarter.faces && quarter.faces > 0,
            "6: coarser levels have fewer faces, and still some");
        // A trunk is a column one voxel thin: at half resolution every stretch of bare trunk (nothing else
        // in its coarse voxel) must still be there. It is only a quarter of that voxel: it counts four times.
        std::size_t bare = 0;
        std::size_t kept = 0;
        const std::vector<float>& wood = *data.find("wood");
        const std::vector<float>& leaf = *data.find("leaf");
        const std::vector<float>& rock = *data.find("solid");
        auto at = [&](int x, int y, int z) { return (static_cast<std::size_t>(z) * n + y) * n + x; };
        for (int z = 0; z + 1 < n; z += 2) {
            for (int y = 0; y + 1 < n; y += 2) {
                for (int x = 0; x + 1 < n; x += 2) {
                    double woods = 0.0;
                    double others = 0.0;
                    for (int dz = 0; dz < 2; ++dz) {
                        for (int dy = 0; dy < 2; ++dy) {
                            for (int dx = 0; dx < 2; ++dx) {
                                const std::size_t i = at(x + dx, y + dy, z + dz);
                                woods += wood[i];
                                others += leaf[i] + rock[i];
                            }
                        }
                    }
                    if (woods > 0.0 && others == 0.0) {
                        ++bare;
                        kept += coarse.at(x / 2, y / 2, z / 2).kind == VoxelCell::Opaque ? 1 : 0;
                    }
                }
            }
        }
        std::cout << "  bare trunk voxels at half resolution: " << kept << " of " << bare << "\n";
        check(bare > 0, "6: this chunk has bare trunk to look at");
        check(kept == bare, "6: at half resolution a tree keeps its trunk");
    }

    // 8. Seams between levels of detail.
    {
        // A flat plain cut by the box's edge: border cells are solid, so nothing shows on the edge
        // unless the skirt says so.
        VoxelBox plain;
        plain.resize(8, 8, 8);
        for (int z = 0; z < 4; ++z) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    plain.at(x, y, z).kind = VoxelCell::Opaque;
                }
            }
        }
        ChunkMesh without;
        ChunkMesh with;
        mesh_box(plain, 1, 0.5, without);
        mesh_box(plain, 1, 0.5, with, 2);
        check(without.faces == 36, "8: without a skirt a flat plain is only its top");
        // The two top layers, around the rim of the 6x6 interior: 4 sides * 6 cells * 2 layers.
        check(with.faces == 36 + 48, "8: a skirt adds the side faces near the surface, and no deeper");

        // The sea: water to cell 12 inclusive, over rock to cell 9.
        VoxelBox sea;
        sea.resize(16, 16, 40);
        for (int z = 0; z <= 12; ++z) {
            for (int y = 0; y < 16; ++y) {
                for (int x = 0; x < 16; ++x) {
                    VoxelCell& cell = sea.at(x, y, z);
                    cell.kind = z <= 9 ? VoxelCell::Opaque : VoxelCell::Water;
                    cell.fill = 255;
                }
            }
        }
        auto surface = [](const ChunkMesh& mesh) {
            float top = 0.0f;
            for (const MeshVertex& v : mesh.vertices) {
                top = std::max(top, v.z);
            }
            return top;
        };
        ChunkMesh fine_sea;
        mesh_box(sea, 0, 0.5, fine_sea);
        for (const int f : {2, 4}) {
            VoxelBox coarse;
            downsample(sea, f, coarse);
            ChunkMesh coarse_sea;
            mesh_box(coarse, 0, 0.5 * f, coarse_sea);
            check(std::abs(surface(coarse_sea) - surface(fine_sea)) < 0.05f,
                "8: the sea stands at the same height at 1/" + std::to_string(f) + " resolution (" +
                    std::to_string(surface(coarse_sea)) + " vs " + std::to_string(surface(fine_sea)) + ")");
        }
    }

    // 9. Breaking and burning, by spells.
    {
        Playground fresh = open_world();
        for (int i = 0; i < 30; ++i) {
            fresh.step(Control{});
        }
        auto spell = [&](const char* name) -> const Pour& {
            for (const Pour& s : fresh.pours()) {
                if (s.name == name) {
                    return s;
                }
            }
            check(false, std::string("9: spells.law has ") + name);
            return fresh.pours().front();
        };
        const double stone = fresh.total("solid");
        const double rubble = fresh.total("rubble");
        const Vec3 feet = fresh.player().position;
        const std::optional<Vec3> ground =
            fresh.pick(Vec3{feet.x + 3.0, feet.y, feet.z + 6.0}, Vec3{0.0, 0.0, -1.0});
        check(ground.has_value(), "9: there is ground to aim at");
        for (int k = 0; k < 3 && ground; ++k) {
            fresh.pour(spell("force"), *ground - Vec3{0.0, 0.0, 0.2}, Vec3{0.0, 0.0, -1.0});
            for (int i = 0; i < 30; ++i) {
                fresh.step(Control{});
            }
        }
        std::cout << "  force: stone " << stone << " -> " << fresh.total("solid") << ", rubble " << rubble << " -> "
                  << fresh.total("rubble") << "\n";
        check(fresh.total("solid") < stone - 1.0, "9: force breaks the generated ground");
        check(fresh.total("rubble") > rubble + 1.0, "9: into rubble");

        const int wood_id = fresh.solver().field("wood");
        const int leaf_id = fresh.solver().field("leaf");
        // The heart of a crown: the top of a trunk, as high as trunks go in the window.
        Vec3 crown{};
        bool found = false;
        for (int z = g.nz - 1; z >= 0 && !found; --z) {
            for (int y = 0; y < g.ny && !found; ++y) {
                for (int x = 0; x < g.nx && !found; ++x) {
                    if (fresh.fields().values(wood_id)[g.index(x, y, z)] > 0.5f) {
                        crown = Vec3{(x + 0.5) * g.cell, (y + 0.5) * g.cell, (z + 0.5) * g.cell};
                        found = true;
                    }
                }
            }
        }
        check(found, "9: a tree in the window");
        const double wood = fresh.total("wood");
        const double leaves = fresh.total("leaf");
        (void)wood_id;
        for (int k = 0; k < 2; ++k) {
            fresh.pour(spell("kindle"), crown, Vec3{1.0, 0.0, 0.0});
            for (int i = 0; i < 60; ++i) {
                fresh.step(Control{});
            }
        }
        for (int second = 0; second < 8; ++second) {
            for (int i = 0; i < 60; ++i) {
                fresh.step(Control{});
            }
            double hottest = 0.0;
            for (const float v : fresh.fields().values(fresh.solver().field("T"))) {
                hottest = std::max(hottest, static_cast<double>(v));
            }
            std::cout << "    t+" << second + 2 << "s leaves " << fresh.total("leaf") << " wood " << fresh.total("wood")
                      << " hottest " << hottest << "\n";
        }
        std::cout << "  fire: leaves " << leaves << " -> " << fresh.total("leaf") << ", wood " << wood << " -> "
                  << fresh.total("wood") << "\n";
        check(fresh.total("leaf") < leaves - 3.0, "9: fire burns the crown");
        check(fresh.total("wood") < wood - 2.0, "9: and the fire walks down the trunk");
        for (const Violation& v : fresh.violations()) {
            std::cerr << "  contract: " << to_string(v.kind) << " [" << v.term << "] " << v.field << ": " << v.detail << "\n";
        }
        check(fresh.violations().empty(), "9: no contract broke");
    }

    // 10. The law on its own thread.
    {
        auto async_world = [&]() {
            std::vector<LawSource> laws;
            std::string spells;
            std::string path;
            std::vector<std::string> errors;
            check(Playground::read_default_law(laws, spells, path, errors), "10: the law files are there");
            Playground::Config config;
            config.open_world = true;
            config.async_fields = true;
            std::optional<Playground> w = Playground::create(laws, spells, errors, config);
            check(w.has_value(), "10: an async world builds");
            if (!w) {
                std::exit(fieldengine::gate::report("voxel_open_world"));
            }
            w->build_open_world();
            return std::move(*w);
        };
        auto play = [](Playground& w, int ticks, bool slow) {
            for (int i = 0; i < ticks; ++i) {
                Control control{.move_x = 0.8, .move_y = 0.5, .jump = i % 45 == 0};
                if (i == 30) {
                    fieldengine::gate::pour_toward(w, 0, w.player().position + Vec3{5.0, 2.0, 0.0}); // fire ahead
                }
                if (i == 60) {
                    fieldengine::gate::pour_toward(w, 4, w.player().position + Vec3{5.0, 2.0, 0.0}); // water ahead
                }
                w.step(control);
                if (slow && i % 7 == 0) {
                    // A busy machine: the main thread falls behind the worker, or the worker behind it.
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        };
        Playground a = async_world();
        Playground b = async_world();
        play(a, 500, false);
        play(b, 500, true);
        check(a.digest() == b.digest(), "10: the same inputs, a differently busy machine: the same bits");
        check(a.window_cx() != async_world().window_cx() || a.window_cy() != async_world().window_cy(),
            "10: the window moved with you");
        check(a.violations().empty(), "10: no contract broke");

        // What a body writes reaches the law, and shows at once.
        Playground c = async_world();
        for (int i = 0; i < 6; ++i) {
            c.step(Control{});
        }
        const int fire = c.fields().find("mana_ember");
        const int temperature = c.fields().find("T");
        const Pour* burn = nullptr;
        for (const Pour& s : c.pours()) {
            burn = s.name == "kindle" ? &s : burn;
        }
        const Vec3 spot = c.player().position + Vec3{6.0, 0.0, 4.0};
        c.pour(*burn, spot, Vec3{1.0, 0.0, 0.0});
        const double at_once = c.fields().sample(fire, spot.x, spot.y, spot.z);
        check(at_once > 1.0, "10: a pour shows in what bodies read at once");
        double hottest = 0.0;
        for (int i = 0; i < 30; ++i) {
            c.step(Control{});
            hottest = std::max(hottest, static_cast<double>(c.fields().sample(temperature, spot.x, spot.y, spot.z)));
        }
        check(hottest > 400.0, "10: and the law acts on it (" + std::to_string(hottest) + " K)");

        // A frame does not pay for the solver: main-thread cost of a tick, sync against async.
        // At a frame's pace, as in the client (flat out, the main thread just waits for the worker).
        struct Cost {
            double mean = 0.0;
            double worst = 0.0;
        };
        auto cost = [](Playground& w) {
            Cost out;
            for (int i = 0; i < 240; ++i) {
                const auto start = std::chrono::steady_clock::now();
                w.step(Control{});
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                out.mean += ms / 240.0;
                out.worst = std::max(out.worst, ms);
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            return out;
        };
        Playground inline_world = open_world();
        Playground threaded_world = async_world();
        const Cost in_line = cost(inline_world);
        const Cost threaded = cost(threaded_world);
        std::cout << "  main thread per tick at 60 Hz: in line " << in_line.mean << " ms (worst " << in_line.worst
                  << "), async " << threaded.mean << " ms (worst " << threaded.worst << ")\n";
        check(threaded.mean < in_line.mean, "10: the frame pays less for the law when it runs on its own thread");
    }

    // 7. Determinism.
    {
        Playground a = open_world();
        Playground b = open_world();
        walk(a, 400, 0.6, 0.8);
        walk(b, 400, 0.6, 0.8);
        check(a.digest() == b.digest(), "7: the same walk gives the same world, bit for bit");
        check(a.window_cx() == b.window_cx() && a.window_cy() == b.window_cy(), "7: and the same window");
    }
    for (const Violation& v : world.violations()) {
        std::cerr << "  contract: " << to_string(v.kind) << " [" << v.term << "] " << v.field << ": " << v.detail << "\n";
    }
    check(world.violations().empty(), "the law's contracts hold in the open world");
    return fieldengine::gate::report("voxel_open_world");
}

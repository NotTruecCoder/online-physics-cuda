// THE WORLD IS ALWAYS DRAWN, AT THE DETAIL THE SCREEN CAN SHOW. Level follows the pixels a voxel covers;
// what the view wants is exactly covered, with no holes while it changes; a build never holds a frame up; and
// nothing the mesher does reaches the simulation.
//
// Falsified by: not sliding the copies up and down, keeping a chunk's last picture when it leaves the window (10
// fails); ignoring the hysteresis in lod_for_zoom (1 fails); wrong regions in view (3 fails); dropping
// `covered` so an old mesh goes before its replacement came (5 fails); leaving the re-ask after a stale build out,
// replan_needed_ (6 fails); never reclassifying a live chunk (6 fails); treating an edited region as untouched
// ground, `changed_in` (7 fails); a far column sampling the wrong height (8 fails).
#include "tests/Gate.hpp"

#include "world/WorldMesher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <thread>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::optional<Playground> make_world()
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    Playground::Config c;
    c.open_world = true;
    c.window_chunks = 3;
    c.field_every = 4;
    std::optional<Playground> world = Playground::create(laws, spells, errors, c);
    world->build_open_world();
    return world;
}

// A stand-in for the renderer: what was uploaded, by handle.
struct Sink {
    int next = 0;
    std::map<int, std::size_t> faces;
    int uploads = 0;
    WorldMesher::Upload upload = [this](int handle, const ChunkMesh& mesh) {
        const int h = handle >= 0 ? handle : next++;
        faces[h] = mesh.faces;
        ++uploads;
        return h;
    };
    WorldMesher::Destroy destroy = [this](int handle) { faces.erase(handle); };
};

LodView view_of(const Playground& world, double pixels_per_metre, double radius)
{
    const Vec3 me = world.player().position + world.origin();
    LodView view;
    view.pixels_per_metre = pixels_per_metre;
    view.centre_x = me.x;
    view.centre_y = me.y;
    view.radius_metres = radius;
    const double cx = me.x;
    const double cy = me.y;
    view.visible = [cx, cy, radius](const double lo[3], const double hi[3]) {
        const double dx = std::max({lo[0] - cx, 0.0, cx - hi[0]});
        const double dy = std::max({lo[1] - cy, 0.0, cy - hi[1]});
        return dx * dx + dy * dy <= radius * radius;
    };
    return view;
}

// Runs frames until the mesher has caught up; false if it never does.
bool settle(WorldMesher& mesher, const Playground& world, const LodView& view, Sink& sink, double* worst_ms = nullptr)
{
    for (int frame = 0; frame < 20000; ++frame) {
        const auto start = std::chrono::steady_clock::now();
        mesher.update(world, view, sink.upload, sink.destroy);
        if (worst_ms != nullptr) {
            *worst_ms = std::max(*worst_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        if (mesher.settled()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

void check_levels()
{
    const double cell = 0.5;
    int level = 0;
    for (double px = 64.0; px > 0.001; px *= 0.9) {
        const int next = lod_for_zoom(px, cell, 4.0, level);
        check(next >= level, "1: zooming out never asks for more detail");
        level = next;
        check(px * cell * lod_level(level).factor >= 4.0 || level == kLodLevels - 1, "1: a voxel keeps at least its pixels on screen");
    }
    check(level == kLodLevels - 1, "1: far enough out, the coarsest level");
    // Held at a boundary, a small wobble does not rebuild the world back and forth.
    const double boundary = 4.0 / (cell * 2.0); // level 1 is just enough here
    const int above = lod_for_zoom(boundary * 1.02, cell, 4.0, 2);
    check(above == 2, "1: a level is kept until the finer one clears the bar by a margin");
    check(lod_for_zoom(boundary * 1.3, cell, 4.0, 2) == 1, "1: ...and then given up");
    for (int l = 0; l < kLodLevels; ++l) {
        const LodLevel d = lod_level(l);
        check(d.span == d.factor, "2: a region is always 16 voxels wide, at every level");
        check(16 % d.factor_z == 0 && d.factor_z <= d.factor, "2: height stays at least as fine as width");
    }
}

// Every chunk the view wants is drawn by exactly one mesh, at the level asked for.
void check_cover(const WorldMesher& mesher, const Playground& world, const LodView& view, int level, const char* what)
{
    const TerrainConfig& t = world.config().terrain;
    const double chunk_m = t.chunk * t.cell;
    std::map<std::pair<int, int>, int> times;
    std::map<std::pair<int, int>, int> overlay; // the live window at full detail over the coarse level (clipped out of it)
    const std::array<int, 4> box = mesher.window_box();
    const auto in_window = [&](int cx, int cy) { return cx >= box[0] && cx < box[2] && cy >= box[1] && cy < box[3]; };
    bool right_level = true;
    mesher.for_each_drawn([&](int, int l, int rx, int ry) {
        if (l == 0 && level != 0 && mesher.live_overlay() && in_window(rx, ry)) {
            ++overlay[{rx, ry}];
            return;
        }
        right_level = right_level && l == level;
        const int span = lod_level(l).span;
        for (int cy = ry; cy < ry + span; ++cy) {
            for (int cx = rx; cx < rx + span; ++cx) {
                ++times[{cx, cy}];
            }
        }
    });
    check(right_level, std::string("3: all at the level the zoom asks: ") + what);
    const int reach = static_cast<int>(view.radius_metres / chunk_m) + 2;
    const int ccx = static_cast<int>(std::floor(view.centre_x / chunk_m));
    const int ccy = static_cast<int>(std::floor(view.centre_y / chunk_m));
    int holes = 0;
    for (int cy = ccy - reach; cy <= ccy + reach; ++cy) {
        for (int cx = ccx - reach; cx <= ccx + reach; ++cx) {
            const double lo[3] = {cx * chunk_m, cy * chunk_m, 0.0};
            const double hi[3] = {(cx + 1) * chunk_m, (cy + 1) * chunk_m, t.nz * t.cell};
            if (view.visible(lo, hi) && times[{cx, cy}] == 0) {
                ++holes;
            }
        }
    }
    check(holes == 0, std::string("3: nothing in view is missing: ") + what);
    bool twice = false;
    for (const auto& [chunk, count] : times) {
        twice = twice || count > 1;
    }
    check(!twice, std::string("3: nothing is drawn twice: ") + what);
    if (mesher.live_overlay() && level != 0) {
        bool whole = true;
        for (int cy = box[1]; cy < box[3]; ++cy) {
            for (int cx = box[0]; cx < box[2]; ++cx) {
                whole = whole && overlay[{cx, cy}] == 1;
            }
        }
        check(whole, std::string("3: the live window is drawn once at full detail over the coarse level: ") + what);
    }
}

void check_view()
{
    std::optional<Playground> world = make_world();
    WorldMesher mesher;
    world->set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, world->config().terrain); });
    Sink sink;
    double worst = 0.0;
    std::size_t most_regions = 0;
    std::size_t least_regions = 1u << 30;
    // The radius a screen shows grows as the pixels per metre fall.
    for (const double px : {30.0, 6.0, 3.0, 1.5, 0.8}) {
        const LodView view = view_of(*world, px, 640.0 / px);
        const auto start = std::chrono::steady_clock::now();
        check(settle(mesher, *world, view, sink, &worst), "3: the mesher catches up");
        const double took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const WorldMesher::Stats& s = mesher.stats();
        std::cout << "  " << px << " px/m: level " << s.level << ", " << s.regions_drawn << " regions, " << s.faces << " faces, built in " << took << " ms\n";
        check_cover(mesher, *world, view, s.level, "after a zoom");
        most_regions = std::max(most_regions, s.regions_drawn);
        least_regions = std::min(least_regions, s.regions_drawn);
    }
    std::cout << "  worst update: " << worst << " ms\n";
    check(most_regions < 4 * least_regions + 200, "4: about the same number of meshes at every zoom");
    check(worst < 120.0, "4: a frame never waits for a build");
    // Static scene, no change: nothing more is built.
    const int uploads = sink.uploads;
    const LodView last = view_of(*world, 0.8, 800.0);
    for (int frame = 0; frame < 20; ++frame) {
        mesher.update(*world, last, sink.upload, sink.destroy);
    }
    check(sink.uploads == uploads, "4: a settled view builds nothing");
}

// While the zoom changes, what was there stays until what replaces it is there.
void check_no_holes_on_zoom()
{
    std::optional<Playground> world = make_world();
    WorldMesher mesher;
    Sink sink;
    const LodView fine = view_of(*world, 30.0, 25.0);
    check(settle(mesher, *world, fine, sink), "5: the first view settles");
    const LodView coarse = view_of(*world, 6.0, 25.0); // the same place, one level coarser or two
    bool holes = false;
    for (int frame = 0; frame < 20000 && (frame == 0 || !mesher.settled()); ++frame) {
        mesher.update(*world, coarse, sink.upload, sink.destroy);
        std::set<std::pair<int, int>> drawn;
        mesher.for_each_drawn([&](int, int l, int rx, int ry) {
            const int span = lod_level(l).span;
            for (int cy = ry; cy < ry + span; ++cy) {
                for (int cx = rx; cx < rx + span; ++cx) {
                    drawn.insert({cx, cy});
                }
            }
        });
        const TerrainConfig& t = world->config().terrain;
        const double chunk_m = t.chunk * t.cell;
        const int ccx = static_cast<int>(std::floor(coarse.centre_x / chunk_m));
        const int ccy = static_cast<int>(std::floor(coarse.centre_y / chunk_m));
        holes = holes || !drawn.contains({ccx, ccy});
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(!holes, "5: the spot you look at is drawn in every frame of a zoom");
    check(mesher.settled(), "5: the new level arrives");
}

// What the law does shows, and keeps showing when the chunk has left the window.
void check_law_shows()
{
    std::optional<Playground> world = make_world();
    WorldMesher mesher;
    world->set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, world->config().terrain); });
    Sink sink;
    LodView view = view_of(*world, 30.0, 30.0);
    check(settle(mesher, *world, view, sink), "6: settles");
    const int before = sink.uploads;
    // A pit in the middle chunk.
    const TerrainConfig& t = world->config().terrain;
    const int cx = world->window_cx() + 1;
    const int cy = world->window_cy() + 1;
    {
        FieldSet& f = world->solver().fields();
        const GridShape& g = f.shape();
        for (const char* name : {"solid", "wood", "leaf", "water", "grass"}) {
            std::vector<float>& v = f.values(f.find(name));
            for (int z = 12; z < g.nz; ++z) {
                for (int y = t.chunk + 2; y < t.chunk + 14; ++y) {
                    for (int x = t.chunk + 2; x < t.chunk + 14; ++x) {
                        v[g.index(x, y, z)] = 0.0f;
                    }
                }
            }
        }
        world->solver().refresh_derived();
    }
    for (int tick = 0; tick < 8; ++tick) {
        world->step(Control{});
    }
    check(settle(mesher, *world, view, sink), "6: settles after a change");
    check(sink.uploads > before, "6: the changed chunk is meshed again");
    // The mesh is what a build from the current cells makes.
    std::uint64_t drawn_faces = 0;
    int level0 = 0;
    mesher.for_each_drawn([&](int handle, int l, int rx, int ry) {
        if (l == 0 && rx == cx && ry == cy) {
            drawn_faces = sink.faces[handle];
            level0 = 1;
        }
    });
    ChunkMesh direct;
    WorldMesher::build_region(*world, 0, cx, cy, direct);
    check(level0 == 1 && drawn_faces == direct.faces, "6: what is drawn is the current state");

    // Walk away: the pit leaves the window, and a far view still shows it.
    Control walk;
    walk.move_x = 1.0;
    for (int tick = 0; tick < 800; ++tick) {
        world->step(walk);
    }
    check(!world->column_in_window(cx, cy), "7: the pit has left the window");
    const int level = 2;
    const LodLevel l = lod_level(level);
    const int rx = cx >= 0 ? cx / l.span * l.span : -((-cx + l.span - 1) / l.span) * l.span;
    const int ry = cy >= 0 ? cy / l.span * l.span : -((-cy + l.span - 1) / l.span) * l.span;
    check(world->store().changed_in(rx, ry, rx + l.span - 1, ry + l.span - 1), "7: the store knows that region is not the generator's");
    ChunkMesh untouched;
    {
        std::optional<Playground> fresh = make_world();
        WorldMesher::build_region(*fresh, level, rx, ry, untouched);
    }
    LodView far = view_of(*world, 3.0, 300.0);
    check(settle(mesher, *world, far, sink), "7: a far view settles");
    std::size_t faces = 0;
    mesher.for_each_drawn([&](int handle, int lv, int x, int y) {
        if (lv == level && x == rx && y == ry) {
            faces = sink.faces[handle];
        }
    });
    ChunkMesh exact;
    WorldMesher::build_region(*world, level, rx, ry, exact);
    check(faces > 0 && faces == exact.faces, "7: what is drawn is what a build from the current state makes");
    check(untouched.faces != exact.faces, "7: an edited region is drawn from what is there, not from the generator's columns");
}

// Untouched far ground looks like the same ground: the columns stand where the chunks would.
void check_far_columns()
{
    std::optional<Playground> world = make_world();
    const TerrainConfig& t = world->config().terrain;
    for (const int level : {2, 3}) {
        const LodLevel l = lod_level(level);
        double total = 0.0;
        int regions = 0;
        for (int k = 0; k < 6; ++k) {
            const int rx = (world->window_cx() / l.span + 8 + k * 3) * l.span;
            const int ry = (world->window_cy() / l.span - 16 + k * 2) * l.span;
            ChunkMesh exact;
            ChunkMesh far;
            WorldMesher::build_region_from_chunks(*world, level, rx, ry, exact);
            WorldMesher::build_region(*world, level, rx, ry, far);
            auto mean_top = [](const ChunkMesh& m) {
                double sum = 0.0;
                for (const MeshVertex& v : m.vertices) {
                    sum += v.z;
                }
                return m.vertices.empty() ? 0.0 : sum / static_cast<double>(m.vertices.size());
            };
            check(far.faces > 0, "8: far ground has faces");
            total += std::abs(mean_top(exact) - mean_top(far));
            ++regions;
        }
        const double error = total / regions;
        std::cout << "  level " << level << ": far columns against whole chunks, mean height off by " << error << " m (a voxel is "
                  << l.factor_z * t.cell << " m tall)\n";
        check(error < 3.0, "8: far columns stand about where the chunks would");
    }
}

// A chunk changing again while its mesh is being built gets built again: what ends up drawn is the last state.
void check_stale_builds()
{
    std::optional<Playground> world = make_world();
    WorldMesher::Settings settings;
    settings.threads = 1; // so builds are in flight when the next change comes
    WorldMesher mesher(settings);
    world->set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, world->config().terrain); });
    Sink sink;
    const LodView view = view_of(*world, 30.0, 30.0);
    check(settle(mesher, *world, view, sink), "6: settles before the burst");
    const TerrainConfig& t = world->config().terrain;
    for (int round = 0; round < 12; ++round) {
        {
            FieldSet& f = world->solver().fields();
            const GridShape& g = f.shape();
            std::vector<float>& solid = f.values(f.find("solid"));
            for (int cy = 0; cy < 3; ++cy) {
                for (int cx = 0; cx < 3; ++cx) {
                    for (int z = 16; z < g.nz; ++z) {
                        solid[g.index(cx * t.chunk + 2 + round, cy * t.chunk + 8, z)] = 0.0f;
                    }
                }
            }
        }
        for (int tick = 0; tick < 4; ++tick) {
            world->step(Control{});
        }
        mesher.update(*world, view, sink.upload, sink.destroy);
    }
    check(settle(mesher, *world, view, sink), "6: settles after the burst");
    bool same = true;
    mesher.for_each_drawn([&](int handle, int level, int rx, int ry) {
        if (world->column_in_window(rx, ry)) {
            ChunkMesh direct;
            WorldMesher::build_region(*world, level, rx, ry, direct);
            same = same && sink.faces[handle] == direct.faces;
        }
    });
    check(same, "6: what is drawn is the last state, however the changes fell against the builds");
}

// The window moves up and down too, and the copies the mesher compares with go with it.
void check_vertical_moves()
{
    std::optional<Playground> world = make_world();
    WorldMesher mesher;
    world->set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, world->config().terrain); });
    Sink sink;
    const LodView view = view_of(*world, 30.0, 30.0);
    check(settle(mesher, *world, view, sink), "10: settles");
    const int layer = world->window_cz();
    world->player().position = world->player().position + Vec3{0.0, 0.0, 30.0};
    for (int tick = 0; tick < 180 && world->window_cz() == layer; ++tick) {
        world->step(Control{});
    }
    check(world->window_cz() != layer, "10: the window went up");
    // Something changes high in the window, in a chunk that slid.
    {
        FieldSet& f = world->solver().fields();
        const GridShape& g = f.shape();
        std::vector<float>& solid = f.values(f.find("solid"));
        for (int y = 20; y < 28; ++y) {
            for (int x = 20; x < 28; ++x) {
                solid[g.index(x, y, 20)] = 1.0f;
            }
        }
        world->solver().refresh_derived();
    }
    for (int tick = 0; tick < 8; ++tick) {
        world->step(Control{});
    }
    check(settle(mesher, *world, view, sink), "10: settles after the move");
    bool same = true;
    mesher.for_each_drawn([&](int handle, int level, int rx, int ry) {
        if (world->column_in_window(rx, ry)) {
            ChunkMesh direct;
            WorldMesher::build_region(*world, level, rx, ry, direct);
            same = same && sink.faces[handle] == direct.faces;
        }
    });
    check(same, "10: what is drawn is what is there, after the window moved up");
}

// The picture is a reader.
void check_sim_untouched()
{
    std::optional<Playground> plain = make_world();
    std::optional<Playground> watched = make_world();
    WorldMesher mesher;
    watched->set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, watched->config().terrain); });
    Sink sink;
    Control walk;
    walk.move_x = 1.0;
    for (int tick = 0; tick < 300; ++tick) {
        plain->step(walk);
        watched->step(walk);
        mesher.update(*watched, view_of(*watched, 30.0, 30.0), sink.upload, sink.destroy);
    }
    check(plain->digest() == watched->digest(), "9: meshing changes nothing in the world");
}

} // namespace

int main()
{
    check_levels();
    check_view();
    check_no_holes_on_zoom();
    check_law_shows();
    check_far_columns();
    check_stale_builds();
    check_vertical_moves();
    check_sim_untouched();
    return fieldengine::gate::report("voxel_lod");
}

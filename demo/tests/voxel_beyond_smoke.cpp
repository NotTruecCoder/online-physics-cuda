// GATE: BEYOND THE LIVE WINDOW THE WORLD IS STILL THERE.
//
// The law runs in a window of chunks around a point of interest; the camera can see far past it. Claims:
//   1. A ball of voxels thrown out past the window's edge lands on the map's ground there, not in nothing; it rests, it is kept
//      in the chunk it lies in, and when the window comes there it is stone in the fields where it lay.
//   2. A ray past the window stops on the map's ground (what the cursor points at, out there).
//   3. The people of a town beyond the window are where their day has them, without bodies: some are out, none of them is also a
//      body, none is inside the window, the same moment gives the same places, and they walk at a walking pace.
//   4. A person beyond the window that the window comes to becomes a body where its day had it: the same person, in the same place.
//   5. Zoomed out, the live window is drawn at full detail over the coarser regions around it, every chunk column of it.
//
// Falsify by: taking ground_at out of pieces_turn (1: the ball falls through the ground, nothing is stored); dropping the
// lattice check in people_beyond (3: proxies inside the window); spawning at the door in bring_person (4: far from the proxy);
// overlay_ always false in WorldMesher::plan (5).

#include "world/Playground.hpp"
#include "world/WorldMesher.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::optional<Playground> make_world(bool people)
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string pours;
    std::string path;
    if (!Playground::read_default_law(laws, pours, path, errors)) {
        return std::nullopt;
    }
    Playground::Config config;
    config.open_world = true;
    config.window_chunks = 3;
    config.window_layers = 3;
    config.far_life = false;
    config.people = people;
    config.observer = true;
    std::optional<Playground> w = Playground::create(laws, pours, errors, config);
    for (const std::string& e : errors) {
        std::cout << "  " << e << "\n";
    }
    check(w.has_value(), "the open world loads");
    if (w) {
        w->build_open_world();
    }
    return w;
}

double solid_in(const ChunkData& chunk)
{
    const std::vector<float>* v = chunk.find("solid");
    double sum = 0.0;
    if (v != nullptr) {
        for (const float x : *v) {
            sum += x;
        }
    }
    return sum;
}

int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

} // namespace

int main(int argc, char** argv)
{
    // With arguments, only the sections named: ball (1, 2), people (3, 4), view (5); "debug" traces the ball.
    const std::string only = argc > 1 ? argv[1] : "";
    const auto wanted = [&](const char* name) { return only.empty() || only == name || (only == "debug" && std::string(name) == "ball"); };
    const auto clock_start = std::chrono::steady_clock::now();
    const auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_start).count(); };
    // ---- 1 and 2. a ball out past the edge, and a ray -------------------------------------------------------------------
    if (wanted("ball")) {
        std::optional<Playground> made = make_world(false);
        if (!made) {
            return fieldengine::gate::report("voxel_beyond");
        }
        Playground& w = *made;
        const TerrainConfig& t = w.config().terrain;
        const double chunk_m = t.chunk * t.cell;
        const Vec3 o = w.origin();
        const GridShape& g = w.fields().shape();
        // Open, level ground past the window's +x edge (no crown or roof over it, nothing to roll off): the nearest such place.
        double wx = 0.0;
        double wy = 0.0;
        bool found = false;
        const double mid_x = o.x + g.nx * g.cell * 0.5;
        const double mid_y = o.y + g.ny * g.cell * 0.5;
        for (double away = g.nx * g.cell * 0.5 + 12.0; away < 400.0 && !found; away += 4.0) {
            for (int k = 0; k < 64 && !found; ++k) {
                const double a = 6.283185307 * k / 64.0;
                const double x = mid_x + std::cos(a) * away;
                const double y = mid_y + std::sin(a) * away;
                const double h = w.ground_at(x, y);
                bool level = h == w.terrain_at(x, y) && h > t.sea_level * t.cell + 1.0;
                for (double dx = -4.0; dx <= 4.0 && level; dx += 1.0) {
                    for (double dy = -4.0; dy <= 4.0 && level; dy += 1.0) {
                        level = std::abs(w.ground_at(x + dx, y + dy) - h) < 0.6;
                    }
                }
                if (level) {
                    wx = x;
                    wy = y;
                    found = true;
                }
            }
        }
        check(found, "1: there is open, level ground beyond the window");
        std::cout << "  thrown at " << wx - mid_x << ", " << wy - mid_y << " m from the window's middle" << std::endl;
        const double ground = w.ground_at(wx, wy);

        const std::optional<Vec3> hit = w.pick_world(Vec3{wx, wy, ground + 200.0}, Vec3{0.0, 0.0, -1.0});
        check(hit.has_value() && std::abs(hit->z - ground) < 0.1, "2: a ray past the window stops on the map's ground");

        int cx = floor_div(static_cast<int>(std::floor(wx / t.cell)), t.chunk);
        int cy = floor_div(static_cast<int>(std::floor(wy / t.cell)), t.chunk);
        int cz = floor_div(static_cast<int>(std::floor(ground / t.cell)), t.chunk);
        check(!w.in_window(cx, cy, cz), "1: that is beyond the window");
        ChunkData before;
        w.read_chunk(cx, cy, cz, before);
        double stone_before = solid_in(before);
        ChunkData above_before;
        w.read_chunk(cx, cy, cz + 1, above_before);
        stone_before += solid_in(above_before);
        const std::size_t stored_before = w.store().size();

        check(w.throw_ball(Vec3{wx, wy, ground + 6.0}, Vec3{0.0, 0.0, -2.0}, 1.0, BallKind::Stone, 3), "1: a ball can be thrown out there");
        int ticks = 0;
        double lowest = 1e30;
        Vec3 rest{wx, wy, ground};
        while (!w.pieces().empty() && ticks < 900) {
            w.step(Control{});
            if (only == "debug" && ticks % 10 == 0 && !w.pieces().empty()) {
                const FallingPiece& p = w.pieces()[0];
                std::cout << "    t " << ticks << " at " << p.at.x - w.origin().x << " " << p.at.y - w.origin().y << " z " << p.at.z << " ground " << w.ground_at(p.at.x, p.at.y) << " vz " << p.velocity.z << std::endl;
            }
            for (const FallingPiece& p : w.pieces()) {
                lowest = std::min(lowest, p.at.z);
                rest = p.at;
            }
            ++ticks;
        }
        std::cout << "  ground at " << ground << " m; the ball's middle went down to " << lowest << " m; at rest after " << ticks << " ticks\n";
        check(w.pieces().empty(), "1: it comes to rest");
        check(lowest > w.ground_at(rest.x, rest.y) - 0.5, "1: on the ground, not through it");
        check(w.store().size() > stored_before, "1: the chunk it lies in is kept");
        // Where it came to rest: that chunk and the one over it (it may lie across their border).
        check(cx == floor_div(static_cast<int>(std::floor(rest.x / t.cell)), t.chunk) && cy == floor_div(static_cast<int>(std::floor(rest.y / t.cell)), t.chunk),
            "1: it rests in the chunk it was thrown at (level ground)");
        ChunkData after;
        w.read_chunk(cx, cy, cz, after);
        ChunkData above_after;
        w.read_chunk(cx, cy, cz + 1, above_after);
        const double added = solid_in(after) + solid_in(above_after) - stone_before;
        std::cout << "  stone added out there: " << added << " of 32\n";
        check(added >= 0.6 * 32.0, "1: as stone");

        // The window comes there: the stone is in the fields.
        w.center_window_on(cx, cy, cz);
        const Vec3 l = Vec3{rest.x, rest.y, ground + 0.7} - w.origin();
        const int solid = w.fields().find("solid");
        const GridShape& lg = w.fields().shape();
        double near_stone = 0.0;
        for (int dz = -2; dz <= 4; ++dz) {
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dx = -3; dx <= 3; ++dx) {
                    const int x = static_cast<int>(l.x / lg.cell) + dx;
                    const int y = static_cast<int>(l.y / lg.cell) + dy;
                    const int z = static_cast<int>(l.z / lg.cell) + dz;
                    if (lg.inside(x, y, z)) {
                        near_stone += w.fields().values(solid)[lg.index(x, y, z)];
                    }
                }
            }
        }
        check(near_stone >= 0.6 * 32.0, "1: the window come there, it is stone in the fields where it lay (" + std::to_string(near_stone) + ")");
        check(w.violations().empty(), "1: no contract broken");
    }

    std::cout << "  [" << seconds() << " s]\n";
    // ---- 3 and 4. people beyond the window ------------------------------------------------------------------------------
    if (wanted("people")) {
        std::optional<Playground> made = make_world(true);
        if (!made || made->society() == nullptr) {
            check(false, "3: the open world has people");
            return fieldengine::gate::report("voxel_beyond");
        }
        Playground& w = *made;
        const TerrainConfig& t = w.config().terrain;
        // The largest living town, with the window at its middle.
        const Settlement* town = nullptr;
        for (const Settlement& s : w.map()->settlements()) {
            if (s.fell < 0 && (town == nullptr || s.people > town->people)) {
                town = &s;
            }
        }
        check(town != nullptr, "3: there is a town");
        if (town == nullptr) {
            return fieldengine::gate::report("voxel_beyond");
        }
        w.center_window_on(town->x / t.chunk, town->y / t.chunk, (town->ground + 4) / t.chunk);
        w.player().position = Vec3{(town->x + 0.5) * t.cell, (town->y + 0.5) * t.cell, (town->ground + 1.5) * t.cell} - w.origin();
        for (int i = 0; i < 130; ++i) {
            w.step(Control{});
        }
        const double cx = (town->x + 0.5) * t.cell;
        const double cy = (town->y + 0.5) * t.cell;
        std::vector<Playground::PersonProxy> out;
        w.people_beyond(cx, cy, town->radius * t.cell + 30.0, 2000, out);
        std::vector<Playground::PersonProxy> again;
        w.people_beyond(cx, cy, town->radius * t.cell + 30.0, 2000, again);
        std::set<std::uint64_t> bodies;
        for (const Creature& c : w.creatures()) {
            if (c.town >= 0) {
                bodies.insert(c.who.pid);
            }
        }
        int inside = 0;
        int twice = 0;
        const GridShape& g = w.fields().shape();
        for (const Playground::PersonProxy& p : out) {
            const Vec3 l = p.at - w.origin();
            inside += l.x > 0.0 && l.y > 0.0 && l.x < g.nx * g.cell && l.y < g.ny * g.cell ? 1 : 0;
            twice += bodies.count(p.pid) != 0 ? 1 : 0;
        }
        bool same = out.size() == again.size();
        for (std::size_t i = 0; same && i < out.size(); ++i) {
            same = out[i].pid == again[i].pid && out[i].at.x == again[i].at.x && out[i].at.y == again[i].at.y;
        }
        std::cout << "  " << town->name << ", " << town->people << " people: " << out.size() << " out of doors beyond the window, " << bodies.size() << " bodies in it\n";
        check(!out.empty(), "3: some of its people are out beyond the window");
        check(inside == 0, "3: none of them is inside the window");
        check(twice == 0, "3: none of them is also a body");
        check(same, "3: the same moment gives the same people in the same places");
        // A walking pace: where its day has someone, a second later.
        double fastest = 0.0;
        for (std::size_t i = 0; i < std::min<std::size_t>(out.size(), 40); ++i) {
            const Person who = w.society()->person(out[i].town, out[i].index);
            Vec3 a{}, b{};
            double yaw = 0.0, speed = 0.0;
            Creature::Doing doing{};
            if (w.day_place(who, out[i].town, w.world_seconds(), a, yaw, speed, doing) && w.day_place(who, out[i].town, w.world_seconds() + 1.0, b, yaw, speed, doing)) {
                fastest = std::max(fastest, std::hypot(b.x - a.x, b.y - a.y));
            }
        }
        std::cout << "  the fastest of them goes " << fastest << " m in a second\n";
        check(fastest < 3.0, "3: they walk, at a walking pace");

        // 4. The window comes to one of them.
        std::sort(out.begin(), out.end(), [&](const auto& a, const auto& b) { return std::hypot(a.at.x - cx, a.at.y - cy) < std::hypot(b.at.x - cx, b.at.y - cy); });
        int became = 0;
        int tried = 0;
        double worst = 0.0;
        for (std::size_t i = 0; i < out.size() && tried < 4; ++i) {
            const Playground::PersonProxy p = out[i];
            ++tried;
            const int pcx = floor_div(static_cast<int>(std::floor(p.at.x / t.cell)), t.chunk);
            const int pcy = floor_div(static_cast<int>(std::floor(p.at.y / t.cell)), t.chunk);
            w.center_window_on(pcx, pcy, (town->ground + 4) / t.chunk);
            w.player().position = p.at - w.origin() + Vec3{0.0, 0.0, 1.0};
            Vec3 where{};
            double yaw = 0.0, speed = 0.0;
            Creature::Doing doing{};
            const Person who = w.society()->person(p.town, p.index);
            for (int k = 0; k < 75; ++k) {
                w.step(Control{});
            }
            w.day_place(who, p.town, w.world_seconds(), where, yaw, speed, doing);
            for (const Creature& c : w.creatures()) {
                if (c.who.pid == p.pid) {
                    const double off = std::hypot(c.body.position.x + w.origin().x - where.x, c.body.position.y + w.origin().y - where.y);
                    worst = std::max(worst, off);
                    became += off < 6.0 ? 1 : 0;
                }
            }
        }
        std::cout << "  of " << tried << " the window came to, " << became << " became bodies where their day had them (furthest " << worst << " m)\n";
        check(became >= 1, "4: a person the window comes to is a body where its day had it");
    }

    std::cout << "  [" << seconds() << " s]\n";
    // ---- 5. the live window drawn whole over coarser regions -----------------------------------------------------------
    if (wanted("view")) {
        std::optional<Playground> made = make_world(false);
        if (!made) {
            return fieldengine::gate::report("voxel_beyond");
        }
        Playground& w = *made;
        WorldMesher mesher(WorldMesher::Settings{.min_pixels = 5.0, .budget_ms = 1000.0, .threads = 3});
        w.set_field_listener([&](const FieldStep& step) { mesher.on_fields(step, w.config().terrain); });
        const TerrainConfig& t = w.config().terrain;
        const double chunk_m = t.chunk * t.cell;
        LodView view;
        view.pixels_per_metre = 2.5; // a voxel of 0.5 m is 1.25 px: level 2
        view.centre_x = (w.window_cx() + w.window_chunks() * 0.5) * chunk_m;
        view.centre_y = (w.window_cy() + w.window_chunks() * 0.5) * chunk_m;
        view.radius_metres = 100.0;
        // What a camera would see: the disc of that radius. With no `visible` the mesher takes everything to be on
        // screen and builds the whole continent (6.5 M regions; this gate ran for most of an hour on it).
        view.visible = [&](const double lo[3], const double hi[3]) {
            const double dx = std::max({lo[0] - view.centre_x, 0.0, view.centre_x - hi[0]});
            const double dy = std::max({lo[1] - view.centre_y, 0.0, view.centre_y - hi[1]});
            return dx * dx + dy * dy <= view.radius_metres * view.radius_metres;
        };
        view.stamp = 1;
        int next_handle = 0;
        const auto upload = [&](int handle, const ChunkMesh&) { return handle >= 0 ? handle : next_handle++; };
        const auto destroy = [](int) {};
        int updates = 0;
        for (int k = 0; k < 2000 && !mesher.settled(); ++k, ++updates) {
            w.step(Control{});
            mesher.update(w, view, upload, destroy);
        }
        std::cout << "  the view settled after " << updates << " updates [" << seconds() << " s]\n";
        check(mesher.settled(), "5: the view settles");
        check(mesher.stats().level >= 1 && mesher.live_overlay(), "5: zoomed out, the live window is drawn over coarser regions");
        std::set<std::pair<int, int>> fine;
        mesher.for_each_drawn([&](int, int level, int rx, int ry) {
            if (level == 0) {
                fine.insert({rx, ry});
            }
        });
        int missing = 0;
        for (int cy = w.window_cy(); cy < w.window_cy() + w.window_chunks(); ++cy) {
            for (int cx = w.window_cx(); cx < w.window_cx() + w.window_chunks(); ++cx) {
                missing += fine.count({cx, cy}) == 0 ? 1 : 0;
            }
        }
        std::cout << "  level " << mesher.stats().level << "; " << fine.size() << " full-detail regions, " << missing << " window columns without one\n";
        check(missing == 0, "5: every column of the window at full detail");
        const std::array<int, 4> box = mesher.window_box();
        check(box[0] == w.window_cx() && box[2] == w.window_cx() + w.window_chunks(), "5: and the box the coarse ones step aside in is the window");
    }
    return fieldengine::gate::report("voxel_beyond");
}

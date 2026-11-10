// WHAT COMES DOWN, AND WHAT A BODY IS.
//   1. Walking into a wall a body stays on the ground; a real step it still goes up.
//   2. A trunk cut through comes away in one piece (Falling.cpp): it tips, falls to the side, and lies there as wood. Struck from
//      one side it goes over toward that side, because the blow lands under its centre of mass. Nothing is written about which
//      way; all its leaves come with it, nobody stands inside it, and it is the world again soon.
//   6. A body is only a body, any clothes fit either, and who someone is comes from their parents and the sun.
// Falsified by: `a_step` always true (1); removing the blow's impulse in lift_piece (2, direction).
#include "field/GpuSolver.hpp"
#include "world/Playground.hpp"
#include "world/Terrain.hpp"
#include "world/VoxelFigure.hpp"
#include "world/VoxelMesh.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::optional<Playground> make_world(bool open, int chunks = 3)
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string pours;
    std::string path;
    if (!Playground::read_default_law(laws, pours, path, errors)) {
        return std::nullopt;
    }
    Playground::Config config;
    config.open_world = open;
    config.window_chunks = chunks;
    config.window_layers = 3;
    config.far_life = false;
    config.falling = true;
    std::optional<Playground> w = Playground::create(laws, pours, errors, config);
    for (const std::string& e : errors) {
        std::cout << e << "\n";
    }
    return w;
}

void arena_walk()
{
    std::optional<Playground> w = make_world(false);
    w->build_arena();
    // The long wall stands at x = 15 m; the player walks straight into it, a long way from its ends.
    PlayBody& you = w->player();
    you.position = Vec3{12.5, 9.0, 1.4};
    for (int t = 0; t < 120; ++t) {
        w->step(Control{});
    }
    const double floor_z = w->player().position.z;
    double highest = floor_z;
    double furthest = 0.0;
    Control go;
    go.move_x = 1.0;
    int off_ground = 0;
    for (int t = 0; t < 60 * 4; ++t) {
        w->step(go);
        highest = std::max(highest, w->player().position.z);
        furthest = std::max(furthest, w->player().position.x);
        off_ground += w->player().grounded ? 0 : 1;
    }
    std::cout << "  into a wall: reached x " << furthest << ", rose " << highest - floor_z << " m, off the ground " << off_ground << " ticks of 240\n";
    check(furthest > 13.0, "1: it does walk up to the wall");
    check(highest - floor_z < 0.1, "1: walking into a wall you do not rise");
    check(off_ground < 5, "1: and you stay on the ground");
}

// A trunk near you, with room around it.
bool find_tree(Playground& w, int& tx, int& ty, int& tz)
{
    const FieldSet& f = w.fields();
    const GridShape& g = f.shape();
    const int wood = w.law().field_id("wood");
    const int solid = w.law().field_id("solid");
    const int built = w.law().field_id("built");
    const std::vector<float>& v = f.values(wood);
    const std::vector<float>& st = f.values(solid);
    const std::vector<float>& b = f.values(built);
    double best = 1e30;
    bool found = false;
    const Vec3 me = w.player().position;
    for (int y = 6; y < g.ny - 6; ++y) {
        for (int x = 6; x < g.nx - 6; ++x) {
            for (int z = 1; z < g.nz - 12; ++z) {
                if (v[g.index(x, y, z)] > 0.5f && st[g.index(x, y, z - 1)] > 0.5f && b[g.index(x, y, z)] < 0.5f) {
                    int height = 0;
                    while (z + height < g.nz && v[g.index(x, y, z + height)] > 0.5f) {
                        ++height;
                    }
                    const double d = std::hypot((x + 0.5) * g.cell - me.x, (y + 0.5) * g.cell - me.y);
                    if (height >= 6 && d < best) {
                        best = d;
                        tx = x;
                        ty = y;
                        tz = z;
                        found = true;
                    }
                    break;
                }
            }
        }
    }
    return found;
}

// Stands beside a tree and fells it: a blow from the -x side or the +x side, and the trunk cut through at the foot. How far its
// middle goes, and which way.
void fell_tree(double from_side, Vec3& moved, double& before_wood, double& after_wood, int& pieces_seen, double& lying_flat, double& in_the_air, double& pushed_clear, int& leaves_left)
{
    std::optional<Playground> made = make_world(true, 5);
    Playground& w = *made;
    w.build_open_world();
    for (int t = 0; t < 40; ++t) {
        w.step(Control{});
    }
    int tx = 0, ty = 0, tz = 0;
    if (!find_tree(w, tx, ty, tz)) {
        check(false, "2: there is a tree");
        return;
    }
    const GridShape& g = w.fields().shape();
    // In the world's metres: the window goes where you go.
    const Vec3 trunk_world = Vec3{(tx + 0.5) * g.cell, (ty + 0.5) * g.cell, (tz + 0.5) * g.cell} + w.origin();
    const auto wood_of_tree = [&]() {
        const FieldSet& f = w.fields();
        const int wood = w.law().field_id("wood");
        const Vec3 o = w.origin();
        double sum = 0.0;
        for (int z = 0; z < g.nz; ++z) {
            for (int y = 0; y < g.ny; ++y) {
                for (int x = 0; x < g.nx; ++x) {
                    const double wx = (x + 0.5) * g.cell + o.x;
                    const double wy = (y + 0.5) * g.cell + o.y;
                    if (std::hypot(wx - trunk_world.x, wy - trunk_world.y) < 14.0) {
                        sum += f.values(wood)[g.index(x, y, z)];
                    }
                }
            }
        }
        return sum;
    };
    w.player().position = trunk_world - w.origin() + Vec3{from_side * 1.3, 0.0, 0.5};
    for (int t = 0; t < 60; ++t) {
        w.player().position = trunk_world - w.origin() + Vec3{from_side * 1.3, 0.0, 0.5};
        w.player().velocity = Vec3{};
        w.step(Control{});
    }
    before_wood = wood_of_tree();
    w.player().velocity = Vec3{};
    Vec3 first{};
    Vec3 last{};
    std::size_t most_cells = 0;
    int blow = 0;
    int probing = -1;
    for (int t = 0; t < 60 * 30; ++t) {
        Control c;
        const Vec3 trunk = trunk_world - w.origin();
        c.aim = trunk + Vec3{0.0, 0.0, 0.3};
        if (most_cells < 20) {
            w.player().position = trunk + Vec3{from_side * 1.3, 0.0, 0.5};
            w.player().velocity = Vec3{};
            if (t == 1) {
                // A blow from this side (pushing the other way), and the trunk cut through at its foot, as a felling stroke does.
                w.note_force(trunk_world + Vec3{0.0, 0.0, 0.3}, Vec3{-from_side * 200.0, 0.0, 0.0});
                FieldSet& f = w.solver().fields();
                const int wood = f.find("wood");
                const Vec3 l = trunk_world - w.origin();
                const int cx = static_cast<int>(l.x / g.cell);
                const int cy = static_cast<int>(l.y / g.cell);
                const int cz = static_cast<int>(l.z / g.cell) + 1;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (g.inside(cx + dx, cy + dy, cz)) {
                            f.values(wood)[g.index(cx + dx, cy + dy, cz)] = 0.0f;
                        }
                    }
                }
                w.solver().refresh_derived();
                ++blow;
            }
        }
        w.player().hp = w.player().hp_max;
        w.step(c);
        w.clear_events();
        if (!w.pieces().empty()) {
            // The largest: what stood, not what came off it.
            const FallingPiece* big = &w.pieces()[0];
            for (const FallingPiece& p : w.pieces()) {
                big = p.cells.size() > big->cells.size() ? &p : big;
            }
            in_the_air += big->cells.size() >= 20 ? Playground::kTick : 0.0;
            // Stand in it: it is as solid in the air as it will be on the ground.
            if (big->cells.size() >= 20 && probing < 0) {
                probing = t;
                const FallingCell& inside = big->cells[big->cells.size() / 3];
                const Vec3 local{inside.x, inside.y, inside.z};
                const Vec3 world{big->at.x + big->turned[0][0] * local.x + big->turned[0][1] * local.y + big->turned[0][2] * local.z, big->at.y + big->turned[1][0] * local.x + big->turned[1][1] * local.y + big->turned[1][2] * local.z, big->at.z + big->turned[2][0] * local.x + big->turned[2][1] * local.y + big->turned[2][2] * local.z};
                w.player().position = world - w.origin();
                w.player().velocity = Vec3{};
            }
            if (probing >= 0 && t == probing + 12) {
                double closest = 1e30;
                const Vec3 me = w.player().position + w.origin();
                for (const FallingCell& c : big->cells) {
                    const Vec3 l{c.x, c.y, c.z};
                    const Vec3 at{big->at.x + big->turned[0][0] * l.x + big->turned[0][1] * l.y + big->turned[0][2] * l.z, big->at.y + big->turned[1][0] * l.x + big->turned[1][1] * l.y + big->turned[1][2] * l.z, big->at.z + big->turned[2][0] * l.x + big->turned[2][1] * l.y + big->turned[2][2] * l.z};
                    closest = std::min(closest, length(at - me));
                }
                pushed_clear = closest;
            }
            if (big->cells.size() > most_cells) {
                most_cells = big->cells.size();
                first = big->at;
            }
            if (big->cells.size() == most_cells) {
                last = big->at;
                lying_flat = std::abs(big->turned[2][2]);
            }
            pieces_seen = most_cells >= 20 ? 1 : 0;
        } else if (most_cells >= 20) {
            break;
        }
    }
    std::cout << "  " << blow << " blows from " << (from_side < 0 ? "-x" : "+x") << "; centre went from (" << first.x << ", " << first.y << ", " << first.z << ") to (" << last.x << ", " << last.y << ", " << last.z << ")\n";
    moved = last - first;
    for (int t = 0; t < 30; ++t) {
        w.step(Control{});
    }
    after_wood = wood_of_tree();
    // Up where the crown was: leaves that lead back to no wood, hanging on nothing.
    {
        const FieldSet& f = w.fields();
        const int wood = w.law().field_id("wood");
        const int leaf = w.law().field_id("leaf");
        const int solid = w.law().field_id("solid");
        const Vec3 o = w.origin();
        const int x0 = std::max(0, static_cast<int>((trunk_world.x - o.x) / g.cell) - 18);
        const int x1 = std::min(g.nx - 1, static_cast<int>((trunk_world.x - o.x) / g.cell) + 18);
        const int y0 = std::max(0, static_cast<int>((trunk_world.y - o.y) / g.cell) - 18);
        const int y1 = std::min(g.ny - 1, static_cast<int>((trunk_world.y - o.y) / g.cell) + 18);
        const int z0 = std::max(0, static_cast<int>((trunk_world.z - o.z) / g.cell) + 10);
        std::map<std::size_t, int> reached;
        std::vector<std::array<int, 4>> front;
        for (int z = 0; z < g.nz; ++z) {
            for (int y = std::max(0, y0 - 9); y <= std::min(g.ny - 1, y1 + 9); ++y) {
                for (int x = std::max(0, x0 - 9); x <= std::min(g.nx - 1, x1 + 9); ++x) {
                    if (f.values(wood)[g.index(x, y, z)] > 0.5f || f.values(solid)[g.index(x, y, z)] > 0.5f) {
                        front.push_back({x, y, z, 0});
                    }
                }
            }
        }
        for (std::size_t q = 0; q < front.size(); ++q) {
            const auto [x, y, z, d] = front[q];
            if (d >= 8) {
                continue;
            }
            for (const auto& [dx, dy, dz] : std::array<std::array<int, 3>, 6>{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}}) {
                const int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= g.nx || ny >= g.ny || nz >= g.nz) {
                    continue;
                }
                const std::size_t i = g.index(nx, ny, nz);
                if (f.values(leaf)[i] > 0.5f && reached.count(i) == 0) {
                    reached[i] = d + 1;
                    front.push_back({nx, ny, nz, d + 1});
                }
            }
        }
        leaves_left = 0;
        for (int z = z0; z < g.nz; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const std::size_t i = g.index(x, y, z);
                    leaves_left += f.values(leaf)[i] > 0.5f && reached.count(i) == 0 ? 1 : 0;
                }
            }
        }
    }
}

void trees()
{
    Vec3 moved_a{}, moved_b{};
    double before = 0.0, after = 0.0, flat_a = 1.0, flat_b = 1.0;
    int seen_a = 0, seen_b = 0;
    double air_a = 0.0, air_b = 0.0, clear_a = 0.0, clear_b = 0.0;
    int left_a = 0, left_b = 0;
    fell_tree(-1.0, moved_a, before, after, seen_a, flat_a, air_a, clear_a, left_a);
    double before_b = 0.0, after_b = 0.0;
    fell_tree(1.0, moved_b, before_b, after_b, seen_b, flat_b, air_b, clear_b, left_b);
    std::cout << "  wood " << before << " -> " << after << "; up axis at the end " << flat_a << " and " << flat_b << "\n";
    std::cout << "  in the air " << air_a << " s and " << air_b << " s; put inside it you end up " << clear_a << " m and " << clear_b << " m from the nearest of it; leaves left hanging on nothing " << left_a << " and " << left_b << "\n";
    check(air_a < 7.0 && air_b < 7.0, "10: it is the world again soon");
    check(clear_a > 0.3 && clear_b > 0.3, "10: nobody stands inside what is falling");
    check(left_a <= 3 && left_b <= 3, "11: all its leaves came with it");
    check(seen_a == 1 && seen_b == 1, "2: cut through, it comes away in one piece");
    check(std::hypot(moved_a.x, moved_a.y) > 1.0, "2: it falls to the side");
    check(flat_a < 0.8, "2: it has tipped over");
    check(moved_a.z < 0.0, "2: and down");
    // (Its own lean, and whatever stands around it, count as well: this is a forest. Only the difference is the blow's.)
    check(length(moved_a - moved_b) > 1.0, "2: where it goes depends on where it was struck from");
    check(moved_a.x < moved_b.x, "2: struck from one side it goes further over to that side");
    check(after > before * 0.8, "2: it is still there, as wood");
}

void bodies()
{
    std::vector<std::string> errors;
    int figures = 0;
    std::set<std::size_t> sizes;
    for (int sex = 0; sex < 2; ++sex) {
        for (int hair = 0; hair < kHairCount; ++hair) {
            for (int face = 0; face < kFaceCount; ++face) {
                Looks l;
                l.sex = sex;
                l.hair = hair;
                l.face = face;
                l.beard = sex == 1 ? face : 0;
                const VoxelFigure body = VoxelFigure::parse(body_model(l), errors);
                FigurePose pose;
                Vec3 hand{};
                check(body.socket_at(pose, "hand_r", hand), "6: a body has a hand");
                sizes.insert(body.voxels());
                ++figures;
            }
        }
    }
    check(errors.empty(), "6: every body reads");
    check(sizes.size() > 10, "6: and they are not all one body");
    for (const std::string& e : errors) {
        std::cout << e << "\n";
    }
    std::vector<std::string> item_errors;
    const ItemBook book = load_items(item_errors);
    for (const std::string& e : item_errors) {
        std::cout << e << "\n";
    }
    check(item_errors.empty(), "6: items read");
    for (const ItemDef& def : book.items) {
        if (def.kind != ItemKind::Clothes) {
            continue;
        }
        std::ifstream in("assets/world/clothes/" + def.model + ".model");
        std::ostringstream all;
        all << in.rdbuf();
        check(!all.str().empty(), "6: " + def.model + " is there");
        for (int sex = 0; sex < 2; ++sex) {
            std::vector<std::string> cloth_errors;
            const VoxelFigure cloth = VoxelFigure::parse(for_sex(all.str(), sex), cloth_errors);
            check(cloth_errors.empty() && !cloth.empty(), "6: " + def.model + " fits");
        }
    }
    // Hot and cold, and a family.
    double hot = 0.0, cold = 0.0;
    int women = 0;
    std::set<int> hair_styles;
    for (std::uint64_t i = 0; i < 400; ++i) {
        hot += looks_of_person(i * 7 + 1, i, 3, 30.0).melanin / 400.0;
        cold += looks_of_person(i * 7 + 1, i, 3, -8.0).melanin / 400.0;
        const Looks l = looks_of_person(i * 7 + 1, i, static_cast<int>(i % 8), 12.0);
        women += l.sex == 0 ? 1 : 0;
        hair_styles.insert(l.hair);
    }
    const auto spread = [](std::uint64_t family_a, std::uint64_t family_b) {
        double within = 0.0, between = 0.0;
        for (std::uint64_t i = 0; i < 40; ++i) {
            within += std::abs(looks_of_person(1000 + i, family_a, 2, 12.0).melanin - looks_of_person(2000 + i, family_a, 2, 12.0).melanin);
            between += std::abs(looks_of_person(1000 + i, family_a + i, 2, 12.0).melanin - looks_of_person(2000 + i, family_b + i * 3, 2, 12.0).melanin);
        }
        return std::pair<double, double>{within, between};
    };
    const auto [within, between] = spread(5, 90);
    std::cout << "  " << figures << " bodies; skin where it is hot " << hot << ", where it is cold " << cold << "; " << women << " women of 400; " << hair_styles.size() << " ways of hair; kin differ by " << within << " and strangers by " << between << "\n";
    check(hot > cold + 0.25, "6: the sun is in the skin");
    check(women > 140 && women < 260, "6: about half are women");
    check(hair_styles.size() >= 6, "6: there are ways of wearing your hair");
    check(within < between, "6: kin look like kin");
}

} // namespace

int main(int argc, char** argv)
{
    // A part alone, by name, when you are working on that part.
    const std::string only = argc > 1 ? argv[1] : "";
    const auto wanted = [&](const char* name) { return only.empty() || only == name; };
    if (wanted("bodies")) {
        bodies();
    }
    if (wanted("walk")) {
        arena_walk();
    }
    if (wanted("trees")) {
        trees();
    }
    return fieldengine::gate::report("voxel_falling");
}

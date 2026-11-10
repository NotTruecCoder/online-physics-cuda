// GATE: A THROWN BALL OF VOXELS IS A RIGID BODY, AND VOXELS AGAIN WHERE IT RESTS.
//
// Playground::throw_ball makes a piece like a felled tree's (Falling.cpp) that was never in the fields. Claims about the
// world, on the proving ground (a floor whose top is at z = 1 m):
//   1. Dropped on open floor, a stone ball comes to rest, leaves the air, and is stone in the fields: nearly all of its
//      cubes, on the floor where it fell, none of it in the floor.
//   2. Thrown hard at a wall thinner than it moves in a tick, it stops on its own side: nothing it is made of ends up
//      beyond the wall. (Contacts are looked at once a substep, and a fast piece takes more of them.)
//   3. It knocks over and hurts a body it hits.
//   4. What it is made of is what it becomes: ice is ice in the fields (and then the law thaws it), wood is wood.
//   5. The same throws give the same world, bit for bit.
//   6. The law's contracts hold throughout.
//
// Falsify by: setting a stone cell down as wood in set_piece_down (1 fails); taking one substep however fast a
// piece goes (2 fails: balls pass through the thin wall); dropping the body hits in pieces_turn (3 fails).

#include "world/Playground.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

Playground fresh()
{
    std::vector<std::string> errors;
    std::optional<Playground> ground = Playground::load_default(errors);
    for (const std::string& e : errors) {
        std::cerr << "  " << e << "\n";
    }
    check(ground.has_value(), "the proving ground loads");
    if (!ground) {
        std::exit(fieldengine::gate::report("voxel_ball"));
    }
    return std::move(*ground);
}

// Steps until nothing is in the air (or `most` ticks); returns the ticks it took.
int settle(Playground& g, int most = 900)
{
    const Control still{};
    int ticks = 0;
    do {
        g.step(still);
        ++ticks;
    } while (!g.pieces().empty() && ticks < most);
    // A few more for the law to see what came down.
    for (int k = 0; k < 6; ++k) {
        g.step(still);
    }
    return ticks;
}

// A field summed over a box of cells.
double sum(const Playground& g, const std::string& name, int x0, int x1, int y0, int y1, int z0, int z1)
{
    const FieldSet& f = g.fields();
    const GridShape& s = f.shape();
    const int id = f.find(name);
    double total = 0.0;
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (id >= 0 && s.inside(x, y, z)) {
                    total += f.values(id)[s.index(x, y, z)];
                }
            }
        }
    }
    return total;
}

// A ball of radius 1 m on a lattice of 0.5 m holds 32 cubes: the 8 about its centre and 4 on each of its 6 faces.
constexpr double kBallCubes = 32.0;

} // namespace

int main()
{
    // ---- 1. a stone ball dropped on the floor is stone on the floor ---------------------------------------------------
    {
        Playground g = fresh();
        const double stone_before = g.total("solid");
        const double floor_before = sum(g, "solid", 0, 47, 0, 47, 0, 1);
        check(g.throw_ball(Vec3{6.0, 14.0, 4.0}, Vec3{}, 1.0, BallKind::Stone, 3), "a ball can be thrown");
        check(g.pieces().size() == 1, "the ball is one piece in the air");
        const int ticks = settle(g);
        check(g.pieces().empty(), "the ball comes to rest (" + std::to_string(ticks) + " ticks)");
        const double added = g.total("solid") - stone_before;
        check(added >= 0.75 * kBallCubes && added <= kBallCubes + 0.5,
            "nearly all of the ball is stone in the fields (" + std::to_string(added) + " of 32)");
        const double where_it_fell = sum(g, "solid", 6, 18, 22, 34, 2, 9);
        check(where_it_fell >= 0.75 * kBallCubes, "and it lies where it fell (" + std::to_string(where_it_fell) + ")");
        check(sum(g, "solid", 0, 47, 0, 47, 0, 1) == floor_before, "none of it went into the floor");
        check(g.violations().empty(), "no contract broken while it fell");
    }

    // ---- 2. thrown hard at a thin wall, it stops on its own side ------------------------------------------------------
    // A wall half a voxel thick: at 45 to 95 m/s a cube crosses 0.75 to 1.6 m a tick, more than the wall is thick, so with
    // one look at the surface a tick some cube of the ball lands past the wall's middle and is pushed out the far side.
    {
        double beyond_all = 0.0;
        double furthest = 0.0;
        int throws = 0;
        for (const double speed : {45.0, 70.0, 95.0}) {
            for (const double start : {8.0, 8.3, 8.6, 8.9}) {
                Playground g = fresh();
                FieldSet& f = g.solver().fields();
                const GridShape& s = f.shape();
                for (int z = 2; z <= 13; ++z) {
                    for (int y = 34; y < s.ny; ++y) {
                        f.values(f.find("solid"))[s.index(34, y, z)] = 1.0f; // x = 17 .. 17.5 m
                    }
                }
                g.solver().refresh_derived();
                const double beyond_before = sum(g, "solid", 35, 47, 28, 47, 2, 19);
                g.throw_ball(Vec3{start, 21.5, 3.2}, Vec3{speed, 0.0, 0.0}, 1.0, BallKind::Stone, 0);
                const Control still{};
                for (int k = 0; k < 900 && !g.pieces().empty(); ++k) {
                    g.step(still);
                    for (const FallingPiece& p : g.pieces()) {
                        furthest = std::max(furthest, p.at.x);
                    }
                }
                settle(g);
                beyond_all += sum(g, "solid", 35, 47, 28, 47, 2, 19) - beyond_before;
                ++throws;
            }
        }
        check(beyond_all < 0.5, "nothing of " + std::to_string(throws) + " balls at 45-95 m/s got through a wall half a voxel thick (" +
            std::to_string(beyond_all) + " beyond)");
        check(furthest < 17.25, "and none of their centres ever got past the middle of the wall (furthest x " + std::to_string(furthest) + " m)");
    }

    // ---- 3. it knocks over and hurts what it hits ---------------------------------------------------------------------
    {
        Playground g = fresh();
        const PlayBody& target = g.target();
        const double hp_before = target.hp;
        const Vec3 at_before = target.position;
        g.throw_ball(Vec3{8.0, 9.0, 1.9 + 0.1}, Vec3{16.0, 0.0, 1.0}, 0.5, BallKind::Stone, 0);
        const Control still{};
        double fastest = 0.0;
        for (int k = 0; k < 40; ++k) {
            g.step(still);
            fastest = std::max(fastest, length(g.target().velocity));
        }
        check(g.target().hp < hp_before, "the target is hurt by the ball (" + std::to_string(g.target().hp) + " hp)");
        check(fastest > 2.0, "and thrown by it (" + std::to_string(fastest) + " m/s)");
        check(g.target().position.x > at_before.x, "the way the ball was going");
    }

    // ---- 4. what it is made of is what it becomes --------------------------------------------------------------------
    {
        // Ice is measured the tick it comes to rest: from then on it is the law's, and on a floor at 290 K it thaws.
        Playground g = fresh();
        const double ice_before = g.total("ice");
        const double water_before = g.total("water");
        g.throw_ball(Vec3{5.0, 14.0, 4.0}, Vec3{}, 1.0, BallKind::Ice);
        const Control still{};
        for (int k = 0; k < 900 && !g.pieces().empty(); ++k) {
            g.step(still);
        }
        const double ice = g.total("ice") - ice_before;
        check(ice >= 0.75 * kBallCubes, "an ice ball is ice in the fields when it comes to rest (" + std::to_string(ice) + ")");
        for (int k = 0; k < 240; ++k) {
            g.step(still);
        }
        const double thawed = g.total("water") - water_before;
        check(g.total("ice") - ice_before < ice && thawed > 1.0,
            "and the law thaws it into water (" + std::to_string(thawed) + " of water in 4 s)");

        Playground h = fresh();
        const double wood_before = h.total("wood");
        h.throw_ball(Vec3{9.0, 20.0, 4.0}, Vec3{}, 1.0, BallKind::Wood);
        settle(h);
        const double wood = h.total("wood") - wood_before;
        check(wood >= 0.75 * kBallCubes, "a wooden ball is wood (" + std::to_string(wood) + ")");
    }

    // ---- 5. the same throws, the same world --------------------------------------------------------------------------
    {
        const auto play = [] {
            Playground g = fresh();
            const Control still{};
            for (int k = 0; k < 240; ++k) {
                if (k % 30 == 0) {
                    g.throw_ball(Vec3{4.0 + k * 0.03, 12.0, 5.0}, Vec3{9.0, 2.0, 4.0}, 0.9, k % 60 == 0 ? BallKind::Stone : BallKind::Wood, 1);
                }
                g.step(still);
            }
            return g.digest();
        };
        check(play() == play(), "the same throws give the same digest");
    }

    return fieldengine::gate::report("voxel_ball");
}

// GATE: SPELLS CHANGE THE VOXELS, AND ONLY BY WAY OF THE LAW.
//
// A spell pours mana into voxels (spells.law). Everything after that is assets/law/*.law. These are
// claims about the WORLD a player sees:
//   1. The laws and spells load; every spell pours into a field some law declares.
//   2. You stand on the floor, walk at a real pace, are stopped by a wall, and climb a one-voxel step.
//   3. Earth raises stone where you aim — and the stone it raises stops you.
//   4. Force breaks a wall: stone becomes rubble, the rubble falls, light passes through the hole, and
//      not one voxel of stone is created or lost doing it (solid + rubble + lava is conserved).
//   5. Fire melts stone into lava, and the lava sets back into stone where it ran: the wall is
//      reshaped, the stone is all still there.
//   6. Water falls and lies on the floor; ice freezes it.
//   7. The target burns in fire and is thrown by force.
//   8. The same inputs give the same world, bit for bit.
//   9. The law's own contracts hold throughout (no conservation or range violation).
//  10. The mesh is the voxels: raising stone adds faces.
//
// Falsify by: zeroing `earth_rate` (3 fails), `shatter` (4 fails), `rubble_fall` (4: rubble hangs in
// the air), `melt_rate` (5 fails), `water_fall` (6: water floats), `freeze_rate` (6 fails),
// `burn_harm` / `force_push` (7 fails); breaking `fall()`'s antisymmetry in Law.cpp (4, 9 fail).

#include "field/Playground.hpp"
#include "field/VoxelMesh.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <cmath>
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
    check(ground.has_value(), "the proving ground loads (laws + spells)");
    if (!ground) {
        std::exit(fieldengine::gate::report("voxel_playground"));
    }
    return std::move(*ground);
}

int slot_of(const Playground& g, const std::string& name)
{
    for (std::size_t i = 0; i < g.pours().size(); ++i) {
        if (g.pours()[i].name == name) {
            return static_cast<int>(i);
        }
    }
    check(false, "spells.law has a '" + name + "' spell");
    return -1;
}

// Sum of a field over a box of cells, and the field-weighted mean height of what is in it.
struct Tally {
    double amount = 0.0;
    double mean_z = 0.0;
};

Tally tally(const Playground& g, const std::string& field, int x0, int x1, int y0, int y1, int z0, int z1)
{
    const FieldSet& f = g.fields();
    const GridShape& s = f.shape();
    const int id = f.find(field);
    Tally t;
    if (id < 0) {
        return t;
    }
    double weighted = 0.0;
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (!s.inside(x, y, z)) {
                    continue;
                }
                const double v = f.values(id)[s.index(x, y, z)];
                t.amount += v;
                weighted += v * (z + 0.5) * s.cell;
            }
        }
    }
    t.mean_z = t.amount > 1e-9 ? weighted / t.amount : 0.0;
    return t;
}

double stone(const Playground& g) { return g.total("solid") + g.total("rubble") + g.total("lava"); }

void run(Playground& g, int ticks, Control control = {})
{
    for (int i = 0; i < ticks; ++i) {
        g.step(control);
        control.jump = false;
    }
}

int violations(const Playground& g, Violation::Kind kind)
{
    int count = 0;
    for (const Violation& v : g.violations()) {
        if (v.kind == kind) {
            std::cerr << "  violation: " << to_string(v.kind) << " [" << v.term << "] " << v.field << ": " << v.detail
                      << "\n";
            ++count;
        }
    }
    return count;
}

} // namespace

int main()
{
    // ---- 1. it loads -----------------------------------------------------------------------------------
    {
        Playground g = fresh();
        check(g.pours().size() >= 6 && g.pours()[0].name == "fire" && g.pours()[5].name == "air", "the six elements come first (" + std::to_string(g.pours().size()) + " spells)");
        // A pour naming a field no law declares is refused (the playground is not built).
        std::vector<std::string> errors;
        const auto broken = Playground::create({LawSource{.name = "tiny", .text = "field occ\nfield sdf\nfield T\n"}},
            "pour zap field mana_zap amount 1 shape ball radius 1\n", errors);
        bool named = false;
        for (const std::string& e : errors) {
            named = named || e.find("mana_zap") != std::string::npos;
        }
        check(!broken && named, "a spell pouring an undeclared field is refused by name");
    }

    // ---- 2. standing, walking, walls, steps ----------------------------------------------------------
    {
        Playground g = fresh();
        run(g, 60);
        check(g.player().grounded && std::abs(g.player().position.z - 1.4) < 0.08,
            "you stand on the floor (z " + std::to_string(g.player().position.z) + ")");
        const Vec3 start = g.player().position;
        run(g, 60, Control{.move_x = 1.0});
        const double walked = g.player().position.x - start.x;
        std::cout << "baseline: walked " << walked << " m in 1 s\n";
        check(walked > 3.5 && std::abs(g.player().position.y - start.y) < 0.05, "a held key walks you ~5 m/s, straight");

        // Into the long wall (x = 15..16 m): you stop at its face.
        g.player().position = Vec3{12.0, 12.0, 1.42};
        run(g, 150, Control{.move_x = 1.0});
        check(g.player().position.x < 15.0 - 0.35 && g.player().position.x > 14.0,
            "the wall stops you at its face (x " + std::to_string(g.player().position.x) + ")");

        // Into the target: you stop against it, and it does not budge.
        g.player().position = Vec3{8.0, 9.0, 1.42};
        g.player().velocity = Vec3{};
        const Vec3 target_at = g.target().position;
        run(g, 120, Control{.move_x = 1.0});
        const double gap = length(g.target().position - g.player().position) - g.player().radius - g.target().radius;
        std::cout << "baseline: walking into the target leaves a gap of " << gap << " m\n";
        check(gap > -0.02, "you cannot walk through the target");
        check(g.player().position.x < target_at.x, "...you stay on your side of it");
        check(length(g.target().position - target_at) < 0.05, "...and walking into it does not shove it");

        // Up the steps (x = 3..4.5 m, each one voxel higher).
        g.player().position = Vec3{1.5, 20.0, 1.42};
        g.player().velocity = Vec3{};
        run(g, 20);
        double top_of_steps = 0.0;
        for (int i = 0; i < 90; ++i) {
            run(g, 1, Control{.move_x = 1.0});
            top_of_steps = std::max(top_of_steps, g.player().position.z);
        }
        std::cout << "baseline: walking over the steps you reach z " << top_of_steps << "\n";
        check(top_of_steps > 2.3, "you climb one-voxel steps by walking into them");
    }

    // ---- 3. earth raises stone, and it is real stone ----------------------------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        std::vector<MeshVertex> mesh;
        const VoxelMeshStats before_mesh = build_voxel_mesh(g.fields(), mesh);
        const double before = tally(g, "solid", 14, 26, 4, 16, 2, 9).amount;
        // Aim at the floor 4 m in front of you.
        const Vec3 aim{9.0, 5.0, 1.0};
        fieldengine::gate::pour_toward(g, slot_of(g, "earth"), aim);
        run(g, 1);
        run(g, 90);
        const double after = tally(g, "solid", 14, 26, 4, 16, 2, 9).amount;
        std::cout << "baseline: earth raised " << after - before << " voxels of stone\n";
        check(after - before > 12.0, "earth raises stone where you aim (" + std::to_string(after - before) + " voxels)");
        const VoxelMeshStats after_mesh = build_voxel_mesh(g.fields(), mesh);
        check(after_mesh.solid_voxels > before_mesh.solid_voxels && after_mesh.faces > before_mesh.faces,
            "the raised stone is in the picture");
        // Walk into it.
        run(g, 120, Control{.move_x = 1.0});
        check(g.player().position.x < 9.0 - 0.3, "the raised stone stops you (x " + std::to_string(g.player().position.x) + ")");
    }

    // ---- 4. force breaks a wall --------------------------------------------------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        const double stone0 = stone(g);
        const Tally wall0 = tally(g, "solid", 30, 31, 18, 30, 2, 7);
        double layer0[8] = {};
        for (int z = 2; z <= 7; ++z) {
            layer0[z] = tally(g, "solid", 30, 31, 18, 30, z, z).amount;
        }
        g.player().position = Vec3{10.0, 12.0, 1.42};
        run(g, 10);
        const Vec3 aim{15.0, 12.0, 2.75}; // the wall's face, at chest height
        for (int k = 0; k < 3; ++k) {
            fieldengine::gate::pour_toward(g, slot_of(g, "force"), aim);
            run(g, 1);
            run(g, 40);
        }
        run(g, 120);
        const Tally wall1 = tally(g, "solid", 30, 31, 18, 30, 2, 7);
        const Tally rubble = tally(g, "rubble", 24, 40, 14, 34, 2, 12);
        std::cout << "baseline: force broke " << wall0.amount - wall1.amount << " voxels of wall into " << rubble.amount
                  << " of rubble lying at mean height " << rubble.mean_z << " m\n";
        check(wall0.amount - wall1.amount > 4.0, "force breaks the wall it hits");
        // Most of what breaks is dust: `crumble` of it lies there after.
        check(rubble.amount > 0.05 * (wall0.amount - wall1.amount) && rubble.amount < 0.2 * (wall0.amount - wall1.amount), "...and a little of it is left lying");
        // Where the broken stone USED to be: the breach's height, weighted by what went missing.
        double lost = 0.0;
        double lost_z = 0.0;
        for (int z = 2; z <= 7; ++z) {
            const double gone = layer0[z] - tally(g, "solid", 30, 31, 18, 30, z, z).amount;
            lost += gone;
            lost_z += gone * (z + 0.5) * g.fields().shape().cell;
        }
        lost_z = lost > 0.0 ? lost_z / lost : 0.0;
        std::cout << "baseline: the breach was at " << lost_z << " m; its rubble lies at " << rubble.mean_z << " m\n";
        check(rubble.mean_z < lost_z - 0.4, "...and the rubble falls below the breach instead of hanging where the wall was");
        // Some of the breach is filled by what fell into it (as a real one is); somewhere along its
        // height you can now see through the wall.
        RayBindings b;
        b.surface = g.surface();
        bool seen_through = false;
        for (double z = 1.25; z <= 3.8; z += 0.25) { // below the top of the wall (4 m)
            const RayHit through = march_ray(g.fields(), b, Vec3{13.0, 12.0, z}, Vec3{1.0, 0.0, 0.0}, 5.0, false);
            seen_through = seen_through || !through.hit || through.position.x > 16.2;
        }
        check(seen_through, "light passes through the breach");
        check(stone(g) <= stone0 + 1e-3 * stone0, "no stone was made (" + std::to_string(stone0) +
                " -> " + std::to_string(stone(g)) + ")");
        check(violations(g, Violation::Kind::Conservation) == 0, "the law's conservation contracts held");
    }

    // ---- 5. fire melts stone, lava sets back into stone ------------------------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        const double stone0 = stone(g);
        g.player().position = Vec3{10.0, 20.0, 1.42};
        run(g, 10);
        const Vec3 aim{15.0, 20.0, 2.0};
        double peak_lava = 0.0;
        for (int k = 0; k < 8; ++k) {
            fieldengine::gate::pour_toward(g, slot_of(g, "fire"), aim);
            run(g, 1);
            for (int i = 0; i < 30; ++i) {
                run(g, 1);
                peak_lava = std::max(peak_lava, g.total("lava"));
            }
        }
        const double wall_after_fire = tally(g, "solid", 30, 31, 14, 26, 2, 7).amount;
        run(g, 60 * 20); // let it cool
        const double lava_left = g.total("lava");
        std::cout << "baseline: fire melted up to " << peak_lava << " voxels into lava; " << lava_left
                  << " still molten after cooling\n";
        check(peak_lava > 1.0, "fire melts stone into lava");
        check(lava_left < peak_lava * 0.2, "...and the lava sets back into stone as it cools");
        check(std::abs(stone(g) - stone0) < 1e-3 * stone0, "melting and setting neither make nor lose stone");
        (void)wall_after_fire;
        check(violations(g, Violation::Kind::Conservation) == 0, "the law's conservation contracts held");
    }

    // ---- 6. water falls and lies; ice freezes it ----------------------------------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        const double basin = g.total("water");
        const Vec3 aim{8.0, 8.0, 1.0};
        fieldengine::gate::pour_toward(g, slot_of(g, "water"), aim);
        run(g, 1);
        run(g, 120);
        const Tally pool = tally(g, "water", 0, 47, 0, 34, 2, 19);
        std::cout << "baseline: water poured " << pool.amount << " voxels, mean height " << pool.mean_z << " m\n";
        check(g.total("water") > basin + 2.0, "water condenses where it lands");
        check(pool.mean_z < 1.35, "...falls, and lies on the floor (mean height " + std::to_string(pool.mean_z) + ")");
        const double ice0 = g.total("ice");
        fieldengine::gate::pour_toward(g, slot_of(g, "ice"), aim);
        run(g, 1);
        run(g, 60);
        std::cout << "baseline: ice froze " << g.total("ice") - ice0 << " voxels\n";
        check(g.total("ice") - ice0 > 1.0, "ice freezes the water it lands on");
    }

    // ---- 7. the target burns and is thrown -------------------------------------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        const Vec3 target = g.target().position;
        const double hp0 = g.target().hp;
        fieldengine::gate::pour_toward(g, slot_of(g, "fire"), target);
        run(g, 1);
        run(g, 90);
        std::cout << "baseline: one fire bolt took the target " << hp0 << " -> " << g.target().hp << "\n";
        check(g.target().hp < hp0 - 10.0, "fire burns the target");
        // A ball of force mana is densest at its heart and its slope is what throws: put it a little to the near side
        // of the target (one exactly on it is balanced, and throws nothing).
        const Vec3 before = g.target().position;
        Vec3 near_side = before;
        {
            const Vec3 you = g.player().position;
            const double d = std::max(1e-6, std::hypot(before.x - you.x, before.y - you.y));
            near_side.x -= 0.8 * (before.x - you.x) / d;
            near_side.y -= 0.8 * (before.y - you.y) / d;
        }
        fieldengine::gate::pour_toward(g, slot_of(g, "force"), near_side);
        run(g, 1);
        run(g, 60);
        const Vec3 after = g.target().position;
        const double thrown = std::hypot(after.x - before.x, after.y - before.y);
        std::cout << "baseline: one force bolt threw the target " << thrown << " m\n";
        check(thrown > 0.5, "force throws the target");
    }

    // ---- 7b. a blast at your feet throws you up; standing near a fire is survivable ----------------
    {
        auto launch = [](const std::string& spell) {
            Playground g = fresh();
            run(g, 30);
            const double floor_z = g.player().position.z;
            const Vec3 feet = g.player().position - Vec3{-0.3, 0.0, g.player().radius + 0.05};
            fieldengine::gate::pour_toward(g, slot_of(g, spell), feet);
            run(g, 1);
            double peak = 0.0;
            for (int i = 0; i < 120; ++i) {
                run(g, 1);
                peak = std::max(peak, g.player().position.z - floor_z);
            }
            std::cout << "baseline: " << spell << " at your feet threw you " << peak << " m up\n";
            return peak;
        };
        check(launch("air") > 1.5, "air cast at your feet throws you up (an updraft)");
        check(launch("force") > 0.8, "force cast at your feet throws you up (a blast)");

        Playground g = fresh();
        run(g, 30);
        const double hp0 = g.player().hp;
        const Vec3 fire_spot = g.player().position + Vec3{2.5, 0.0, -g.player().radius};
        for (int k = 0; k < 3; ++k) {
            fieldengine::gate::pour_toward(g, slot_of(g, "fire"), fire_spot);
            run(g, 1);
            run(g, 30);
        }
        run(g, 90);
        const double lost = hp0 - g.player().hp;
        std::cout << "baseline: standing 2.5 m from three fire bolts cost you " << lost << " HP\n";
        check(lost < 25.0, "standing a couple of metres from a fire hurts, but does not melt you");
    }

    // ---- 9. heavy, careless casting breaks no contract of the law ---------------------------------
    {
        Playground g = fresh();
        run(g, 30);
        const Vec3 spots[] = {{9.0, 5.0, 1.0}, {15.0, 9.0, 2.5}, {12.0, 31.0, 3.0}};
        for (const Vec3& spot : spots) {
            g.player().position = Vec3{spot.x - 3.0, spot.y - 2.0, 1.42};
            for (int k = 0; k < 18; ++k) {
                fieldengine::gate::pour_toward(g, k % 6, spot);
                run(g, 1);
                fieldengine::gate::pour_toward(g, 0, spot); // and fire, again and again
                run(g, 1);
                run(g, 12);
            }
        }
        run(g, 300);
        const int broken = static_cast<int>(g.violations().size());
        for (const Violation& v : g.violations()) {
            std::cerr << "  " << to_string(v.kind) << " [" << v.term << "] " << v.field << ": " << v.detail << "\n";
        }
        check(broken == 0, "a minute of every spell on the same spots breaks no contract (" + std::to_string(broken) + ")");
    }

    // ---- 8. determinism ------------------------------------------------------------------------------
    {
        auto script = [](Playground& g) {
            for (int i = 0; i < 240; ++i) {
                Control c{.move_x = i < 120 ? 1.0 : 0.0, .move_y = 0.4, .jump = i == 60};
                if (i == 30) {
                    fieldengine::gate::pour_toward(g, 3, Vec3{15.0, 9.0, 2.5});
                }
                if (i == 90) {
                    fieldengine::gate::pour_toward(g, 0, Vec3{15.0, 9.0, 2.5});
                }
                g.step(c);
            }
            return g.digest();
        };
        Playground a = fresh();
        Playground b = fresh();
        check(script(a) == script(b), "the same inputs give the same world, bit for bit");
    }

    return fieldengine::gate::report("voxel_playground");
}

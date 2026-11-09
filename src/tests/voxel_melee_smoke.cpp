// THE HAND. Claims about the blows bodies fight with (Melee.cpp), whoever's body it is:
//   1. Moves are read from text, refused with the line when wrong, and a stronger move costs more.
//   2. A clip knows nothing of a skin: one clip, two bodies of different size, the same angles; something held goes with the hand.
//   3. A blow lands where it says: in reach, in its arc, once, and not through a wall. It leaves force mana if it breaks things.
//   4. A guard takes most of a blow from in front, none from behind; raised just in time, it turns the blow back.
//  10. A fighter of the law (life.law) comes, swings and lands by the same rules; a guard works on it, and it can be beaten.
#include "field/FigureAnim.hpp"
#include "field/Melee.hpp"
#include "field/Playground.hpp"
#include "field/VoxelFigure.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::optional<Playground> make_world(MeleeBook& book)
{
    std::vector<std::string> errors;
    std::optional<Playground> world = Playground::load_default(errors, false);
    check(world.has_value(), "the proving ground loads");
    if (world) {
        book = load_moves(errors);
        check(errors.empty(), "the moves read");
        world->set_melee(book);
        // Open ground around the middle, and the target in front of you.
        world->player().position = Vec3{5.0, 5.0, world->fields().shape().cell * 2.0 + 0.42};
        world->target().position = world->player().position + Vec3{1.5, 0.0, 0.0};
        world->target().velocity = Vec3{};
    }
    return world;
}

void run(Playground& world, int ticks, const Control& control = Control{})
{
    for (int k = 0; k < ticks; ++k) {
        world.step(control);
    }
}

// A blow of the player's body, with a weapon's move (by its place among those of the weapon), toward `aim`; then a tick.
void strike(Playground& world, const MeleeBook& book, const char* weapon, int slot, Vec3 aim)
{
    const std::vector<int> moves = book.known(weapon, 99);
    if (slot < static_cast<int>(moves.size())) {
        world.swing(world.player(), moves[static_cast<std::size_t>(slot)], aim - world.player().position);
    }
    world.step(Control{});
}

double sum_of(const Playground& world, const std::string& name)
{
    return world.total(name);
}

void check_moves()
{
    std::vector<std::string> errors;
    const MeleeBook book = load_moves(errors);
    check(errors.empty(), "1: moves.world reads");
    check(book.weapons.size() >= 4 && book.moves.size() >= 10, "1: several weapons, several moves");
    for (const WeaponDef& weapon : book.weapons) {
        check(book.known(weapon.name, 0).size() >= 1, "1: each weapon has a first move: " + weapon.name);
    }
    check(book.known("sword", 0).size() < book.known("sword", 99).size(), "1: more skill knows more moves");
    std::vector<std::string> wrong;
    parse_moves("weapon w\nmove a weapon w anim x windup -1 active 0.1 recover 0.1 reach 1 arc 90 hit 1\n", wrong);
    check(!wrong.empty() && wrong[0].find(":2:") != std::string::npos, "1: a bad move is refused by its line");
    wrong.clear();
    parse_moves("move a weapon nothing anim x\n", wrong);
    check(!wrong.empty(), "1: a move with no such weapon is refused");
    MoveDef weak;
    weak.hit = 5.0;
    MoveDef strong = weak;
    strong.hit = 30.0;
    check(move_stamina(strong) > move_stamina(weak), "1: harder costs more");
    strong = weak;
    strong.breaks = 3.0;
    check(move_stamina(strong) > move_stamina(weak), "1: and so does breaking things");
}

void check_skins()
{
    std::vector<std::string> errors;
    const VoxelFigure mage = VoxelFigure::load("assets/world/skins/mage.model", errors);
    const VoxelFigure knight = VoxelFigure::load("assets/world/skins/knight.model", errors);
    const VoxelFigure sword = VoxelFigure::load("assets/world/weapons/sword.model", errors);
    const ClipBook clips = ClipBook::load("assets/world/anim/body.anim", errors);
    for (const std::string& e : errors) {
        std::cout << "  " << e << "\n";
    }
    check(errors.empty(), "2: skins, a weapon and clips read");
    check(knight.width() > mage.width() * 1.1, "2: the skins differ in size");
    std::vector<std::string> wrong;
    VoxelFigure::parse("voxel 0.1\npart wing pivot 0 0 0\n", wrong);
    check(!wrong.empty(), "2: a skin with no such bone is refused");
    wrong.clear();
    ClipBook::parse("clip a\nkey 0 wing 1\n", wrong);
    check(!wrong.empty(), "2: a clip with no such bone is refused");
    // Every move's clip is there and every move plays on both.
    std::vector<std::string> ignored;
    const MeleeBook book = load_moves(ignored);
    for (const MoveDef& move : book.moves) {
        check(clips.find(move.anim) != nullptr, "2: there is a clip for " + move.name);
    }
    for (const VoxelFigure* skin : {&mage, &knight}) {
        FigurePose at_rest;
        FigurePose raised;
        clips.play("thrust", 1.5, 1.0, raised);
        Vec3 low{};
        Vec3 high{};
        check(skin->socket_at(at_rest, "hand_r", low) && skin->socket_at(raised, "hand_r", high), "2: the skin has a right hand");
        check(high.z > low.z + 0.1, "2: the same clip raises the hand of either");
        std::vector<MeshVertex> with;
        std::vector<MeshVertex> without;
        skin->mesh(raised, without);
        with = without;
        check(skin->mesh_held(raised, sword, "hand_r", with) && with.size() > without.size(), "2: and it can hold a sword");
        // The sword goes with the hand: its grip is at the hand, wherever that went.
        double nearest = 1e9;
        for (std::size_t i = without.size(); i < with.size(); ++i) {
            nearest = std::min(nearest, length(Vec3{with[i].x, with[i].y, with[i].z} - high));
        }
        check(nearest < 0.15, "2: and the sword is at the hand");
    }
    // The clip is the same numbers for both: it has no idea which it is on.
    FigurePose a;
    FigurePose b;
    clips.play("slash", 1.2, 1.0, a);
    clips.play("slash", 1.2, 1.0, b);
    check(a.bones[kArmR].swing == b.bones[kArmR].swing, "2: a clip is angles, the same for anyone");
    FigurePose over;
    over.bones[kLegL].swing = 0.7;
    clips.play("cast_r", 0.0, 1.0, over);
    check(over.bones[kLegL].swing == 0.7 && over.bones[kArmR].swing > 1.0, "2: a clip leaves alone the bones it does not name");
}

void check_blows()
{
    MeleeBook book;
    std::optional<Playground> w = make_world(book);
    if (!w) {
        return;
    }
    // In reach.
    const double hp = w->target().hp;
    strike(*w, book, "sword", 0, w->target().position);
    run(*w, 2);
    check(w->target().hp == hp, "3: no harm while it winds up");
    check(w->player().swing.move >= 0, "3: the blow is under way");
    run(*w, 60);
    const double once = hp - w->target().hp;
    const MoveDef& slash = book.moves[static_cast<std::size_t>(book.move_slot("slash"))];
    std::cout << "  slash took " << once << " of " << slash.hit << std::endl;
    check(std::abs(once - slash.hit) < 1e-6, "3: it lands, once");
    check(w->player().swing.move < 0, "3: and it ends");
    // Out of reach.
    w->target().hp = hp;
    w->target().position = w->player().position + Vec3{6.0, 0.0, 0.0};
    w->target().velocity = Vec3{};
    strike(*w, book, "sword", 0, w->target().position);
    run(*w, 60);
    check(w->target().hp == hp, "3: too far is too far");
    // Out of its arc: behind you.
    w->target().position = w->player().position + Vec3{-1.5, 0.0, 0.0};
    w->target().velocity = Vec3{};
    strike(*w, book, "sword", 0, w->player().position + Vec3{3.0, 0.0, 0.0});
    run(*w, 60);
    check(w->target().hp == hp, "3: what is behind you is not hit");
    // Through a wall: no.
    w->target().position = w->player().position + Vec3{2.0, 0.0, 0.0};
    w->target().velocity = Vec3{};
    {
        FieldSet& f = w->solver().fields();
        const GridShape& g = f.shape();
        const Vec3 at = w->player().position + Vec3{1.0, 0.0, 0.0};
        const int solid = f.find("solid");
        const int x = static_cast<int>(at.x / g.cell);
        const int y0 = static_cast<int>(at.y / g.cell);
        for (int z = 0; z < g.nz; ++z) {
            for (int y = y0 - 6; y <= y0 + 6; ++y) {
                f.values(solid)[g.index(x, y, z)] = 1.0f;
            }
        }
        w->solver().refresh_derived();
    }
    strike(*w, book, "sword", 0, w->target().position);
    run(*w, 60);
    check(w->target().hp == hp, "3: not through a wall");
    // A move that breaks things leaves force mana on the wall; one that does not, does not.
    const double force_before = sum_of(*w, "mana_force");
    check(force_before < 1e-6, "3: no force about yet");
    strike(*w, book, "axe", 0, w->target().position);
    bool force_seen = false;
    for (int k = 0; k < 40 && !force_seen; ++k) {
        w->step(Control{});
        force_seen = sum_of(*w, "mana_force") > 0.01;
    }
    check(force_seen, "3: a chop leaves force on what it strikes");
}

void check_guard()
{
    MeleeBook book;
    std::optional<Playground> w = make_world(book);
    if (!w) {
        return;
    }
    const auto blow_with_guard = [&](double raised_for, double facing_yaw) {
        w->target().hp = 300.0;
        w->target().position = w->player().position + Vec3{1.5, 0.0, 0.0};
        w->target().velocity = Vec3{};
        w->target().guard_yaw = facing_yaw;
        w->player().stagger = 0.0;
        strike(*w, book, "sword", 0, w->target().position);
        for (int k = 0; k < 60; ++k) {
            w->target().guard = raised_for;
            w->step(Control{});
        }
        w->target().guard = -1.0;
        return 300.0 - w->target().hp;
    };
    const double facing_you = 3.14159265358979;
    const double open = blow_with_guard(-1.0, facing_you);
    const double guarded = blow_with_guard(5.0, facing_you);
    const double back_turned = blow_with_guard(5.0, 0.0);
    std::cout << "  open " << open << ", guarded " << guarded << ", guard the wrong way " << back_turned << std::endl;
    check(guarded < open * 0.5 && guarded > 0.0, "4: a guard takes most of it");
    check(back_turned == open, "4: not from behind");
    const double parried = blow_with_guard(0.05, facing_you);
    check(parried == 0.0, "4: just in time, none of it");
}

// 10. Someone who hits back, with the moves you have, by the same rules.
void check_fighters()
{
    MeleeBook book;
    std::optional<Playground> w = make_world(book);
    if (!w) {
        return;
    }
    check(w->spawn("bandit", w->player().position + Vec3{3.0, 0.0, 0.3}), "10: a fighter comes");
    check(w->creatures().size() >= 1, "10: and is there");
    const double hp0 = w->player().hp;
    double most_hurt = 0.0;
    bool swung = false;
    for (int k = 0; k < 60 * 6; ++k) {
        w->step(Control{});
        most_hurt = std::max(most_hurt, hp0 - w->player().hp);
        for (const Creature& c : w->creatures()) {
            swung = swung || c.body.swing.move >= 0;
        }
    }
    std::cout << "  a fighter took " << most_hurt << " off you in 6 s" << std::endl;
    check(swung, "10: it swings");
    check(most_hurt > 0.0, "10: and it lands");
    check(w->target().hp >= w->target().hp_max, "10: on you, and on nothing else");
    // A guard works on it: raised and kept toward it.
    w->player().hp = w->player().hp_max;
    double guarded_hurt = 0.0;
    const double before = w->player().hp;
    for (int k = 0; k < 60 * 6; ++k) {
        if (!w->creatures().empty()) {
            const Vec3 to = w->creatures()[0].body.position - w->player().position;
            w->player().guard_yaw = std::atan2(to.y, to.x);
        }
        w->player().guard = 1.0;
        w->step(Control{});
        guarded_hurt = std::max(guarded_hurt, before - w->player().hp);
    }
    w->player().guard = -1.0;
    std::cout << "  behind a guard: " << guarded_hurt << std::endl;
    check(guarded_hurt < most_hurt * 0.6, "10: behind a guard it takes much less");
    // And it can be beaten.
    w->player().hp = w->player().hp_max;
    bool gone = false;
    for (int k = 0; k < 60 * 25 && !gone; ++k) {
        w->player().hp = w->player().hp_max;
        if (!w->creatures().empty() && k % 12 == 0 && w->player().swing.move < 0) {
            strike(*w, book, "sword", 0, w->creatures()[0].body.position);
        } else {
            w->step(Control{});
        }
        gone = w->creatures().empty();
    }
    check(gone, "10: and it can be beaten");
}

} // namespace

int main()
{
    check_moves();
    check_skins();
    check_blows();
    check_guard();
    check_fighters();
    return fieldengine::gate::report("voxel_melee");
}

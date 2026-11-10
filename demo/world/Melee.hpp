#pragma once

#include <string>
#include <vector>

// A HAND AND WHAT IT HOLDS. A move is one simple blow, read from assets/world/moves.world: which moves someone knows comes with
// skill in the weapon. People and beasts fight with them (Melee.cpp): windup, the blow across its arc, recovery; a guard raised
// in time turns it back on who struck.
namespace fieldengine::field {

struct WeaponDef {
    std::string name;
    std::string model; // empty: nothing in the hand
    double mass = 0.5;   // kg, as far as a blow is concerned
    double length = 0.8; // metres it adds to an arm
    // As a tool: cells of each a blow takes. Below 1 a cell takes several blows; 0, it does nothing to it.
    double tool_wood = 0.0;
    double tool_stone = 0.0;
    double tool_leaf = 0.0;
};

struct MoveDef {
    std::string name;
    std::string weapon;
    std::string anim;
    double windup = 0.1;
    double active = 0.1;
    double recover = 0.2;
    double cancel = 0.5; // how far into the recovery the next thing can begin, 0 to 1
    int on_key = -1;     // the key it starts on, if not by its place in the list
    double walk = 0.25;  // of your speed, what you keep while it is under way; above this, nothing carries you for it
    int rounds = 1;      // how many times the blow goes across its arc, reaching everyone again each time
    double reach = 1.5;  // metres from the edge of the body
    double arc = 90.0;   // degrees, around where it faces
    double hit = 0.0;
    double knock = 0.0;
    double lift = 0.0;
    double step = 0.0;   // m/s, while it lands
    double breaks = 0.0; // force mana left where it strikes the world
    double shock = 0.0;  // metres: what it does to the ground around it when it lands
    int mastery = 0;     // skill with the weapon it takes to know it
    double stamina = 0.0; // move_stamina

    [[nodiscard]] double length() const { return windup + active + recover; }
    [[nodiscard]] double free_from() const { return windup + active + recover * cancel; }
};

// What a move costs follows from what it does, so a stronger one is never also a cheaper one.
double move_stamina(const MoveDef& move);

struct MeleeBook {
    std::vector<WeaponDef> weapons;
    std::vector<MoveDef> moves;

    [[nodiscard]] int weapon_slot(const std::string& name) const;
    [[nodiscard]] int move_slot(const std::string& name) const;
    // A move's place among this weapon's, 0 first; the moves of it that this much skill knows, in order.
    [[nodiscard]] std::vector<int> known(const std::string& weapon, int skill) const;
};

// Errors name the line. What it knows of animation is only the clip's name.
MeleeBook parse_moves(const std::string& text, std::vector<std::string>& errors);

// From the first root that has it.
MeleeBook load_moves(std::vector<std::string>& errors);

// A blow under way, in a body.
struct Swing {
    int move = -1;
    double t = 0.0;
    double dir_x = 1.0; // where it faces
    double dir_y = 0.0;
    bool landed_world = false;
    double power = 1.0; // how far along a string it comes, and what that adds to it
    bool finisher = false; // the last blow of a string
    bool stepped = false;
    int round = 0;
    std::vector<int> struck; // who it has hit already: one blow, once each
};

} // namespace fieldengine::field

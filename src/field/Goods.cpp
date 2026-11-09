#include "field/Playground.hpp"

// WHAT PEOPLE WEAR AND HOLD: assets/world/items.world. Clothes are figures on the same bones as bodies, chosen by weather and work
// (People.cpp, outfit); what is in a hand is the weapon its moves are (Melee.cpp). Nothing here is carried, bought or used up.
namespace fieldengine::field {

void Playground::start_goods()
{
    // With rigid pieces on, what loses its support comes down whole (Falling.cpp), not as the law's rubble.
    if (config_.falling) {
        set_param("fibre_falls", 0.0);
    }
    std::vector<std::string> errors;
    items_ = load_items(errors);
    for (const std::string& e : errors) {
        note(e);
    }
}

} // namespace fieldengine::field

#pragma once

#include "field/Playground.hpp"

#include <string>

// Pouring a named pour (assets/law/pours.law) the way a body facing it would: at the point aimed at, a wall across the way from
// the body to it, a beam from the body to it. What the gates used to cast, now that nothing casts.
namespace fieldengine::gate {

inline int pour_named(const field::Playground& g, const std::string& name)
{
    return g.pour_slot(name);
}

inline void pour_toward(field::Playground& g, int slot, field::Vec3 aim)
{
    using namespace fieldengine::field;
    if (slot < 0 || slot >= static_cast<int>(g.pours().size())) {
        return;
    }
    const Pour& p = g.pours()[static_cast<std::size_t>(slot)];
    const Vec3 from = g.player().position;
    Vec3 way{aim.x - from.x, aim.y - from.y, 0.0};
    if (length(way) < 1e-9) {
        way = Vec3{1.0, 0.0, 0.0};
    }
    if (p.shape == Pour::Shape::Beam) {
        g.pour_at(slot, from, normalize(aim - from) * p.range);
    } else {
        g.pour_at(slot, aim, normalize(way));
    }
}

} // namespace fieldengine::gate

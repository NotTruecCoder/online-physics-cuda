#pragma once

#include "field/LawNode.hpp"

#include <limits>
#include <vector>

namespace fieldengine::field {

// How far an expression can go, given how far the fields go. A term whose every equation is exactly [0, 0] has nothing to
// do this step, and is not dispatched: stone does not melt in a world with nothing near the melting point. Nothing in here
// knows what a term is about. Whatever cannot be bounded is everything.
struct Bounds {
    double lo = -std::numeric_limits<double>::infinity();
    double hi = std::numeric_limits<double>::infinity();
    [[nodiscard]] bool none() const { return lo == 0.0 && hi == 0.0; }
};

// Times, with 0 * infinity as nothing.
double times(double a, double b);

// Exactly zero everywhere, given which fields are (per field: every cell 0): a term about mana nobody has cast is never
// dispatched. Shared by the GPU backends.
bool always_zero(const Node& n, const std::vector<char>& zero, const std::vector<double>& params);

Bounds bounds_of(const Node& n, const std::vector<Bounds>& fields, const std::vector<double>& params, double h);

// What a term of explicit Euler substeps (`substeps` of dt / substeps) may have done to the bounds of the fields it
// writes, for the terms after it: carried substep by substep, never narrower than the cells. Clipped to the declared
// range only for a field that is clamped to it. Shared by the GPU backends.
void flow_bounds(const std::vector<Equation>& equations, double dt, int substeps, const Law& law, double h, std::vector<Bounds>& flowing);

} // namespace fieldengine::field

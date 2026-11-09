#pragma once

namespace fieldengine::render::units {

// Gate G2 (see src/render/UNITS.md). The simulation space is normalized [0,1]^2 for all of E2; the
// arena models a 100 m x 100 m zone (GDD §14.7). These constants are the single source of truth for
// the metres-per-unit conversion the Jolt bridge will use in E3. E2 does not depend on them for
// anything load-bearing; they exist so E3 is not relitigated.
inline constexpr double metres_per_sim_unit = 100.0;
inline constexpr double sim_units_per_metre = 1.0 / metres_per_sim_unit;

constexpr double to_metres(double sim_units) { return sim_units * metres_per_sim_unit; }
constexpr double to_sim_units(double metres) { return metres * sim_units_per_metre; }

} // namespace fieldengine::render::units

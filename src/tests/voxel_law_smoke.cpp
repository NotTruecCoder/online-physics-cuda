// GATE: THE LAW IS TEXT, AND THE ENGINE HONOURS WHAT THE TEXT PROMISES.
//
// Properties (each a claim about the world, not about a function):
//   1. The shipped laws (assets/law/world.law + magic.law) compile, and what each term reads and
//      writes is DERIVED from its equations — checked against a roll call written out by hand here.
//   2. A broken law is refused with the source and line that broke it.
//   3. A term that promises `conserves T` moves heat and never makes it — and a term that breaks the
//      promise is reported by name (falsified in place with a leaky term).
//   4. The author's `max_dt` is what keeps the world bounded: with it, diffusion never raises the
//      hottest cell (maximum principle); ignoring it (the falsification hook) blows that up.
//   5. `range` is a contract: a write outside it is reported with the field and the cell, unless the
//      field asked to be clamped.
//   6. Same law, same start -> same bits; a one-kelvin difference -> different bits (anti-vacuity).
//   7. Magic needs no C++: adding magic.law makes mana kindle fuel and add to gravity's source; the
//      same world without it does neither.
//   8. Fire is not a rule, it is conduction + combustion: a flame walks down a fuse of fuel.
//   9. Mass pulls light: a dense voxel is the bottom of the potential and the top of the optical index.
//
// Falsify by: deleting the conservation check in Solver::apply_term (3 fails); deleting the substep
// computation (4 fails); clamping every field (5 fails); reading an uninitialised buffer (6 fails);
// removing magic.law's `mana_kindles` term (7 fails); setting `flame = 0` in world.law (8 fails);
// flipping the sign in `derive n` (9 fails).

#include "field/Law.hpp"
#include "field/Solver.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::vector<LawSource> shipped(bool with_magic)
{
    std::vector<LawSource> sources;
    for (const char* path : {"assets/law/world.law", "assets/law/magic.law"}) {
        if (!with_magic && std::string(path).find("magic") != std::string::npos) {
            continue;
        }
        const std::optional<LawSource> source = read_law_file(path);
        check(source.has_value(), std::string("law file is present: ") + path);
        if (source) {
            sources.push_back(*source);
        }
    }
    return sources;
}

Law compile_or_report(const std::vector<LawSource>& sources, const std::string& label)
{
    CompileResult result = compile_law(sources);
    for (const std::string& error : result.errors) {
        std::cerr << "  " << label << ": " << error << "\n";
    }
    check(result.law.has_value(), label + " compiles");
    return result.law ? *result.law : Law{};
}

bool refused_with(const std::string& text, const std::string& needle)
{
    const CompileResult result = compile_law({LawSource{.name = "probe", .text = text}});
    if (result.law) {
        return false;
    }
    for (const std::string& error : result.errors) {
        if (error.find(needle) != std::string::npos && error.find("probe:") == 0) {
            return true;
        }
    }
    std::cerr << "  (refused, but not with '" << needle << "'; first error: "
              << (result.errors.empty() ? "<none>" : result.errors.front()) << ")\n";
    return false;
}

std::set<std::string> names(const Law& law, const std::vector<int>& ids)
{
    std::set<std::string> out;
    for (const int id : ids) {
        out.insert(law.fields[static_cast<std::size_t>(id)].name);
    }
    return out;
}

const Term* find_term(const Law& law, const std::string& name)
{
    for (const Term& term : law.terms) {
        if (term.name == name) {
            return &term;
        }
    }
    return nullptr;
}

bool has_violation(const Solver& solver, Violation::Kind kind, const std::string& field)
{
    for (const Violation& v : solver.violations()) {
        if (v.kind == kind && v.field == field) {
            return true;
        }
    }
    return false;
}

void report_violations(const Solver& solver, const std::string& label)
{
    for (const Violation& v : solver.violations()) {
        std::cerr << "  " << label << ": " << to_string(v.kind) << " [" << v.term << "] " << v.field << ": " << v.detail
                  << "\n";
    }
}

float max_of(const std::vector<float>& values) { return *std::max_element(values.begin(), values.end()); }
float min_of(const std::vector<float>& values) { return *std::min_element(values.begin(), values.end()); }

void set_cell(Solver& solver, const std::string& field, int x, int y, int z, float value)
{
    const GridShape& shape = solver.fields().shape();
    solver.fields().values(solver.field(field))[shape.index(x, y, z)] = value;
}

float cell(const Solver& solver, const std::string& field, int x, int y, int z)
{
    const GridShape& shape = solver.fields().shape();
    return solver.fields().values(solver.field(field))[shape.index(x, y, z)];
}

} // namespace

int main()
{
    // ---- 1. the shipped laws compile; reads/writes come from the equations ----------------------------
    const Law world = compile_or_report(shipped(false), "world.law");
    const Law full = compile_or_report(shipped(true), "world.law + magic.law");
    if (const Term* combustion = find_term(full, "combustion")) {
        check(names(full, combustion->reads) == std::set<std::string>{"T", "fuel"},
            "combustion reads exactly T and fuel (derived from its equations)");
        check(names(full, combustion->writes) == std::set<std::string>{"T", "fuel"},
            "combustion writes exactly T and fuel");
    } else {
        check(false, "world.law has a combustion term");
    }
    if (const Term* kindle = find_term(full, "mana_kindles")) {
        check(names(full, kindle->reads) == std::set<std::string>{"mana", "fuel"}, "mana_kindles reads mana and fuel");
        check(names(full, kindle->writes) == std::set<std::string>{"T"}, "mana_kindles writes only T");
    } else {
        check(false, "magic.law has a mana_kindles term");
    }
    check(find_term(world, "mana_kindles") == nullptr, "the base law alone knows nothing of mana");

    // ---- 2. a broken law is refused with its line --------------------------------------------------
    check(refused_with("field a\nterm t\n  d b = 1\n", "probe:3"), "an undeclared target is refused at its line");
    check(refused_with("field a static\nterm t\n  d a = 1\n", "writes static field 'a'"),
        "a term may not write a static field");
    check(refused_with("field a\nfield b\nterm t\n  d a = b\n  conserves b\n", "conserves 'b' but never writes it"),
        "a vacuous conservation promise is refused");
    check(refused_with("field a\nterm t\n  d a = frobnicate(a)\n", "unknown function"), "unknown functions are refused");
    check(refused_with("field a\nterm t\n  d a = lap(a, a)\n", "takes 1 argument"), "arity is checked");
    check(refused_with("field a\nderive s = distance(a)\nterm t\n  d s = 1\n", "derived field 's'"),
        "a term may not write a derived field");
    check(refused_with("field a\nterm t\n  d a = 1\n  max_dt = 1 / a\n", "max_dt must be a number"),
        "max_dt cannot silently depend on a cell");

    // ---- 3. conservation is a checked promise --------------------------------------------------------
    const GridShape shape{.nx = 16, .ny = 16, .nz = 4, .cell = 0.5};
    {
        Solver solver(world, shape);
        // Stone in one half: conductivity varies, so the flux form is actually exercised.
        for (int z = 0; z < shape.nz; ++z) {
            for (int y = 0; y < shape.ny; ++y) {
                for (int x = 8; x < shape.nx; ++x) {
                    set_cell(solver, "solid", x, y, z, 1.0f);
                }
            }
        }
        solver.set_param("cool", 0.0); // radiation is not conservative by design; isolate conduction
        solver.refresh_derived();
        set_cell(solver, "T", 7, 7, 1, 3000.0f);
        set_cell(solver, "T", 9, 3, 2, 2000.0f);
        for (int i = 0; i < 30; ++i) {
            solver.step(0.25);
        }
        report_violations(solver, "conduction");
        check(!has_violation(solver, Violation::Kind::Conservation, "T"), "conduction moves heat without making it");
    }
    {
        const Law leaky = compile_or_report({LawSource{.name = "leaky",
                                                .text = "field T init 290\n"
                                                        "term leak\n  d T = lap(T) - 0.1 * T\n  conserves T\n"
                                                        "  max_dt = 0.1 * h * h\n"}},
            "leaky law");
        Solver solver(leaky, shape);
        solver.step(0.1);
        check(has_violation(solver, Violation::Kind::Conservation, "T"),
            "a term that leaks heat while promising not to is reported (falsification)");
    }

    // ---- 4. max_dt is what keeps it bounded ----------------------------------------------------------
    {
        const std::string text = "field T init 0 range -1e9 1e9\n"
                                 "param k = 1.0\n"
                                 "term diffusion\n  d T = diffuse(k, T)\n  conserves T\n  max_dt = 0.15 * h * h / k\n";
        const Law law = compile_or_report({LawSource{.name = "diffusion", .text = text}}, "diffusion law");
        for (const bool respect : {true, false}) {
            Solver solver(law, shape);
            solver.set_respect_max_dt(respect);
            set_cell(solver, "T", 5, 5, 1, 1000.0f);
            set_cell(solver, "T", 6, 5, 1, -500.0f);
            bool bounded = true;
            for (int i = 0; i < 20; ++i) {
                solver.step(1.0); // 26x the stable step at h = 0.5
                const auto& t = solver.fields().values(solver.field("T"));
                bounded = bounded && max_of(t) <= 1000.0f + 1e-2f && min_of(t) >= -500.0f - 1e-2f;
            }
            if (respect) {
                check(bounded, "with the declared max_dt, no cell ever gets hotter than the hottest start");
                check(solver.last_substeps().front() > 1, "the declared max_dt actually subdivides a long step");
            } else {
                check(!bounded, "ignoring max_dt breaks the maximum principle (the contract is load-bearing)");
            }
        }
    }

    // ---- 5. range is reported, or clamped on request -------------------------------------------------
    {
        const Law law = compile_or_report({LawSource{.name = "range",
                                              .text = "field a init 0 range 0 1\nfield b init 0 range 0 1 clamp\n"
                                                      "term pump\n  d a = 2\n  d b = 2\n"}},
            "range law");
        Solver solver(law, GridShape{4, 4, 1, 1.0});
        solver.step(1.0);
        check(has_violation(solver, Violation::Kind::Range, "a"), "a write past the declared range is reported");
        bool names_cell = false;
        for (const Violation& v : solver.violations()) {
            names_cell = names_cell || (v.field == "a" && v.detail.find("(0,0,0)") != std::string::npos);
        }
        check(names_cell, "the report names the cell");
        check(!has_violation(solver, Violation::Kind::Range, "b") && cell(solver, "b", 2, 2, 0) == 1.0f,
            "a field that asked for clamp is clamped, not reported");
    }

    // ---- 6. determinism and its anti-vacuity twin ----------------------------------------------------
    {
        auto run = [&](float extra) {
            Solver solver(full, shape);
            for (int x = 2; x < 12; ++x) {
                set_cell(solver, "fuel", x, 4, 1, 1.0f);
            }
            set_cell(solver, "T", 2, 4, 1, 900.0f + extra);
            set_cell(solver, "mana", 10, 10, 2, 50.0f);
            set_cell(solver, "mass", 12, 12, 2, 500.0f);
            solver.refresh_derived();
            for (int i = 0; i < 25; ++i) {
                solver.step(0.2);
            }
            return solver.fields().digest();
        };
        const auto a = run(0.0f);
        const auto b = run(0.0f);
        const auto c = run(1.0f);
        check(a == b, "the same law from the same start gives the same bits");
        check(a != c, "a one-kelvin difference at the start changes the bits (the digest sees the world)");
    }

    // ---- 7. magic is more law ------------------------------------------------------------------------
    {
        auto kindled = [&](const Law& law) {
            Solver solver(law, shape);
            set_cell(solver, "fuel", 4, 4, 1, 1.0f);
            if (solver.field("mana") >= 0) {
                set_cell(solver, "mana", 4, 4, 1, 100.0f);
            }
            // The PEAK temperature: a kindled voxel burns out, and by the end of the run it may
            // already be cooling — what matters is that it caught.
            float peak = 0.0f;
            for (int i = 0; i < 40; ++i) {
                solver.step(0.1);
                peak = std::max(peak, cell(solver, "T", 4, 4, 1));
            }
            return std::pair{peak, cell(solver, "fuel", 4, 4, 1)};
        };
        const auto [t_magic, fuel_magic] = kindled(full);
        const auto [t_plain, fuel_plain] = kindled(world);
        check(t_magic > 600.0f && fuel_magic < 0.9f, "with magic.law, mana in a fuel voxel kindles it");
        check(t_plain < 400.0f && fuel_plain == 1.0f, "the same voxel without magic.law never burns");

        Solver solver(full, shape);
        set_cell(solver, "mana", 8, 8, 2, 400.0f);
        solver.refresh_derived();
        check(cell(solver, "gsrc", 8, 8, 2) > 0.0f, "mana adds to gravity's source through `derive gsrc +=`");
        check(cell(solver, "n", 8, 8, 2) > cell(solver, "n", 1, 1, 0), "...and so it bends light toward itself");
    }

    // ---- 8. a flame walks down a fuse -----------------------------------------------------------------
    {
        // A one-voxel-wide fuse in a 1-D world: heat leaks only forward and back. (In open air the
        // same thread of fuel loses heat sideways faster than it burns and goes out — which is also
        // the law speaking, and why a campfire needs a pile, not a line.)
        Solver solver(world, GridShape{24, 1, 1, 0.5});
        for (int x = 0; x < 24; ++x) {
            set_cell(solver, "fuel", x, 0, 0, 1.0f);
        }
        set_cell(solver, "T", 0, 0, 0, 1500.0f);
        int front = 0;
        for (int i = 0; i < 10; ++i) {
            solver.step(0.1);
        }
        check(cell(solver, "fuel", 23, 0, 0) == 1.0f && cell(solver, "T", 23, 0, 0) < 400.0f,
            "a second in, the far end of the fuse has not heard of the fire (finite speed)");
        for (int i = 0; i < 190; ++i) {
            solver.step(0.1);
        }
        for (int x = 0; x < 24; ++x) {
            if (cell(solver, "fuel", x, 0, 0) < 0.5f) {
                front = x;
            }
        }
        report_violations(solver, "fuse");
        check(front >= 4, "fire spreads along the fuel from conduction + combustion alone (front at voxel " +
                std::to_string(front) + ")");
    }

    // ---- 9. mass pulls light ---------------------------------------------------------------------------
    {
        Solver solver(world, shape);
        set_cell(solver, "mass", 8, 8, 2, 2000.0f);
        for (int i = 0; i < 8; ++i) {
            solver.refresh_derived(); // warm-started potential settles over a few refreshes
        }
        const float phi_centre = cell(solver, "phi", 8, 8, 2);
        const float phi_far = cell(solver, "phi", 1, 1, 0);
        check(phi_centre < phi_far && phi_centre < 0.0f, "a dense voxel is the bottom of the potential well");
        check(cell(solver, "n", 8, 8, 2) > cell(solver, "n", 1, 1, 0), "...and the top of the optical index");
    }

    // ---- baseline (printed, not gated) ----------------------------------------------------------------
    {
        const GridShape big{.nx = 64, .ny = 64, .nz = 16, .cell = 0.5};
        Solver solver(full, big);
        const auto start = std::chrono::steady_clock::now();
        const int steps = 10;
        for (int i = 0; i < steps; ++i) {
            solver.step(1.0 / 30.0);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "baseline: " << big.cells() << " cells x " << full.terms.size() << " terms + "
                  << full.derives.size() << " derives: " << (seconds / steps) * 1000.0 << " ms/step (CPU, 1 thread)\n";
    }

    return fieldengine::gate::report("voxel_law");
}

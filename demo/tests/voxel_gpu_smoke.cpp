// THE LAW ON THE GPU IS THE SAME LAW. The CPU solver is the reference: from one start the two must stay
// together, op by op and over the whole shipped law with a world on fire, and every contract must still be
// reported by name. Without a hardware GPU the gate says so and passes; there is nothing to test.
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include "field/GpuSolver.hpp"
#include "world/Playground.hpp"
#include "field/Solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <string>
#include <vector>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

struct Compare {
    double worst = 0.0;     // |a - b|
    double worst_rel = 0.0; // relative to the field's own scale
    std::size_t off = 0;
    std::string where;
};

// How far the two lattices are apart, field by field.
Compare compare(const Solver& a, const Solver& b, double tolerance)
{
    Compare out;
    for (std::size_t id = 0; id < a.fields().count(); ++id) {
        const std::vector<float>& x = a.fields().values(static_cast<int>(id));
        const std::vector<float>& y = b.fields().values(static_cast<int>(id));
        double scale = 1.0;
        for (const float v : x) {
            scale = std::max(scale, static_cast<double>(std::abs(v)));
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = std::abs(static_cast<double>(x[i]) - y[i]);
            if (d > out.worst) {
                out.worst = d;
            }
            if (d / scale > out.worst_rel) {
                out.worst_rel = d / scale;
                out.where = a.law().fields[id].name + " cell " + std::to_string(i) + ": " + std::to_string(x[i]) + " vs " + std::to_string(y[i]);
            }
            if (d / scale > tolerance) {
                ++out.off;
            }
        }
    }
    return out;
}

Law must_compile(const std::string& text)
{
    CompileResult compiled = compile_law({LawSource{"gate", text}});
    for (const std::string& e : compiled.errors) {
        std::cout << "  " << e << std::endl;
    }
    return *compiled.law;
}

void seed_blob(FieldSet& fields, int id, float value)
{
    const GridShape& s = fields.shape();
    for (int z = 0; z < s.nz; ++z) {
        for (int y = 0; y < s.ny; ++y) {
            for (int x = 0; x < s.nx; ++x) {
                const double dx = x - s.nx * 0.4;
                const double dy = y - s.ny * 0.6;
                const double dz = z - s.nz * 0.5;
                if (dx * dx + dy * dy + dz * dz < 12.0) {
                    fields.values(id)[s.index(x, y, z)] = value * static_cast<float>(1.0 + 0.1 * x);
                }
            }
        }
    }
}

const GridShape kSmall{.nx = 24, .ny = 19, .nz = 13, .cell = 0.5};

// Every op of the language, alone, against the reference.
void check_every_op()
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"arith", "d A = (B + 2) * (B - 1) / (B + 3) - 0.1 * A"},
        {"div0", "d A = 1 / B"},
        {"funcs", "d A = 0.01 * (abs(B - 1) + sqrt(B) + exp(0.1 * B) + log(B + 1) + step(B - 1) + pos(B - 2))"},
        {"minmax", "d A = min(B, 2) - max(A, 1) + clamp(B, 1, 3) - clamp(B, 3, 1)"},
        {"pow", "d A = 0.01 * ((B - 1)^2 + (B + 1)^0.5 + B^0 + (B - 1)^3)"},
        {"lap", "d A = 0.05 * lap(B)"},
        {"grad", "d A = 0.1 * (gx(B) + gy(B) + gz(B))"},
        {"diffuse", "d A = diffuse(0.1 + 0.1 * B, B)"},
        {"adv", "d A = adv(B, 0.5 - 0.2 * B, 0.3, -0.4)"},
        {"shift", "d A = shift(B, 2, -1, 0) - shift(B, 0, 0, -3)"},
        {"fall", "d A = fall(A + B, 0.1 * B, 2)"},
        {"spread", "d A = spread(A + B, 0.1 * B, 0.05)"},
        {"topple", "d A = topple(B, step(1 - B), 0.5)"},
        {"topple over a drop", "d A = topple(B, 0.2 * B, 0.5)"},
        {"reduce", "d A = 0.01 * (maxof(B) - minof(A) + 0.001 * sumof(B))"},
    };
    for (const auto& [name, equation] : cases) {
        const std::string text = "field A init 0.5 range -1e9 1e9\nfield B init 0 range -1e9 1e9\nterm t\n  " + equation + "\n  max_dt = 0.02\n";
        Solver cpu(must_compile(text), kSmall);
        Solver gpu(must_compile(text), kSmall);
        std::string why;
        check(gpu.use_gpu(why), ("the op compiles for the GPU: " + name + " " + why).c_str());
        if (!gpu.on_gpu()) {
            continue;
        }
        seed_blob(cpu.fields(), cpu.field("B"), 3.0f);
        seed_blob(gpu.fields(), gpu.field("B"), 3.0f);
        for (int s = 0; s < 5; ++s) {
            cpu.step(0.05);
            gpu.step(0.05);
        }
        const Compare c = compare(cpu, gpu, 2e-3);
        if (c.off != 0) {
            std::cout << "  " << name << ": " << c.off << " cells off, worst " << c.worst_rel << " (" << c.where << ")" << std::endl;
        }
        check(c.off == 0, ("the GPU computes the same as the reference: " + name).c_str());
        check(gpu.last_substeps().front() == cpu.last_substeps().front(), ("the same substeps: " + name).c_str());
    }
}

// A field that is zero when the step starts can stop being zero during it: the term after the one that wrote it
// is not idle, in the same step.
void check_chain_in_one_step()
{
    const std::string text = "field X init 1 range 0 1e9\nfield Y init 0 range 0 1e9\nfield Z init 0 range 0 1e9\n"
                             "term first\n  d Y = X\nterm second\n  d Z = Y\n";
    Solver cpu(must_compile(text), kSmall);
    Solver gpu(must_compile(text), kSmall);
    std::string why;
    check(gpu.use_gpu(why), "the chain compiles for the GPU");
    cpu.step(0.1);
    gpu.step(0.1);
    check(gpu.fields().values(gpu.field("Z"))[0] > 0.0f, "what one term wakes, the next term reads in the same step");
    check(compare(cpu, gpu, 1e-5).off == 0, "the chain is the same as the reference");
}

// distance is exact: the same cells, the same metres, whatever the rock looks like, and again after it moves.
void check_globals()
{
    const std::string text = "field M init 0 range 0 1\nfield Q init 0 range -1e9 1e9\nfield R init 0 range -1e9 1e9\n"
                             "derive D = distance(M)\nderive P = poisson(Q, 16)\nderive P3 = poisson(R, 7)\n";
    CompileResult compiled = compile_law({LawSource{"gate", text}});
    check(compiled.law.has_value(), "a law of globals compiles");
    Solver cpu(*compiled.law, kSmall);
    Solver gpu(*compiled.law, kSmall);
    std::string why;
    check(gpu.use_gpu(why), "a law of globals compiles for the GPU");
    if (!gpu.on_gpu()) {
        return;
    }
    auto both = [&](auto&& fn) {
        fn(cpu);
        fn(gpu);
        cpu.refresh_derived();
        gpu.refresh_derived();
    };
    auto distances_equal = [&](const char* what) {
        const std::vector<float>& a = cpu.fields().values(cpu.field("D"));
        const std::vector<float>& b = gpu.fields().values(gpu.field("D"));
        std::size_t off = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            off += std::abs(a[i] - b[i]) > 1e-5f ? 1 : 0;
        }
        if (off != 0) {
            std::cout << "  " << off << " cells of distance differ" << std::endl;
        }
        check(off == 0, what);
    };
    both([](Solver&) {});
    distances_equal("distance with no rock at all");
    both([](Solver& s) { std::fill(s.fields().values(s.field("M")).begin(), s.fields().values(s.field("M")).end(), 1.0f); });
    distances_equal("distance when all of it is rock");
    both([](Solver& s) {
        std::fill(s.fields().values(s.field("M")).begin(), s.fields().values(s.field("M")).end(), 0.0f);
        seed_blob(s.fields(), s.field("M"), 1.0f);
    });
    distances_equal("distance to a ball of rock");
    both([](Solver& s) { s.fields().values(s.field("M"))[s.fields().shape().index(1, 1, 1)] = 1.0f; });
    distances_equal("distance after one cell was added: the change is noticed");
    both([](Solver& s) {
        seed_blob(s.fields(), s.field("Q"), 2.0f);
        seed_blob(s.fields(), s.field("R"), -1.0f);
    });
    const Compare c = compare(cpu, gpu, 1e-3);
    if (c.off != 0) {
        std::cout << "  poisson: " << c.off << " cells off, worst " << c.worst_rel << " (" << c.where << ")" << std::endl;
    }
    check(c.off == 0, "poisson, with an even and an odd number of sweeps");
}

// The contracts hold on the card too, named as they are on the CPU.
void check_contracts()
{
    auto first = [](const Solver& s, Violation::Kind kind, const std::string& term) {
        for (const Violation& v : s.violations()) {
            if (v.kind == kind && v.term == term) {
                return true;
            }
        }
        return false;
    };
    auto run = [&](const std::string& text, double dt = 0.1) {
        Solver gpu(must_compile(text), kSmall);
        std::string why;
        gpu.use_gpu(why);
        seed_blob(gpu.fields(), gpu.field("A"), 1.0f);
        gpu.step(dt);
        return gpu;
    };
    {
        Solver s = run("field A init 0 range 0 1\nterm grows\n  d A = 100\n");
        check(first(s, Violation::Kind::Range, "grows"), "a broken range is reported, with its term");
    }
    {
        Solver s = run("field A init 0 range -1e9 1e9\nterm leaks\n  d A = 0.5\n  conserves A\n");
        check(first(s, Violation::Kind::Conservation, "leaks"), "a broken conservation is reported, with its term");
    }
    {
        Solver s = run("field A init 0 range -1e9 1e9\nterm moves\n  d A = diffuse(0.1, A)\n  conserves A\n");
        check(s.violations().empty(), "an honest diffusion is not reported");
    }
    {
        Solver s = run("field A init 0 range -1e9 1e9\nterm blows\n  d A = exp(80) * exp(80) * A\n");
        check(first(s, Violation::Kind::NotFinite, "blows"), "an infinity is reported");
    }
    {
        Solver s = run("field A init 0 range 0 1 clamp\nterm grows\n  d A = 100\n");
        check(s.violations().empty(), "a clamped field is not a violation");
        const std::vector<float>& a = s.fields().values(s.field("A"));
        check(*std::max_element(a.begin(), a.end()) == 1.0f, "a clamped field holds its range");
    }
    {
        Solver s = run("field A init 0 range -1e9 1e9\nterm stiff\n  d A = -A\n  max_dt = 1e-6\n", 1.0);
        check(first(s, Violation::Kind::SubstepCap, "stiff"), "a substep cap is reported");
    }
}

Playground::Config world_config(bool gpu)
{
    Playground::Config c;
    c.open_world = true;
    c.window_chunks = 3;
    c.field_every = 4;
    c.gpu_fields = gpu;
    return c;
}

std::optional<Playground> make_world(bool gpu)
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    check(Playground::read_default_law(laws, spells, path, errors), "the law files read");
    std::optional<Playground> world = Playground::create(laws, spells, errors, world_config(gpu));
    for (const std::string& e : errors) {
        std::cout << "  " << e << std::endl;
    }
    world->build_open_world();
    return world;
}

// Every element, into the ground and the air around you.
void cast_everything(Playground& world, double spot)
{
    const Vec3 p = world.player().position;
    int k = 0;
    for (const Pour& spell : world.pours()) {
        const double a = k++ * 1.1;
        for (const double down : {-1.0, 0.5}) {
            world.pour(spell, p + Vec3{spot * std::cos(a), spot * std::sin(a), down}, Vec3{1, 0, 0});
        }
    }
}

double peak(const Playground& world, const char* name)
{
    const std::vector<float>& v = world.fields().values(world.fields().find(name));
    return *std::max_element(v.begin(), v.end());
}

// The shipped law, the generated world, every element cast at once.
void check_whole_law()
{
    std::optional<Playground> cpu = make_world(false);
    std::optional<Playground> gpu = make_world(true);
    check(gpu->solver().on_gpu(), "the world runs on the GPU");
    if (!gpu->solver().on_gpu()) {
        return;
    }
    std::cout << "  on " << gpu->solver().gpu_name() << std::endl;
    check(cpu->digest() != 0 && compare(cpu->solver(), gpu->solver(), 1e-3).off == 0, "the worlds start the same");
    // Magma under your feet, so stone melts, flows, meets water and sets.
    for (Playground* world : {&*cpu, &*gpu}) {
        FieldSet& f = world->solver().fields();
        const GridShape& g = f.shape();
        const int T = f.find("T");
        const Vec3 p = world->player().position;
        const int cx = static_cast<int>(p.x / g.cell) + 10;
        const int cy = static_cast<int>(p.y / g.cell);
        const int cz = static_cast<int>(p.z / g.cell);
        for (int z = std::max(0, cz - 6); z < std::min(g.nz, cz + 2); ++z) {
            for (int y = cy - 4; y <= cy + 4; ++y) {
                for (int x = cx - 4; x <= cx + 4; ++x) {
                    if (g.inside(x, y, z)) {
                        f.values(T)[g.index(x, y, z)] = 3000.0f;
                    }
                }
            }
        }
    }
    double worst_share = 0.0;
    double hottest = 0.0;
    double rubble = 0.0;
    double lava = 0.0;
    for (int tick = 1; tick <= 240; ++tick) {
        if (tick % 20 == 1) {
            cast_everything(*cpu, 5.0);
            cast_everything(*gpu, 5.0);
        }
        cpu->step(Control{});
        gpu->step(Control{});
        hottest = std::max(hottest, peak(*gpu, "T"));
        rubble = std::max(rubble, gpu->total("rubble"));
        lava = std::max(lava, gpu->total("lava"));
        if (tick % 40 == 0) {
            const Compare c = compare(cpu->solver(), gpu->solver(), 0.02);
            const double share = static_cast<double>(c.off) / static_cast<double>(cpu->fields().shape().cells());
            worst_share = std::max(worst_share, share);
            std::cout << "  tick " << tick << ": " << c.off << " of cells*fields past 2%, worst " << c.worst_rel << " (" << c.where << ")" << std::endl;
        }
    }
    // What this comparison is worth: the scene has to have done the things it compares.
    check(hottest > 1400.0 && lava > 0.5, "the scene melted stone");
    check(rubble > 0.5, "the scene broke stone");
    check(worst_share < 0.002, "over four seconds of everything at once, the GPU stays with the reference");
    for (const char* name : {"solid", "rubble", "water", "lava", "T", "grass"}) {
        const double a = cpu->total(name);
        const double b = gpu->total(name);
        std::cout << "  total " << name << ": " << a << " vs " << b << std::endl;
        check(std::abs(a - b) <= 0.01 * (std::abs(a) + 10.0), (std::string("the same amount of ") + name).c_str());
    }
    check(gpu->violations().size() == cpu->violations().size(), "the same contracts held");
    for (const Violation& v : gpu->violations()) {
        std::cout << "  GPU: " << v.term << " " << v.field << ": " << v.detail << std::endl;
    }
}

// Two runs on one card are the same to the bit, and the sim above it is deterministic as before.
void check_determinism()
{
    std::optional<Playground> a = make_world(true);
    std::optional<Playground> b = make_world(true);
    cast_everything(*a, 4.0);
    cast_everything(*b, 4.0);
    for (int tick = 0; tick < 120; ++tick) {
        Control walk;
        walk.move_x = 1.0;
        a->step(walk);
        b->step(walk);
    }
    check(a->digest() == b->digest(), "two runs on one card agree to the bit");
}

// The world around the solver works as before: bodies and window, edits, a law changed while it runs.
void check_living_world()
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    Playground::Config config = world_config(true);
    config.async_fields = true;
    std::optional<Playground> world = Playground::create(laws, spells, errors, config);
    world->build_open_world();
    const int x0 = world->window_cx();
    Control walk;
    walk.move_x = 1.0;
    const double heat0 = world->total("T");
    cast_everything(*world, 3.0);
    for (int tick = 0; tick < 30; ++tick) {
        world->step(Control{});
    }
    check(world->total("T") != heat0, "the law ran");
    for (int tick = 0; tick < 700; ++tick) {
        walk.jump = tick % 30 == 0; // over the trees in the way
        world->step(walk);
    }
    check(world->window_cx() != x0, "walking moves the window with the fields on the GPU");
    {
        // What was cast while the window moved lands where it was cast, in the lattice the window moved to.
        // Walking +x, a trail behind you is right; an edit landed in the old lattice's cell would lie a whole
        // chunk ahead of you.
        const Pour& fire = world->pours().front();
        const int mana = world->fields().find(fire.field);
        const std::uint64_t version = world->window_version();
        double ahead = -100.0;
        for (int tick = 0; tick < 500 && world->window_version() < version + 2; ++tick) {
            const std::uint64_t seen = world->window_version();
            world->player().hp = world->player().hp_max; // standing in your own fire: dying would take you home, and that is not what is tested
            world->pour(fire, world->player().position + Vec3{0.0, 0.0, 0.5}, Vec3{1, 0, 0});
            world->step(walk);
            if (world->window_version() != seen) {
                const FieldSet& f = world->fields();
                const GridShape& g = f.shape();
                const double me = world->player().position.x;
                const std::vector<float>& v = f.values(mana);
                for (int z = 0; z < g.nz; ++z) {
                    for (int y = 0; y < g.ny; ++y) {
                        for (int x = 0; x < g.nx; ++x) {
                            if (v[g.index(x, y, z)] > 0.5f) {
                                ahead = std::max(ahead, (x + 0.5) * g.cell - me);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  mana poured during window moves: at most " << ahead << " m ahead of you\n";
        check(ahead > -100.0 && ahead < 4.0, "what is cast during a window move stays where it was cast");
    }
    for (int tick = 0; tick < 40; ++tick) {
        world->step(walk);
    }
    {
        // Read through the settle: the state and the picture bodies read are one.
        const Solver& settled = world->solver();
        check(settled.fields().digest() == world->fields().digest(), "the picture is the state, field by field");
    }
    check(world->player().position.z > 0.0 && world->player().hp > 0.0, "you stand on what the GPU says is there");
    std::vector<LawSource> changed = laws;
    changed[0].text += "\nparam gate_param = 2\nfield gate_field init 3 range 0 10\nterm gate_term\n  d gate_field = gate_param\n";
    errors.clear();
    check(world->reload_law(changed, spells, errors), "the law changes while the world runs on the GPU");
    check(world->solver().on_gpu(), "the changed law is still on the GPU");
    // After the walk the window may still be catching up, and on the law's thread a window move takes a field step's
    // turn (dt 0): wait until the law has had steps of its own, or give up after a few seconds of world time.
    const int gate_field = world->solver().field("gate_field");
    for (int tick = 0; tick < 240 && world->fields().values(gate_field)[0] <= 3.0f; ++tick) {
        world->step(Control{});
    }
    const std::vector<float>& added = world->fields().values(gate_field);
    check(added[0] > 3.0f, "the new term runs");
}

// A window move takes the worker's turn from a field step; the law makes up the time. A clock in the law
// counts it: the window goes back and forth over one chunk edge, so the cell it is read in never leaves it.
void check_time_is_kept()
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    laws[0].text += "\nfield gate_clock init 0 range 0 1e9\nterm gate_clock_runs\n  d gate_clock = 1\n";
    Playground::Config config = world_config(true);
    config.async_fields = true;
    std::optional<Playground> world = Playground::create(laws, spells, errors, config);
    world->build_open_world();
    const TerrainConfig& t = world->config().terrain;
    const double chunk_m = t.chunk * t.cell;
    const std::uint64_t first = world->window_version();
    int ticks = 0;
    bool forth = true;
    for (; ticks < 1500; ++ticks) {
        if (ticks % 40 == 0) {
            // Over the edge of the middle chunk, one way and then the other, at the height of the middle of the window.
            PlayBody& you = world->player();
            you.position = Vec3{forth ? 2.0 * chunk_m + 1.5 : chunk_m - 1.5, 1.5 * chunk_m, 1.5 * chunk_m};
            you.velocity = Vec3{};
            forth = !forth;
        }
        world->step(Control{});
    }
    const std::uint64_t version = world->window_version();
    const std::vector<float>& clock = world->fields().values(world->fields().find("gate_clock"));
    const double most = *std::max_element(clock.begin(), clock.end());
    const double elapsed = ticks * Playground::kTick;
    std::cout << "  " << version - first << " window moves: the law's clock says " << most << " s of " << elapsed << " s\n";
    check(version - first >= 8, "the window went back and forth");
    check(most <= elapsed + 1e-3 && most > elapsed - 0.4, "window moves cost the law no time");
}

// A field all zero everywhere does not slide with the window. One that fills when a chunk comes back must still show.
void check_zero_field_returns()
{
    Playground::Config c = world_config(true);
    c.window_chunks = 3;
    c.async_fields = true;
    c.far_life = false;
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    std::optional<Playground> world = Playground::create(laws, spells, errors, c);
    world->build_open_world();
    const int mass = world->fields().find("mass");
    const GridShape& g = world->fields().shape();
    const TerrainConfig& t = world->config().terrain;
    const double chunk_m = t.chunk * t.cell;
    world->solver().edit_cell(mass, g.index(4, 4, 4)) = 5.0f;
    world->solver().refresh_derived();
    const auto walk = [&](double direction, int moves) {
        // Past the edge of the middle chunk, again and again, as the moves come on the worker.
        for (int k = 0; k < moves; ++k) {
            // Held there, well past the edge: if it is in rock, it gets pushed out as far as a cell a tick.
            for (int tick = 0; tick < 16; ++tick) {
                world->player().position = Vec3{direction > 0.0 ? 2.0 * chunk_m + 3.0 : chunk_m - 3.0, 1.5 * chunk_m, 1.5 * chunk_m};
                world->player().velocity = Vec3{};
                world->step(Control{});
            }
        }
    };
    for (int tick = 0; tick < 8; ++tick) {
        world->step(Control{});
    }
    const int start = world->window_cx();
    walk(1.0, 6);
    check(world->window_cx() >= start + 3, "the window moved away");
    check(world->total("mass") == 0.0, "a field left behind with its chunk");
    for (int tries = 0; tries < 24 && world->window_cx() > start; ++tries) {
        walk(-1.0, 1);
    }
    for (int tick = 0; tick < 16; ++tick) {
        world->step(Control{});
    }
    check(world->total("mass") > 4.0, "what went out of the window comes back, in the picture");
    double in_law = 0.0;
    for (const float v : std::as_const(world->solver()).fields().values(mass)) {
        in_law += v;
    }
    check(in_law > 4.0, "and in the law");
}

void report_speed()
{
    for (const bool gpu : {false, true}) {
        Playground::Config c = world_config(gpu);
        c.window_chunks = 5;
        std::vector<std::string> errors;
        std::vector<LawSource> laws;
        std::string spells;
        std::string path;
        Playground::read_default_law(laws, spells, path, errors);
        const auto t0 = std::chrono::steady_clock::now();
        std::optional<Playground> world = Playground::create(laws, spells, errors, c);
        world->build_open_world();
        const double setup = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        cast_everything(*world, 5.0);
        const auto t1 = std::chrono::steady_clock::now();
        for (int s = 0; s < 30; ++s) {
            world->solver().step(1.0 / 15.0);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 30.0;
        std::cout << "  5x5 window, " << (gpu ? "GPU" : "CPU") << ": " << ms << " ms a field step, set up in " << setup << " ms" << std::endl;
    }
}

// The client's window, walking: what a window move costs the worker, and what a frame ever waits for it.
void report_walk()
{
    Playground::Config c = world_config(true);
    c.window_chunks = 9;
    c.async_fields = true;
    c.field_every = 4;
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    Playground::read_default_law(laws, spells, path, errors);
    std::optional<Playground> world = Playground::create(laws, spells, errors, c);
    world->build_open_world();
    double law_sum = 0.0;
    double law_worst = 0.0;
    double move_sum = 0.0;
    double move_worst = 0.0;
    double tick_worst = 0.0;
    int laws_seen = 0;
    int moves = 0;
    std::uint64_t version = world->window_version();
    int layer_moves = 0;
    int layer = world->window_cz();
    for (int tick = 0; tick < 1200; ++tick) {
        const auto start = std::chrono::steady_clock::now();
        world->step(Control{.move_x = 0.8, .move_y = (tick / 120) % 2 == 0 ? 0.6 : -0.6, .jump = tick % 45 == 0});
        tick_worst = std::max(tick_worst, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        if (tick % 4 == 3) {
            const double ms = world->field_ms();
            if (world->window_version() != version) {
                version = world->window_version();
                move_sum += ms;
                move_worst = std::max(move_worst, ms);
                ++moves;
                layer_moves += world->window_cz() != layer ? 1 : 0;
                layer = world->window_cz();
            } else {
                law_sum += ms;
                law_worst = std::max(law_worst, ms);
                ++laws_seen;
            }
        }
    }
    std::cout << "  9x9x3 window on the GPU, walking: law " << law_sum / std::max(laws_seen, 1) << " ms (worst " << law_worst << "), " << moves << " moves (" << layer_moves
              << " up or down) " << move_sum / std::max(moves, 1) << " ms (worst " << move_worst << "), worst tick on the main thread " << tick_worst << " ms" << std::endl;
}

} // namespace

int main()
{
    if (const std::string why = GpuSolver::unavailable(); !why.empty()) {
        std::cout << "voxel_gpu_smoke: SKIPPED, no GPU: " << why << std::endl;
        return 0;
    }
    check_every_op();
    check_chain_in_one_step();
    check_globals();
    check_contracts();
    check_whole_law();
    check_determinism();
    check_living_world();
    check_time_is_kept();
    check_zero_field_returns();
    report_speed();
    report_walk();
    return fieldengine::gate::report("voxel_gpu");
}

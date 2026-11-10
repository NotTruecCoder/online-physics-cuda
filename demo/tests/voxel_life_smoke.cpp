// GATE: CREATURES ARE LAW, AND THE LAW CAN CHANGE WHILE THE WORLD RUNS.
//
// Claims:
//   1. A grazer walks to turf and eats it: the turf goes down, its hunger goes down. With `graze` at 0
//      (control) the turf is untouched and it stays hungry.
//   2. A grazer runs from your scent: it ends further from you than it started, and afraid. With
//      `scent_fear` at 0 and no seek on scent (control, a law without the coupling) it does not.
//   3. Fire makes it afraid, and heat hurts it; hunger with nothing to eat starves it: it dies, and the
//      world says so.
//   4. The language refuses a creature that reads a lattice op, and a shift by a fraction of a cell;
//      shift() reads the neighbouring cell.
//   5. The law changes while the world runs: fields keep their values by name, a new field starts at its
//      init, a new term acts at once, a changed param takes effect, creatures keep their state. A law
//      that does not compile changes nothing and is reported by line.
//   6. The inspector's equations are the running law: every term that writes T is on the right of
//      dT/dt, printed as mathematics; a changed law prints differently.
//   7. The bits do not depend on the number of threads.
//   8. Fire walks through turf, makes smoke, stops at bare ground, and dies out. Turf that gives a third of
//      the heat (control) cannot keep a front alive: the soil drinks it, and the fire stays where it was lit.
//   9. A hunter follows a herd's musk, kills, and lives on it. Control: no prey, it only gets hungrier, and
//      a grazer with no hunter near lives.
//  10. Well-fed grazers breed; a starving one does not; the population stops at the world's limit.
//  11. `show` lines compile into what the picture draws, and bad ones are refused by line.
//
// Falsify by: dropping apply_flows from think (1 fails); negating the seek gradient (1, 2 fail);
// clearing fields in reload_law instead of copying them (5 fails); returning a fixed list from
// law_equations (6 fails); any cell's value depending on the thread that computed it (7 fails); `grass_heat`
// at 900 in life.law (8 fails); `emit claw` at 0 (9 fails); `breed` ignored (10 fails).

#include "world/Playground.hpp"
#include "tests/Gate.hpp"
#include "tests/Pours.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

std::vector<LawSource> g_laws;
std::string g_spells;

// A flat meadow of soil, with you parked in a far corner.
Playground meadow(const std::vector<LawSource>& laws = g_laws, int max_creatures = 64)
{
    std::vector<std::string> errors;
    Playground::Config config;
    config.max_creatures = max_creatures;
    config.shape = GridShape{.nx = 64, .ny = 32, .nz = 10, .cell = 0.5};
    std::optional<Playground> world = Playground::create(laws, g_spells, errors, config);
    for (const std::string& e : errors) {
        std::cerr << "  " << e << "\n";
    }
    check(world.has_value(), "the laws compile");
    if (!world) {
        std::exit(fieldengine::gate::report("voxel_life"));
    }
    FieldSet& f = world->solver().fields();
    const GridShape& g = f.shape();
    const int solid = f.find("solid");
    const int mat = f.find("material");
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x) {
                f.values(solid)[g.index(x, y, z)] = 1.0f;
                f.values(mat)[g.index(x, y, z)] = static_cast<float>(material::soil);
            }
        }
    }
    world->solver().refresh_derived();
    world->player().position = Vec3{1.0, 1.0, 1.45};
    world->player().controlled = true;
    world->target().position = Vec3{1.0, 14.0, 1.5};
    return std::move(*world);
}

Vec3 horizontal_of(Vec3 v) { return Vec3{v.x, v.y, 0.0}; }

// A patch of turf, x cells [x0, x1], the whole width.
void turf(Playground& world, int x0, int x1)
{
    FieldSet& f = world.solver().fields();
    const GridShape& g = f.shape();
    const int grass = f.find("grass");
    for (int y = 0; y < g.ny; ++y) {
        for (int x = x0; x <= x1; ++x) {
            f.values(grass)[g.index(x, y, 2)] = 1.0f;
        }
    }
    world.solver().refresh_derived();
}

double local(const Playground& world, const Creature& c, const std::string& name)
{
    const CreatureDef& def = world.solver().law().creatures[static_cast<std::size_t>(c.def)];
    const auto it = std::find(def.locals.begin(), def.locals.end(), name);
    return it == def.locals.end() ? -1.0 : c.locals[static_cast<std::size_t>(it - def.locals.begin())];
}

void run(Playground& world, int ticks)
{
    for (int i = 0; i < ticks; ++i) {
        world.step(Control{});
    }
}

// The turf a grazer can walk to; returns {turf eaten, hunger at the end, x at the end}.
struct Meal {
    double eaten = 0.0;
    double hunger = 0.0;
    double x = 0.0;
};
Meal graze_run(double graze)
{
    Playground world = meadow();
    world.solver().set_param("grass_grow", 0.0); // what changes the turf here is the grazer alone
    world.solver().set_param("graze", graze);
    turf(world, 30, 40); // 15..20 m
    check(world.spawn("grazer", Vec3{12.5, 8.0, 1.35}), "1: the law has grazers");
    const double before = world.total("grass");
    double reached = 0.0;
    double hunger = 0.0;
    for (int i = 0; i < 60 * 25; ++i) {
        world.step(Control{});
        if (!world.creatures().empty()) {
            reached = std::max(reached, world.creatures()[0].body.position.x);
            hunger = local(world, world.creatures()[0], "hunger");
        }
    }
    return Meal{before - world.total("grass"), hunger, reached};
}

} // namespace

int main()
{
    {
        std::string path;
        std::vector<std::string> errors;
        check(Playground::read_default_law(g_laws, g_spells, path, errors), "the law files are there");
    }

    // 1. Eating.
    {
        const Meal fed = graze_run(0.5);
        const Meal control = graze_run(0.0);
        std::cout << "  fed: ate " << fed.eaten << ", hunger " << fed.hunger << ", reached x " << fed.x << "\n";
        std::cout << "  control: ate " << control.eaten << ", hunger " << control.hunger << ", reached x " << control.x
                  << "\n";
        check(fed.x > 14.5, "1: the grazer walks to the turf it senses");
        check(fed.eaten > 0.5, "1: the turf goes down as it eats");
        check(fed.hunger < control.hunger - 0.1, "1: eating lowers its hunger");
        check(control.eaten < 1e-3, "1: control: with graze = 0 the turf is untouched");
    }

    // 2. Scent.
    {
        auto flee = [](bool coupled) {
            std::vector<LawSource> laws = g_laws;
            if (!coupled) {
                // The same creature with nothing tying it to your scent.
                for (LawSource& law : laws) {
                    const std::size_t at = law.text.find("  seek = ");
                    if (at != std::string::npos && law.text.find("creature grazer") != std::string::npos) {
                        const std::size_t end = law.text.find('\n', at);
                        law.text.replace(at, end - at, "  seek = (0.2 + hunger) * grass");
                    }
                }
            }
            Playground world = meadow(laws);
            world.solver().set_param("scent_fear", coupled ? 4.0 : 0.0);
            world.player().position = Vec3{16.0, 8.0, 1.45};
            world.spawn("grazer", Vec3{18.5, 8.0, 1.35});
            double peak = 0.0;
            for (int i = 0; i < 60 * 12; ++i) {
                world.step(Control{});
                peak = std::max(peak, local(world, world.creatures().at(0), "fear"));
            }
            const Creature& c = world.creatures().at(0);
            const Vec3 d = c.body.position - world.player().position;
            std::cout << "  " << (coupled ? "coupled" : "uncoupled") << ": distance " << length(horizontal_of(d))
                      << ", peak fear " << peak << "\n";
            return std::pair<double, double>{length(horizontal_of(d)), peak};
        };
        const auto [far, fear] = flee(true);
        const auto [near, calm] = flee(false);
        check(far > 5.0, "2: it runs from your scent");
        check(fear > 0.2, "2: and it is afraid");
        check(calm < 1e-6, "2: control: without the coupling, no fear");
        check(far > near + 1.0, "2: control: without the coupling it does not run as far");
    }

    // 3. Fire and hunger.
    {
        Playground world = meadow();
        world.spawn("grazer", Vec3{20.0, 8.0, 1.35});
        run(world, 30);
        const double hp = world.creatures().at(0).body.hp;
        const int fire = world.solver().field("mana_fire");
        check(fire >= 0, "3: the law has fire");
        for (const Pour& spell : world.pours()) {
            if (spell.name == "kindle") {
                world.pour(spell, world.creatures().at(0).body.position, Vec3{1.0, 0.0, 0.0});
            }
        }
        run(world, 40);
        check(!world.creatures().empty() && local(world, world.creatures()[0], "fear") > 0.2, "3: fire frightens it");
        check(!world.creatures().empty() && world.creatures()[0].body.hp < hp, "3: heat hurts it");

        Playground hungry = meadow();
        hungry.solver().set_param("hunger_rate", 2.0);
        hungry.solver().set_param("starve", 60.0);
        hungry.spawn("grazer", Vec3{20.0, 8.0, 1.35});
        run(hungry, 120);
        check(hungry.creatures().empty(), "3: with nothing to eat it starves");
        const bool told = std::any_of(hungry.log().begin(), hungry.log().end(),
            [](const std::string& line) { return line.find("died") != std::string::npos; });
        check(told, "3: and the world says so");
    }

    // 4. The language.
    {
        const std::string base = "field a init 0\nfield b init 0\nderive occ = a\nderive sdf = a\nfield T init 1\n";
        CompileResult bad = compile_law({LawSource{.name = "x", .text = base + "creature c\n  seek = lap(a)\n"}});
        check(!bad.law.has_value(), "4: a creature cannot read a lattice op");
        CompileResult fraction = compile_law({LawSource{.name = "x", .text = base + "derive s = shift(a, 0.5, 0, 0)\n"}});
        check(!fraction.law.has_value(), "4: shift moves whole cells");
        CompileResult state = compile_law({LawSource{.name = "x", .text = base + "creature c\n  d nope = 1\n"}});
        check(!state.law.has_value() && !state.errors.empty() && state.errors[0].find("x:7") == 0,
            "4: a bad creature line is reported by file and line");

        CompileResult good = compile_law({LawSource{.name = "x", .text = "field a init 0\nderive below = shift(a, 0, 0, -1)\n"}});
        check(good.law.has_value(), "4: shift compiles");
        if (good.law) {
            Solver solver(*good.law, GridShape{.nx = 2, .ny = 2, .nz = 3, .cell = 1.0});
            FieldSet& f = solver.fields();
            f.values(0)[f.shape().index(1, 1, 1)] = 5.0f;
            solver.refresh_derived();
            const int below = solver.field("below");
            check(f.values(below)[f.shape().index(1, 1, 2)] == 5.0f, "4: shift(a, 0, 0, -1) reads the cell below");
            check(f.values(below)[f.shape().index(1, 1, 1)] == 0.0f, "4: and not the cell itself");
        }
    }

    // 5. The law, changed in flight.
    {
        Playground world = meadow();
        turf(world, 30, 40);
        world.spawn("grazer", Vec3{12.5, 8.0, 1.35});
        for (const Pour& spell : world.pours()) {
            if (spell.name == "kindle") {
                world.pour(spell, Vec3{25.0, 8.0, 1.5}, Vec3{1.0, 0.0, 0.0});
            }
        }
        // Hungrier than it was born: what the reload must keep is not what a fresh creature would have.
        world.solver().set_param("hunger_rate", 0.12);
        run(world, 120);
        const double soil = world.total("solid");
        const double heat = world.total("T");
        const double hunger = local(world, world.creatures().at(0), "hunger");
        check(hunger > 0.4, "5: (it has got hungrier than a newborn: " + std::to_string(hunger) + ")");
        const double hp = world.creatures().at(0).body.hp;
        const std::size_t terms = world.solver().law().terms.size();

        // A law that does not compile changes nothing.
        std::vector<LawSource> broken = world.law_sources();
        broken[0].text += "\nterm oops\n  d T = nonsense(1)\n";
        std::vector<std::string> errors;
        check(!world.reload_law(broken, world.pours_text(), errors), "5: a broken law is refused");
        check(!errors.empty() && errors[0].find(":") != std::string::npos && errors[0].find("nonsense") != std::string::npos,
            "5: and the error names the line and the fault");
        check(world.solver().law().terms.size() == terms, "5: the running law is untouched");
        run(world, 3);

        // A law with a new field, a new term, a changed param, a new creature state.
        std::vector<LawSource> next = world.law_sources();
        for (LawSource& law : next) {
            const std::size_t at = law.text.find("param hunger_rate = 0.012");
            if (at != std::string::npos) {
                law.text.replace(at, 25, "param hunger_rate = 0.5  ");
                const std::size_t state = law.text.find("  state fear ");
                law.text.insert(state, "  state joy init 0.25 range 0 1\n");
            }
        }
        next.push_back(LawSource{.name = "mod.law", .text = "field glow init 7 range 0 1e6\nterm glow_grows\n  d glow = 2\n"});
        const double soil_now = world.total("solid");
        errors.clear();
        const bool applied = world.reload_law(next, world.pours_text(), errors);
        for (const std::string& e : errors) {
            std::cerr << "  " << e << "\n";
        }
        check(applied, "5: a good law is applied while the world runs");
        check(world.total("solid") == soil_now, "5: every field keeps its values (stone)");
        check(std::abs(soil_now - soil) < soil * 0.01, "5: nothing happened to the stone meanwhile");
        check(std::abs(world.total("T") - heat) < heat * 0.05, "5: every field keeps its values (heat)");
        const int glow = world.solver().field("glow");
        check(glow >= 0 && world.fields().values(glow)[0] == 7.0f, "5: a new field starts at its init");
        check(world.solver().param("hunger_rate").value_or(0.0) == 0.5, "5: a changed param has the new text's value");
        check(!world.creatures().empty(), "5: the creature lives through it");
        const Creature& c = world.creatures().at(0);
        check(std::abs(local(world, c, "hunger") - hunger) < 0.05, "5: and keeps the state the new law still has");
        check(local(world, c, "joy") == 0.25, "5: a new state starts at its init");
        check(std::abs(c.body.hp - hp) < 1.0, "5: and its body");
        run(world, 60);
        check(world.fields().values(glow)[0] > 7.5f, "5: the new term acts at once");
        check(local(world, world.creatures().at(0), "hunger") > hunger + 0.2, "5: the changed param acts at once");
        check(world.violations().empty(), "5: no contract broke across the change");
    }

    // 6. The equations are the running law.
    {
        Playground world = meadow();
        std::vector<LawEquation> eqs = law_equations(world.solver().law());
        auto find = [&](const std::string& lhs) -> const LawEquation* {
            for (const LawEquation& e : eqs) {
                if (e.lhs == lhs) {
                    return &e;
                }
            }
            return nullptr;
        };
        auto joined = [](const LawEquation& e) {
            std::string all;
            for (const std::string& p : e.parts) {
                all += p + " ; ";
            }
            return all;
        };
        const LawEquation* dT = find("∂T/∂t");
        check(dT != nullptr, "6: the temperature has an equation");
        if (dT != nullptr) {
            const std::string all = joined(*dT);
            std::cout << "  dT/dt = " << all.substr(0, 60) << "...\n";
            check(all.find("∇·(k∇T)") != std::string::npos, "6: conduction prints as div(k grad T)");
            check(all.find("H(T − ignite)") != std::string::npos, "6: a step prints as a Heaviside");
            // Written by hand: every term the law files say write T.
            for (const char* term : {"conduction", "combustion", "radiation", "melting", "wood_burns", "leaves_burn",
                     "freezing", "thawing", "boiling", "fire_burns", "ice_chills", "grass_burns"}) {
                check(std::find(dT->source.begin(), dT->source.end(), term) != dT->source.end(),
                    std::string("6: term ") + term + " is on the right of dT/dt");
            }
        }
        const LawEquation* occ = find("occ");
        check(occ != nullptr && joined(*occ).find("wood") != std::string::npos, "6: derived fields print too");
        const LawEquation* hunger = find("grazer: dhunger/dt");
        check(hunger != nullptr && joined(*hunger).find("eaten") != std::string::npos, "6: a creature's equations print");

        std::vector<LawSource> next = world.law_sources();
        next.push_back(LawSource{.name = "mod.law", .text = "term extra_cooling\n  d T = -3 * T\n"});
        std::vector<std::string> errors;
        check(world.reload_law(next, world.pours_text(), errors), "6: the mod compiles");
        eqs = law_equations(world.solver().law());
        const LawEquation* changed = find("∂T/∂t");
        check(changed != nullptr && std::find(changed->source.begin(), changed->source.end(), "extra_cooling") !=
                changed->source.end(),
            "6: a changed law prints differently");
    }

    // 8. A grass fire.
    {
        auto burn = [](double conduct, double& far_side, double& beyond_gap, double& smoke_peak, double& hottest_end) {
            Playground world = meadow();
            world.solver().set_param("grass_grow", 0.0);
            if (conduct > 0.0) {
                world.solver().set_param("grass_heat", conduct); // else: what the law file says
            }
            turf(world, 6, 40);   // 3..20 m
            turf(world, 49, 63);  // past a bare gap of 4 m
            const FieldSet& f = world.fields();
            const GridShape& g = f.shape();
            const int grass = f.find("grass");
            auto strip = [&](int x0, int x1) {
                double sum = 0.0;
                for (int y = 0; y < g.ny; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        sum += f.values(grass)[g.index(x, y, 2)];
                    }
                }
                return sum / (g.ny * (x1 - x0 + 1));
            };
            for (const Pour& spell : world.pours()) {
                if (spell.name == "kindle") {
                    world.pour(spell, Vec3{10.0, 8.0, 1.25}, Vec3{1.0, 0.0, 0.0});
                }
            }
            smoke_peak = 0.0;
            for (int i = 0; i < 60 * 30; ++i) {
                world.step(Control{});
                if (i % 30 == 0) {
                    smoke_peak = std::max(smoke_peak, world.total("smoke"));
                }
            }
            far_side = strip(34, 40);    // 7..10 m from where it was lit
            beyond_gap = strip(49, 63);
            hottest_end = 0.0;
            for (const float v : f.values(f.find("T"))) {
                hottest_end = std::max(hottest_end, static_cast<double>(v));
            }
            check(world.violations().empty(), "8: no contract broke in the fire");
        };
        double far_side = 0.0;
        double beyond = 0.0;
        double smoke = 0.0;
        double end_heat = 0.0;
        burn(-1.0, far_side, beyond, smoke, end_heat);
        double control_far = 0.0;
        double control_beyond = 0.0;
        double control_smoke = 0.0;
        double control_heat = 0.0;
        burn(900.0, control_far, control_beyond, control_smoke, control_heat);
        std::cout << "  grass fire: turf left 7-10 m away " << far_side << ", past the gap " << beyond << ", smoke peak "
                  << smoke << ", hottest at the end " << end_heat << "; control turf left " << control_far << "\n";
        check(far_side < 0.1, "8: fire walks through turf");
        check(beyond > 0.9, "8: bare ground stops it");
        check(smoke > 1.0, "8: it makes smoke");
        check(end_heat < 500.0, "8: and it dies out");
        check(control_far > 0.9, "8: control: with a third of the heat it stays where it was lit");
    }

    // 9. A hunter.
    {
        auto hunt = [](bool prey, bool hunter, double& grazer_hp, double& hunter_hunger) {
            Playground world = meadow();
            turf(world, 0, 63);
            world.solver().set_param("litter", 0.0); // one grazer, so its health is its own
            if (prey) {
                world.spawn("grazer", Vec3{20.0, 8.0, 1.35});
            }
            if (hunter) {
                check(world.spawn("hunter", Vec3{15.0, 8.0, 1.45}), "9: the law has hunters");
            }
            run(world, 60 * 45);
            grazer_hp = 0.0;
            hunter_hunger = -1.0;
            for (const Creature& c : world.creatures()) {
                if (c.body.name == "grazer") {
                    grazer_hp = std::max(grazer_hp, c.body.hp);
                } else if (c.body.name == "hunter") {
                    hunter_hunger = local(world, c, "hunger");
                }
            }
        };
        double hp = 0.0;
        double fed = 0.0;
        hunt(true, true, hp, fed);
        double control_hp = 0.0;
        double unused = 0.0;
        hunt(true, false, control_hp, unused);
        double unused_hp = 0.0;
        double starving = 0.0;
        hunt(false, true, unused_hp, starving);
        std::cout << "  hunter: grazer hp " << hp << " (no hunter " << control_hp << "), hunter hunger " << fed
                  << " (no prey " << starving << ")\n";
        check(hp < 1.0, "9: the hunter kills the grazer");
        check(control_hp > 35.0, "9: control: with no hunter it lives");
        check(fed < starving - 0.02, "9: and lives on it: fed, against a hunter with no prey");
    }

    // 10. Breeding.
    {
        auto herd = [](double hunger_rate, int limit, double litter) {
            Playground world = meadow(g_laws, limit);
            turf(world, 0, 63);
            world.solver().set_param("hunger_rate", hunger_rate);
            world.solver().set_param("litter", litter);
            world.solver().set_param("scent_fear", 0.0);
            world.spawn("grazer", Vec3{20.0, 8.0, 1.35});
            run(world, 60 * 45);
            return world.creatures().size();
        };
        const std::size_t fed = herd(0.012, 64, 0.2);
        const std::size_t starved = herd(0.6, 64, 0.2);
        const std::size_t capped = herd(0.012, 3, 1.0);
        std::cout << "  grazers after 45 s: fed " << fed << ", starving " << starved << ", capped " << capped << "\n";
        check(fed >= 3, "10: well fed, they breed");
        check(starved <= 1, "10: starving, they do not");
        check(capped == 3, "10: and never past the world's limit");
    }

    // 11. show.
    {
        std::vector<std::string> errors;
        std::string spells;
        Playground world = meadow();
        auto shows = [&](const char* name) {
            const int id = world.law().field_id(name);
            for (const FieldVisual& v : world.law().visuals) {
                if (v.field == id) {
                    return true;
                }
            }
            return false;
        };
        for (const char* name : {"T", "smoke", "mana_fire", "mana_ice", "mana_earth", "mana_force", "mana_water", "mana_air"}) {
            check(shows(name), std::string("11: the law shows ") + name);
        }
        const std::string base = "field a init 0\n";
        CompileResult unknown = compile_law({LawSource{.name = "x", .text = base + "show nope colour 1 1 1 above 0 full 1\n"}});
        check(!unknown.law.has_value() && unknown.errors.at(0).find("x:2") == 0, "11: showing an unknown field is refused by line");
        CompileResult upside = compile_law({LawSource{.name = "x", .text = base + "show a colour 1 1 1 above 2 full 1\n"}});
        check(!upside.law.has_value(), "11: and so is full below above");
    }

    // 7. Threads.
    {
        auto digest_with = [](int threads) {
            omp_set_num_threads(threads);
            std::vector<std::string> errors;
            std::optional<Playground> world = Playground::load_default(errors, true);
            check(world.has_value(), "7: the open world loads");
            for (int i = 0; i < 120; ++i) {
                Control control{.move_x = 1.0, .move_y = 0.3};
                if (i == 20) {
                    fieldengine::gate::pour_toward(*world, 0, world->player().position + Vec3{4.0, 0.0, 0.0}); // fire, so the busy terms run
                }
                if (i == 40) {
                    fieldengine::gate::pour_toward(*world, 4, world->player().position + Vec3{-3.0, 2.0, 0.0}); // water
                }
                world->step(control);
            }
            return world->digest();
        };
        const std::uint64_t one = digest_with(1);
        const std::uint64_t four = digest_with(4);
        const std::uint64_t seven = digest_with(7);
        check(one == four && four == seven, "7: 1, 4 and 7 threads give the same world, bit for bit");
    }
    return fieldengine::gate::report("voxel_life");
}

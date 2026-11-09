// Gate: A WORLD MODEL LEARNS AND PLANS. A forager's model (life.law's `creature forager`) is fed what happens to it in a
// made-up world where going east fills it (hunger falls with ax) and grass lies east: its prediction error falls by
// learning, and once it knows, MPPI sends it east; the same plan comes from the same tick (the noise is a hash), and the
// cost it plans against is the law's plan line, the same on the card and the host.
//
// Falsified by (each run once): Adam's sign flipped (the error grows); the cost's sign flipped in MPPI's weights (it
// goes west); the noise keyed by the clock instead of the tick (the two plans differ).
#include "tests/Gate.hpp"

#include "field/Law.hpp"
#include "field/mind/WorldModel.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>

using namespace fieldengine::field;
using fieldengine::gate::check;

int main()
{
    std::vector<LawSource> sources;
    for (const char* p : {"assets/law/world.law", "assets/law/magic.law", "assets/law/life.law"}) {
        if (std::optional<LawSource> s = read_law_file(p)) {
            sources.push_back(*s);
        }
    }
    CompileResult compiled = compile_law(sources);
    check(compiled.law.has_value(), "the laws with a forager compile");
    if (!compiled.law) {
        return fieldengine::gate::report("mind");
    }
    const Law& law = *compiled.law;
    const int forager = law.creature_id("forager");
    check(forager >= 0 && law.creatures[static_cast<std::size_t>(forager)].plan != nullptr, "a forager plans");
    WorldModel mind(law, forager);
    const int F = mind.features();
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(forager)];
    const auto local = [&](const char* name) {
        return static_cast<int>(std::find(def.locals.begin(), def.locals.end(), name) - def.locals.begin());
    };
    const int hunger = local("hunger");
    std::cout << "  " << F << " features, on " << mind.backend() << (mind.why_cpu().empty() ? "" : " (" + mind.why_cpu() + ")") << "\n";

    // The made-up world: going east fills it (hunger falls by 0.4 ax a second, rises by 0.05 otherwise); nothing else
    // moves. 64 foragers live in it; what each does is what it planned.
    std::vector<float> s(static_cast<std::size_t>(F), 0.0f);
    const double dt = def.plan_dt;
    std::vector<float> h(65, 0.0f);
    for (int id = 1; id <= 64; ++id) {
        h[static_cast<std::size_t>(id)] = 0.2f + 0.6f * std::fmod(0.37f * static_cast<float>(id), 1.0f);
    }
    double first_loss = -1.0;
    double east = 0.0;
    int counted = 0;
    std::vector<float> actions;
    for (int round = 0; round < 900; ++round) {
        std::vector<float> obs;
        std::vector<std::uint32_t> ids;
        for (std::uint32_t id = 1; id <= 64; ++id) {
            std::vector<float> o = s;
            o[static_cast<std::size_t>(hunger)] = h[id];
            mind.remember(id, o.data(), dt);
            obs.insert(obs.end(), o.begin(), o.end());
            ids.push_back(id);
        }
        mind.plan(obs, ids, static_cast<std::uint64_t>(round), actions);
        for (std::uint32_t id = 1; id <= 64; ++id) {
            const float ax = actions[2 * (id - 1)];
            float& hv = h[id];
            hv = std::clamp(hv + static_cast<float>(dt) * (-0.4f * ax + 0.05f), 0.0f, 1.0f);
            if (hv <= 0.02f || hv >= 0.98f) {
                hv = 0.2f + 0.6f * std::fmod(0.37f * static_cast<float>(id) + 0.07f * static_cast<float>(round), 1.0f);
                mind.forget(id); // a new life: its last observation is no transition
            }
            if (round >= 850) {
                east += ax;
                ++counted;
            }
        }
        if (round % 4 == 3) {
            mind.learn(6);
            if (first_loss < 0.0 && mind.loss() > 0.0) {
                first_loss = mind.loss();
            }
        }
    }
    east /= std::max(counted, 1);
    std::cout << "  prediction loss: " << first_loss << " at first, " << mind.loss() << " after " << mind.updates() << " updates; "
              << mind.transitions() << " transitions; plan " << mind.plan_ms() << " ms, learn " << mind.learn_ms() << " ms\n";
    std::cout << "  once it has learned, it goes east at " << east << " of its speed on average\n";
    check(mind.transitions() > 1000, "it remembers what happened");
    check(first_loss > 0.0 && mind.loss() < 0.5 * first_loss, "learning online lowers its prediction error");
    check(east > 0.3, "it plans its way to what lowers its cost (east fills it)");

    // The cost it plans against is the law's line, on the card as on the host.
    std::vector<float> probe = s;
    probe[static_cast<std::size_t>(hunger)] = 0.5f;
    const float c = mind.cost(probe.data());
    check(std::abs(c - 0.5f) < 1e-6f, "the plan cost is the law's plan line (hunger + 2 fear - 0.5 eaten)");
    check(mind.cost_code().find("wm_cost") != std::string::npos, "the plan line is compiled into the planner's kernel");

    // The same tick plans the same way.
    std::vector<float> a1;
    std::vector<float> a2;
    mind.forget(9000);
    mind.plan(probe, {9000}, 77, a1);
    mind.forget(9000);
    mind.plan(probe, {9000}, 77, a2);
    check(a1 == a2, "the same world plans the same way (the noise is a hash of creature, tick, future, step)");
    return fieldengine::gate::report("mind");
}

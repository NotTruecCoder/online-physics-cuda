// THE PEOPLE OF THE WORLD.
//   1. Every living place has people, and their work differs with the land, with no list of jobs.
//   2. The same person comes back the same.
//   3. A rumour goes down the roads, at a finite speed, and what matters goes further than what does not.
//   4. Walking into a town, bodies come out; they go about their day; they talk among themselves.
//   5. Struck by someone, a person fights back or runs, and a watchman comes for whoever struck.
//   6. Killed, it stays dead: written down, told, the town one fewer.
// Falsified by: a body that ignores its anger; opinions read from the town and not the person; a wave with no damping.
#include "field/Playground.hpp"
#include "field/Terrain.hpp"
#include "tests/Gate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

Vec3 horizontal(Vec3 v)
{
    return Vec3{v.x, v.y, 0.0};
}

std::optional<Playground> make_world()
{
    std::vector<std::string> errors;
    std::vector<LawSource> laws;
    std::string spells;
    std::string path;
    if (!Playground::read_default_law(laws, spells, path, errors)) {
        return std::nullopt;
    }
    Playground::Config config;
    config.open_world = true;
    config.window_chunks = 5;
    config.window_layers = 3;
    config.far_life = false;
    config.people = true;
    config.max_creatures = 110;
    std::optional<Playground> w = Playground::create(laws, spells, errors, config);
    for (const std::string& e : errors) {
        std::cout << e << "\n";
    }
    return w;
}

// You, at a town's market.
void walk_into(Playground& w, const Settlement& s)
{
    const TerrainConfig& t = w.config().terrain;
    w.center_window_on(s.x / t.chunk, s.y / t.chunk, (s.ground + 4) / t.chunk);
    w.player().position = Vec3{(s.x + 0.5) * t.cell, (s.y + 0.5) * t.cell, (s.ground + 3) * t.cell} - w.origin();
    w.player().velocity = Vec3{};
}

void tick_for(Playground& w, double seconds, Control control = {})
{
    for (int i = 0; i < static_cast<int>(seconds * 60.0); ++i) {
        w.player().hp = w.player().hp_max;
        w.step(control);
    }
}

const Settlement* largest(const WorldMap& map, const Society& society, int nth = 0)
{
    std::vector<const Settlement*> all;
    for (const Town& t : society.towns()) {
        const Settlement& s = map.settlements()[static_cast<std::size_t>(t.settlement)];
        if (s.tier >= 1 && !s.buildings.empty()) {
            all.push_back(&s);
        }
    }
    std::sort(all.begin(), all.end(), [](const Settlement* a, const Settlement* b) { return a->people > b->people; });
    return all.empty() ? nullptr : all[static_cast<std::size_t>(nth) % all.size()];
}

int local(const Playground& w, const Creature& c, const char* name)
{
    const CreatureDef& def = w.law().creatures[static_cast<std::size_t>(c.def)];
    const auto it = std::find(def.locals.begin(), def.locals.end(), name);
    return it == def.locals.end() ? -1 : static_cast<int>(it - def.locals.begin());
}

} // namespace

// Nobody stays in a wall, and nobody goes on walking into one.
void not_stuck(Playground& w, const Settlement& s)
{
    walk_into(w, s);
    tick_for(w, 10.0);
    std::map<std::uint32_t, Vec3> was;
    std::map<std::uint32_t, int> still;
    const int occ = w.law().field_id("occ");
    int longest = 0;
    int inside_ticks = 0;
    int ticks = 0;
    for (int look = 0; look < 12; ++look) {
        tick_for(w, 5.0);
        for (const Creature& c : w.creatures()) {
            if (c.town < 0) {
                continue;
            }
            const FieldSet& f = w.fields();
            ++ticks;
            inside_ticks += f.sample(occ, c.body.position.x, c.body.position.y, c.body.position.z) > 0.5f ? 1 : 0;
            const auto it = was.find(c.id);
            const bool wants = c.doing != Creature::Doing::Talking && c.doing != Creature::Doing::Looking && length(horizontal(c.goal - c.body.position)) > 2.5;
            if (it != was.end() && wants && length(horizontal(c.body.position - it->second)) < 0.5) {
                longest = std::max(longest, ++still[c.id]);
            } else {
                still[c.id] = 0;
            }
            was[c.id] = c.body.position;
        }
    }
    std::cout << "  of " << ticks << " looks at someone " << inside_ticks << " found them inside something; the longest anyone wanted to go somewhere and did not: " << longest * 5 << " s\n";
    check(inside_ticks * 50 <= ticks, "stuck: hardly anyone is in a wall, and not for long");
    check(longest * 5 <= 20, "stuck: nobody walks into the same wall for long");
}

int main()
{
    std::optional<Playground> made = make_world();
    if (!made) {
        check(false, "the world makes");
        return fieldengine::gate::report("voxel_people");
    }
    Playground& w = *made;
    const auto started = std::chrono::steady_clock::now();
    w.build_open_world();
    {
        std::vector<std::string> errors;
        w.set_melee(load_moves(errors));
    }
    std::cout << "  world and towns built in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
    Society* society = w.society();
    check(society != nullptr, "there are people");
    if (society == nullptr) {
        return fieldengine::gate::report("voxel_people");
    }
    const WorldMap& map = *w.map();
    int living = 0;
    for (const Settlement& s : map.settlements()) {
        living += s.fell < 0 && s.people > 0 ? 1 : 0;
    }
    std::cout << "  " << society->towns().size() << " towns of " << living << " living places, " << society->facts().size() << " facts\n";
    check(static_cast<int>(society->towns().size()) == living, "1: every living place is a town");

    // 1. Work.
    {
        std::map<std::string, int> top_jobs;
        double bad = 0.0;
        double low_food = 0.0;
        int shown = 0;
        for (const Town& t : society->towns()) {
            int best = 0;
            double sum = 0.0;
            for (int g = 0; g < static_cast<int>(society->goods().size()); ++g) {
                const double q = society->share(t, g);
                sum += q;
                bad += std::isfinite(q) && q >= 0.0 ? 0.0 : 1.0;
                best = q > society->share(t, best) ? g : best;
            }
            bad += std::abs(sum - 1.0) < 1e-6 ? 0.0 : 1.0;
            low_food += society->state(t, "food") < 0.5 ? 1.0 : 0.0;
            ++top_jobs[society->goods()[static_cast<std::size_t>(best)]];
            if (shown++ < 6) {
                std::cout << "  " << t.name << ": mostly " << society->goods()[static_cast<std::size_t>(best)] << " " << society->share(t, best) << ", food " << society->state(t, "food")
                          << ", unrest " << society->state(t, "unrest") << ", people " << society->people_in(t) << ", camps " << society->camps_of(t).size() << "\n";
            }
        }
        std::cout << "  what most people do, by town:";
        for (const auto& [job, n] : top_jobs) {
            std::cout << " " << job << " " << n;
        }
        std::cout << "; hungry towns " << low_food << " of " << society->towns().size() << "\n";
        check(bad == 0.0, "1: shares of the work are shares");
        check(top_jobs.size() >= 3, "1: what people do differs by place");
        check(low_food < 0.5 * society->towns().size(), "1: most places can feed themselves");
    }

    // 2. The same person.
    {
        const Person a = society->person(0, 7);
        for (int i = 0; i < 20 * 60 * 20; ++i) {
            society->tick(Playground::kTick / 20.0 * 20.0);
        }
        const Person b = society->person(0, 7);
        std::cout << "  " << a.name << " the " << a.job << " is " << b.name << " the " << b.job << "\n";
        check(a.name == b.name && a.bold == b.bold && a.house == b.house, "2: a person comes back the same");
        const Person c = society->person(0, 8);
        check(c.name != a.name, "2: and someone else is someone else");
        check(!a.name.empty() && a.name.size() > 3, "2: with a name");
    }

    // 3. A rumour.
    {
        // Far from everything: the middle of the longest chain the roads give.
        const int from = 0;
        std::vector<int> hops(society->towns().size(), -1);
        std::vector<int> queue{from};
        hops[from] = 0;
        for (std::size_t q = 0; q < queue.size(); ++q) {
            for (const auto& [to, good] : society->towns()[static_cast<std::size_t>(queue[q])].roads) {
                (void)good;
                if (hops[static_cast<std::size_t>(to)] < 0) {
                    hops[static_cast<std::size_t>(to)] = hops[static_cast<std::size_t>(queue[q])] + 1;
                    queue.push_back(to);
                }
            }
        }
        const Person who = society->person(from, 1);
        const int big = society->deed("killed", from, who.name, 1.0, -1.0, {}, 0.5, who.pid + 1);
        const Person other = society->person(from, 2);
        const int small = society->deed("threat", from, other.name, 0.15, -0.3, {}, 0.5, other.pid + 1);
        const auto reach_of = [&](int fact, double& far_hop) {
            int towns_with = 0;
            far_hop = 0.0;
            for (std::size_t t = 0; t < society->towns().size(); ++t) {
                for (const FactRow& row : society->towns()[t].rows) {
                    if (row.fact == fact && row.now > 0.1f) {
                        ++towns_with;
                        far_hop = std::max(far_hop, static_cast<double>(hops[t]));
                    }
                }
            }
            return towns_with;
        };
        double last_far = 0.0;
        bool slow = true;
        for (int minute = 1; minute <= 8; ++minute) {
            for (int i = 0; i < 60 * 60; ++i) {
                society->tick(Playground::kTick);
            }
            double far_big = 0.0, far_small = 0.0;
            const int a = reach_of(big, far_big);
            const int b = reach_of(small, far_small);
            std::cout << "  after " << minute << " min of play: the killing in " << a << " towns, " << far_big << " roads off; the threat in " << b << ", " << far_small << " roads off\n";
            slow = slow && far_big <= last_far + 15.0;
            last_far = far_big;
        }
        double far_big = 0.0, far_small = 0.0;
        reach_of(big, far_big);
        reach_of(small, far_small);
        check(far_big >= 3.0, "3: a killing is heard of a few roads away");
        check(slow, "3: and word takes its time");
        check(far_small < far_big, "3: what matters less goes less far");
        // Who has heard stays who has heard.
        int said = 0;
        int asked = 0;
        for (int i = 0; i < 300; ++i) {
            const Person p = society->person(from, i);
            const bool now = society->knows(p.pid, from, big);
            const bool again = society->knows(p.pid, from, big);
            said += now == again ? 1 : 0;
            ++asked;
        }
        check(said == asked, "3: asking twice gets the same answer");
    }

    // 4. Into a town.
    const Settlement* place = largest(map, *society);
    check(place != nullptr, "there is a town to walk into");
    if (place == nullptr) {
        return fieldengine::gate::report("voxel_people");
    }
    const int town = society->town_at(*place);
    std::cout << "  " << place->name << ", " << place->people << " people\n";
    walk_into(w, *place);
    tick_for(w, 4.0);
    const auto persons = [&]() {
        std::vector<const Creature*> out;
        for (const Creature& c : w.creatures()) {
            if (c.town >= 0) {
                out.push_back(&c);
            }
        }
        return out;
    };
    const std::size_t bodies = persons().size();
    std::cout << "  bodies about: " << bodies << ", creatures " << w.creatures().size() << ", window " << w.window_cx() << " " << w.window_cy() << " " << w.window_cz() << "; town at " << place->x / w.config().terrain.chunk << " " << place->y / w.config().terrain.chunk << " " << place->ground / w.config().terrain.chunk << "; tier " << place->tier << ", buildings " << place->buildings.size() << ", wanted " << society->towns()[static_cast<std::size_t>(town)].wanted << "\n";
    for (const std::string& line : w.log()) {
        std::cout << "    " << line << "\n";
    }
    check(bodies >= 4, "4: walking into a town, there are people about");
    if (bodies == 0) {
        return fieldengine::gate::report("voxel_people");
    }
    {
        // They go about: they walk off from where they came out.
        std::map<std::uint32_t, Vec3> were;
        for (const Creature* c : persons()) {
            were[c->id] = c->body.position;
        }
        tick_for(w, 12.0);
        double walked = 0.0;
        int count = 0;
        std::map<std::string, int> doing;
        for (const Creature* c : persons()) {
            ++doing[w.doing_words(*c)];
            const auto it = were.find(c->id);
            if (it != were.end()) {
                walked += length(horizontal(c->body.position - it->second));
                ++count;
            }
        }
        std::cout << "  walked " << walked / std::max(count, 1) << " m on average in 12 s;";
        for (const auto& [what, n] : doing) {
            std::cout << " " << n << " " << what << ",";
        }
        std::cout << "\n";
        check(walked / std::max(count, 1) > 1.5, "4: they go about their day");
        for (const Creature* c : persons()) {
            check(!c->hostile, "4: nobody has anything against you yet");
        }
    }
    // They talk among themselves: what is said over their heads.
    {
        bool said = !w.barks().empty();
        for (int i = 0; i < 60 * 20 && !said; ++i) {
            w.step(Control{});
            said = !w.barks().empty();
        }
        check(said, "4: they talk among themselves");
    }

    // 5. Someone strikes someone. (Nobody here is a player: the blows are a body's like any other.)
    const auto live = [&](std::uint32_t id) -> Creature* {
        for (Creature& c : const_cast<std::vector<Creature>&>(w.creatures())) {
            if (c.id == id) {
                return &c;
            }
        }
        return nullptr;
    };
    const std::size_t facts_before = society->facts().size();
    const Creature* peasant = nullptr;
    const Creature* brute = nullptr;
    for (const Creature* c : persons()) {
        const CreatureDef& def = w.law().creatures[static_cast<std::size_t>(c->def)];
        if (def.name == "villager" && peasant == nullptr) {
            peasant = c;
        } else if (def.name == "villager" && brute == nullptr) {
            brute = c;
        }
    }
    check(peasant != nullptr && brute != nullptr, "5: two villagers, one to strike the other");
    if (peasant == nullptr || brute == nullptr) {
        return fieldengine::gate::report("voxel_people");
    }
    const std::uint32_t victim = peasant->id;
    const std::uint32_t attacker = brute->id;
    const std::uint64_t victim_pid = peasant->who.pid;
    const std::vector<int> sword = w.melee().known("sword", 99);
    check(!sword.empty(), "5: the book has blows for a sword");
    int blows = 0;
    bool answered = false;
    bool watchman_came = false;
    bool blocked = false;
    for (int t = 0; t < 60 * 40 && w.creature_with(victim) != nullptr && !sword.empty(); ++t) {
        Creature* a = live(attacker);
        Creature* v = live(victim);
        if (a == nullptr || v == nullptr) {
            break;
        }
        a->foe = victim;
        a->body.hp = a->body.hp_max;
        const Vec3 off = horizontal(v->body.position - a->body.position);
        if (length(off) > 2.0) {
            a->body.position = v->body.position - normalize(off) * 1.6;
        }
        if (t % 40 == 0 && a->body.swing.move < 0) {
            w.swing(a->body, sword[0], v->body.position - a->body.position);
            ++blows;
        }
        w.step(Control{});
        for (const Creature* p : persons()) {
            const CreatureDef& def = w.law().creatures[static_cast<std::size_t>(p->def)];
            answered = answered || (p->id == victim && (p->foe == attacker || p->doing == Creature::Doing::Fleeing || p->flee_left > 0.0));
            watchman_came = watchman_came || (def.name == "watchman" && p->foe == attacker);
            blocked = blocked || (p->id == victim && p->body.guard >= 0.0);
        }
        w.clear_events();
    }
    std::cout << "  " << blows << " blows; answered " << answered << ", a watchman came " << watchman_came << ", the struck guarded " << blocked << "\n";
    check(answered, "5: struck, it fights back or runs");
    check(watchman_came, "5: the watch comes for whoever struck");

    // 6. And the dead stay dead.
    check(w.creature_with(victim) == nullptr, "6: enough blows kill someone");
    tick_for(w, 3.0);
    check(society->dead(victim_pid), "6: the dead are written down");
    check(society->facts().size() > facts_before, "6: and it is known");
    for (const Creature* p : persons()) {
        check(p->id != victim, "6: it does not come back");
    }
    if (const Settlement* quiet = largest(*w.map(), *w.society(), 3)) {
        not_stuck(w, *quiet);
    }
    check(w.violations().empty(), "the law held");
    return fieldengine::gate::report("voxel_people");
}

#include "field/Society.hpp"

#include "field/Terrain.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

namespace {

std::uint64_t mix(std::uint64_t a, std::uint64_t b)
{
    std::uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x1234567ull) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return x;
}

double hash01(std::uint64_t a, std::uint64_t b)
{
    return static_cast<double>(mix(a, b) >> 11) / 9007199254740992.0;
}

std::uint64_t text_hash(const std::string& text)
{
    std::uint64_t h = 1469598103934665603ull;
    for (const char c : text) {
        h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
    return h;
}

void replace_all(std::string& text, const std::string& what, const std::string& with)
{
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + with.size())) {
        text.replace(at, what.size(), with);
    }
}

std::uint64_t pid_of(int town, int index)
{
    return (static_cast<std::uint64_t>(town + 1) << 24) | static_cast<std::uint64_t>(index);
}

} // namespace

std::string Society::read_words()
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/world/people.world");
        if (in) {
            std::ostringstream all;
            all << in.rdbuf();
            return all.str();
        }
    }
    return "";
}

void Society::set_words(const std::string& text)
{
    words_.clear();
    lists_.clear();
    syllables_.clear();
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream parts(line);
        std::string topic;
        parts >> topic;
        std::string rest;
        std::getline(parts, rest);
        rest.erase(0, rest.find_first_not_of(' '));
        if (topic == "syllable") {
            std::istringstream each(rest);
            std::string s;
            while (each >> s) {
                syllables_.push_back(s);
            }
            continue;
        }
        if (topic == "word") {
            std::istringstream each(rest);
            std::string list;
            each >> list;
            for (std::string w; each >> w;) {
                lists_[list].push_back(w);
            }
            continue;
        }
        if (topic == "job" || topic == "good" || topic == "event" || topic == "deed" || topic == "local") {
            std::istringstream two(rest);
            std::string key;
            two >> key;
            std::string words;
            std::getline(two, words);
            words.erase(0, words.find_first_not_of(' '));
            words_[topic + "/" + key].push_back(words);
            continue;
        }
        words_[topic].push_back(rest);
    }
}

std::string Society::words(const std::string& topic, std::uint64_t seed) const
{
    const auto it = words_.find(topic);
    if (it == words_.end() || it->second.empty()) {
        return "";
    }
    return it->second[static_cast<std::size_t>(mix(seed, text_hash(topic)) % it->second.size())];
}

bool Society::word_on(const std::string& list, const std::string& word) const
{
    const auto it = lists_.find(list);
    return it != lists_.end() && std::find(it->second.begin(), it->second.end(), word) != it->second.end();
}

std::string Society::word_of(const std::string& list, const std::string& key) const
{
    const std::string found = words(list + "/" + key, 0);
    return found.empty() ? key : found;
}

std::string Society::say(const std::string& topic, const Person& person, std::uint64_t seed, const std::string& fact, const std::string& good, const std::string& other, int count) const
{
    std::string line = words(topic, seed);
    replace_all(line, "{name}", person.name);
    replace_all(line, "{job}", person.job);
    replace_all(line, "{town}", person.town >= 0 ? towns_[static_cast<std::size_t>(person.town)].name : "");
    replace_all(line, "{fact}", fact);
    replace_all(line, "{good}", good);
    replace_all(line, "{other}", other);
    replace_all(line, "{count}", std::to_string(count));
    if (!line.empty()) {
        line[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(line[0])));
    }
    return line;
}

double Society::param(const char* name, double fallback) const
{
    const int id = law_->param_id(name);
    return id >= 0 ? law_->param_values[static_cast<std::size_t>(id)] : fallback;
}

int Society::local(const char* name) const
{
    if (def_ < 0) {
        return -1;
    }
    const CreatureDef& def = law_->creatures[static_cast<std::size_t>(def_)];
    const auto found = std::find(def.locals.begin(), def.locals.end(), name);
    return found == def.locals.end() ? -1 : static_cast<int>(found - def.locals.begin());
}

void Society::bind()
{
    def_ = law_->creature_id("town");
    goods_.clear();
    q_.clear();
    p_.clear();
    res_.clear();
    cult_.clear();
    stock_.clear();
    if (def_ < 0) {
        return;
    }
    const CreatureDef& def = law_->creatures[static_cast<std::size_t>(def_)];
    for (std::size_t i = 0; i < def.locals.size(); ++i) {
        if (def.locals[i].rfind("q_", 0) == 0) {
            const std::string good = def.locals[i].substr(2);
            goods_.push_back(good);
            q_.push_back(static_cast<int>(i));
            p_.push_back(local(("p_" + good).c_str()));
            res_.push_back(local(("res_" + good).c_str()));
            cult_.push_back(local(("cult_" + good).c_str()));
            stock_.push_back(local(("stock_" + good).c_str()));
        }
    }
    i_people_ = local("people");
    i_wealth_ = local("wealth");
    i_camps_ = local("camps");
    i_ill_ = local("ill_will");
    i_good_ = local("good_will");
    i_tolerance_ = local("tolerance");
    i_guard_ = local("q_guard");
    month_seconds_ = param("month_seconds", 20.0);
    rumour_every_ = param("rumour_every", 6.0);
    rumour_damp_ = param("rumour_damp", 0.3);
    rumour_speed_ = param("rumour_speed", 0.25);
    rumour_retell_ = param("rumour_retell", 0.25);
    rumour_line_ = param("rumour_line", 0.05);
    rumour_forget_ = param("rumour_forget", 0.015);
    rumour_far_ = param("rumour_far", 0.8);
    rumour_age_ = param("rumour_age", 0.0006);
    belief_mix_ = param("belief_mix", 0.02);
}

void Society::bring_land(Town& town, const WorldMap& map, const Settlement& s, const Law& law)
{
    (void)law;
    // What lies around it, walking out as far as one would go to work: a ring of columns.
    double soil = 0.0, water = 0.0, forest = 0.0, rock = 0.0, ore = 0.0, grass = 0.0;
    int count = 0;
    double low = 1e30, high = -1e30;
    const int reach = s.radius * 2 + 80;
    for (int ring = 1; ring <= 4; ++ring) {
        for (int k = 0; k < 12; ++k) {
            const double a = 6.283185307 * k / 12.0 + ring;
            const int wx = s.x + static_cast<int>(std::cos(a) * reach * ring / 4);
            const int wy = s.y + static_cast<int>(std::sin(a) * reach * ring / 4);
            const Column c = map.column(wx, wy);
            const double rain = map.rain_at(wx, wy);
            const double warm = std::clamp((map.temperature_at(wx, wy) + 5.0) / 25.0, 0.0, 1.0);
            low = std::min(low, static_cast<double>(c.top));
            high = std::max(high, static_cast<double>(c.top));
            ++count;
            water += c.water >= 0 ? 1.0 : 0.0;
            switch (c.biome) {
            case Biome::Grassland: soil += rain * warm; grass += 1.0; break;
            case Biome::Savanna: soil += 0.3 * rain; grass += 0.8; break;
            case Biome::Forest: forest += 1.0; soil += 0.3 * rain * warm; grass += 0.2; break;
            case Biome::Jungle: forest += 0.9; soil += 0.3; break;
            case Biome::Taiga: forest += 0.9; grass += 0.2; break;
            case Biome::Tundra: grass += 0.4; ore += 0.1; rock += 0.2; break;
            case Biome::Snow: rock += 0.3; ore += 0.15; break;
            case Biome::Swamp: forest += 0.3; soil += 0.2; break;
            case Biome::Mountain: rock += 1.0; ore += 0.6; grass += 0.1; break;
            case Biome::Desert: rock += 0.3; ore += 0.15; break;
            default: break;
            }
        }
    }
    const double steep = std::clamp((high - low) / 80.0, 0.0, 1.0);
    const auto set = [&](const char* name, double v) {
        if (const int id = local(name); id >= 0) {
            town.locals[static_cast<std::size_t>(id)] = std::clamp(v, 0.0, 1.0);
        }
    };
    set("land_soil", soil / count);
    set("land_water", std::min(1.0, 3.0 * water / count));
    set("land_forest", forest / count);
    set("land_rock", rock / count + 0.3 * steep);
    set("land_ore", ore / count + 0.2 * steep);
    set("land_grass", grass / count);
}

Society::Society(const WorldMap& map, const Law& law, const std::string& words) : map_(&map), law_(&law)
{
    set_words(words);
    bind();
    if (def_ < 0) {
        return;
    }
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(def_)];
    locals_now_ = def.locals;
    const std::vector<Settlement>& all = map.settlements();
    town_of_settlement_.assign(all.size(), -1);
    for (std::size_t i = 0; i < all.size(); ++i) {
        const Settlement& s = all[i];
        if (s.fell >= 0 || s.people <= 0) {
            continue;
        }
        Town town;
        town.settlement = static_cast<int>(i);
        town.name = s.name;
        town.culture = map.cultures()[static_cast<std::size_t>(s.culture)].name;
        town.locals = def.init;
        town.crossed.assign(def.events.size(), 0);
        if (i_people_ >= 0) {
            town.locals[static_cast<std::size_t>(i_people_)] = s.people;
        }
        if (const int room = local("room"); room >= 0) {
            town.locals[static_cast<std::size_t>(room)] = s.people * 1.4 + 10.0;
        }
        bring_land(town, map, s, law);
        // What this people holds in honour comes with the people, and each place has a little of its own.
        for (std::size_t g = 0; g < goods_.size(); ++g) {
            const double a = hash01(static_cast<std::uint64_t>(s.culture) + 77u, text_hash(goods_[g])) - 0.5;
            const double b = hash01(static_cast<std::uint64_t>(i) + 991u, text_hash(goods_[g])) - 0.5;
            if (cult_[g] >= 0) {
                town.locals[static_cast<std::size_t>(cult_[g])] = 1.2 * a + 0.4 * b;
            }
        }
        // Opinions begin as a wide lump, off-centre as the culture has it.
        const double centre = 0.3 + 0.4 * hash01(static_cast<std::uint64_t>(s.culture), 555u);
        double sum = 0.0;
        for (int b = 0; b < Town::kBeliefs; ++b) {
            const double x = (b + 0.5) / Town::kBeliefs;
            const double wobble = 0.8 + 0.4 * hash01(i * 37u + 1u, static_cast<std::uint64_t>(b));
            town.beliefs[static_cast<std::size_t>(b)] = static_cast<float>(wobble * std::exp(-(x - centre) * (x - centre) / (2.0 * 0.28 * 0.28)));
            sum += town.beliefs[static_cast<std::size_t>(b)];
        }
        for (float& v : town.beliefs) {
            v = static_cast<float>(v / sum);
        }
        town.wanted = std::clamp(static_cast<int>(std::sqrt(static_cast<double>(s.people))) + 3, 3, 16);
        town_of_settlement_[i] = static_cast<int>(towns_.size());
        towns_.push_back(std::move(town));
    }
    // The roads: each place to the nearest few, both ways. Further is worse. (A graph, of the living.)
    for (std::size_t a = 0; a < towns_.size(); ++a) {
        const Settlement& sa = all[static_cast<std::size_t>(towns_[a].settlement)];
        std::vector<std::pair<double, int>> near_ones;
        for (std::size_t b = 0; b < towns_.size(); ++b) {
            if (a != b) {
                const Settlement& sb = all[static_cast<std::size_t>(towns_[b].settlement)];
                near_ones.push_back({std::hypot(static_cast<double>(sa.x - sb.x), static_cast<double>(sa.y - sb.y)), static_cast<int>(b)});
            }
        }
        const std::size_t keep = std::min<std::size_t>(3, near_ones.size());
        std::partial_sort(near_ones.begin(), near_ones.begin() + static_cast<std::ptrdiff_t>(keep), near_ones.end());
        for (std::size_t k = 0; k < keep; ++k) {
            const auto [dist, other] = near_ones[k];
            const float good = static_cast<float>(std::clamp(1.0 / (1.0 + dist / 1500.0), 0.2, 1.0));
            const auto has = [&](const Town& t, int to) {
                return std::any_of(t.roads.begin(), t.roads.end(), [&](const auto& r) { return r.first == to; });
            };
            if (!has(towns_[a], other)) {
                towns_[a].roads.push_back({other, good});
                towns_[static_cast<std::size_t>(other)].roads.push_back({static_cast<int>(a), good});
            }
        }
    }
    // And nowhere is cut off for good: a tree over everyone, the shortest it can be, so word can get anywhere.
    if (towns_.size() > 1) {
        std::vector<char> joined(towns_.size(), 0);
        std::vector<double> best(towns_.size(), 1e30);
        std::vector<int> from(towns_.size(), -1);
        std::size_t at = 0;
        for (std::size_t round = 1; round < towns_.size(); ++round) {
            joined[at] = 1;
            const Settlement& sa = all[static_cast<std::size_t>(towns_[at].settlement)];
            std::size_t next = 0;
            double lowest = 1e30;
            for (std::size_t b = 0; b < towns_.size(); ++b) {
                if (joined[b]) {
                    continue;
                }
                const Settlement& sb = all[static_cast<std::size_t>(towns_[b].settlement)];
                const double d = std::hypot(static_cast<double>(sa.x - sb.x), static_cast<double>(sa.y - sb.y));
                if (d < best[b]) {
                    best[b] = d;
                    from[b] = static_cast<int>(at);
                }
                if (best[b] < lowest) {
                    lowest = best[b];
                    next = b;
                }
            }
            Town& a = towns_[static_cast<std::size_t>(from[next])];
            const int to = static_cast<int>(next);
            if (std::none_of(a.roads.begin(), a.roads.end(), [&](const auto& r) { return r.first == to; })) {
                const float good = static_cast<float>(std::clamp(1.0 / (1.0 + lowest / 1500.0), 0.2, 1.0));
                a.roads.push_back({to, good});
                towns_[next].roads.push_back({from[next], good});
            }
            at = next;
        }
    }
    // What the map's history remembers is known where it happened, the later the better.
    int last_age = 0;
    for (const Chronicle& c : map.chronicle()) {
        last_age = std::max(last_age, c.age);
    }
    for (const Chronicle& c : map.chronicle()) {
        if (c.settlement < 0 || c.settlement >= static_cast<int>(town_of_settlement_.size()) || town_of_settlement_[static_cast<std::size_t>(c.settlement)] < 0) {
            continue;
        }
        const int t = town_of_settlement_[static_cast<std::size_t>(c.settlement)];
        // What is long past is told less; it is the old stories that make room.
        const float recent = static_cast<float>(1.0 / (1.0 + 0.5 * (last_age - c.age)));
        const float weight = static_cast<float>(0.2 + 0.4 * recent * hash01(text_hash(c.text), 3u));
        Fact fact;
        fact.text = c.text + " (" + towns_[static_cast<std::size_t>(t)].name + ")";
        fact.town = t;
        fact.weight = weight;
        fact.month = (c.age - last_age) * 12 * 25;
        const int id = add_fact(std::move(fact));
        add_row(towns_[static_cast<std::size_t>(t)], id, std::clamp(0.9f - 0.1f * static_cast<float>(last_age - c.age), 0.2f, 0.9f), weight);
    }
    // Enough months for each place to come to what its land and people make of it.
    for (int m = 0; m < 240; ++m) {
        for (Town& town : towns_) {
            take_month(town);
        }
        ++month_;
        if (m % 8 == 0) {
            wave_step();
        }
    }
}

void Society::rebind(const Law& law)
{
    const std::vector<std::string> old_names = locals_now_;
    law_ = &law;
    bind();
    if (def_ < 0) {
        return;
    }
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(def_)];
    locals_now_ = def.locals;
    for (Town& town : towns_) {
        std::vector<double> locals = def.init;
        for (std::size_t i = 0; i < def.locals.size(); ++i) {
            const auto it = std::find(old_names.begin(), old_names.end(), def.locals[i]);
            if (it != old_names.end()) {
                locals[i] = town.locals[static_cast<std::size_t>(it - old_names.begin())];
            }
        }
        town.locals = std::move(locals);
        town.crossed.resize(def.events.size(), 0);
    }
}

int Society::town_at(const Settlement& settlement) const
{
    const auto index = static_cast<std::size_t>(&settlement - map_->settlements().data());
    return index < town_of_settlement_.size() ? town_of_settlement_[index] : -1;
}

double Society::state(const Town& town, const char* name) const
{
    const int id = local(name);
    return id >= 0 ? town.locals[static_cast<std::size_t>(id)] : 0.0;
}

double Society::share(const Town& town, int good) const
{
    return town.locals[static_cast<std::size_t>(q_[static_cast<std::size_t>(good)])];
}

double Society::price(const Town& town, int good) const
{
    const int id = p_[static_cast<std::size_t>(good)];
    return id >= 0 ? town.locals[static_cast<std::size_t>(id)] : 1.0;
}

int Society::people_in(const Town& town) const
{
    return i_people_ >= 0 ? static_cast<int>(town.locals[static_cast<std::size_t>(i_people_)]) : 0;
}

int Society::guards_in(const Town& town) const
{
    return i_guard_ >= 0 ? static_cast<int>(std::round(town.locals[static_cast<std::size_t>(i_guard_)] * people_in(town))) : 0;
}

// BELIEFS: every opinion moves to the mean of those near it, and what is near is `tolerance`. So the camps come by themselves,
// and it is the mass in a bin that moves, split between the two bins it lands between.
void Society::beliefs_step(Town& town)
{
    const int n = Town::kBeliefs;
    const double eps = i_tolerance_ >= 0 ? town.locals[static_cast<std::size_t>(i_tolerance_)] : 0.12;
    std::array<double, Town::kBeliefs + 1> cm{}, cf{};
    for (int k = 0; k < n; ++k) {
        const double x = (k + 0.5) / n;
        cm[static_cast<std::size_t>(k + 1)] = cm[static_cast<std::size_t>(k)] + town.beliefs[static_cast<std::size_t>(k)];
        cf[static_cast<std::size_t>(k + 1)] = cf[static_cast<std::size_t>(k)] + town.beliefs[static_cast<std::size_t>(k)] * x;
    }
    const auto at = [&](const std::array<double, Town::kBeliefs + 1>& c, double x) {
        const double p = std::clamp(x, 0.0, 1.0) * n;
        const int k = std::min(static_cast<int>(p), n - 1);
        return c[static_cast<std::size_t>(k)] + (c[static_cast<std::size_t>(k + 1)] - c[static_cast<std::size_t>(k)]) * (p - k);
    };
    std::array<float, Town::kBeliefs> next{};
    for (int k = 0; k < n; ++k) {
        const float mass = town.beliefs[static_cast<std::size_t>(k)];
        if (mass <= 0.0f) {
            continue;
        }
        const double x = (k + 0.5) / n;
        const double m = at(cm, x + eps) - at(cm, x - eps);
        const double f = at(cf, x + eps) - at(cf, x - eps);
        const double to = m > 1e-9 ? x + 0.2 * (f / m - x) : x;
        const double p = std::clamp(to * n - 0.5, 0.0, n - 1.0);
        const int low = std::min(static_cast<int>(p), n - 2);
        const float share_high = static_cast<float>(p - low);
        next[static_cast<std::size_t>(low)] += mass * (1.0f - share_high);
        next[static_cast<std::size_t>(low + 1)] += mass * share_high;
    }
    // A little comes in by the roads. So one place's quarrels can start in another.
    if (!town.roads.empty()) {
        for (const auto& [other, good] : town.roads) {
            const Town& o = towns_[static_cast<std::size_t>(other)];
            const float mix_in = static_cast<float>(belief_mix_) * good / static_cast<float>(town.roads.size());
            for (int k = 0; k < n; ++k) {
                next[static_cast<std::size_t>(k)] += mix_in * (o.beliefs[static_cast<std::size_t>(k)] - next[static_cast<std::size_t>(k)]);
            }
        }
    }
    float total = 0.0f;
    for (const float v : next) {
        total += v;
    }
    for (int k = 0; k < n; ++k) {
        town.beliefs[static_cast<std::size_t>(k)] = next[static_cast<std::size_t>(k)] / std::max(total, 1e-6f);
    }
}

std::vector<std::pair<float, float>> Society::camps_of(const Town& town) const
{
    // A camp is a run of opinions anyone holds, with a gap after it, and enough people in it.
    std::vector<std::pair<float, float>> out;
    double mass = 0.0, moment = 0.0;
    int gap = 0;
    const auto close = [&]() {
        if (mass >= 0.05) {
            out.push_back({static_cast<float>(moment / mass), static_cast<float>(mass)});
        }
        mass = 0.0;
        moment = 0.0;
    };
    for (int k = 0; k < Town::kBeliefs; ++k) {
        const float v = town.beliefs[static_cast<std::size_t>(k)];
        if (v > 0.008f) {
            mass += v;
            moment += v * (k + 0.5) / Town::kBeliefs;
            gap = 0;
        } else if (++gap == 1) {
            close();
        }
    }
    close();
    return out;
}

// A TOWN'S MONTH: what comes from outside goes into its inputs, and the law's lines do the rest.
void Society::take_month(Town& town)
{
    const CreatureDef& def = law_->creatures[static_cast<std::size_t>(def_)];
    std::vector<double>& locals = town.locals;
    double wealth = 0.0;
    for (std::size_t g = 0; g < goods_.size(); ++g) {
        // What the month's work fetches: the same sum the law's lines take for what is made.
        const double q = locals[static_cast<std::size_t>(q_[g])];
        const double p = p_[g] >= 0 ? locals[static_cast<std::size_t>(p_[g])] : 1.0;
        const double r = res_[g] >= 0 ? locals[static_cast<std::size_t>(res_[g])] : 1.0;
        wealth += p * r * std::sqrt(q);
    }
    if (i_wealth_ >= 0) {
        locals[static_cast<std::size_t>(i_wealth_)] = wealth;
    }
    beliefs_step(town);
    if (i_camps_ >= 0) {
        locals[static_cast<std::size_t>(i_camps_)] = static_cast<double>(camps_of(town).size());
    }
    double ill = 0.0, good = 0.0;
    for (const FactRow& row : town.rows) {
        const float about = facts_[static_cast<std::size_t>(row.fact)].about_you;
        if (about < 0.0f) {
            ill += -about * row.now * row.weight;
        } else {
            good += about * row.now * row.weight;
        }
    }
    if (i_ill_ >= 0) {
        locals[static_cast<std::size_t>(i_ill_)] = ill;
    }
    if (i_good_ >= 0) {
        locals[static_cast<std::size_t>(i_good_)] = good;
    }
    const PointContext at{.fields = nullptr, .params = &law_->param_values, .locals = &locals, .x = 0.0, .y = 0.0, .z = 0.0};
    std::vector<double> next = locals;
    for (std::size_t i = static_cast<std::size_t>(kFirstState); i < def.locals.size(); ++i) {
        if (def.rates[i]) {
            next[i] = std::clamp(locals[i] + evaluate_at(*def.rates[i], at), def.lo[i], def.hi[i]);
        }
    }
    locals = std::move(next);
    // Shares of the work add up: this is the mean the replicator takes away.
    double total = 0.0;
    for (const int q : q_) {
        total += locals[static_cast<std::size_t>(q)];
    }
    for (const int q : q_) {
        locals[static_cast<std::size_t>(q)] /= std::max(total, 1e-9);
    }
    // Only what one thing fetches against another means anything: left alone, every price drifts together.
    double logs = 0.0;
    for (const int price_at : p_) {
        logs += price_at >= 0 ? std::log(locals[static_cast<std::size_t>(price_at)]) : 0.0;
    }
    const double level = std::exp(logs / std::max<std::size_t>(p_.size(), 1));
    for (const int price_at : p_) {
        if (price_at >= 0) {
            locals[static_cast<std::size_t>(price_at)] /= level;
        }
    }
    // A state past its line is something that happened.
    for (std::size_t e = 0; e < def.events.size(); ++e) {
        const CreatureEvent& event = def.events[e];
        const double v = locals[static_cast<std::size_t>(event.state)];
        const bool past = event.above ? v > event.line : v < event.line;
        if (past && !town.crossed[e]) {
            town.crossed[e] = 1;
            if (month_ >= 24) { // what a place began as is not news
                // The same thing again, soon, is the same story.
                const auto again = std::find_if(town.rows.begin(), town.rows.end(), [&](const FactRow& row) {
                    const Fact& old = facts_[static_cast<std::size_t>(row.fact)];
                    return old.kind == event.name && old.town == static_cast<int>(&town - towns_.data());
                });
                if (again != town.rows.end()) {
                    again->now = std::max(again->now, 0.6f);
                    facts_[static_cast<std::size_t>(again->fact)].month = month_;
                    facts_[static_cast<std::size_t>(again->fact)].value = v;
                } else {
                Fact fact;
                fact.text = word_of("event", event.name);
                replace_all(fact.text, "{town}", town.name);
                fact.town = static_cast<int>(&town - towns_.data());
                fact.weight = static_cast<float>(event.weight);
                fact.month = month_;
                fact.kind = event.name;
                // A price is a number someone can act on, and be wrong about.
                if (event.name.rfind("dear_", 0) == 0 || event.name.rfind("cheap_", 0) == 0) {
                    fact.good = good_index(event.name.substr(event.name.find('_') + 1));
                    fact.value = v;
                }
                const int id = add_fact(std::move(fact));
                add_row(town, id, 0.6f, static_cast<float>(event.weight));
                }
            }
            if (event.sets >= 0) {
                locals[static_cast<std::size_t>(event.sets)] = event.to;
            }
        } else if (!past) {
            // Well clear of it, it can happen again.
            const double margin = std::abs(event.line) * 0.1 + 0.02;
            if (event.above ? v < event.line - margin : v > event.line + margin) {
                town.crossed[e] = 0;
            }
        }
    }
}

int Society::add_fact(Fact fact)
{
    facts_.push_back(std::move(fact));
    return static_cast<int>(facts_.size()) - 1;
}

void Society::add_row(Town& town, int fact, float now, float weight)
{
    for (FactRow& row : town.rows) {
        if (row.fact == fact) {
            row.now = std::max(row.now, now);
            row.before = std::max(row.before, now);
            row.weight = std::max(row.weight, weight);
            return;
        }
    }
    FactRow fresh{.fact = fact, .now = now, .before = now, .weight = weight};
    if (static_cast<int>(town.rows.size()) < kRows) {
        town.rows.push_back(fresh);
        return;
    }
    // No room: what matters least goes, if that is less than this. Something new that matters is news however few have it yet.
    auto weakest = std::min_element(town.rows.begin(), town.rows.end(), [](const FactRow& a, const FactRow& b) { return a.weight < b.weight; });
    if (weakest->weight < weight) {
        *weakest = fresh;
    }
}

// THE RUMOURS TAKE A STEP: for each fact a town has, a telegraph equation over its roads. A fact a town lacks is 0 there;
// what is pushed to it past the line becomes a row. Below the line a thing is not passed on, and dies out.
void Society::wave_step()
{
    struct Next {
        int town;
        int fact;
        float now;
        float weight;
    };
    std::vector<Next> arriving;
    std::vector<std::vector<FactRow>> nexts(towns_.size());
    const float damp = static_cast<float>((1.0 - rumour_damp_) / (1.0 + rumour_damp_));
    const float push = static_cast<float>(rumour_speed_ / (1.0 + rumour_damp_));
    const float line = static_cast<float>(rumour_line_);
    const auto amount_in = [&](const Town& t, int fact) {
        for (const FactRow& r : t.rows) {
            if (r.fact == fact) {
                return r.now;
            }
        }
        return 0.0f;
    };
    for (std::size_t a = 0; a < towns_.size(); ++a) {
        const Town& t = towns_[a];
        float degree = 0.0f;
        for (const auto& road : t.roads) {
            degree += road.second;
        }
        nexts[a].reserve(t.rows.size());
        for (const FactRow& row : t.rows) {
            float flow = 0.0f;
            for (const auto& [other, good] : t.roads) {
                const float there = amount_in(towns_[static_cast<std::size_t>(other)], row.fact);
                flow += good * (row.now - there);
                // Heard well enough to go on, and it matters enough there to be worth telling.
                if (there == 0.0f && row.now > line * 2.0f && row.weight * static_cast<float>(rumour_far_) > 0.1f) {
                    const float over = push * good * row.now / std::max(degree, 0.01f);
                    if (over > 0.0f) {
                        arriving.push_back({other, row.fact, over, row.weight * static_cast<float>(rumour_far_)});
                    }
                }
            }
            const float a_now = std::clamp(row.now, 0.0f, 1.0f);
            float next = row.now + (row.now - row.before) * damp - push * flow / std::max(degree, 0.01f);
            next += row.now > line ? static_cast<float>(rumour_retell_) * row.weight * a_now * (1.0f - a_now) : -0.1f * row.now;
            next -= static_cast<float>(rumour_forget_) * row.now;
            next = std::clamp(next, 0.0f, 1.0f);
            if (next > 0.005f) {
                nexts[a].push_back(FactRow{row.fact, next, row.now, row.weight * (1.0f - static_cast<float>(rumour_age_))});
            }
        }
    }
    for (std::size_t a = 0; a < towns_.size(); ++a) {
        towns_[a].rows = std::move(nexts[a]);
    }
    for (const Next& n : arriving) {
        add_row(towns_[static_cast<std::size_t>(n.town)], n.fact, n.now, n.weight);
    }
}

void Society::tick(double dt)
{
    if (def_ < 0 || towns_.empty()) {
        return;
    }
    // As many towns a tick as it takes for each to have its month, every month.
    const double per_second = static_cast<double>(towns_.size()) / std::max(month_seconds_, 0.1);
    owed_months_ += per_second * dt;
    while (owed_months_ >= 1.0) {
        owed_months_ -= 1.0;
        take_month(towns_[next_town_]);
        next_town_ = (next_town_ + 1) % towns_.size();
        if (next_town_ == 0) {
            ++month_;
        }
    }
    caravan_owed_ += dt;
    while (caravan_owed_ >= 1.0) {
        caravan_owed_ -= 1.0;
        caravans_step(1.0);
    }
    wave_left_ -= dt;
    if (wave_left_ <= 0.0) {
        wave_left_ = rumour_every_;
        wave_step();
    }
}

Person Society::person(int town, int index) const
{
    Person p;
    p.town = town;
    p.index = index;
    p.pid = pid_of(town, index);
    const Town& t = towns_[static_cast<std::size_t>(town)];
    // A house is a handful of people: same family name.
    p.house = index / 5;
    const std::uint64_t family = pid_of(town, p.house * 5) ^ 0xABCDEFull;
    std::string given;
    std::string kin;
    if (!syllables_.empty()) {
        const auto syllable = [&](std::uint64_t a, std::uint64_t b) { return syllables_[static_cast<std::size_t>(mix(a, b) % syllables_.size())]; };
        given = syllable(p.pid, 1) + syllable(p.pid, 2);
        kin = syllable(family, 3) + syllable(family, 4) + (hash01(family, 5) < 0.4 ? syllable(family, 6) : "");
        given[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(given[0])));
        kin[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(kin[0])));
    }
    p.name = given + " " + kin;
    p.bold = 0.5 * hash01(p.pid, 11) + 0.5 * hash01(p.pid, 12);
    p.temper = 0.5 * hash01(p.pid, 13) + 0.5 * hash01(family, 14);
    p.kind = 0.5 * hash01(p.pid, 15) + 0.5 * hash01(family, 16);
    // Its work: a stretch of the shares, a stretch that moves as the shares do.
    const double u = hash01(p.pid, 21);
    double upto = 0.0;
    p.good = 0;
    for (std::size_t g = 0; g < goods_.size(); ++g) {
        upto += share(t, static_cast<int>(g));
        if (u <= upto) {
            p.good = static_cast<int>(g);
            break;
        }
    }
    p.job = goods_.empty() ? "" : word_of("job", goods_[static_cast<std::size_t>(p.good)]);
    if (goods_[static_cast<std::size_t>(p.good)] == "guard") {
        p.bold = 0.6 + 0.4 * p.bold;
    }
    const double v = hash01(p.pid, 31);
    double up = 0.0;
    p.belief = 0.5;
    for (int k = 0; k < Town::kBeliefs; ++k) {
        up += t.beliefs[static_cast<std::size_t>(k)];
        if (v <= up) {
            p.belief = (k + 0.5) / Town::kBeliefs;
            break;
        }
    }
    const std::vector<std::pair<float, float>> camps = camps_of(t);
    float best = 2.0f;
    for (std::size_t c = 0; c < camps.size(); ++c) {
        const float d = std::abs(camps[c].first - static_cast<float>(p.belief));
        if (d < best) {
            best = d;
            p.camp = static_cast<int>(c);
        }
    }
    return p;
}

bool Society::dead(std::uint64_t pid) const
{
    const auto it = records_.find(pid);
    return it != records_.end() && it->second.dead;
}

const Record* Society::record(std::uint64_t pid) const
{
    const auto it = records_.find(pid);
    return it == records_.end() ? nullptr : &it->second;
}

Record& Society::record_of(std::uint64_t pid)
{
    return records_[pid];
}

bool Society::knows(std::uint64_t pid, int town, int fact) const
{
    if (const Record* r = record(pid)) {
        if (std::find(r->saw.begin(), r->saw.end(), fact) != r->saw.end() || std::find(r->said.begin(), r->said.end(), fact) != r->said.end()) {
            return true;
        }
    }
    for (const FactRow& row : towns_[static_cast<std::size_t>(town)].rows) {
        if (row.fact == fact) {
            return hash01(pid, static_cast<std::uint64_t>(fact) + 1000u) < row.now;
        }
    }
    return false;
}

std::vector<int> Society::known(std::uint64_t pid, int town) const
{
    std::vector<std::pair<float, int>> out;
    const Record* r = record(pid);
    if (r != nullptr) {
        for (const int f : r->saw) {
            out.push_back({facts_[static_cast<std::size_t>(f)].weight + 1.0f, f});
        }
        for (const int f : r->said) {
            out.push_back({facts_[static_cast<std::size_t>(f)].weight, f});
        }
    }
    for (const FactRow& row : towns_[static_cast<std::size_t>(town)].rows) {
        if (hash01(pid, static_cast<std::uint64_t>(row.fact) + 1000u) < row.now) {
            if (std::none_of(out.begin(), out.end(), [&](const auto& e) { return e.second == row.fact; })) {
                out.push_back({row.weight, row.fact});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> facts;
    for (const auto& e : out) {
        facts.push_back(e.second);
    }
    return facts;
}

double Society::opinion(const Person& person, int* because) const
{
    double sum = 0.0;
    double most = 0.0;
    if (because != nullptr) {
        *because = -1;
    }
    for (const int f : known(person.pid, person.town)) {
        const Fact& fact = facts_[static_cast<std::size_t>(f)];
        if (fact.about_you == 0.0f) {
            continue;
        }
        const double w = static_cast<double>(fact.about_you) * fact.weight * std::min(3, fact.count);
        sum += w;
        if (std::abs(w) > most) {
            most = std::abs(w);
            if (because != nullptr) {
                *because = f;
            }
        }
    }
    if (const Record* r = record(person.pid)) {
        sum += r->own;
    }
    // The hard think worse of a stranger who has done wrong, and the kind forgive more.
    return std::clamp(sum * (sum < 0.0 ? 1.3 - 0.6 * person.kind : 0.7 + 0.6 * person.kind), -1.0, 1.0);
}

int Society::deed(const std::string& kind, int town, const std::string& name, double weight, double about_you, const std::vector<std::uint64_t>& witnesses, double town_share, std::uint64_t who, const std::string& other)
{
    Town& t = towns_[static_cast<std::size_t>(town)];
    // Done again, in the same place, soon: it is the same story, worse.
    int id = -1;
    if (kind == "killed") {
        for (const FactRow& row : t.rows) {
            const Fact& old = facts_[static_cast<std::size_t>(row.fact)];
            if (old.town == town && (old.kind == "killed" || old.kind == "killed_many") && month_ - old.month < 12) {
                id = row.fact;
                break;
            }
        }
    }
    if (id >= 0) {
        Fact& old = facts_[static_cast<std::size_t>(id)];
        old.kind = "killed_many";
        old.count += 1;
        old.weight = std::min(1.0f, old.weight + 0.1f);
        old.month = month_;
        old.about_you = std::max(-1.0f, old.about_you - 0.1f);
        old.text = word_of("deed", "killed_many");
        replace_all(old.text, "{town}", t.name);
        replace_all(old.text, "{count}", std::to_string(old.count));
    } else {
        Fact fact;
        fact.text = word_of("deed", kind);
        replace_all(fact.text, "{name}", name);
        replace_all(fact.text, "{town}", t.name);
        replace_all(fact.text, "{other}", other);
        fact.town = town;
        fact.weight = static_cast<float>(weight);
        fact.about_you = static_cast<float>(about_you);
        fact.month = month_;
        fact.kind = kind;
        fact.who = who;
        id = add_fact(std::move(fact));
    }
    add_row(t, id, static_cast<float>(town_share), static_cast<float>(weight));
    for (const std::uint64_t w : witnesses) {
        std::vector<int>& saw = record_of(w).saw;
        if (std::find(saw.begin(), saw.end(), id) == saw.end()) {
            saw.push_back(id);
            if (saw.size() > kSawKept) {
                saw.erase(saw.begin());
            }
        }
    }
    return id;
}

int Society::local_fact(const std::string& kind, int town, const std::string& name, const std::string& other, double weight, double about_you, const std::vector<std::uint64_t>& witnesses, std::uint64_t who)
{
    Fact fact;
    fact.text = word_of("local", kind);
    replace_all(fact.text, "{name}", name);
    replace_all(fact.text, "{town}", towns_[static_cast<std::size_t>(town)].name);
    replace_all(fact.text, "{other}", other);
    fact.town = town;
    fact.weight = static_cast<float>(weight);
    fact.about_you = static_cast<float>(about_you);
    fact.month = month_;
    fact.kind = kind;
    fact.who = who;
    fact.local = true;
    const int id = add_fact(std::move(fact));
    for (const std::uint64_t w : witnesses) {
        std::vector<int>& saw = record_of(w).saw;
        saw.push_back(id);
        if (saw.size() > kSawKept) {
            saw.erase(saw.begin());
        }
    }
    return id;
}

int Society::said_fact(const std::string& words, int town, double about_you, const std::vector<std::uint64_t>& hearers)
{
    Fact fact;
    fact.text = word_of("local", "said");
    replace_all(fact.text, "{town}", town >= 0 ? towns_[static_cast<std::size_t>(town)].name : "");
    replace_all(fact.text, "{other}", words);
    fact.town = town;
    fact.weight = 0.2f;
    fact.about_you = static_cast<float>(about_you);
    fact.month = month_;
    fact.kind = "said";
    fact.local = true;
    fact.quote = words;
    const int id = add_fact(std::move(fact));
    for (const std::uint64_t w : hearers) {
        std::vector<int>& said = record_of(w).said;
        said.push_back(id);
        if (said.size() > kSaidKept) {
            said.erase(said.begin());
        }
    }
    return id;
}

// Time passes in one go: every month, every step of the rumours, as if it had been played.
void Society::months_pass(double seconds)
{
    const double step = std::max(0.05, rumour_every_);
    for (double left = seconds; left > 0.0; left -= step) {
        tick(std::min(left, step));
    }
}

bool Society::told(const Person& person, int fact)
{
    const bool news = !knows(person.pid, person.town, fact);
    if (facts_[static_cast<std::size_t>(fact)].local) {
        // It goes from mouth to mouth, and not over the roads.
        if (news) {
            std::vector<int>& saw = record_of(person.pid).saw;
            saw.push_back(fact);
            if (saw.size() > kSawKept) {
                saw.erase(saw.begin());
            }
        }
        return news;
    }
    Town& t = towns_[static_cast<std::size_t>(person.town)];
    // One more tells the others: it counts for how well it is told by how much it matters.
    float was = 0.0f;
    for (const FactRow& row : t.rows) {
        if (row.fact == fact) {
            was = row.now;
        }
    }
    const float people = std::max(1.0f, static_cast<float>(people_in(t)));
    add_row(t, fact, std::min(1.0f, was + std::max(0.1f, 6.0f / people)), facts_[static_cast<std::size_t>(fact)].weight);
    std::vector<int>& saw = record_of(person.pid).saw;
    if (std::find(saw.begin(), saw.end(), fact) == saw.end()) {
        saw.push_back(fact);
        if (saw.size() > kSawKept) {
            saw.erase(saw.begin());
        }
    }
    return news;
}

// ---- the market ---------------------------------------------------------------------------------------------------------------

double Society::stock(const Town& town, int good) const
{
    const int at = stock_[static_cast<std::size_t>(good)];
    return at >= 0 ? town.locals[static_cast<std::size_t>(at)] : 1.0;
}

void Society::add_stock(int town, int good, double amount)
{
    const int at = good >= 0 && good < static_cast<int>(stock_.size()) ? stock_[static_cast<std::size_t>(good)] : -1;
    if (at >= 0) {
        double& v = towns_[static_cast<std::size_t>(town)].locals[static_cast<std::size_t>(at)];
        v = std::max(0.0, v + amount);
    }
}

int Society::good_index(const std::string& name) const
{
    const auto it = std::find(goods_.begin(), goods_.end(), name);
    return it == goods_.end() ? -1 : static_cast<int>(it - goods_.begin());
}

bool Society::on_the_road(std::uint64_t pid) const
{
    return std::any_of(caravans_.begin(), caravans_.end(), [&](const Caravan& c) { return c.pid == pid && c.next >= 0 && !c.gone; });
}

std::vector<MarketNews> Society::take_news()
{
    std::vector<MarketNews> out;
    out.swap(news_);
    return out;
}

// One in a place that has anything like a trader: who it is depends on the shares of the work, and it is the same one from then on.
void Society::make_caravans()
{
    caravans_.clear();
    const int trade = good_index("trade");
    if (trade < 0) {
        return;
    }
    for (std::size_t t = 0; t < towns_.size(); ++t) {
        const Settlement& s = map_->settlements()[static_cast<std::size_t>(towns_[t].settlement)];
        if (s.tier < 1 || towns_[t].roads.empty()) {
            continue;
        }
        for (int index = 0; index < std::min(people_in(towns_[t]), 80); ++index) {
            const Person p = person(static_cast<int>(t), index);
            if (p.good != trade) {
                continue;
            }
            Caravan c;
            c.pid = p.pid;
            c.home = static_cast<int>(t);
            c.index = index;
            c.town = static_cast<int>(t);
            c.x = s.x;
            c.y = s.y;
            c.wait = 5.0 + 60.0 * hash01(p.pid, 301);
            caravans_.push_back(c);
            break;
        }
    }
}

// The first town on the way, over the roads. -1: there is no way.
int Society::hop_toward(int from, int to) const
{
    if (from == to) {
        return to;
    }
    // The first step of the way each town was reached by, spreading out from here.
    std::vector<int> first(towns_.size(), -1);
    std::vector<int> front{from};
    first[static_cast<std::size_t>(from)] = from;
    for (std::size_t k = 0; k < front.size(); ++k) {
        const int here = front[k];
        for (const auto& road : towns_[static_cast<std::size_t>(here)].roads) {
            if (first[static_cast<std::size_t>(road.first)] < 0) {
                first[static_cast<std::size_t>(road.first)] = here == from ? road.first : first[static_cast<std::size_t>(here)];
                if (road.first == to) {
                    return first[static_cast<std::size_t>(road.first)];
                }
                front.push_back(road.first);
            }
        }
    }
    return -1;
}

// What it does in a place: it looks at what it knows. Dear somewhere, and here it can be had for less: it buys and goes. Cheap somewhere:
// it goes to see. Nothing: it takes a little of what is cheapest here to the next town, because you do not hear anything by staying.
void Society::caravan_decides(Caravan& c)
{
    const Town& here = towns_[static_cast<std::size_t>(c.town)];
    const double cap = param("caravan_load", 0.8);
    int best_good = -1;
    int best_town = -1;
    double best = 0.25;
    bool to_buy = false;
    for (const int f : known(c.pid, c.town)) {
        const Fact& fact = facts_[static_cast<std::size_t>(f)];
        if (fact.good < 0 || fact.town == c.town || fact.town < 0) {
            continue;
        }
        const double age = std::max(0, month_ - fact.month);
        const double trust = std::exp(-0.005 * age);
        const double p_here = price(here, fact.good);
        if (fact.kind.rfind("dear_", 0) == 0) {
            if (stock(here, fact.good) < 0.3) {
                continue;
            }
            const double margin = fact.value * trust - p_here;
            if (margin > best) {
                best = margin;
                best_good = fact.good;
                best_town = fact.town;
                to_buy = false;
            }
        } else if (fact.kind.rfind("cheap_", 0) == 0) {
            const double margin = (p_here - fact.value) * trust * 0.5;
            if (margin > best) {
                best = margin;
                best_good = fact.good;
                best_town = fact.town;
                to_buy = true;
            }
        }
    }
    if (best_town >= 0 && hop_toward(c.town, best_town) < 0) {
        best_town = -1;
    }
    ++c.trips;
    if (best_town < 0) {
        // Nothing to go on. A town next door, with a little of what is cheapest here.
        const std::uint64_t pick = static_cast<std::uint64_t>(mix(c.pid, static_cast<std::uint64_t>(c.trips)));
        best_town = here.roads[static_cast<std::size_t>(pick % here.roads.size())].first;
        double lowest = 0.9;
        best_good = -1;
        for (int g = 0; g < static_cast<int>(goods_.size()); ++g) {
            if (goods_[static_cast<std::size_t>(g)] != "guard" && goods_[static_cast<std::size_t>(g)] != "trade" && price(here, g) < lowest && stock(here, g) > 0.5) {
                lowest = price(here, g);
                best_good = g;
            }
        }
        if (best_good >= 0) {
            const double p = price(here, best_good);
            const double load = std::min({cap * 0.4, c.coins / std::max(p, 0.05), stock(here, best_good) - 0.3});
            if (load > 0.02) {
                add_stock(c.town, best_good, -load);
                c.coins -= load * p;
                c.load = load;
                c.good = best_good;
                c.paid = p;
            }
        }
    } else if (!to_buy) {
        const double p = price(here, best_good);
        const double load = std::min({cap, c.coins / std::max(p, 0.05), stock(here, best_good) - 0.2});
        if (load > 0.02) {
            add_stock(c.town, best_good, -load);
            c.coins -= load * p;
            c.load = load;
            c.good = best_good;
            c.paid = p;
        }
    }
    c.goal = best_town;
    c.next = hop_toward(c.town, best_town);
    if (c.next >= 0) {
        MarketNews line;
        line.town = c.town;
        line.text = words("news_left", c.pid + static_cast<std::uint64_t>(c.trips));
        replace_all(line.text, "{town}", here.name);
        replace_all(line.text, "{other}", towns_[static_cast<std::size_t>(c.goal)].name);
        replace_all(line.text, "{good}", c.good >= 0 ? word_of("good", goods_[static_cast<std::size_t>(c.good)]) : "nothing yet");
        news_.push_back(line);
    }
}

void Society::caravan_arrives(Caravan& c)
{
    // What it heard on the way goes into the place with it, and it knows what it knew.
    const std::vector<int> knew = known(c.pid, c.town);
    std::vector<int>& saw = record_of(c.pid).saw;
    int carried = 0;
    for (const int f : knew) {
        const Fact& fact = facts_[static_cast<std::size_t>(f)];
        if (fact.local || !fact.quote.empty() || ++carried > 6) {
            continue;
        }
        add_row(towns_[static_cast<std::size_t>(c.next)], f, 0.08f, fact.weight * static_cast<float>(rumour_far_));
        if (std::find(saw.begin(), saw.end(), f) == saw.end()) {
            saw.push_back(f);
            if (saw.size() > kSawKept) {
                saw.erase(saw.begin());
            }
        }
    }
    c.town = c.next;
    c.next = -1;
    const Town& here = towns_[static_cast<std::size_t>(c.town)];
    if (c.town != c.goal) {
        c.next = hop_toward(c.town, c.goal); // on its way
        return;
    }
    c.goal = -1;
    MarketNews line;
    line.town = c.town;
    if (c.good >= 0 && c.load > 0.0) {
        // It sells for what it finds, whatever it had heard.
        const double p = price(here, c.good);
        add_stock(c.town, c.good, c.load);
        c.coins += c.load * p;
        c.earned += c.load * (p - c.paid);
        line.text = words("news_arrived", c.pid + static_cast<std::uint64_t>(c.trips));
        replace_all(line.text, "{good}", word_of("good", goods_[static_cast<std::size_t>(c.good)]));
    } else {
        line.text = words("news_arrived_empty", c.pid + static_cast<std::uint64_t>(c.trips));
    }
    replace_all(line.text, "{town}", here.name);
    news_.push_back(line);
    c.load = 0.0;
    c.good = -1;
    c.wait = param("caravan_wait", 25.0);
}

void Society::caravans_step(double dt)
{
    if (!caravans_made_) {
        caravans_made_ = true;
        make_caravans();
    }
    const double speed = param("caravan_speed", 120.0);
    if (speed <= 0.0) {
        return;
    }
    for (Caravan& c : caravans_) {
        if (c.gone) {
            c.gone_left -= dt;
            if (c.gone_left <= 0.0) {
                // Someone else takes it up, from home.
                c = Caravan{.pid = person(c.home, c.index + 5).pid, .home = c.home, .index = c.index + 5, .town = c.home,
                    .x = static_cast<double>(map_->settlements()[static_cast<std::size_t>(towns_[static_cast<std::size_t>(c.home)].settlement)].x),
                    .y = static_cast<double>(map_->settlements()[static_cast<std::size_t>(towns_[static_cast<std::size_t>(c.home)].settlement)].y),
                    .wait = 30.0};
            }
            continue;
        }
        if (c.next < 0) {
            c.wait -= dt;
            if (c.wait <= 0.0) {
                c.wait = 20.0;
                caravan_decides(c);
            }
            continue;
        }
        const Settlement& to = map_->settlements()[static_cast<std::size_t>(towns_[static_cast<std::size_t>(c.next)].settlement)];
        const double dx = to.x - c.x;
        const double dy = to.y - c.y;
        const double away = std::hypot(dx, dy);
        if (away < std::max(to.radius * 0.5, 8.0)) {
            caravan_arrives(c);
            continue;
        }
        if (!c.body) {
            const double step = std::min(speed * dt, away);
            c.x += dx / away * step;
            c.y += dy / away * step;
        }
    }
}

void Society::caravan_lost(int i)
{
    Caravan& c = caravans_[static_cast<std::size_t>(i)];
    c.gone = true;
    c.gone_left = 200.0;
    c.load = 0.0;
    c.good = -1;
    c.next = -1;
    c.body = false;
}

int Society::price_fact(int good, int town, double value)
{
    Fact fact;
    const std::string kind = std::string(value >= 1.0 ? "dear_" : "cheap_") + goods_[static_cast<std::size_t>(good)];
    fact.text = word_of("event", kind);
    replace_all(fact.text, "{town}", towns_[static_cast<std::size_t>(town)].name);
    fact.town = town;
    fact.weight = 0.4f;
    fact.month = month_;
    fact.kind = kind;
    fact.good = good;
    fact.value = value;
    return add_fact(std::move(fact));
}

void Society::lost(int town)
{
    Town& t = towns_[static_cast<std::size_t>(town)];
    if (i_people_ >= 0) {
        t.locals[static_cast<std::size_t>(i_people_)] = std::max(0.0, t.locals[static_cast<std::size_t>(i_people_)] - 1.0);
    }
}

} // namespace fieldengine::field

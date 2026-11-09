#include "field/Playground.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

// WHAT GOES ON AMONG THE PEOPLE. Local things are real: a theft is one person running from another with the watch after them, a
// brawl is two bodies with moves, and what gets told is what happened, to those who were there and whom they tell (Society).
// The chat is the log of it.
namespace fieldengine::field {

namespace {

constexpr double kHearing = 12.0; // metres, for words
constexpr double kAround = 70.0;  // metres, for what happens to happen near you

Vec3 horizontal(Vec3 v)
{
    return Vec3{v.x, v.y, 0.0};
}

double hash01(std::uint64_t a, std::uint64_t b)
{
    std::uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x1234567ull) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}

double law_param(const Law& law, const char* name, double fallback)
{
    const int id = law.param_id(name);
    return id >= 0 ? law.param_values[static_cast<std::size_t>(id)] : fallback;
}

double read_local(const CreatureDef& def, const Creature& c, const char* name)
{
    const auto it = std::find(def.locals.begin(), def.locals.end(), name);
    return it == def.locals.end() ? 0.0 : c.locals[static_cast<std::size_t>(it - def.locals.begin())];
}

void write_local(const CreatureDef& def, Creature& c, const char* name, double value)
{
    const auto it = std::find(def.locals.begin(), def.locals.end(), name);
    if (it != def.locals.end()) {
        const std::size_t id = static_cast<std::size_t>(it - def.locals.begin());
        c.locals[id] = std::clamp(value, def.lo[id], def.hi[id]);
    }
}

} // namespace

Creature* Playground::creature_mut(std::uint32_t id)
{
    for (Creature& c : creatures_) {
        if (c.id == id) {
            return &c;
        }
    }
    return nullptr;
}

void Playground::chat_line(ChatLine::Kind kind, const std::string& who, const std::string& text)
{
    if (text.empty()) {
        return;
    }
    chat_.push_back(ChatLine{kind, who, text, tick_});
    if (chat_.size() > 200) {
        chat_.erase(chat_.begin());
    }
}

double Playground::day_phase() const
{
    const double day = law_param(solver_.law(), "day_seconds", 480.0);
    return std::fmod((static_cast<double>(tick_) * kTick + day_shift_) / day, 1.0);
}

// ---- words ------------------------------------------------------------------------------------------------------------------

// ---- things that happen -------------------------------------------------------------------------------------------------------

void Playground::set_on(Creature& c, const Creature& foe)
{
    c.foe = foe.id;
    c.body.foe = foe.id;
    c.plan_left = 0.0;
    c.flee_left = 0.0;
}

void Playground::fight_over(Creature& c)
{
    c.foe = 0;
    c.body.foe = 0;
    c.body.guard = -1.0;
    if (c.brawling) {
        c.brawling = false;
        c.arm = c.arm_before;
    }
}

// One of them has had enough. Who was hit more knows it, and so does everyone who watched.
void Playground::end_brawl(Creature& c)
{
    Creature* other = creature_mut(c.foe);
    const std::uint32_t was = c.foe;
    (void)was;
    if (other == nullptr || !other->brawling) {
        fight_over(c);
        return;
    }
    Creature& won = c.body.hp >= other->body.hp ? c : *other;
    Creature& lost = c.body.hp >= other->body.hp ? *other : c;
    society_->record_of(won.who.pid).others[lost.who.pid] = -0.3;
    society_->record_of(lost.who.pid).others[won.who.pid] = -0.6;
    std::vector<std::uint64_t> saw = witnesses_of(lost, false);
    saw.push_back(won.who.pid);
    saw.push_back(lost.who.pid);
    const int fact = society_->local_fact("brawl_end", won.town, won.who.name, lost.who.name, 0.35, 0.0, saw);
    chat_line(ChatLine::Kind::Local, "", society_->facts()[static_cast<std::size_t>(fact)].text);
    lost.flee_left = 6.0;
    lost.flee_from = won.body.position;
    Creature& a = c;
    Creature& b = *other;
    fight_over(a);
    fight_over(b);
}

bool Playground::make_incident(const std::string& kind)
{
    if (society_ == nullptr) {
        return false;
    }
    const Law& law = solver_.law();
    const PlayBody& you = player();
    std::vector<Creature*> about;
    for (Creature& c : creatures_) {
        if (c.town >= 0 && !c.hostile && c.foe == 0 && c.flee_left <= 0.0 && length(c.body.position - you.position) < kAround) {
            about.push_back(&c);
        }
    }
    if (about.empty()) {
        return false;
    }
    std::sort(about.begin(), about.end(), [&](const Creature* a, const Creature* b) { return length(a->body.position - you.position) < length(b->body.position - you.position); });
    const int town = about[0]->town;
    about.erase(std::remove_if(about.begin(), about.end(), [&](const Creature* c) { return c->town != town; }), about.end());
    const Town& place = society_->towns()[static_cast<std::size_t>(town)];
    std::string what = kind;
    if (what.empty()) {
        // What a place is like says what goes wrong in it.
        const double unrest = society_->state(place, "unrest");
        const double hungry = std::max(0.0, 1.0 - society_->state(place, "food"));
        const double theft = 0.3 + unrest + hungry;
        const double brawl = 0.25 + unrest;
        const double beast = 0.25 + society_->state(place, "land_grass") + 0.5 * society_->state(place, "land_forest");
        const double roll = hash01(tick_, 91) * (theft + brawl + beast);
        what = roll < theft ? "theft" : roll < theft + brawl ? "brawl" : "beast";
    }
    const auto seed = [&](std::uint64_t k) { return hash01(tick_ + k, static_cast<std::uint64_t>(town)); };
    if (what == "beast") {
        const int wolf = law.creature_id("wolf");
        if (wolf < 0 || about.empty()) {
            return false;
        }
        Creature& victim = *about[static_cast<std::size_t>(seed(1) * about.size())];
        const GridShape& g = fields().shape();
        for (int tries = 0; tries < 12; ++tries) {
            const double a = 6.283185307 * seed(2 + static_cast<std::uint64_t>(tries));
            const Vec3 top{victim.body.position.x + 13.0 * std::cos(a), victim.body.position.y + 13.0 * std::sin(a), g.nz * g.cell - 0.1};
            if (top.x < 2.0 || top.y < 2.0 || top.x > g.nx * g.cell - 2.0 || top.y > g.ny * g.cell - 2.0) {
                continue;
            }
            const std::optional<Vec3> ground = pick(top, Vec3{0.0, 0.0, -1.0});
            if (!ground || std::abs(ground->z - victim.body.position.z) > 3.0) {
                continue;
            }
            Creature beast = make_creature(wolf, *ground + Vec3{0.0, 0.0, law.creatures[static_cast<std::size_t>(wolf)].radius + 0.1});
            beast.foe = victim.id;
            beast.body.foe = victim.id;
            beast.prey_hold = 40.0;
            creatures_.push_back(std::move(beast));
            // creatures_ may have moved.
            const Creature* v = nullptr;
            for (const Creature& c : creatures_) {
                if (c.town == town && c.who.pid != 0 && c.id != creatures_.back().id && length(c.body.position - creatures_.back().body.position) < 14.0) {
                    v = &c;
                    break;
                }
            }
            std::vector<std::uint64_t> saw = v != nullptr ? witnesses_of(*v, false) : std::vector<std::uint64_t>{};
            if (v != nullptr) {
                saw.push_back(v->who.pid);
            }
            const int fact = society_->local_fact("beast_came", town, v != nullptr ? v->who.name : "", law.creatures[static_cast<std::size_t>(wolf)].name, 0.45, 0.0, saw);
                chat_line(ChatLine::Kind::Local, "", society_->facts()[static_cast<std::size_t>(fact)].text);
            return true;
        }
        return false;
    }
    if (what == "theft") {
        if (about.size() < 2) {
            return false;
        }
        Creature* thief = nullptr;
        for (Creature* c : about) {
            if (society_->goods()[static_cast<std::size_t>(c->who.good)] != "guard" && (thief == nullptr || c->who.kind < thief->who.kind)) {
                thief = c;
            }
        }
        if (thief == nullptr) {
            return false;
        }
        Creature* victim = nullptr;
        double closest = 1e9;
        for (Creature* c : about) {
            const double d = length(c->body.position - thief->body.position);
            if (c != thief && d < closest) {
                closest = d;
                victim = c;
            }
        }
        if (victim == nullptr) {
            return false;
        }
        thief->stolen_from = victim->who.pid + 1;
        thief->flee_left = 25.0;
        thief->flee_from = victim->body.position;
        thief->began_it = true;
        society_->record_of(victim->who.pid).others[thief->who.pid] = -0.8;
        bark(*victim, society_->say("robbed", victim->who, tick_));
        std::vector<std::uint64_t> saw = witnesses_of(*thief, false);
        saw.push_back(victim->who.pid);
        const int fact = society_->local_fact("theft", town, thief->who.name, "a purse from " + victim->who.name, 0.45, 0.0, saw, thief->who.pid + 1);
        chat_line(ChatLine::Kind::Local, "", society_->facts()[static_cast<std::size_t>(fact)].text);
        // Whoever is brave goes after it, and the watch has to.
        if (victim->who.bold > 0.55) {
            set_on(*victim, *thief);
        }
        for (Creature* c : about) {
            if (c != thief && c != victim && society_->goods()[static_cast<std::size_t>(c->who.good)] == "guard" && length(c->body.position - thief->body.position) < 30.0) {
                set_on(*c, *thief);
            }
        }
        return true;
    }
    if (what == "brawl") {
        Creature* a = nullptr;
        for (Creature* c : about) {
            if (society_->goods()[static_cast<std::size_t>(c->who.good)] != "guard" && (a == nullptr || c->who.temper > a->who.temper)) {
                a = c;
            }
        }
        if (a == nullptr) {
            return false;
        }
        Creature* b = nullptr;
        double closest = 25.0;
        for (Creature* c : about) {
            const double d = length(c->body.position - a->body.position);
            if (c != a && society_->goods()[static_cast<std::size_t>(c->who.good)] != "guard" && d < closest) {
                closest = d;
                b = c;
            }
        }
        if (b == nullptr) {
            return false;
        }
        for (Creature* c : {a, b}) {
            c->brawling = true;
            c->arm_before = c->arm;
            c->arm = "fists";
        }
        set_on(*a, *b);
        set_on(*b, *a);
        bark(*a, society_->say("brawl_start", a->who, tick_, "", "", b->who.name));
        std::vector<std::uint64_t> saw = witnesses_of(*a, false);
        saw.push_back(a->who.pid);
        saw.push_back(b->who.pid);
        const int fact = society_->local_fact("brawl", town, a->who.name, b->who.name, 0.3, 0.0, saw);
        chat_line(ChatLine::Kind::Local, "", society_->facts()[static_cast<std::size_t>(fact)].text);
        return true;
    }
    return false;
}

void Playground::local_life_turn()
{
    if (society_ == nullptr || !config_.local_life) {
        return;
    }
    incident_left_ -= kTick;
    if (incident_left_ <= 0.0) {
        incident_left_ = law_param(solver_.law(), "incident_every", 60.0) * (0.6 + 0.8 * hash01(tick_, 92));
        make_incident("");
    }
}

// ---- buying and selling --------------------------------------------------------------------------------------------------------

// ---- commands ----------------------------------------------------------------------------------------------------------------

} // namespace fieldengine::field

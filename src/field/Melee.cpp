#include "field/Melee.hpp"

#include "field/Playground.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

namespace {

Vec3 horizontal(Vec3 v) { return Vec3{v.x, v.y, 0.0}; }

// 0 to 1, by who it is and when: what stands in for a whim. No clocks, no generator.
double wander_unit(std::uint64_t a, std::uint64_t b)
{
    std::uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x77ull) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}

} // namespace

double move_stamina(const MoveDef& move)
{
    return 2.0 + 0.45 * move.hit + 0.35 * move.knock + 0.5 * move.lift + 0.5 * move.step * (1.0 + move.active) + 3.0 * move.breaks + 0.02 * move.arc + 2.5 * move.shock;
}

int MeleeBook::weapon_slot(const std::string& name) const
{
    for (std::size_t i = 0; i < weapons.size(); ++i) {
        if (weapons[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int MeleeBook::move_slot(const std::string& name) const
{
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (moves[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<int> MeleeBook::known(const std::string& weapon, int skill) const
{
    std::vector<int> out;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (moves[i].weapon == weapon && moves[i].mastery <= skill) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

MeleeBook parse_moves(const std::string& text, std::vector<std::string>& errors)
{
    MeleeBook book;
    std::istringstream stream(text);
    std::string line;
    int number = 0;
    while (std::getline(stream, line)) {
        ++number;
        std::istringstream in(line.substr(0, line.find('#')));
        std::vector<std::string> w;
        for (std::string word; in >> word;) {
            w.push_back(word);
        }
        if (w.empty()) {
            continue;
        }
        const std::string where = "moves.world:" + std::to_string(number) + ": ";
        if (w[0] == "weapon" && w.size() >= 2 && w.size() % 2 == 0) {
            WeaponDef weapon;
            weapon.name = w[1];
            try {
                for (std::size_t k = 2; k + 1 < w.size(); k += 2) {
                    if (w[k] == "model") {
                        weapon.model = w[k + 1];
                    } else if (w[k] == "mass") {
                        weapon.mass = std::stod(w[k + 1]);
                    } else if (w[k] == "length") {
                        weapon.length = std::stod(w[k + 1]);
                    } else if (w[k] == "wood") {
                        weapon.tool_wood = std::stod(w[k + 1]);
                    } else if (w[k] == "stone") {
                        weapon.tool_stone = std::stod(w[k + 1]);
                    } else if (w[k] == "leaf") {
                        weapon.tool_leaf = std::stod(w[k + 1]);
                    } else {
                        errors.push_back(where + "no such word: " + w[k]);
                    }
                }
            } catch (...) {
                errors.push_back(where + "a number was expected");
            }
            book.weapons.push_back(std::move(weapon));
            continue;
        }
        if (w[0] != "move" || w.size() < 2 || w.size() % 2 != 0) {
            errors.push_back(where + "expected weapon NAME [model FILE], or move NAME and then pairs of a word and its value");
            continue;
        }
        MoveDef move;
        move.name = w[1];
        bool bad = false;
        for (std::size_t k = 2; k + 1 < w.size(); k += 2) {
            const std::string& key = w[k];
            const std::string& value = w[k + 1];
            try {
                if (key == "weapon") {
                    move.weapon = value;
                } else if (key == "anim") {
                    move.anim = value;
                } else if (key == "windup") {
                    move.windup = std::stod(value);
                } else if (key == "active") {
                    move.active = std::stod(value);
                } else if (key == "walk") {
                    move.walk = std::stod(value);
                } else if (key == "rounds") {
                    move.rounds = std::stoi(value);
                } else if (key == "on") {
                    move.on_key = std::stoi(value);
                } else if (key == "recover") {
                    move.recover = std::stod(value);
                } else if (key == "cancel") {
                    move.cancel = std::stod(value);
                } else if (key == "reach") {
                    move.reach = std::stod(value);
                } else if (key == "arc") {
                    move.arc = std::stod(value);
                } else if (key == "hit") {
                    move.hit = std::stod(value);
                } else if (key == "knock") {
                    move.knock = std::stod(value);
                } else if (key == "lift") {
                    move.lift = std::stod(value);
                } else if (key == "step") {
                    move.step = std::stod(value);
                } else if (key == "breaks") {
                    move.breaks = std::stod(value);
                } else if (key == "shock") {
                    move.shock = std::stod(value);
                } else if (key == "mastery") {
                    move.mastery = std::stoi(value);
                } else {
                    errors.push_back(where + "no such word: " + key);
                    bad = true;
                }
            } catch (...) {
                errors.push_back(where + key + ": a number was expected");
                bad = true;
            }
        }
        if (book.weapon_slot(move.weapon) < 0) {
            errors.push_back(where + move.name + ": no such weapon: " + move.weapon);
            bad = true;
        }
        if (move.anim.empty()) {
            errors.push_back(where + move.name + " has no anim");
            bad = true;
        }
        if (move.windup <= 0.0 || move.active <= 0.0 || move.recover <= 0.0 || move.cancel < 0.0 || move.cancel > 1.0 || move.reach <= 0.0) {
            errors.push_back(where + move.name + ": its times, its reach or its cancel make no sense");
            bad = true;
        }
        if (book.move_slot(move.name) >= 0) {
            errors.push_back(where + "a second move called " + move.name);
            bad = true;
        }
        if (!bad) {
            move.stamina = move_stamina(move);
            book.moves.push_back(std::move(move));
        }
    }
    return book;
}

MeleeBook load_moves(std::vector<std::string>& errors)
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/world/moves.world", std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            return parse_moves(text.str(), errors);
        }
    }
    errors.push_back("assets/world/moves.world not found");
    return MeleeBook{};
}

// ---- a blow, in the world -----------------------------------------------------------------------------------

void Playground::set_melee(MeleeBook book)
{
    melee_ = std::move(book);
    player().swing = Swing{};
}

double Playground::phase_of(const PlayBody& body) const
{
    const Swing& swing = body.swing;
    if (swing.move < 0 || swing.move >= static_cast<int>(melee_.moves.size())) {
        return -1.0;
    }
    const MoveDef& move = melee_.moves[static_cast<std::size_t>(swing.move)];
    if (swing.t < move.windup) {
        return swing.t / move.windup;
    }
    if (swing.t < move.windup + move.active) {
        return 1.0 + (swing.t - move.windup) / move.active;
    }
    return std::min(3.0, 2.0 + (swing.t - move.windup - move.active) / move.recover);
}

void Playground::swing(PlayBody& body, int move, Vec3 way)
{
    if (move >= 0 && move < static_cast<int>(melee_.moves.size())) {
        const Vec3 flat = horizontal(way);
        start_swing(body, move, length(flat) > 1e-9 ? normalize(flat) : Vec3{1.0, 0.0, 0.0});
    }
}

void Playground::start_swing(PlayBody& body, int slot, Vec3 way)
{
    body.swing = Swing{};
    body.swing.move = slot;
    body.swing.dir_x = way.x;
    body.swing.dir_y = way.y;
}

// A blow on a body. Whoever turns to meet it takes less, and whoever meets it just as it comes takes none, and it is the one
// who struck who pays. Nothing here is about who is who: NPCs will fight with this too.
void Playground::land_blow(PlayBody& from, PlayBody& to, const MoveDef& move)
{
    const Vec3 o = origin();
    const Vec3 way = normalize(horizontal(to.position - from.position));
    const double power = from.swing.power * from.strength;
    double hit = move.hit * power;
    double knock = move.knock * power;
    double lift = move.lift * std::sqrt(power);
    const bool finisher = from.swing.finisher;
    if (to.guard >= 0.0) {
        const Vec3 facing{std::cos(to.guard_yaw), std::sin(to.guard_yaw), 0.0};
        // It has to come from in front.
        if (dot(facing, way) < -0.2) {
            if (to.guard < config_.parry_window) {
                from.swing = Swing{};
                from.stagger = config_.parry_stagger;
                from.velocity = from.velocity + way * -4.0;
                event(PlayEvent::Kind::Parried, -1, to.position + o, from.position + o, 1.0);
                return;
            }
            hit *= config_.guard_keeps;
            knock *= config_.guard_keeps;
            lift = 0.0;
            event(PlayEvent::Kind::Blocked, -1, to.position + o, from.position + o, hit);
        }
    }
    hit *= 100.0 / (100.0 + to.armour);
    to.hp = std::max(0.0, to.hp - hit);
    to.struck_by = &from == &player() ? 0u : from.id;
    to.velocity = to.velocity + way * knock + Vec3{0.0, 0.0, lift + (knock > 0.0 ? 0.25 * knock : 0.0)};
    if (knock > 0.0 || lift > 0.0) {
        to.grounded = false;
        to.stagger = std::max(to.stagger, 0.35 + 0.012 * knock); // the harder, the longer before its feet catch it
    }
    event(PlayEvent::Kind::MeleeHit, -1, to.position + o, from.position + o, hit);
    events_.back().hand = finisher ? 1 : 0;
    events_.back().yours = &from == &player() || &to == &player();
}

void Playground::swing_turn(PlayBody& you)
{
    Swing& swing = you.swing;
    if (swing.move < 0) {
        return;
    }
    const MoveDef move = melee_.moves[static_cast<std::size_t>(swing.move)];
    const double before = swing.t;
    swing.t += kTick * you.quick;
    const double from = move.windup;
    const double to = move.windup + move.active;
    const Vec3 o = origin();
    const Vec3 way{swing.dir_x, swing.dir_y, 0.0};
    const Vec3 chest = you.position + Vec3{0.0, 0.0, 0.25};
    if (before < from && swing.t >= from) {
        event(PlayEvent::Kind::Swing, -1, chest + o, chest + way * (you.radius + move.reach) + o, move.arc);
        // Every blow goes into it: a step of its own if it has one, and enough to reach whoever it turned to.
        double step = std::max(move.step, 1.5);
        for (PlayBody* body : all_bodies()) {
            if (body == &you || body->parked) {
                continue;
            }
            const Vec3 off = horizontal(body->position - you.position);
            const double d = length(off);
            if (d > 1e-6 && d < move.reach + 3.0 && dot(off * (1.0 / d), way) > 0.8) {
                step = std::max(step, std::min(9.0, (d - you.radius - body->radius - move.reach * 0.6) / move.active));
            }
        }
        if ((you.grounded || move.step > 0.0) && move.walk <= 0.25) {
            you.velocity = way * step;
            you.surge = move.active;
        }
        if (move.shock > 0.0) {
            const Vec3 feet = you.position - Vec3{0.0, 0.0, you.radius} + way * (you.radius + move.reach * 0.6);
            int key = 0;
            for (PlayBody* body : all_bodies()) {
                const int this_key = key++;
                if (body == &you || body->parked || (&you != &player() && (you.foe == 0 ? body != &player() : body->id != you.foe))) {
                    continue;
                }
                const Vec3 off = body->position - feet;
                const double d = length(horizontal(off));
                if (d < move.shock + body->radius && std::abs(off.z) < move.shock) {
                    const double near = 1.0 - d / (move.shock + body->radius + 0.01);
                    body->hp = std::max(0.0, body->hp - move.hit * 0.45 * near * swing.power);
                    body->struck_by = &you == &player() ? 0u : you.id;
                    body->velocity = body->velocity + (d > 1e-6 ? horizontal(off) * (1.0 / d) : way) * (move.knock * near) + Vec3{0.0, 0.0, 3.0 + 4.0 * near};
                    body->grounded = false;
                    body->stagger = std::max(body->stagger, 0.35);
                    swing.struck.push_back(this_key);
                }
            }
            const int force = solver_.field("mana_force");
            if (force >= 0 && move.breaks > 0.0) {
                Pour blow;
                blow.radius = move.shock * 0.7;
                pour_shape(force, move.breaks * 0.6, blow, feet, way);
            }
            event(PlayEvent::Kind::Shock, -1, feet + o, feet + o, move.shock);
        }
    }
    if (swing.t >= from && before < to) {
        const double half = 0.5 * move.arc * 3.14159265358979 / 180.0;
        // Across the arc as it goes: the blade passes each angle once.
        double along = std::clamp((swing.t - from) / move.active, 0.0, 1.0);
        // Round again: everyone is there to be hit again.
        if (move.rounds > 1) {
            const double rounds_in = along * move.rounds;
            const int round = std::min(static_cast<int>(rounds_in), move.rounds - 1);
            if (round != swing.round) {
                swing.round = round;
                swing.struck.clear();
                swing.landed_world = false;
            }
            along = rounds_in - round;
        }
        const double angle = half * (1.0 - 2.0 * along);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const Vec3 tip_way{way.x * c - way.y * s, way.x * s + way.y * c, 0.0};
        const double tip = you.radius + move.reach;
        // Who is in reach, in front, on the same level, and not behind a wall.
        int key = 0;
        for (PlayBody* body : all_bodies()) {
            const int this_key = key++;
            // Who is hit by whom: you, anyone; anyone else, only the one it is fighting.
            if (body == &you || body->parked || (&you != &player() && (you.foe == 0 ? body != &player() : body->id != you.foe)) || std::find(swing.struck.begin(), swing.struck.end(), this_key) != swing.struck.end()) {
                continue;
            }
            const Vec3 off = body->position - you.position;
            const Vec3 flat = horizontal(off);
            const double d = length(flat);
            if (d > tip + body->radius || std::abs(off.z) > 1.4 + body->radius) {
                continue;
            }
            if (d > 1e-6 && dot(flat * (1.0 / d), way) < std::cos(std::min(half + 0.15, 3.14159265358979))) {
                continue;
            }
            const Vec3 target = body->position;
            // Any of it will do, its middle or its head: on a step, the edge of the step is not a wall.
            bool reachable = false;
            for (const double up : {0.0, 0.7}) {
                const Vec3 line = target + Vec3{0.0, 0.0, up} - chest;
                const double clear = length(line);
                const std::optional<Vec3> wall = clear > 1e-6 ? pick(chest, line * (1.0 / clear)) : std::nullopt;
                reachable = reachable || !wall || length(*wall - chest) >= clear - body->radius;
            }
            if (!reachable) {
                continue;
            }
            swing.struck.push_back(this_key);
            land_blow(you, *body, move);
            if (you.swing.move < 0) {
                return; // parried
            }
        }
        // And the world, once: it takes what the blow leaves.
        if (!swing.landed_world) {
            const std::optional<Vec3> wall = pick(chest, tip_way);
            if (wall && length(*wall - chest) <= tip) {
                swing.landed_world = true;
                const int force = solver_.field("mana_force");
                // A heavy blow leaves force where it strikes the world, and the law breaks what that can break.
                if (move.breaks > 0.0 && force >= 0) {
                    Pour blow;
                    blow.radius = 0.9;
                    pour_shape(force, move.breaks, blow, *wall, tip_way);
                }
                event(PlayEvent::Kind::Clang, -1, *wall + o, chest + o, move.breaks);
            }
        }
    }
    if (swing.t >= move.length()) {
        swing = Swing{};
    }
}

// Whom it is fighting. A beast picks: the nearest of what it goes for, and it does not change its mind every moment.
PlayBody* Playground::foe_of(Creature& c, const CreatureDef& def)
{
    if (c.town < 0 && !def.prey.empty() && c.prey_hold <= 0.0 && (tick_ + c.id) % 30u == 0u) {
        const double far_off = def.sense * 8.0;
        double best = far_off;
        std::uint32_t chosen = 0;
        bool any = false;
        if (std::find(def.prey.begin(), def.prey.end(), "player") != def.prey.end()) {
            const double d = length(horizontal(player().position - c.body.position));
            if (d < best) {
                best = d;
                any = true;
            }
        }
        const Law& law = solver_.law();
        for (const Creature& other : creatures_) {
            if (other.id == c.id) {
                continue;
            }
            const std::string& name = law.creatures[static_cast<std::size_t>(other.def)].name;
            if (std::find(def.prey.begin(), def.prey.end(), name) == def.prey.end()) {
                continue;
            }
            const double d = length(horizontal(other.body.position - c.body.position));
            if (d < best) {
                best = d;
                chosen = other.id;
                any = true;
            }
        }
        c.foe = any ? chosen : c.id; // itself: nobody
    }
    if (c.foe == c.id) {
        return nullptr;
    }
    if (c.foe != 0) {
        Creature* other = creature_mut(c.foe);
        if (other == nullptr || other->body.hp <= 0.0) {
            c.foe = c.town < 0 && !def.prey.empty() ? c.id : 0;
            c.prey_hold = 0.0;
            return c.town < 0 ? (c.foe == 0 ? &player() : nullptr) : nullptr;
        }
        return &other->body;
    }
    return &player();
}

// A creature that fights, this tick. It does not follow a script: every so often it looks at how things stand and does
// what is worth most, by what it can see the other doing, by what it knows, and by who it is. The weights are the law's.
//   strike a move    what the move does a second, more if the other is open, less if something is coming and it is slow
//   guard            something is coming and it has seen it; the more it knows, the sooner it sees and the closer it cuts it
//   back away        hurt, and timid
//   close, circle    out of reach; in reach with nothing better
// It has the same moves, and goes through the same doors, as you do. False: it has nothing to say about where it walks.
bool Playground::fight_turn(Creature& c, const CreatureDef& def, Vec3& want)
{
    if (def.fights.empty() || melee_.moves.empty()) {
        return false;
    }
    PlayBody& body = c.body;
    body.surge = std::max(0.0, body.surge - kTick); // only your own runs out in move_player
    swing_turn(body);
    c.fight_wait = std::max(0.0, c.fight_wait - kTick);
    c.guard_wait = std::max(0.0, c.guard_wait - kTick);
    c.prey_hold = std::max(0.0, c.prey_hold - kTick);
    PlayBody* other = foe_of(c, def);
    body.foe = c.foe;
    if (other == nullptr) {
        body.guard = -1.0;
        return false;
    }
    const PlayBody& you = *other;
    const Vec3 off = horizontal(you.position - body.position);
    const double d = length(off);
    if (d > def.sense * 8.0 || std::abs(you.position.z - body.position.z) > 3.0 || d < 1e-6) {
        return false;
    }
    const Vec3 way = off * (1.0 / d);
    c.face = std::atan2(way.y, way.x);
    std::vector<int> known = melee_.known(c.arm.empty() ? def.fights : c.arm, def.skill);
    if (known.empty()) {
        known = melee_.known(def.fights, def.skill);
    }
    if (known.empty()) {
        return false;
    }
    const Law& law = solver_.law();
    const auto weight = [&](const char* name, double fallback) {
        const int id = law.param_id(name);
        return id >= 0 ? law.param_values[static_cast<std::size_t>(id)] : fallback;
    };
    const double bold = c.town >= 0 ? c.who.bold : 0.3 + 0.5 * wander_unit(c.id, 1);
    const double temper = c.town >= 0 ? c.who.temper : 0.3 + 0.5 * wander_unit(c.id, 2);
    const double skill = def.skill;
    // What it can see of the other.
    const MoveDef* theirs = you.swing.move >= 0 ? &melee_.moves[static_cast<std::size_t>(you.swing.move)] : nullptr;
    const double seen_after = weight("fight_sees", 0.09) / (1.0 + skill);
    const bool coming = theirs != nullptr && you.swing.t < theirs->windup + theirs->active && you.swing.t >= seen_after && d < body.radius + you.radius + theirs->reach + 1.5;
    const double to_land = theirs != nullptr ? theirs->windup - you.swing.t : 1.0;
    const bool open = (theirs != nullptr && you.swing.t >= theirs->windup + theirs->active) || you.stagger > 0.0;
    c.plan_left -= kTick;
    // Something coming is worth looking up for: it does not wait for its next thought.
    if (coming && c.plan != Creature::Plan::Guard && c.guard_wait <= 0.0) {
        c.plan_left = std::min(c.plan_left, 0.03);
    }
    if (c.plan_left <= 0.0 && body.swing.move < 0) {
        c.plan_left = 0.12 + 0.12 * wander_unit(c.id, tick_);
        const double whim = 0.2 * wander_unit(c.id + 3u, tick_);
        double best = (0.25 + 0.3 * (1.0 - temper)) * weight("fight_circle", 1.0) + 0.5 * whim;
        c.plan = Creature::Plan::Circle;
        c.plan_move = -1;
        const double hurt = 1.0 - body.hp / std::max(body.hp_max, 1.0);
        // Hurt and timid, it keeps out of reach: not out of the fight, or it would only ever be chased.
        const double back = weight("fight_back", 1.5) * hurt * hurt * (1.0 - bold) * (d < body.radius + you.radius + 2.5 ? 1.0 : 0.0);
        if (back > best) {
            best = back;
            c.plan = Creature::Plan::Back;
        }
        double reach_needed = 0.0;
        for (const int slot : known) {
            const MoveDef& m = melee_.moves[static_cast<std::size_t>(slot)];
            const double in_reach = body.radius + m.reach * 0.9 + you.radius;
            reach_needed = reach_needed == 0.0 ? in_reach : std::min(reach_needed, in_reach);
            if (d > in_reach || c.fight_wait > 0.0 || body.stagger > 0.0) {
                continue;
            }
            double worth = weight("fight_strike", 1.0) * (0.5 + m.hit / m.length() / 90.0 + 0.6 * temper) + whim;
            worth += open ? weight("fight_open", 1.0) * (0.2 + 0.25 * skill) * (0.25 / m.windup) : 0.0;
            worth -= coming ? weight("fight_risk", 1.5) * (1.2 - bold) * m.windup * 4.0 : 0.0;
            worth += you.guard >= 0.0 ? 0.3 * m.knock / 15.0 : 0.0;
            if (worth > best) {
                best = worth;
                c.plan = Creature::Plan::Strike;
                c.plan_move = slot;
            }
        }
        if (coming && c.guard_wait <= 0.0) {
            const double guard = weight("fight_guard", 1.3) * (0.45 + 0.35 * skill) * (1.2 - 0.5 * temper) + whim;
            if (guard > best) {
                best = guard;
                c.plan = Creature::Plan::Guard;
                c.guard_wait = 1.1 - 0.2 * skill + 0.5 * wander_unit(c.id + 5u, tick_);
            }
        }
        if (d > reach_needed && c.plan == Creature::Plan::Circle) {
            c.plan = Creature::Plan::Close;
        }
    }
    // A guard goes up when it is worth it: who knows more cuts it closer, and that is a parry.
    const bool cut_close = def.skill >= 2;
    if (c.plan == Creature::Plan::Guard && coming && body.swing.move < 0 && (!cut_close || to_land < config_.parry_window * 0.8)) {
        body.guard = body.guard < 0.0 ? (cut_close ? 0.0 : 0.25) : body.guard + kTick; // only who knows how meets it in time to turn it back
        body.guard_yaw = std::atan2(way.y, way.x);
    } else if (c.plan != Creature::Plan::Guard || !coming) {
        body.guard = -1.0;
        if (c.plan == Creature::Plan::Guard) {
            c.plan_left = 0.0;
        }
    }
    if (c.plan == Creature::Plan::Strike && c.plan_move >= 0 && body.swing.move < 0 && body.guard < 0.0 && body.stagger <= 0.0 && c.fight_wait <= 0.0) {
        start_swing(body, c.plan_move, way);
        c.fight_wait = (1.0 - 0.6 * temper) * (1.3 - 0.15 * skill) + 0.25 * wander_unit(c.id, tick_);
        c.plan_left = 0.0;
        c.plan = Creature::Plan::Circle;
    }
    if (body.swing.move >= 0) {
        // Most blows take all of you. One that keeps you moving keeps you coming.
        const double walk = melee_.moves[static_cast<std::size_t>(body.swing.move)].walk;
        want = walk > 0.25 ? way * (def.speed * walk) : Vec3{};
        return true;
    }
    const Vec3 side{-way.y, way.x, 0.0};
    const double circling = (c.id & 1u) != 0u ? 0.35 : -0.35;
    switch (c.plan) {
    case Creature::Plan::Back:
        want = (way * -1.0 + side * circling) * (def.speed * 0.8);
        break;
    case Creature::Plan::Close:
    case Creature::Plan::Strike:
        want = (way + side * circling) * def.speed;
        break;
    case Creature::Plan::Guard:
        want = Vec3{};
        break;
    case Creature::Plan::Circle:
    default:
        want = side * (circling * def.speed) + way * ((d > body.radius + you.radius + 1.6 ? 0.5 : 0.0) * def.speed);
        break;
    }
    return true;
}

} // namespace fieldengine::field

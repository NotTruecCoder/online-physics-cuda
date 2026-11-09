#include "field/Playground.hpp"

#include <algorithm>
#include <cmath>
#include <set>

// THE PEOPLE AROUND YOU, IN BODIES. A town near the window sends out a few of its people; going out of it they are numbers
// again (Society.hpp). What a body does is read from its anger and fear, which are lines of the law (assets/law/people.law):
// nothing here decides that anybody is angry.
namespace fieldengine::field {

namespace {

constexpr double kSeeFar = 24.0;   // metres, for it to mind you
constexpr double kBlowFar = 28.0;  // to have seen a blow

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

int local_of(const CreatureDef& def, const char* name)
{
    const auto it = std::find(def.locals.begin(), def.locals.end(), name);
    return it == def.locals.end() ? -1 : static_cast<int>(it - def.locals.begin());
}

double read_local(const CreatureDef& def, const Creature& c, const char* name)
{
    const int id = local_of(def, name);
    return id >= 0 ? c.locals[static_cast<std::size_t>(id)] : 0.0;
}

void write_local(const CreatureDef& def, Creature& c, const char* name, double value)
{
    const int id = local_of(def, name);
    if (id >= 0) {
        c.locals[static_cast<std::size_t>(id)] = std::clamp(value, def.lo[static_cast<std::size_t>(id)], def.hi[static_cast<std::size_t>(id)]);
    }
}

double law_param(const Law& law, const char* name, double fallback)
{
    const int id = law.param_id(name);
    return id >= 0 ? law.param_values[static_cast<std::size_t>(id)] : fallback;
}

} // namespace

void Playground::start_people()
{
    if (!config_.people || !config_.open_world || map_ == nullptr || solver_.law().creature_id("town") < 0) {
        return;
    }
    society_ = std::make_shared<Society>(*map_, solver_.law(), Society::read_words());
}

const Creature* Playground::creature_with(std::uint32_t id) const
{
    for (const Creature& c : creatures_) {
        if (c.id == id) {
            return &c;
        }
    }
    return nullptr;
}

const char* Playground::doing_words(const Creature& c) const
{
    switch (c.doing) {
    case Creature::Doing::Working: return "at work";
    case Creature::Doing::Market: return "at the market";
    case Creature::Doing::Home: return "at home";
    case Creature::Doing::Patrol: return "keeping watch";
    case Creature::Doing::Talking: return "talking to you";
    case Creature::Doing::Looking: return "looking at you";
    case Creature::Doing::Fleeing: return "running from you";
    case Creature::Doing::Fighting: return "coming for you";
    }
    return "";
}

void Playground::bark(const Creature& c, const std::string& text, bool to_chat)
{
    if (text.empty()) {
        return;
    }
    if (to_chat && length(c.body.position - player().position) < 16.0) {
        chat_line(ChatLine::Kind::Person, c.body.name, text);
    }
    for (Bark& b : barks_) {
        if (b.id == c.id) {
            b.text = text;
            b.left = 3.5;
            return;
        }
    }
    barks_.push_back(Bark{c.id, text, 3.5});
}

bool Playground::blocked_at(Vec3 p) const
{
    return occ_ >= 0 && sample(occ_, p) > 0.5;
}

Vec3 Playground::door_of(const Person& who, const Settlement& s) const
{
    const TerrainConfig& t = config_.terrain;
    std::vector<const Building*> homes;
    for (const Building& b : s.buildings) {
        if (b.has_rooms()) {
            homes.push_back(&b);
        }
    }
    if (homes.empty()) {
        const double a = 6.283185307 * hash01(who.pid, 41);
        return Vec3{(s.x + std::cos(a) * s.radius * 0.5) * t.cell, (s.y + std::sin(a) * s.radius * 0.5) * t.cell, (s.ground + 1) * t.cell};
    }
    const Building& b = *homes[static_cast<std::size_t>(who.house) % homes.size()];
    int x = b.door;
    int y = b.door;
    switch (b.facing) {
    case 0: x = b.x1 + 2; break;
    case 1: x = b.x0 - 2; break;
    case 2: y = b.y1 + 2; break;
    default: y = b.y0 - 2; break;
    }
    return Vec3{(x + 0.5) * t.cell, (y + 0.5) * t.cell, (b.ground + 1) * t.cell};
}

bool Playground::bring_person(int town, int index)
{
    const Person who = society_->person(town, index);
    const Law& law = solver_.law();
    const bool watches = society_->goods()[static_cast<std::size_t>(who.good)] == "guard";
    const int def_id = law.creature_id(watches ? "watchman" : "villager");
    if (def_id < 0) {
        return false;
    }
    const Settlement& s = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(town)].settlement)];
    // Where its day has it, if that is in the window: someone walking in from beyond it goes on from where it was. Else its door.
    Vec3 at = door_of(who, s) - origin();
    {
        Vec3 place{};
        double yaw = 0.0;
        double speed = 0.0;
        Creature::Doing doing = Creature::Doing::Working;
        const GridShape& lattice = fields().shape();
        if (day_place(who, town, world_seconds(), place, yaw, speed, doing)) {
            const Vec3 l = place - origin();
            if (l.x > 1.0 && l.y > 1.0 && l.x < lattice.nx * lattice.cell - 1.0 && l.y < lattice.ny * lattice.cell - 1.0 &&
                !blocked_at(l + Vec3{0.0, 0.0, 0.4})) {
                at = l;
            }
        }
    }
    const GridShape& g = fields().shape();
    if (at.x < 1.0 || at.y < 1.0 || at.x > g.nx * g.cell - 1.0 || at.y > g.ny * g.cell - 1.0 || at.z < 0.5 || at.z > g.nz * g.cell - 2.0) {
        return false;
    }
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(def_id)];
    // The ground may have been changed since the map was drawn: the first open place going up from where it was.
    Vec3 where = at + Vec3{0.0, 0.0, def.radius + 0.15};
    bool open = false;
    for (double up = 0.0; up < 6.0 && !open; up += 0.25) {
        const Vec3 tried = where + Vec3{0.0, 0.0, up};
        if (!blocked_at(tried) && !blocked_at(tried + Vec3{0.0, 0.0, 0.8})) {
            where = tried;
            open = true;
        }
    }
    if (!open) {
        return false;
    }
    Creature c = make_creature(def_id, where);
    c.town = town;
    c.who = who;
    c.body.name = who.name;
    c.last_hp = c.body.hp;
    c.face = c.heading;
    c.turning = hash01(who.pid, 51) < 0.5 ? 1 : -1;
    if (const Record* r = society_->record(who.pid); r != nullptr && r->hp >= 0.0) {
        c.body.hp = std::clamp(r->hp, 1.0, c.body.hp_max);
        c.last_hp = c.body.hp;
    }
    write_local(def, c, "bold", who.bold);
    write_local(def, c, "temper", who.temper);
    write_local(def, c, "kind", who.kind);
    dress_person(c, s);
    c.opinion = society_->opinion(who);
    creatures_.push_back(std::move(c));
    return true;
}

// A caravan that has come near you is someone on a road, with a pack and what it carries; gone from the window, it is numbers again.
void Playground::caravans_near()
{
    const std::vector<Caravan>& all = society_->caravans();
    std::vector<char> walking(all.size(), 0);
    for (std::size_t k = 0; k < creatures_.size();) {
        Creature& c = creatures_[k];
        if (c.caravan < 0) {
            ++k;
            continue;
        }
        const Caravan& road = all[static_cast<std::size_t>(c.caravan)];
        if (road.next < 0 || road.gone || length(c.body.position - player().position) > 50.0) {
            creatures_.erase(creatures_.begin() + static_cast<std::ptrdiff_t>(k));
            continue;
        }
        walking[static_cast<std::size_t>(c.caravan)] = 1;
        ++k;
    }
    const Vec3 me = player().position + origin();
    const double cell = config_.terrain.cell;
    for (std::size_t i = 0; i < all.size(); ++i) {
        society_->caravan(static_cast<int>(i)).body = walking[i] != 0;
        if (walking[i] == 0 && !all[i].gone && all[i].next >= 0 && std::hypot(all[i].x * cell - me.x, all[i].y * cell - me.y) < 40.0) {
            bring_caravan(static_cast<int>(i));
        }
    }
}

bool Playground::bring_caravan(int i)
{
    Caravan& road = society_->caravan(i);
    const Law& law = solver_.law();
    const int def_id = law.creature_id("villager");
    if (def_id < 0) {
        return false;
    }
    const double cell = config_.terrain.cell;
    const Vec3 o = origin();
    const GridShape& g = fields().shape();
    const Vec3 top{road.x * cell - o.x, road.y * cell - o.y, g.nz * g.cell - 0.1};
    if (top.x < 1.0 || top.y < 1.0 || top.x > g.nx * g.cell - 1.0 || top.y > g.ny * g.cell - 1.0) {
        return false;
    }
    const std::optional<Vec3> ground = pick(top, Vec3{0.0, 0.0, -1.0});
    if (!ground) {
        return false;
    }
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(def_id)];
    Creature c = make_creature(def_id, *ground + Vec3{0.0, 0.0, def.radius + 0.15});
    c.town = road.home;
    c.who = society_->person(road.home, road.index);
    c.body.name = c.who.name;
    c.last_hp = c.body.hp;
    c.caravan = i;
    c.turning = hash01(c.who.pid, 51) < 0.5 ? 1 : -1;
    const Settlement& s = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(road.home)].settlement)];
    dress_person(c, s);
    // On the road: a pack on its back.
    if (const int pack = items_.find("pack"); pack >= 0) {
        c.bag.worn[kWornBack] = pack;
    }
    c.opinion = society_->opinion(c.who);
    creatures_.push_back(std::move(c));
    road.body = true;
    return true;
}

// Who should be standing around here: of the towns the window holds, the nearest first.
void Playground::bring_people()
{
    const TerrainConfig& t = config_.terrain;
    const Vec3 o = origin();
    const GridShape& g = fields().shape();
    const std::vector<Town>& towns = society_->towns();
    const Vec3 me = player().position + o;
    std::vector<std::pair<double, int>> near_towns;
    for (std::size_t i = 0; i < towns.size(); ++i) {
        const Settlement& s = map_->settlements()[static_cast<std::size_t>(towns[i].settlement)];
        const double x = s.x * t.cell - o.x;
        const double y = s.y * t.cell - o.y;
        const double r = s.radius * t.cell;
        if (x + r > 0.0 && y + r > 0.0 && x - r < g.nx * g.cell && y - r < g.ny * g.cell) {
            near_towns.push_back({std::hypot(s.x * t.cell - me.x, s.y * t.cell - me.y), static_cast<int>(i)});
        }
    }
    std::sort(near_towns.begin(), near_towns.end());
    std::set<std::uint64_t> here;
    int bodies = 0;
    for (const Creature& c : creatures_) {
        if (c.town >= 0) {
            here.insert(c.who.pid);
            ++bodies;
        }
    }
    for (const auto& [away, town] : near_towns) {
        (void)away;
        const Town& place = towns[static_cast<std::size_t>(town)];
        int count = 0;
        int guards = 0;
        for (const Creature& c : creatures_) {
            if (c.town == town) {
                ++count;
                guards += c.def == solver_.law().creature_id("watchman") ? 1 : 0;
            }
        }
        const int people = society_->people_in(place);
        const int wanted = std::min(place.wanted, people);
        // The watch come out first, as many as a town of this size has about it.
        const int watch = std::min(std::max(1, wanted / 4), society_->guards_in(place));
        const Settlement& where = map_->settlements()[static_cast<std::size_t>(place.settlement)];
        // A person's door has to be in the window: people first by that, which is cheap, then by who they are.
        const auto door_here = [&](int index) {
            Person cheap;
            cheap.pid = index;
            cheap.house = index / 5;
            const Vec3 at = door_of(cheap, where) - o;
            return at.x > 1.0 && at.y > 1.0 && at.x < g.nx * g.cell - 1.0 && at.y < g.ny * g.cell - 1.0 && at.z > 0.5 && at.z < g.nz * g.cell - 2.0;
        };
        for (int pass = 0; pass < 2; ++pass) {
            for (int index = 0; index < people && (pass == 1 || count < wanted) && bodies < config_.people_most; ++index) {
                if (pass == 0 && guards >= watch) {
                    break;
                }
                // The watch come out first; after them, who is out of doors today (as beyond the window).
                if (pass == 1 && !out_today(town, index, world_seconds())) {
                    continue;
                }
                if (pass == 0 && !door_here(index)) {
                    index += 4 - index % 5; // nobody of this house
                    continue;
                }
                const Person who = society_->person(town, index);
                const bool is_guard = society_->goods()[static_cast<std::size_t>(who.good)] == "guard";
                if ((pass == 0 && !is_guard) || here.count(who.pid) != 0 || society_->dead(who.pid) || society_->on_the_road(who.pid)) {
                    continue;
                }
                if (bring_person(town, index)) {
                    here.insert(who.pid);
                    ++count;
                    ++bodies;
                    guards += is_guard ? 1 : 0;
                    if (pass == 0) {
                        index += 4 - index % 5; // one of a house at a time: a street is not one family
                    }
                }
            }
        }
    }
}

void Playground::people_turn()
{
    if (society_ == nullptr) {
        return;
    }
    society_->use(solver_.law());
    society_->tick(kTick);
    const PlayBody& you = player();
    for (Bark& b : barks_) {
        b.left -= kTick;
    }
    barks_.erase(std::remove_if(barks_.begin(), barks_.end(), [](const Bark& b) { return b.left <= 0.0; }), barks_.end());
    if (tick_ % 60u == 13u) {
        bring_people();
        caravans_near();
    }
    // Who is walking says where a caravan is.
    for (Creature& c : creatures_) {
        if (c.caravan >= 0) {
            const Vec3 at = c.body.position + origin();
            Caravan& road = society_->caravan(c.caravan);
            road.x = at.x / config_.terrain.cell;
            road.y = at.y / config_.terrain.cell;
        }
    }
    // What is happening with the goods, if it is happening near you.
    const Vec3 me = you.position + origin();
    for (const MarketNews& news : society_->take_news()) {
        const Settlement& s = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(news.town)].settlement)];
        if (std::hypot(s.x * config_.terrain.cell - me.x, s.y * config_.terrain.cell - me.y) < 150.0) {
            chat_line(ChatLine::Kind::Local, "", news.text);
        }
    }
    if (tick_ % 150u == 77u) {
        chatter();
    }
}

// What it reads each tick, in the numbers its lines take.
void Playground::person_mind(Creature& c)
{
    const CreatureDef& def = solver_.law().creatures[static_cast<std::size_t>(c.def)];
    const PlayBody& you = player();
    const Vec3 off = you.position - c.body.position;
    const double d = length(off);
    if ((tick_ + c.id) % 12u == 0u) {
        c.sees_you = false;
        if (!config_.observer && d < kSeeFar) { // an observer is nobody: nothing sees it
            const Vec3 eye = c.body.position + Vec3{0.0, 0.0, 0.4};
            const Vec3 line = you.position - eye;
            const double clear = length(line);
            const std::optional<Vec3> wall = clear > 1e-6 ? pick(eye, line * (1.0 / clear)) : std::nullopt;
            c.sees_you = !wall || length(*wall - eye) >= clear - you.radius;
        }
    }
    if ((tick_ + c.id) % 60u == 7u) {
        c.opinion = society_->opinion(c.who);
    }
    write_local(def, c, "you_near", c.sees_you ? std::clamp(1.0 - d / kSeeFar, 0.0, 1.0) : 0.0);
    write_local(def, c, "standing", society_->state(society_->towns()[static_cast<std::size_t>(c.town)], "standing"));
    write_local(def, c, "own", c.opinion);
    c.deed_wait = std::max(0.0, c.deed_wait - kTick);
    c.bark_wait = std::max(0.0, c.bark_wait - kTick);
    c.look_wait = std::max(0.0, c.look_wait - kTick);
    c.flee_left = std::max(0.0, c.flee_left - kTick);
    if (c.body.hp < c.last_hp - 0.01) {
        if (c.body.struck_by != 0) {
            // Not you: someone here, or something. It answers, or it runs; and the watch, if there is one by, comes.
            if (const Creature* by = creature_with(c.body.struck_by); by != nullptr && c.foe == 0) {
                if (def.name == "watchman" || c.who.bold > 0.5) {
                    set_on(c, *by);
                } else if (c.flee_left <= 0.0) {
                    c.flee_left = 6.0;
                    c.flee_from = by->body.position;
                    bark(c, society_->say("afraid", c.who, tick_ + c.id));
                }
                for (Creature& other : creatures_) {
                    if (other.town == c.town && other.foe == 0 && other.id != by->id && solver_.law().creatures[static_cast<std::size_t>(other.def)].name == "watchman" && length(other.body.position - c.body.position) < 25.0) {
                        set_on(other, *by);
                    }
                }
            }
        }
    }
    if (c.brawling) {
        const Creature* other = creature_with(c.foe);
        if (other == nullptr || c.body.hp < c.body.hp_max * 0.45 || other->body.hp < other->body.hp_max * 0.45) {
            end_brawl(c);
        }
    }
    c.last_hp = c.body.hp;
}

std::vector<std::uint64_t> Playground::witnesses_of(const Creature& c, bool blames_you)
{
    std::vector<std::uint64_t> out;
    const Law& law = solver_.law();
    for (Creature& other : creatures_) {
        if (other.town < 0 || other.id == c.id) {
            continue;
        }
        const Vec3 eye = other.body.position + Vec3{0.0, 0.0, 0.4};
        const Vec3 line = c.body.position - eye;
        const double clear = length(line);
        if (clear > kBlowFar || clear < 1e-6) {
            continue;
        }
        const std::optional<Vec3> wall = pick(eye, line * (1.0 / clear));
        if (wall && length(*wall - eye) < clear - c.body.radius) {
            continue;
        }
        out.push_back(other.who.pid);
        if (!blames_you) {
            continue;
        }
        const CreatureDef& def = law.creatures[static_cast<std::size_t>(other.def)];
        write_local(def, other, "blood", read_local(def, other, "blood") + 0.6);
        if (other.bark_wait <= 0.0) {
            const bool watches = def.name == "watchman";
            bark(other, society_->say(watches ? "angry" : "saw_blood", other.who, tick_ + other.id));
            other.bark_wait = 4.0;
        }
    }
    return out;
}

// What they wear comes from how warm it is there, what they do, and who they are.
void Playground::outfit(const Person& who, int town, Looks& looks, Inventory& worn, std::string& arm) const
{
    const Settlement& s = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(town)].settlement)];
    const double temperature = map_->temperature_at(s.x, s.y);
    const std::uint64_t family = (who.pid & ~static_cast<std::uint64_t>(0xFFFFFF)) ^ static_cast<std::uint64_t>(who.house) * 7919u;
    looks = looks_of_person(who.pid, family, s.culture, temperature);
    const std::string& good = society_->goods()[static_cast<std::size_t>(who.good)];
    const std::vector<ItemDef>& book = items_.items;
    const auto wanted_warmth = temperature < 2.0 ? 1.0 : temperature > 24.0 ? -1.0 : 0.0;
    const auto wear = [&](int slot, const std::vector<std::string>& names) {
        // The first of these that suits the weather, a little by chance.
        const std::size_t start = static_cast<std::size_t>(hash01(who.pid, 71 + static_cast<std::uint64_t>(slot)) * names.size());
        int best = -1;
        double closest = 1e9;
        for (std::size_t k = 0; k < names.size(); ++k) {
            const int item = items_.find(names[(start + k) % names.size()]);
            if (item < 0) {
                continue;
            }
            const double d = std::abs(book[static_cast<std::size_t>(item)].warmth - wanted_warmth);
            if (d < closest - 0.01) {
                closest = d;
                best = item;
            }
        }
        if (best >= 0) {
            worn.worn[static_cast<std::size_t>(slot)] = best;
        }
    };
    if (good == "guard") {
        wear(kWornBody, {"cuirass"});
        wear(kWornLegs, {"greaves"});
        wear(kWornHead, {"helmet"});
        wear(kWornFeet, {"boots"});
        if (hash01(who.pid, 73) < 0.5) {
            wear(kWornBack, {"cloak"});
        }
    } else {
        if (looks.sex == 0 && wanted_warmth < 1.0 && hash01(who.pid, 74) < 0.6) {
            wear(kWornBody, {"linen_dress"});
        } else {
            wear(kWornBody, {"linen_shirt", "wool_tunic", "light_robe", "fur_coat"});
            wear(kWornLegs, {"trousers"});
        }
        wear(kWornFeet, {"boots", "sandals"});
        if (wanted_warmth > 0.5) {
            wear(kWornHead, {"fur_hood"});
            if (hash01(who.pid, 75) < 0.5) {
                wear(kWornBack, {"cloak"});
            }
        } else if (good == "grain" && hash01(who.pid, 76) < 0.6) {
            wear(kWornHead, {"straw_hat"});
        }
    }
    // What their work puts in their hands.
    for (std::size_t i = 0; i < items_.items.size(); ++i) {
        const ItemDef& def = items_.items[i];
        if (def.for_job == society_->goods()[static_cast<std::size_t>(who.good)] && def.kind == ItemKind::Weapon && worn.worn[kWornHand] < 0) {
            worn.worn[kWornHand] = static_cast<int>(i);
            arm = def.weapon;
        }
    }
}

void Playground::dress_person(Creature& c, const Settlement& s)
{
    (void)s;
    outfit(c.who, c.town, c.looks, c.bag, c.arm);
    c.body.radius = std::clamp(c.body.radius * (0.6 + 0.4 * c.looks.size), 0.3, 0.45);
    double armour = 0.0;
    for (const int item : c.bag.worn) {
        armour += item >= 0 ? items_.items[static_cast<std::size_t>(item)].armour : 0.0;
    }
    c.body.armour = armour;
}

void Playground::person_died(const Creature& c)
{
    if (c.caravan >= 0) {
        // What it was carrying never gets there, and that gets about.
        society_->caravan_lost(c.caravan);
        const int people = std::max(1, society_->people_in(society_->towns()[static_cast<std::size_t>(c.town)]));
        society_->deed("caravan_lost", c.town, c.who.name, 0.5, 0.0, witnesses_of(c, false), std::min(1.0, 3.0 / people + 0.1), c.who.pid + 1);
    }
    society_->record_of(c.who.pid).dead = true;
    society_->lost(c.town);
    // The world did it (fire, a fall, a stone from the sky) and nobody tells it as a deed; or someone did, and they will.
    if (c.body.struck_by != 0) {
        std::vector<std::uint64_t> seen = witnesses_of(c, false);
        const Creature* by = creature_with(c.body.struck_by);
        const int fact = society_->local_fact(by != nullptr && by->town >= 0 ? "killed_by" : "killed_by_beast", c.town, c.who.name, by != nullptr ? by->body.name : "", 0.5, 0.0, seen);
        chat_line(ChatLine::Kind::Local, "", society_->facts()[static_cast<std::size_t>(fact)].text);
    }
}

void Playground::beast_died(const Creature& c)
{
    const CreatureDef& def = solver_.law().creatures[static_cast<std::size_t>(c.def)];
    // What starves or burns is nobody's deed; a beast someone brought down near a town is told there.
    if (society_ == nullptr || c.body.struck_by == 0 || (def.fights.empty() && def.name != "hunter")) {
        return;
    }
    const TerrainConfig& t = config_.terrain;
    const Vec3 at = c.body.position + origin();
    const int wx = static_cast<int>(std::floor(at.x / t.cell));
    const int wy = static_cast<int>(std::floor(at.y / t.cell));
    const Settlement* s = map_->nearest_settlement(wx, wy);
    if (s == nullptr || std::hypot(static_cast<double>(wx - s->x), static_cast<double>(wy - s->y)) > s->radius * 3.0) {
        return;
    }
    const int town = society_->town_at(*s);
    if (town >= 0) {
        // Near people, someone is glad of it; and glad enough to pay, if it was after them.
        Creature* glad = nullptr;
        double closest = 30.0;
        for (Creature& other : creatures_) {
            const double d = length(other.body.position - c.body.position);
            if (other.town == town && !other.hostile && d < closest && other.id != c.id) {
                closest = d;
                glad = &other;
            }
        }
        if (glad != nullptr && !def.prey.empty()) {
            bark(*glad, society_->say("glad", glad->who, tick_ + glad->id));
        }
        society_->deed("beast", town, "", 0.4, 0.5, witnesses_of(c, false), 0.3, 0, def.name);
    }
}

// Two who stand together say something. If one knows what the other does not, now it does: this is what the wave is, close up.
void Playground::chatter()
{
    for (Creature& a : creatures_) {
        if (a.town < 0 || a.hostile || a.bark_wait > 0.0 || a.doing == Creature::Doing::Fleeing || a.doing == Creature::Doing::Talking) {
            continue;
        }
        for (Creature& b : creatures_) {
            if (b.town != a.town || b.id == a.id || b.hostile || length(b.body.position - a.body.position) > 3.5) {
                continue;
            }
            const std::vector<int> knows = society_->known(a.who.pid, a.town);
            int to_tell = -1;
            for (const int f : knows) {
                if (!society_->knows(b.who.pid, b.town, f)) {
                    to_tell = f;
                    break;
                }
            }
            if (to_tell < 0 && !knows.empty() && (tick_ + a.id) % 5u == 0u) {
                to_tell = knows[static_cast<std::size_t>((tick_ + a.id) % knows.size())];
            }
            if (to_tell >= 0) {
                bark(a, society_->say("chat", a.who, tick_ + a.id, society_->facts()[static_cast<std::size_t>(to_tell)].text));
                society_->told(b.who, to_tell);
            } else {
                const std::vector<std::string>& goods = society_->goods();
                bark(a, society_->say("chat_small", a.who, tick_ + a.id, "", society_->word_of("good", goods[static_cast<std::size_t>(a.who.good)])), false);
            }
            a.bark_wait = 7.0;
            b.bark_wait = 4.0;
            return; // one a time: a town is not a choir
        }
    }
}

// Where its day has someone. A day is a part of the work, the market and home, and everyone's starts a little differently.
Vec3 Playground::goal_at(const Person& who, int town, double seconds, Creature::Doing& doing) const
{
    const TerrainConfig& t = config_.terrain;
    const Settlement& s = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(town)].settlement)];
    const double day = law_param(solver_.law(), "day_seconds", 480.0);
    const double phase = std::fmod(seconds / day + 0.1 * hash01(who.pid, 61), 1.0);
    const std::string& good = society_->goods()[static_cast<std::size_t>(who.good)];
    const double h = hash01(who.pid, 62);
    const double ground = (s.ground + 1) * t.cell;
    const auto around = [&](double angle, double r) { return Vec3{(s.x + std::cos(angle) * r) * t.cell, (s.y + std::sin(angle) * r) * t.cell, ground}; };
    if (good == "guard") {
        doing = Creature::Doing::Patrol;
        const double turning = hash01(who.pid, 51) < 0.5 ? 1.0 : -1.0;
        // Round the town at a walking pace, however big it is: a lap of 90 s, or longer.
        const double r = s.radius * (0.5 + 0.4 * h);
        const double lap = std::max(90.0, 6.283185307 * r * t.cell / 1.3);
        const double a = 6.283185307 * (h + seconds / lap * turning);
        return around(a, r);
    }
    if (phase < 0.55) {
        doing = Creature::Doing::Working;
        // Where this work is done goes with the good, in this place: the same side for all who do it.
        std::uint64_t name = 5381;
        for (const char ch : good) {
            name = name * 33 + static_cast<unsigned char>(ch);
        }
        const double a = 6.283185307 * hash01(name, static_cast<std::uint64_t>(town)) + 0.7 * (h - 0.5);
        return around(a, s.radius * (good == "trade" ? 0.25 : 0.75 + 0.3 * h));
    }
    if (phase < 0.72) {
        doing = Creature::Doing::Market;
        return around(6.283185307 * h, 2.0 + 10.0 * hash01(who.pid, 63));
    }
    doing = Creature::Doing::Home;
    const Vec3 door = door_of(who, s);
    return Vec3{door.x, door.y, ground};
}

Vec3 Playground::goal_of(const Creature& c, Creature::Doing& doing) const
{
    const TerrainConfig& t = config_.terrain;
    const Vec3 o = origin();
    // On the road: only the road.
    if (c.caravan >= 0) {
        const Caravan& road = society_->caravans()[static_cast<std::size_t>(c.caravan)];
        if (road.next >= 0) {
            const Settlement& to = map_->settlements()[static_cast<std::size_t>(society_->towns()[static_cast<std::size_t>(road.next)].settlement)];
            doing = Creature::Doing::Working;
            return Vec3{(to.x + 0.5) * t.cell - o.x, (to.y + 0.5) * t.cell - o.y, 0.0};
        }
    }
    const Vec3 goal = goal_at(c.who, c.town, world_seconds(), doing);
    return Vec3{goal.x - o.x, goal.y - o.y, 0.0};
}

bool Playground::out_today(int town, int index, double seconds) const
{
    const double day = law_param(solver_.law(), "day_seconds", 480.0);
    const auto today = static_cast<std::uint64_t>(std::floor(seconds / day));
    return hash01(static_cast<std::uint64_t>(town) * 1000003u + static_cast<std::uint64_t>(index), 977u + today) < config_.out_share;
}

// From the last place to this one at a walking pace, then about it; at home, indoors.
bool Playground::day_place(const Person& who, int town, double seconds, Vec3& at, double& yaw, double& speed, Creature::Doing& doing) const
{
    const double day = law_param(solver_.law(), "day_seconds", 480.0);
    const Vec3 goal = goal_at(who, town, seconds, doing);
    if (doing == Creature::Doing::Patrol) {
        Creature::Doing later = doing;
        const Vec3 next = goal_at(who, town, seconds + 1.0, later);
        at = goal;
        yaw = std::atan2(next.y - goal.y, next.x - goal.x);
        speed = std::hypot(next.x - goal.x, next.y - goal.y);
        return true;
    }
    const double phase = std::fmod(seconds / day + 0.1 * hash01(who.pid, 61), 1.0);
    const double begun = phase < 0.55 ? 0.0 : phase < 0.72 ? 0.55 : 0.72;
    const double since = (phase - begun) * day;
    Creature::Doing before = doing;
    const Vec3 from = goal_at(who, town, seconds - since - 1.0, before);
    const double way = std::hypot(goal.x - from.x, goal.y - from.y);
    const double pace = 1.1 + 0.4 * hash01(who.pid, 64);
    if (since * pace < way) {
        const double a = since * pace / std::max(way, 1e-6);
        at = from + (goal - from) * a;
        yaw = std::atan2(goal.y - from.y, goal.x - from.x);
        speed = pace;
        return true;
    }
    if (doing == Creature::Doing::Home) {
        return false; // indoors
    }
    // There: about the place, slowly.
    const double turn = seconds * 0.15 + 6.283185307 * hash01(who.pid, 65);
    at = goal + Vec3{std::cos(turn) * 0.8, std::sin(turn) * 0.8, 0.0};
    yaw = turn + 1.5707963;
    speed = 0.12;
    return true;
}

void Playground::people_beyond(double cx, double cy, double radius, std::size_t most, std::vector<PersonProxy>& out) const
{
    out.clear();
    if (society_ == nullptr || map_ == nullptr) {
        return;
    }
    const TerrainConfig& t = config_.terrain;
    const Vec3 o = origin();
    const GridShape& g = fields().shape();
    const double seconds = world_seconds();
    std::set<std::uint64_t> bodies;
    for (const Creature& c : creatures_) {
        if (c.town >= 0) {
            bodies.insert(c.who.pid);
        }
    }
    const std::vector<Town>& towns = society_->towns();
    std::vector<std::pair<double, int>> near;
    for (std::size_t i = 0; i < towns.size(); ++i) {
        const Settlement& s = map_->settlements()[static_cast<std::size_t>(towns[i].settlement)];
        const double away = std::hypot(s.x * t.cell - cx, s.y * t.cell - cy);
        if (s.fell < 0 && away - s.radius * t.cell < radius) {
            near.emplace_back(away, static_cast<int>(i));
        }
    }
    std::sort(near.begin(), near.end());
    for (const auto& [away, town] : near) {
        (void)away;
        const int people = society_->people_in(towns[static_cast<std::size_t>(town)]);
        for (int index = 0; index < people && out.size() < most; ++index) {
            if (!out_today(town, index, seconds)) {
                continue;
            }
            const Person who = society_->person(town, index);
            if (bodies.count(who.pid) != 0 || society_->dead(who.pid) || society_->on_the_road(who.pid)) {
                continue;
            }
            PersonProxy p{.pid = who.pid, .town = town, .index = index};
            if (!day_place(who, town, seconds, p.at, p.yaw, p.speed, p.doing)) {
                continue;
            }
            // In the window everyone out is a body: none of them is drawn twice.
            const Vec3 l = p.at - o;
            if (l.x > 0.0 && l.y > 0.0 && l.x < g.nx * g.cell && l.y < g.ny * g.cell) {
                continue;
            }
            if (std::hypot(p.at.x - cx, p.at.y - cy) > radius) {
                continue;
            }
            out.push_back(p);
        }
        if (out.size() >= most) {
            return;
        }
    }
}

// What it does about what it feels, this tick.
void Playground::person_turn(Creature& c, const CreatureDef& def, Vec3& want, bool& jump, Vec3 pull, double strength)
{
    const Law& law = solver_.law();
    PlayBody& body = c.body;
    PlayBody& you = player();
    const double anger = read_local(def, c, "anger");
    const double fear = read_local(def, c, "fear");
    const double fight_at = law_param(law, "fight_at", 0.55);
    const double flee_at = law_param(law, "flee_at", 0.6);
    const bool was_hostile = c.hostile;
    c.hostile = anger > fight_at || (c.hostile && anger > fight_at * 0.6);
    const bool flees = fear > flee_at && fear > anger;
    if (flees) {
        c.hostile = false;
    }
    if (c.hostile && !was_hostile) {
        bark(c, society_->say(def.name == "watchman" ? "call_watch" : "angry", c.who, tick_ + c.id));
        c.bark_wait = 3.0;
        c.began_it = c.began_it || read_local(def, c, "struck") <= 0.01;
    }
    const Vec3 off = horizontal(you.position - body.position);
    const double d = length(off);
    const Vec3 to_you = d > 1e-6 ? off * (1.0 / d) : Vec3{1.0, 0.0, 0.0};
    // Someone it has gone for who is gone, or has got away: it lets it go.
    if (c.foe != 0) {
        const Creature* other = creature_with(c.foe);
        if (other == nullptr || length(other->body.position - body.position) > 45.0) {
            fight_over(c);
        }
    }
    const bool fighting = c.hostile || c.foe != 0;
    if (fighting) {
        c.doing = Creature::Doing::Fighting;
        c.face = std::atan2(to_you.y, to_you.x);
        const bool fought = fight_turn(c, def, want);
        if (!fought) {
            want = to_you * def.speed;
        }
    } else {
        // At peace: what is in the hand goes on to its end, and no blow starts.
        body.surge = std::max(0.0, body.surge - kTick);
        swing_turn(body);
        body.guard = -1.0;
    }
    Vec3 way{};
    double pace = def.speed * (c.caravan >= 0 ? 0.8 : 0.55);
    if (fighting) {
        way = horizontal(want);
        pace = length(way);
        way = pace > 1e-6 ? way * (1.0 / pace) : Vec3{};
    } else if (c.flee_left > 0.0) {
        c.doing = Creature::Doing::Fleeing;
        Vec3 away = horizontal(body.position - c.flee_from);
        away = length(away) > 1e-6 ? normalize(away) : Vec3{1.0, 0.0, 0.0};
        // A mind that plans (its world model): away, and the way it learned leaves it least afraid.
        way = def.plan && c.planning ? normalize(away + c.planned) : away;
        pace = def.speed * 1.2;
    } else if (flees) {
        if (c.doing != Creature::Doing::Fleeing && c.bark_wait <= 0.0) {
            bark(c, society_->say("afraid", c.who, tick_ + c.id));
            c.bark_wait = 4.0;
        }
        c.doing = Creature::Doing::Fleeing;
        way = def.plan && c.planning && length(c.planned) > 0.1 ? normalize(c.planned) : to_you * -1.0;
        pace = def.speed * 1.2;
    } else {
        // It looks up when you come close, and says something if it has something to say.
        if (c.sees_you && d < 4.0 && c.bark_wait <= 0.0) {
            const double o = c.opinion;
            bark(c, society_->say(o < -0.5 ? "greet_hate" : o < -0.1 ? "greet_dislike" : o < 0.2 ? "greet_none" : o < 0.6 ? "greet_like" : "greet_love", c.who, c.who.pid));
            c.bark_wait = 25.0;
            c.look_wait = 2.5;
        }
        if (c.look_wait > 0.0) {
            c.doing = Creature::Doing::Looking;
            c.face = std::atan2(to_you.y, to_you.x);
        } else {
            if ((tick_ + c.id * 11u) % 30u == 0u || length(c.goal) < 1e-6) {
                c.goal = goal_of(c, c.doing);
            }
            const Vec3 go = horizontal(c.goal - body.position);
            const double left = length(go);
            if (left > 1.5) {
                way = go * (1.0 / left);
            } else if (c.doing == Creature::Doing::Working && (tick_ + c.id * 7u) % 600u < 200u) {
                way = Vec3{std::cos(c.heading), std::sin(c.heading), 0.0}; // about its work
                pace = def.speed * 0.25;
            }
        }
    }
    if (fighting && body.swing.move >= 0) {
        return; // fight_turn has said where it stands
    }
    // Something grew round it, or it was shut in: out, to the nearest place there is room.
    c.walled = blocked_at(body.position) || blocked_at(body.position + Vec3{0.0, 0.0, 0.6}) ? c.walled + 1 : 0;
    if (c.walled > 20) {
        c.walled = 0;
        const GridShape& g = fields().shape();
        const auto free_at = [&](Vec3 p) { return p.x > 1.0 && p.y > 1.0 && p.x < g.nx * g.cell - 1.0 && p.y < g.ny * g.cell - 1.0 && p.z > 0.5 && p.z < g.nz * g.cell - 2.0 && !blocked_at(p) && !blocked_at(p + Vec3{0.0, 0.0, 0.6}) && !blocked_at(p - Vec3{0.0, 0.0, 0.4}); };
        bool moved = false;
        for (double r = 0.5; r <= 6.0 && !moved; r += 0.5) {
            for (int k = 0; k < 16 && !moved; ++k) {
                const double a = 6.283185307 * k / 16.0;
                for (const double up : {0.0, 0.5, 1.0, 2.0, -0.5}) {
                    const Vec3 tried = body.position + Vec3{r * std::cos(a), r * std::sin(a), up};
                    if (free_at(tried)) {
                        body.position = tried;
                        body.velocity = Vec3{};
                        moved = true;
                        break;
                    }
                }
            }
        }
    }
    // Pressing against the same place for no gain: first another way for a moment, then another errand.
    if ((tick_ + c.id) % 30u == 0u) {
        const bool gets_nowhere = length(way) > 1e-6 && length(horizontal(body.position - c.last_place)) < 0.25 * pace * 0.5;
        c.pressed = gets_nowhere ? c.pressed + 1 : 0;
        c.last_place = body.position;
        if (c.pressed >= 3) {
            c.turning = -c.turning;
            c.detour_left = 1.5;
            c.detour_angle = std::atan2(way.y, way.x) + c.turning * (1.2 + 1.2 * hash01(c.who.pid, tick_));
        }
        // That did not help: right away from here for a while, and it can think again after that.
        if (c.pressed >= 6) {
            c.pressed = 0;
            c.goal = Vec3{};
            c.detour_left = 5.0;
            c.detour_angle = 6.283185307 * hash01(c.who.pid + 7, tick_);
        }
    }
    c.detour_left = std::max(0.0, c.detour_left - kTick);
    if (c.detour_left > 0.0 && length(way) > 1e-6) {
        way = Vec3{std::cos(c.detour_angle), std::sin(c.detour_angle), 0.0};
    }
    // Round what is in the way: it tries ahead, then turns further and further to its own side; a step it can take, it jumps.
    if (length(way) > 1e-6) {
        const Vec3 feet = body.position - Vec3{0.0, 0.0, body.radius - 0.25};
        const Vec3 chest = body.position + Vec3{0.0, 0.0, 0.45};
        const double look = body.radius + 0.55;
        Vec3 chosen = way;
        for (const double turn : {0.0, 0.8, 1.6, 2.4}) {
            const double a = turn * c.turning;
            const Vec3 tried{way.x * std::cos(a) - way.y * std::sin(a), way.x * std::sin(a) + way.y * std::cos(a), 0.0};
            const bool high = blocked_at(chest + tried * look);
            if (!high) {
                chosen = tried;
                jump = turn == 0.0 && blocked_at(feet + tried * look);
                break;
            }
        }
        way = chosen;
        if (!fighting) {
            c.face = std::atan2(way.y, way.x);
        }
    }
    // What burns or breaks pulls harder than a day's work.
    want = way * (pace * (1.0 - strength)) + pull * (strength * def.speed);
}

} // namespace fieldengine::field

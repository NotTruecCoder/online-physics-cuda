#include "field/WorldMap.hpp"

#include "field/Terrain.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <queue>
#include <sstream>
#include <tuple>

namespace fieldengine::field {

namespace {

constexpr int kN = WorldMap::kNodes;
constexpr int kTreeGrid = 7;
constexpr int kTreeReach = 34; // cells: as far as any crown stands from its foot
constexpr int kBigEvery = 3;   // a stand of a big tree, every this many grid cells

std::uint64_t mix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::uint64_t hash(std::uint64_t seed, std::int64_t a, std::int64_t b, std::int64_t c = 0)
{
    std::uint64_t h = mix64(seed);
    h = mix64(h ^ static_cast<std::uint64_t>(a));
    h = mix64(h ^ static_cast<std::uint64_t>(b));
    return mix64(h ^ static_cast<std::uint64_t>(c));
}

double unit(std::uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }

struct Rng {
    std::uint64_t state;
    double next()
    {
        state = mix64(state);
        return unit(state);
    }
    int range(int lo, int hi) // inclusive
    {
        return hi <= lo ? lo : lo + static_cast<int>(next() * (hi - lo + 1));
    }
};

double saturate(double v) { return std::clamp(v, 0.0, 1.0); }
double smooth(double t) { return t * t * (3.0 - 2.0 * t); }
double smoothstep(double a, double b, double v) { return smooth(saturate((v - a) / (b - a))); }

double value_noise(std::uint64_t seed, double x, double y)
{
    const double fx = std::floor(x);
    const double fy = std::floor(y);
    const auto ix = static_cast<std::int64_t>(fx);
    const auto iy = static_cast<std::int64_t>(fy);
    const double tx = smooth(x - fx);
    const double ty = smooth(y - fy);
    const double a = unit(hash(seed, ix, iy));
    const double b = unit(hash(seed, ix + 1, iy));
    const double c = unit(hash(seed, ix, iy + 1));
    const double d = unit(hash(seed, ix + 1, iy + 1));
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
}

// Octaves finer than `factor` cells are not asked for: from far away they could not show.
double fbm(std::uint64_t seed, double x, double y, double scale, int octaves, int factor = 1)
{
    double sum = 0.0;
    double amplitude = 1.0;
    double total = 0.0;
    double wavelength = scale;
    for (int o = 0; o < octaves; ++o) {
        // A wave shorter than the step between samples is not a coarser wave, it is noise: it fades to its mean as the samples outgrow it.
        const double kept = std::clamp((wavelength / factor - 1.0) * 0.5, 0.0, 1.0);
        const double wave = factor == 1 || kept >= 1.0 ? value_noise(seed + static_cast<std::uint64_t>(o) * 7919u, x / wavelength, y / wavelength)
            : kept > 0.0 ? 0.5 + kept * (value_noise(seed + static_cast<std::uint64_t>(o) * 7919u, x / wavelength, y / wavelength) - 0.5) : 0.5;
        sum += amplitude * wave;
        total += amplitude;
        amplitude *= 0.5;
        wavelength *= 0.5;
    }
    return sum / total;
}

int at(int x, int y) { return y * kN + x; }

double segment_distance(double px, double py, double ax, double ay, double bx, double by, double& t)
{
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len2 = dx * dx + dy * dy;
    t = len2 > 0.0 ? saturate(((px - ax) * dx + (py - ay) * dy) / len2) : 0.0;
    return std::hypot(px - (ax + dx * t), py - (ay + dy * t));
}

int material_named(const std::string& name)
{
    static const std::map<std::string, int> kNames{{"granite", material::granite}, {"basalt", material::basalt},
        {"slate", material::slate}, {"sandstone", material::sandstone}, {"soil", material::soil}, {"sand", material::sand},
        {"clay", material::clay}, {"chalk", material::chalk}, {"gravel", material::gravel}, {"plaster", material::plaster},
        {"brick", material::brick}, {"thatch", material::thatch}, {"tile", material::tile}, {"timber", material::timber},
        {"char", material::charred}, {"reed", material::reed}};
    const auto it = kNames.find(name);
    return it == kNames.end() ? -1 : it->second;
}

bool biome_named(const std::string& name, Biome& out)
{
    for (int b = 0; b <= static_cast<int>(Biome::Mountain); ++b) {
        if (name == biome_name(static_cast<Biome>(b))) {
            out = static_cast<Biome>(b);
            return true;
        }
    }
    return false;
}

bool stuff_named(const std::string& name, Style::Stuff& out)
{
    if (name == "stone") {
        out = Style::Stuff::Stone;
    } else if (name == "wood") {
        out = Style::Stuff::Wood;
    } else if (name == "leaf") {
        out = Style::Stuff::Leaf;
    } else {
        return false;
    }
    return true;
}

BuildingCell::Kind cell_of(Style::Stuff stuff)
{
    switch (stuff) {
    case Style::Stuff::Wood: return BuildingCell::Wood;
    case Style::Stuff::Leaf: return BuildingCell::Leaf;
    default: return BuildingCell::Stone;
    }
}

const char* const kPlainStyle =
    "style plain\n  biomes grassland\n  wall stone granite\n  roof gable stone slate\n  floors 1 1\n  storey 6\n"
    "  width 12 14\n  depth 13 15\n  base stone 1 granite\n  windows 4\n";

// The lands, as the text writes them.
constexpr std::array<const char*, 12> kBiomeWords = {"Ocean", "Beach", "Desert", "Savanna", "Grassland", "Forest", "Jungle", "Taiga", "Tundra", "Snow", "Swamp", "Mountain"};

std::string read_trees_file()
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/world/trees.world", std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            return text.str();
        }
    }
    return {};
}

std::string read_styles_file()
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/world/styles.world", std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            return text.str();
        }
    }
    return {};
}

} // namespace

const char* biome_name(Biome biome)
{
    switch (biome) {
    case Biome::Ocean: return "ocean";
    case Biome::Beach: return "beach";
    case Biome::Desert: return "desert";
    case Biome::Savanna: return "savanna";
    case Biome::Grassland: return "grassland";
    case Biome::Forest: return "forest";
    case Biome::Jungle: return "jungle";
    case Biome::Taiga: return "taiga";
    case Biome::Tundra: return "tundra";
    case Biome::Snow: return "snow";
    case Biome::Swamp: return "swamp";
    case Biome::Mountain: return "mountain";
    }
    return "?";
}

std::vector<Style> parse_styles(const std::string& text, std::vector<std::string>& errors)
{
    std::vector<Style> styles;
    std::istringstream stream(text);
    std::string line;
    int number = 0;
    auto check_last = [&](int where) {
        if (styles.empty()) {
            return;
        }
        const Style& s = styles.back();
        const std::string head = "styles.world:" + std::to_string(where) + ": style '" + s.name + "': ";
        if (s.biomes.empty()) {
            errors.push_back(head + "belongs to no biome");
        }
        if (s.storey < 5) {
            errors.push_back(head + "a storey under 5 cells cannot hold a door");
        }
        if (s.floors_min < 1 || s.floors_max < s.floors_min) {
            errors.push_back(head + "floors MIN MAX do not make a range");
        }
        if (s.width_min < 12 || s.width_max < s.width_min || s.depth_min < 12 || s.depth_max < s.depth_min) {
            errors.push_back(head + "a house narrower than 12 cells has no room for a person to turn");
        }
        if (s.floors_max > 1 && s.depth_max < s.storey + 7) {
            errors.push_back(head + "more than one floor needs depth for a stair (storey + 7)");
        }
        if (s.raise < 0 || s.raise > 8) {
            errors.push_back(head + "base height out of 0..8");
        }
        if (s.pitch < 1 || s.pitch > 3 || s.eaves < 0 || s.eaves > 2) {
            errors.push_back(head + "pitch out of 1..3 or eaves out of 0..2");
        }
        if (s.wall != Style::Stuff::Stone && s.roof == Style::Roof::Dome) {
            errors.push_back(head + "a dome does not stand on walls that burn away");
        }
    };
    while (std::getline(stream, line)) {
        ++number;
        std::istringstream words(line.substr(0, line.find('#')));
        std::vector<std::string> w;
        for (std::string word; words >> word;) {
            w.push_back(word);
        }
        if (w.empty()) {
            continue;
        }
        const std::string where = "styles.world:" + std::to_string(number) + ": ";
        if (w[0] == "style") {
            check_last(number);
            if (w.size() != 2) {
                errors.push_back(where + "expected 'style NAME'");
            }
            styles.push_back(Style{.name = w.size() > 1 ? w[1] : "?"});
            continue;
        }
        if (styles.empty()) {
            errors.push_back(where + "'" + w[0] + "' outside a style");
            continue;
        }
        Style& s = styles.back();
        auto number_of = [&](std::size_t i, int& out) {
            try {
                out = std::stoi(w.at(i));
            } catch (...) {
                errors.push_back(where + "expected a number for '" + w[0] + "'");
            }
        };
        auto material_of = [&](std::size_t i, int& out) {
            out = i < w.size() ? material_named(w[i]) : -1;
            if (out < 0) {
                errors.push_back(where + "no such material for '" + w[0] + "'");
                out = 0;
            }
        };
        auto stuff_of = [&](std::size_t i, Style::Stuff& out) {
            if (i >= w.size() || !stuff_named(w[i], out)) {
                errors.push_back(where + "expected stone, wood or leaf for '" + w[0] + "'");
            }
        };
        if (w[0] == "biomes") {
            for (std::size_t i = 1; i < w.size(); ++i) {
                Biome b{};
                if (biome_named(w[i], b)) {
                    s.biomes.push_back(b);
                } else {
                    errors.push_back(where + "no such biome '" + w[i] + "'");
                }
            }
        } else if (w[0] == "wall") {
            stuff_of(1, s.wall);
            material_of(2, s.wall_material);
        } else if (w[0] == "roof") {
            if (w.size() > 1 && w[1] == "gable") {
                s.roof = Style::Roof::Gable;
            } else if (w.size() > 1 && w[1] == "hip") {
                s.roof = Style::Roof::Hip;
            } else if (w.size() > 1 && w[1] == "flat") {
                s.roof = Style::Roof::Flat;
            } else if (w.size() > 1 && w[1] == "dome") {
                s.roof = Style::Roof::Dome;
            } else if (w.size() > 1 && w[1] == "spire") {
                s.roof = Style::Roof::Spire;
            } else {
                errors.push_back(where + "expected gable, hip, flat, dome or spire");
            }
            stuff_of(2, s.roof_stuff);
            material_of(3, s.roof_material);
            if (w.size() > 4) {
                number_of(4, s.pitch);
            }
        } else if (w[0] == "floors") {
            number_of(1, s.floors_min);
            number_of(2, s.floors_max);
        } else if (w[0] == "storey") {
            number_of(1, s.storey);
        } else if (w[0] == "width") {
            number_of(1, s.width_min);
            number_of(2, s.width_max);
        } else if (w[0] == "depth") {
            number_of(1, s.depth_min);
            number_of(2, s.depth_max);
        } else if (w[0] == "base") {
            if (w.size() > 1 && (w[1] == "stone" || w[1] == "piles")) {
                s.piles = w[1] == "piles";
            } else {
                errors.push_back(where + "expected stone or piles");
            }
            number_of(2, s.raise);
            material_of(3, s.base_material);
        } else if (w[0] == "windows") {
            number_of(1, s.window_every);
        } else if (w[0] == "ring") {
            material_of(1, s.ring_material);
            s.ring_round = w.size() > 2 && w[2] == "round";
        } else if (w[0] == "upper") {
            stuff_of(1, s.upper);
            material_of(2, s.upper_material);
        } else if (w[0] == "trim") {
            material_of(1, s.trim_material);
        } else if (w[0] == "door") {
            s.door_set = true;
            if (w.size() > 1) {
                s.door = w[1] == "none" ? Style::Door::None : w[1] == "double" ? Style::Door::Double : w[1] == "slide" ? Style::Door::Slide : w[1] == "lift" ? Style::Door::Lift : Style::Door::Plank;
            }
            if (w.size() > 2) {
                s.door_stone = w[2] == "stone";
                material_of(3, s.door_material);
            }
        } else if (w[0] == "eaves") {
            number_of(1, s.eaves);
        } else if (w[0] == "jetty") {
            s.jetty = true;
        } else if (w[0] == "wing") {
            s.wing = true;
        } else if (w[0] == "yard") {
            s.yard = true;
        } else if (w[0] == "chimney") {
            s.chimney = true;
        } else if (w[0] == "towers" && w.size() > 1 && w[1] == "round") {
            s.round_towers = true;
        } else if (w[0] == "layout") {
            static const char* const kLayouts[] = {"grid", "organic", "ring", "ribbon", "clusters"};
            bool known = false;
            for (int k = 0; k < 5; ++k) {
                if (w.size() > 1 && w[1] == kLayouts[k]) {
                    s.layout = static_cast<Style::Layout>(k);
                    known = true;
                }
            }
            if (!known) {
                errors.push_back(where + "no such layout");
            }
        } else if (w[0] == "temple") {
            static const char* const kTemples[] = {"hall", "dome", "stepped", "spire"};
            bool known = false;
            for (int k = 0; k < 4; ++k) {
                if (w.size() > 1 && w[1] == kTemples[k]) {
                    s.temple = static_cast<Style::Temple>(k);
                    known = true;
                }
            }
            if (!known) {
                errors.push_back(where + "no such temple");
            }
        } else {
            errors.push_back(where + "unknown key '" + w[0] + "'");
        }
    }
    check_last(number);
    return styles;
}

namespace {

const char* material_name(int m)
{
    static const char* const kNames[] = {"granite", "basalt", "slate", "sandstone", "soil", "sand", "clay", "chalk", "gravel", "plaster", "brick",
        "thatch", "tile", "timber", "char", "reed"};
    return kNames[std::clamp(m, 0, 15)];
}

const char* stuff_name(Style::Stuff s) { return s == Style::Stuff::Stone ? "stone" : (s == Style::Stuff::Wood ? "wood" : "leaf"); }

const char* roof_name(Style::Roof r)
{
    static const char* const kNames[] = {"gable", "hip", "flat", "dome", "spire"};
    return kNames[static_cast<int>(r)];
}

} // namespace

const char* door_name(Style::Door door)
{
    switch (door) {
    case Style::Door::None: return "none";
    case Style::Door::Double: return "double";
    case Style::Door::Slide: return "slide";
    case Style::Door::Lift: return "lift";
    default: return "plank";
    }
}

int material_id(const std::string& name)
{
    return material_named(name);
}

std::string style_text(const Style& s)
{
    static const char* const kLayouts[] = {"grid", "organic", "ring", "ribbon", "clusters"};
    static const char* const kTemples[] = {"hall", "dome", "stepped", "spire"};
    std::ostringstream o;
    o << "style " << s.name << "\n  biomes";
    for (const Biome b : s.biomes) {
        o << " " << biome_name(b);
    }
    o << "\n  wall " << stuff_name(s.wall) << " " << material_name(s.wall_material);
    if (s.upper_material >= 0) {
        o << "\n  upper " << stuff_name(s.upper) << " " << material_name(s.upper_material);
    }
    if (s.trim_material >= 0) {
        o << "\n  trim " << material_name(s.trim_material);
    }
    o << "\n  roof " << roof_name(s.roof) << " " << stuff_name(s.roof_stuff) << " " << material_name(s.roof_material) << " " << s.pitch;
    o << "\n  eaves " << s.eaves << "\n  floors " << s.floors_min << " " << s.floors_max << "\n  storey " << s.storey;
    o << "\n  width " << s.width_min << " " << s.width_max << "\n  depth " << s.depth_min << " " << s.depth_max;
    o << "\n  base " << (s.piles ? "piles" : "stone") << " " << s.raise << " " << material_name(s.base_material);
    o << "\n  windows " << s.window_every;
    o << "\n  door " << door_name(s.door) << " " << (s.door_stone ? "stone" : "wood") << " " << material_name(s.door_material);
    if (s.ring_material >= 0) {
        o << "\n  ring " << material_name(s.ring_material) << (s.ring_round ? " round" : "");
    }
    if (s.jetty) o << "\n  jetty";
    if (s.wing) o << "\n  wing";
    if (s.yard) o << "\n  yard";
    if (s.chimney) o << "\n  chimney";
    if (s.round_towers) o << "\n  towers round";
    o << "\n  layout " << kLayouts[static_cast<int>(s.layout)] << "\n  temple " << kTemples[static_cast<int>(s.temple)] << "\n";
    return o.str();
}

std::string style_words(const Style& s)
{
    static const char* const kLayouts[] = {"a grid of streets", "winding streets", "rings around the square", "one long street", "scattered farmsteads"};
    std::ostringstream o;
    o << material_name(s.wall_material) << " walls";
    if (s.upper_material >= 0) {
        o << " under " << material_name(s.upper_material);
    }
    o << ", " << roof_name(s.roof) << " roofs of " << material_name(s.roof_material);
    if (s.piles) {
        o << ", on piles";
    } else if (s.raise >= 2) {
        o << ", raised floors";
    }
    if (s.jetty) o << ", jetties";
    if (s.yard) o << ", walled yards";
    o << "; " << kLayouts[static_cast<int>(s.layout)];
    return o.str();
}

void Building::bounds(const std::vector<Style>& styles, int& bx0, int& by0, int& bx1, int& by1, int& top) const
{
    const Style& s = styles[static_cast<std::size_t>(style)];
    const bool plain = !has_rooms();
    const int reach = plain ? 0 : 1 + s.raise + s.eaves + (s.jetty ? 1 : 0);
    bx0 = x0 - reach;
    by0 = y0 - reach;
    bx1 = x1 + reach;
    by1 = y1 + reach;
    if (kind == Kind::Ring) {
        top = ground + 12;
    } else if (kind == Kind::Well || kind == Kind::Yard || kind == Kind::Graves) {
        top = ground + 5;
    } else if (kind == Kind::Monument) {
        top = ground + floors + 2;
    } else if (kind == Kind::Stepped) {
        top = ground + std::min(x1 - x0, y1 - y0) + 4;
    } else {
        const int span = std::min(x1 - x0, y1 - y0) / 2 + s.eaves + 3;
        top = ground + s.raise + 1 + floors * s.storey + span * std::max(s.pitch, 2) + 6;
    }
}

BuildingCell building_cell(const Building& b, const Style& st, int x, int y, int z)
{
    BuildingCell cell;
    const std::uint64_t h = hash(b.seed, x, y, z);
    const bool in_rect = x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
    auto made_of = [&](BuildingCell::Kind kind, int material) {
        if (b.ruin > 0.0f && z > b.ground) {
            const double fall = b.ruin * (0.25 + 0.75 * saturate(static_cast<double>(z - b.ground) / 12.0));
            if (unit(h) < fall) {
                return BuildingCell{BuildingCell::Air, 0};
            }
        }
        if (b.burned && z > b.ground) {
            return BuildingCell{BuildingCell::Stone, material::charred};
        }
        return BuildingCell{kind, material};
    };
    const int cx = (b.x0 + b.x1) / 2;
    const int cy = (b.y0 + b.y1) / 2;
    switch (b.kind) {
    case Building::Kind::Ring: {
        if (z <= b.ground - 2 || z > b.ground + 10 || std::abs(x - cx) < 5 || std::abs(y - cy) < 5) {
            return cell; // the gates are where the main streets run out
        }
        const int r = (b.x1 - b.x0) / 2;
        const bool band = b.part[0] == 1 ? std::hypot(static_cast<double>(x - cx), static_cast<double>(y - cy)) >= r - 2.0 &&
                std::hypot(static_cast<double>(x - cx), static_cast<double>(y - cy)) <= r
                                         : in_rect && (x < b.x0 + 2 || x > b.x1 - 2 || y < b.y0 + 2 || y > b.y1 - 2);
        if (!band || (z == b.ground + 10 && (x + y) % 2 == 0)) {
            return cell;
        }
        return made_of(BuildingCell::Stone, st.ring_material >= 0 ? st.ring_material : st.wall_material);
    }
    case Building::Kind::Well: {
        if (!in_rect || z <= b.ground - 2 || z > b.ground + 2) {
            return cell;
        }
        const bool rim = x == b.x0 || x == b.x1 || y == b.y0 || y == b.y1;
        return rim ? made_of(BuildingCell::Stone, st.base_material) : BuildingCell{BuildingCell::Air, 0};
    }
    case Building::Kind::Monument: {
        if (!in_rect || z <= b.ground - 1) {
            return cell;
        }
        const bool shaft = x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1;
        const int height = shaft ? b.floors : 1;
        return z <= b.ground + height ? made_of(BuildingCell::Stone, st.ring_material >= 0 ? st.ring_material : st.base_material) : cell;
    }
    case Building::Kind::Graves:
        if (in_rect && z == b.ground + 1 && x % 3 == 0 && y % 2 == 0 && unit(hash(b.seed, x, y)) < 0.8) {
            return BuildingCell{BuildingCell::Stone, material::slate};
        }
        return cell;
    case Building::Kind::Yard: {
        if (!in_rect || z <= b.ground - 1 || z > b.ground + 3) {
            return cell;
        }
        const bool border = x == b.x0 || x == b.x1 || y == b.y0 || y == b.y1;
        const bool open = (b.facing == 0 && x == b.x1) || (b.facing == 1 && x == b.x0) || (b.facing == 2 && y == b.y1) || (b.facing == 3 && y == b.y0);
        return border && !open ? made_of(cell_of(st.wall), st.wall_material) : cell;
    }
    case Building::Kind::Stepped: {
        if (!in_rect || z <= b.ground - 2) {
            return cell;
        }
        const int inset = std::min({x - b.x0, b.x1 - x, y - b.y0, b.y1 - y});
        const int tiers = std::min(b.x1 - b.x0, b.y1 - b.y0) / 2;
        int height = (std::min(inset, tiers) / 2 + 1) * 2;
        if (std::abs(x - cx) <= 1) {
            height = std::max(height, std::min(y - b.y0 + 1, (tiers / 2 + 1) * 2)); // a stair up the front
        }
        return z <= b.ground + height ? made_of(BuildingCell::Stone, st.wall_material) : cell;
    }
    default:
        break;
    }
    const int floor0 = b.ground + 1 + st.raise; // the cell of the first floor
    const int w = b.x1 - b.x0 + 1;
    const int d = b.y1 - b.y0 + 1;
    const int ceiling = floor0 + b.floors * st.storey;
    const int jetty = st.jetty && b.kind != Building::Kind::Round && z >= floor0 + st.storey ? 1 : 0;
    const Style::Roof roof_kind = b.roof >= 0 ? static_cast<Style::Roof>(b.roof) : st.roof;
    // The plan of this level: a rectangle, a circle, or an L with the corner away from the stair cut off.
    const double radial = std::hypot(static_cast<double>(x) - (b.x0 + b.x1) * 0.5, static_cast<double>(y) - (b.y0 + b.y1) * 0.5);
    const double radius = std::min(w, d) * 0.5;
    const int cut_x = b.x0 + w * 6 / 10;
    const int cut_y = b.y0 + d * 6 / 10;
    const auto inside_plan = [&](int ax, int ay) {
        if (b.kind == Building::Kind::Round) {
            return std::hypot(static_cast<double>(ax) - (b.x0 + b.x1) * 0.5, static_cast<double>(ay) - (b.y0 + b.y1) * 0.5) <= radius;
        }
        if (ax < b.x0 - jetty || ax > b.x1 + jetty || ay < b.y0 - jetty || ay > b.y1 + jetty) {
            return false;
        }
        return !(b.wing && ax >= cut_x && ay >= cut_y);
    };
    const bool in = inside_plan(x, y);

    if (!in_rect) {
        // The steps up to a raised door.
        if (st.raise > 0 && z > b.ground && z < floor0) {
            int k = 0;
            bool along = false;
            switch (b.facing) {
            case 0: k = x - b.x1; along = y == b.door || y == b.door + 1; break;
            case 1: k = b.x0 - x; along = y == b.door || y == b.door + 1; break;
            case 2: k = y - b.y1; along = x == b.door || x == b.door + 1; break;
            default: k = b.y0 - y; along = x == b.door || x == b.door + 1; break;
            }
            if (along && k >= 1 && k <= st.raise && z <= floor0 - k) {
                return made_of(BuildingCell::Stone, st.base_material);
            }
        }
    }
    if (z < floor0) {
        if (!in || z <= b.ground - 3) {
            return cell;
        }
        if (st.piles) {
            const bool post = (x == b.x0 || x == b.x1) && (y == b.y0 || y == b.y1);
            if (post) {
                return made_of(cell_of(st.wall), st.base_material);
            }
            return z > b.ground ? BuildingCell{BuildingCell::Air, 0} : cell;
        }
        return BuildingCell{BuildingCell::Stone, st.base_material};
    }
    if (z <= ceiling && in) {
        const int h_in = z - floor0;
        const int level = h_in % st.storey;
        const int storey_index = h_in / st.storey;
        const Style::Stuff stuff = storey_index > 0 && st.upper_material >= 0 ? st.upper : st.wall;
        const int wall_material = storey_index > 0 && st.upper_material >= 0 ? st.upper_material : st.wall_material;
        // The stair: flights 2 wide along y in the first room, in turn at x0+1 going up the y axis and at x0+3 coming
        // back, each with two cells of floor at its foot: stacked straight up, one would stand in the other's headroom.
        const int rise = st.storey - 1; // steps in a flight
        const auto flight = [&](int s, int& step) {
            if (s < 0 || s + 1 >= b.floors || y < b.y0 + 3 || y >= b.y0 + 3 + rise) {
                return false;
            }
            const int side = s % 2 == 0 ? b.x0 + 1 : b.x0 + 3;
            step = s % 2 == 0 ? y - (b.y0 + 3) : b.y0 + 2 + rise - y;
            return x == side || x == side + 1;
        };
        int step = 0;
        if (level == 0) {
            if (flight(storey_index - 1, step)) {
                return BuildingCell{BuildingCell::Air, 0}; // the hole the stair comes up through
            }
            return made_of(cell_of(stuff), wall_material);
        }
        if (flight(storey_index, step) && level <= step + 1) {
            return made_of(cell_of(stuff), wall_material);
        }
        const bool border = !inside_plan(x + 1, y) || !inside_plan(x - 1, y) || !inside_plan(x, y + 1) || !inside_plan(x, y - 1);
        if (border) {
            // The door, on the first floor.
            if (storey_index == 0 && level >= 1 && level <= 4) {
                const bool on_side = (b.facing == 0 && x > cx) || (b.facing == 1 && x < cx) || (b.facing == 2 && y > cy) || (b.facing == 3 && y < cy);
                const int along = b.facing < 2 ? y : x;
                if (on_side && (along == b.door || along == b.door + 1)) {
                    if (st.door != Style::Door::None && b.ruin <= 0.0f && !b.burned) {
                        return BuildingCell{st.door_stone ? BuildingCell::Stone : BuildingCell::Wood, st.door_material, true};
                    }
                    return BuildingCell{BuildingCell::Air, 0};
                }
            }
            const bool on_x_side = !inside_plan(x + 1, y) || !inside_plan(x - 1, y);
            const int along = on_x_side ? y - b.y0 : x - b.x0;
            const int length = on_x_side ? d : w;
            if (st.window_every > 0 && (level == 2 || level == 3)) {
                const bool corner = along < 2 || along >= length - 2;
                if (!corner && along % st.window_every == 2) {
                    return BuildingCell{BuildingCell::Air, 0};
                }
            }
            const bool post = (!inside_plan(x + 1, y) || !inside_plan(x - 1, y)) && (!inside_plan(x, y + 1) || !inside_plan(x, y - 1));
            if (post && st.trim_material >= 0) {
                return made_of(cell_of(st.wall), st.trim_material);
            }
            return made_of(cell_of(stuff), wall_material);
        }
        for (int k = 0; k < 2; ++k) {
            if (b.part[k] >= 0 && x == b.part[k]) {
                const bool gap = (y == b.gap[k] || y == b.gap[k] + 1) && level <= 4;
                return gap ? BuildingCell{BuildingCell::Air, 0} : made_of(cell_of(stuff), wall_material);
            }
        }
        if (b.ruin > 0.3f && storey_index == 0 && level == 1 && unit(hash(b.seed ^ 0x51, x, y)) < b.ruin * 0.5) {
            return BuildingCell{BuildingCell::Rubble, 0};
        }
        return BuildingCell{BuildingCell::Air, 0};
    }
    // The roof.
    if (z <= ceiling) {
        return cell;
    }
    if (b.ruin > 0.25f) {
        return cell;
    }
    const int k = z - ceiling;
    const int roof_jetty = st.jetty && b.floors > 1 && b.kind != Building::Kind::Round ? 1 : 0;
    const int e = st.eaves + roof_jetty;
    if (roof_kind == Style::Roof::Flat) {
        const bool edge = in_rect && (x == b.x0 || x == b.x1 || y == b.y0 || y == b.y1) && inside_plan(x, y);
        const bool bent_edge = b.wing && in_rect && inside_plan(x, y) && (!inside_plan(x + 1, y) || !inside_plan(x, y + 1));
        return (edge || bent_edge) && k == 1 ? made_of(cell_of(st.wall), st.wall_material) : cell;
    }
    if (roof_kind == Style::Roof::Dome) {
        const double rd = radius + st.eaves;
        const double rad = rd * rd - static_cast<double>(k * k) * 0.8;
        if (rad <= 0.0 || b.wing) {
            if (!b.wing) {
                return cell;
            }
        }
        if (!b.wing) {
            const double at = std::sqrt(rad);
            if (radial > at) {
                return cell;
            }
            return radial > at - 1.6 ? made_of(cell_of(st.roof_stuff), st.roof_material) : (in ? BuildingCell{BuildingCell::Air, 0} : cell);
        }
    }
    const int rx = std::min(x - (b.x0 - e), b.x1 + e - x);
    const int ry = std::min(y - (b.y0 - e), b.y1 + e - y);
    if (rx < 0 || ry < 0) {
        return cell;
    }
    const bool ridge_along_y = d >= w;
    int inset;
    if (b.kind == Building::Kind::Round) {
        inset = static_cast<int>(radius + e - radial);
        if (inset < 0) {
            return cell;
        }
    } else if (roof_kind == Style::Roof::Gable && !b.wing) {
        inset = ridge_along_y ? rx : ry;
    } else {
        inset = std::min(rx, ry);
        if (b.wing) {
            // An L: the hip follows the cut as well.
            if (x >= cut_x - e && y >= cut_y - e) {
                inset = std::min(inset, std::max(cut_x - e - 1 - x, cut_y - e - 1 - y) + 2 * e);
                if (x >= cut_x + e && y >= cut_y + e) {
                    return cell;
                }
                inset = std::max(inset, 0);
            }
        }
    }
    const int pitch = roof_kind == Style::Roof::Spire ? std::max(st.pitch, 2) + 1 : st.pitch;
    const int surface = (inset + 1) * pitch;
    if (k <= surface && k > surface - pitch - 1) {
        return made_of(cell_of(st.roof_stuff), st.roof_material);
    }
    if (st.chimney && b.kind == Building::Kind::House && !b.wing && x == b.x1 - 1 && y == b.y1 - 1 && k <= surface + 3) {
        return made_of(BuildingCell::Stone, st.base_material);
    }
    if (k <= surface - pitch - 1 && inside_plan(x, y)) {
        // Under the roof: the attic. A gable's end is wall.
        if (roof_kind == Style::Roof::Gable && !b.wing) {
            const bool end = ridge_along_y ? (y == b.y0 || y == b.y1) : (x == b.x0 || x == b.x1);
            if (end) {
                return made_of(cell_of(st.wall), st.wall_material);
            }
        }
        return BuildingCell{BuildingCell::Air, 0};
    }
    return cell;
}

// HOW A TREE GROWS. See assets/world/trees.world for the formulas: everything here follows from the species' numbers and
// a seed. In cells; a branch is walked one step at a time, thick as the taper says, turned by light and bent by weight.
std::vector<TreeVoxel> grow_tree(const TreeSpecies& sp, std::uint64_t seed, int height_cells, double cell)
{
    struct V3 {
        double x, y, z;
    };
    Rng rng{seed};
    std::vector<TreeVoxel> out;
    const auto normal = [](V3 v) {
        const double l = std::max(1e-9, std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z));
        return V3{v.x / l, v.y / l, v.z / l};
    };
    const int limit_z = height_cells + static_cast<int>(sp.leaf * 2.0) + 3;
    const auto put = [&](double x, double y, double z, bool wood, std::uint8_t material) {
        const int ix = static_cast<int>(std::lround(x));
        const int iy = static_cast<int>(std::lround(y));
        const int iz = static_cast<int>(std::lround(z));
        if (std::abs(ix) > kTreeReach || std::abs(iy) > kTreeReach || iz < 1 || iz > limit_z) {
            return;
        }
        out.push_back(TreeVoxel{static_cast<std::int8_t>(ix), static_cast<std::int8_t>(iy), static_cast<std::uint16_t>(iz), material, static_cast<std::uint8_t>(wood ? 1 : 0)});
    };
    // From one point to the next by whole steps along the axes, so that what grows is joined face to face. Wood holds wood
    // through a face, not through a corner.
    const auto join = [&](V3 a, V3 b, bool wood, std::uint8_t material) {
        int x = static_cast<int>(std::lround(a.x));
        int y = static_cast<int>(std::lround(a.y));
        int z = static_cast<int>(std::lround(a.z));
        const int tx = static_cast<int>(std::lround(b.x));
        const int ty = static_cast<int>(std::lround(b.y));
        const int tz = static_cast<int>(std::lround(b.z));
        while (x != tx || y != ty || z != tz) {
            if (z != tz) {
                z += z < tz ? 1 : -1;
            } else if (x != tx) {
                x += x < tx ? 1 : -1;
            } else {
                y += y < ty ? 1 : -1;
            }
            put(x, y, z, wood, material);
        }
    };
    const auto disc = [&](V3 at, double r, std::uint8_t material) {
        const int reach = static_cast<int>(std::ceil(r));
        if (r < 0.65) {
            put(at.x, at.y, at.z, true, material);
            return;
        }
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dx = -reach; dx <= reach; ++dx) {
                if (dx * dx + dy * dy <= r * r) {
                    put(at.x + dx, at.y + dy, at.z, true, material);
                }
            }
        }
    };
    const auto leaves = [&](V3 at, double r, double flat) {
        const int reach = static_cast<int>(std::ceil(r));
        const int reach_z = static_cast<int>(std::ceil(r * flat));
        for (int dz = -reach_z; dz <= reach_z; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    const double h = static_cast<double>(dz) / std::max(flat, 0.2);
                    const double d2 = dx * dx + dy * dy + h * h;
                    // Ragged: the farther out, the fewer.
                    if (d2 <= r * r && rng.next() > 0.02 + 0.35 * d2 * d2 / (r * r * r * r)) {
                        put(at.x + dx, at.y + dy, at.z + dz, false, static_cast<std::uint8_t>(sp.colour));
                    }
                }
            }
        }
    };
    // Leaves are as much as the wood can carry: the crown of a big tree is more than the crown of a small one, as sqrt(H).
    const double foliage = sp.leaf * std::sqrt(std::max(height_cells, 8) / 28.0);
    const double golden = 137.5 * 3.14159265358979 / 180.0;
    const double spread = sp.spread * 3.14159265358979 / 180.0;
    const double height = static_cast<double>(height_cells);
    // d = (H / slender)^(3/2), metres; a radius in cells.
    const double r0 = 0.5 * std::pow(height * cell / sp.slender, 1.5) / cell;
    const auto bark = static_cast<std::uint8_t>(sp.bark);
    // A direction `spread` off `dir`, at this angle round it.
    const auto fork = [&](V3 dir, double angle, double off) {
        V3 side = std::abs(dir.z) < 0.9 ? V3{dir.y, -dir.x, 0.0} : V3{1.0, 0.0, 0.0};
        side = normal(side);
        const V3 other{dir.y * side.z - dir.z * side.y, dir.z * side.x - dir.x * side.z, dir.x * side.y - dir.y * side.x};
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double co = std::cos(off);
        const double so = std::sin(off);
        return normal(V3{dir.x * co + (side.x * c + other.x * s) * so, dir.y * co + (side.y * c + other.y * s) * so, dir.z * co + (side.z * c + other.z * s) * so});
    };
    // One branch and whatever it forks into.
    std::function<void(V3, V3, double, double, int)> branch = [&](V3 at, V3 dir, double length, double r, int level) {
        const int steps = std::max(1, static_cast<int>(std::lround(length)));
        for (int k = 0; k < steps; ++k) {
            const double t = static_cast<double>(k) / steps;
            const V3 from = at;
            at = V3{at.x + dir.x, at.y + dir.y, at.z + dir.z};
            // No crown goes further than the map looks for it: the branch ends here, with its leaves.
            if (std::abs(at.x) > kTreeReach - 6 || std::abs(at.y) > kTreeReach - 6 || at.z > limit_z - 2 || at.z < 2.0) {
                break;
            }
            join(from, at, true, bark);
            disc(at, r * std::pow(1.0 - 0.45 * t, 0.75), bark);
            // Toward the light; and down under its own weight, as a cantilever: (l / r)^2, which a real branch answers by
            // stiffening as it grows, so it levels off.
            const double slenderness = length / (14.0 * (r + 0.8));
            const double bend = sp.droop * 0.05 * std::min(1.0, slenderness * slenderness) * t;
            dir = normal(V3{dir.x + 0.12 * (rng.next() - 0.5), dir.y + 0.12 * (rng.next() - 0.5), dir.z + sp.rise * 0.18 - bend});
            // Twigs and leaves all along the last of it.
            if (level >= sp.levels && sp.leaf > 0.0 && k % 2 == 1) {
                leaves(at, foliage * 0.85, sp.flat);
            }
        }
        if (level >= sp.levels) {
            if (sp.leaf > 0.0) {
                leaves(at, foliage * (1.0 + 0.4 * rng.next()), sp.flat);
                // A hanging kind trails what it has, down from where it ends.
                if (sp.droop > 1.0) {
                    V3 hang = at;
                    for (int k = 0; k < static_cast<int>(height * 0.35); ++k) {
                        hang.z -= 1.0;
                        if (hang.z < 3.0) {
                            break;
                        }
                        leaves(hang, 0.9, 1.0);
                    }
                }
            }
            return;
        }
        const double child_r = r * std::pow(1.0 - 0.45, 0.75) / std::sqrt(static_cast<double>(sp.limbs));
        const double child_l = length * std::pow(std::max(child_r, 0.15) / std::max(r, 0.15), 2.0 / 3.0);
        double angle = rng.next() * 6.283;
        for (int k = 0; k < sp.limbs; ++k) {
            angle += golden;
            branch(at, fork(dir, angle, spread * (0.8 + 0.4 * rng.next())), child_l * (0.8 + 0.4 * rng.next()), child_r, level + 1);
        }
    };
    const V3 up{0.0, 0.0, 1.0};
    switch (sp.form) {
    case TreeSpecies::Form::Spreading:
    case TreeSpecies::Form::Snag: {
        // A trunk with its taper and its flare, up to where it gives out into limbs.
        const double fork_at = height * (sp.rise < 0.06 ? 0.68 : 0.48);
        V3 at{0.0, 0.0, 0.0};
        V3 dir = fork(up, rng.next() * 6.283, 0.03 + 0.05 * rng.next());
        const int steps = static_cast<int>(fork_at);
        for (int k = 0; k < steps; ++k) {
            const double z = static_cast<double>(k) / height;
            const double flare = 1.0 + 0.7 * std::exp(-static_cast<double>(k) * cell);
            const V3 from = at;
            at = V3{at.x + dir.x, at.y + dir.y, at.z + dir.z};
            join(from, at, true, bark);
            disc(at, r0 * std::pow(1.0 - z, 0.75) * flare, bark);
        }
        const double r_top = r0 * std::pow(1.0 - fork_at / height, 0.75);
        const double child_r = r_top / std::sqrt(static_cast<double>(sp.limbs));
        // What is left of the height goes into the limbs; elastic similarity says how long they are against how thick.
        const double child_l = (height - fork_at) * 0.95 / std::max(0.35, std::cos(spread));
        double angle = rng.next() * 6.283;
        for (int k = 0; k < sp.limbs; ++k) {
            angle += golden;
            branch(at, fork(dir, angle, spread * (0.75 + 0.5 * rng.next())), child_l * (0.72 + 0.3 * rng.next()), child_r, 1);
        }
        break;
    }
    case TreeSpecies::Form::Conifer: {
        // The trunk goes all the way up; whorls of branches, shorter the higher they are: the crown is what the light allows.
        V3 at{0.0, 0.0, 0.0};
        double angle = rng.next() * 6.283;
        const int first = static_cast<int>(height * 0.2);
        const int whorl_every = std::max(2, static_cast<int>(std::lround(height / 11.0)));
        for (int k = 1; k <= height_cells; ++k) {
            const double z = static_cast<double>(k) / height;
            at = V3{0.0, 0.0, static_cast<double>(k)};
            disc(at, r0 * std::pow(1.0 - z * 0.98, 0.75) * (1.0 + 0.6 * std::exp(-k * cell)), bark);
            if (k >= first && (k - first) % whorl_every == 0) {
                const double reach = height * 0.30 * std::pow(1.0 - z, 0.8) + 1.0;
                for (int b = 0; b < sp.limbs; ++b) {
                    angle += golden;
                    V3 p = at;
                    V3 d = fork(up, angle, spread);
                    const int len = std::max(1, static_cast<int>(reach));
                    for (int j = 0; j < len; ++j) {
                        const V3 back = p;
                        p = V3{p.x + d.x, p.y + d.y, p.z + d.z};
                        if (j < len / 3) {
                            join(back, p, true, bark);
                            put(p.x, p.y, p.z, true, bark);
                        } else {
                            join(back, p, false, static_cast<std::uint8_t>(sp.colour));
                        }
                        leaves(p, sp.leaf * (0.75 + 0.5 * (1.0 - static_cast<double>(j) / len)), 0.7);
                        d = normal(V3{d.x, d.y, d.z - sp.droop * 0.06 * j + sp.rise * 0.2});
                    }
                }
            }
        }
        leaves(V3{0.0, 0.0, height}, sp.leaf + 0.4, 1.3);
        break;
    }
    case TreeSpecies::Form::Palm: {
        // A leaning, bending stem. Fronds go out and come down as a thrown stone does.
        V3 at{0.0, 0.0, 0.0};
        V3 dir = fork(up, rng.next() * 6.283, 0.18 + 0.2 * rng.next());
        for (int k = 0; k < height_cells; ++k) {
            const V3 from = at;
            at = V3{at.x + dir.x, at.y + dir.y, at.z + dir.z};
            join(from, at, true, bark);
            disc(at, std::max(0.5, r0 * (1.0 - 0.4 * static_cast<double>(k) / height)), bark);
            dir = normal(V3{dir.x, dir.y, dir.z + 0.03});
        }
        double angle = rng.next() * 6.283;
        const double frond = 3.0 + height * 0.28;
        for (int b = 0; b < sp.limbs; ++b) {
            angle += golden;
            const double dx = std::cos(angle);
            const double dy = std::sin(angle);
            V3 last = at;
            for (int j = 1; j <= static_cast<int>(frond); ++j) {
                const double t = j / frond;
                const double z = at.z + 1.0 + frond * 0.5 * t - frond * sp.droop * 0.95 * t * t;
                const V3 here{at.x + dx * j, at.y + dy * j, z};
                join(last, here, false, static_cast<std::uint8_t>(sp.colour));
                last = here;
                put(at.x + dx * j, at.y + dy * j, z, false, static_cast<std::uint8_t>(sp.colour));
                put(at.x + dx * j - dy, at.y + dy * j + dx, z, false, static_cast<std::uint8_t>(sp.colour));
            }
        }
        break;
    }
    case TreeSpecies::Form::Column: {
        // A stem and arms that go out and turn up. No wood: it is all green flesh.
        const auto flesh = static_cast<std::uint8_t>(sp.colour);
        const double r = std::max(0.7, r0);
        const auto ring = [&](V3 at, double radius) {
            const int reach = static_cast<int>(std::ceil(radius));
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        put(at.x + dx, at.y + dy, at.z, false, flesh);
                    }
                }
            }
        };
        for (int k = 1; k <= height_cells; ++k) {
            ring(V3{0.0, 0.0, static_cast<double>(k)}, r);
        }
        double angle = rng.next() * 6.283;
        for (int b = 0; b < sp.limbs; ++b) {
            angle += golden;
            const int from = static_cast<int>(height * (0.35 + 0.3 * rng.next()));
            const double out_by = 2.0 + 2.0 * rng.next();
            const double dx = std::cos(angle);
            const double dy = std::sin(angle);
            for (double j = 0.0; j <= out_by; j += 0.5) {
                ring(V3{dx * j, dy * j, static_cast<double>(from)}, r * 0.7);
            }
            const int rise_by = static_cast<int>(height * 0.3 + 2.0 * rng.next());
            for (int k = 0; k < rise_by; ++k) {
                ring(V3{dx * out_by, dy * out_by, static_cast<double>(from + k)}, r * 0.7);
            }
        }
        break;
    }
    case TreeSpecies::Form::Bush: {
        // Stems out of the ground, each with its lump.
        double angle = rng.next() * 6.283;
        for (int b = 0; b < sp.limbs; ++b) {
            angle += golden;
            const double reach = 0.3 + 0.7 * rng.next();
            V3 at{0.0, 0.0, 0.0};
            const V3 d = fork(up, angle, spread * reach);
            const int len = std::max(1, static_cast<int>(height * (0.6 + 0.4 * rng.next())));
            for (int k = 0; k < len; ++k) {
                at = V3{at.x + d.x, at.y + d.y, at.z + d.z};
                if (k < len / 2) {
                    put(at.x, at.y, at.z, true, bark);
                }
            }
            leaves(at, sp.leaf * (0.75 + 0.5 * rng.next()), sp.flat);
        }
        break;
    }
    }
    // One thing in a place, wood over leaves.
    std::sort(out.begin(), out.end(), [](const TreeVoxel& a, const TreeVoxel& b) {
        return std::tie(a.dz, a.dy, a.dx, b.wood) < std::tie(b.dz, b.dy, b.dx, a.wood);
    });
    out.erase(std::unique(out.begin(), out.end(), [](const TreeVoxel& a, const TreeVoxel& b) { return a.dx == b.dx && a.dy == b.dy && a.dz == b.dz; }), out.end());
    return out;
}

void WorldMap::read_growth(const std::string& text)
{
    if (text.empty()) {
        errors_.push_back("assets/world/trees.world not found");
        return;
    }
    const auto biome_of = [](const std::string& word, Biome& out) {
        for (int b = 0; b < 12; ++b) {
            if (word == biome_name(static_cast<Biome>(b)) || word == kBiomeWords[static_cast<std::size_t>(b)]) {
                out = static_cast<Biome>(b);
                return true;
            }
        }
        return false;
    };
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
        const std::string where = "trees.world:" + std::to_string(number) + ": ";
        try {
            if (w[0] == "land" && w.size() == 8) {
                Biome b{};
                if (!biome_of(w[1], b)) {
                    errors_.push_back(where + "no land called '" + w[1] + "'");
                    continue;
                }
                growth_[static_cast<std::size_t>(b)] = LandGrowth{std::stod(w[3]), std::stod(w[5]), std::stod(w[7])};
            } else if ((w[0] == "tree" || w[0] == "flora") && w.size() >= 4 && w[2] == "biomes") {
                std::vector<Biome> biomes;
                std::size_t i = 3;
                for (Biome b{}; i < w.size() && biome_of(w[i], b); ++i) {
                    biomes.push_back(b);
                }
                if (biomes.empty()) {
                    errors_.push_back(where + "'" + w[1] + "' grows on no land anyone knows");
                    continue;
                }
                if (w[0] == "flora") {
                    FloraKind f{.name = w[1], .biomes = biomes};
                    for (; i + 1 < w.size(); i += 2) {
                        if (w[i] == "height") {
                            f.height = std::stoi(w[i + 1]);
                        } else if (w[i] == "colour") {
                            f.colour = std::stoi(w[i + 1]);
                        } else if (w[i] == "share") {
                            f.share = std::stod(w[i + 1]);
                        } else {
                            errors_.push_back(where + "unknown word '" + w[i] + "'");
                        }
                    }
                    flora_.push_back(f);
                    continue;
                }
                TreeSpecies t{.name = w[1], .biomes = biomes};
                while (i + 1 < w.size()) {
                    const std::string& key = w[i];
                    const std::string& value = w[i + 1];
                    if (key == "height" && i + 2 < w.size()) {
                        t.height_min = std::stod(value);
                        t.height_max = std::stod(w[i + 2]);
                        i += 3;
                        continue;
                    }
                    if (key == "form") {
                        static const std::map<std::string, TreeSpecies::Form> kForms{{"spreading", TreeSpecies::Form::Spreading}, {"conifer", TreeSpecies::Form::Conifer}, {"palm", TreeSpecies::Form::Palm},
                            {"column", TreeSpecies::Form::Column}, {"bush", TreeSpecies::Form::Bush}, {"snag", TreeSpecies::Form::Snag}};
                        const auto found = kForms.find(value);
                        if (found == kForms.end()) {
                            errors_.push_back(where + "no form '" + value + "'");
                        } else {
                            t.form = found->second;
                        }
                    } else if (key == "slender") {
                        t.slender = std::stod(value);
                    } else if (key == "limbs") {
                        t.limbs = std::stoi(value);
                    } else if (key == "levels") {
                        t.levels = std::stoi(value);
                    } else if (key == "spread") {
                        t.spread = std::stod(value);
                    } else if (key == "rise") {
                        t.rise = std::stod(value);
                    } else if (key == "droop") {
                        t.droop = std::stod(value);
                    } else if (key == "leaf") {
                        t.leaf = std::stod(value);
                    } else if (key == "flat") {
                        t.flat = std::stod(value);
                    } else if (key == "bark") {
                        t.bark = std::stoi(value);
                    } else if (key == "colour") {
                        t.colour = std::stoi(value);
                    } else if (key == "share") {
                        t.share = std::stod(value);
                    } else {
                        errors_.push_back(where + "unknown word '" + key + "'");
                    }
                    i += 2;
                }
                if (t.height_max < t.height_min || t.slender <= 0.0 || t.limbs < 1 || t.levels < 1 || t.levels > 4 || t.flat <= 0.0) {
                    errors_.push_back(where + "'" + t.name + "' could not grow");
                    continue;
                }
                species_.push_back(t);
            } else {
                errors_.push_back(where + "expected land, tree or flora");
            }
        } catch (...) {
            errors_.push_back(where + "a number was expected");
        }
    }
    // A big tree stands in its own stand; a small one anywhere. Each land has its list, by share.
    for (int k = 0; k < static_cast<int>(species_.size()); ++k) {
        const TreeSpecies& t = species_[static_cast<std::size_t>(k)];
        for (const Biome b : t.biomes) {
            std::array<std::vector<int>, 12>& list = t.height_max >= 11.0 ? big_of_ : small_of_;
            for (int n = 0; n < static_cast<int>(t.share); ++n) {
                list[static_cast<std::size_t>(b)].push_back(k);
            }
        }
    }
    for (int k = 0; k < static_cast<int>(flora_.size()); ++k) {
        const FloraKind& f = flora_[static_cast<std::size_t>(k)];
        for (const Biome b : f.biomes) {
            for (int n = 0; n < static_cast<int>(f.share); ++n) {
                flora_of_[static_cast<std::size_t>(b)].push_back(k);
            }
        }
    }
}

WorldMap::WorldMap(std::uint64_t seed, int sea_level, int height, int chunk, const std::string& styles)
    : seed_(seed), sea_(sea_level), height_(height), chunk_(chunk)
{
    read_growth(read_trees_file());
    styles_ = parse_styles(styles, errors_);
    // A style that does not say how it shuts its doors: its name decides, and some do not.
    for (Style& s : styles_) {
        if (!s.door_set) {
            std::uint64_t h = 1469598103934665603ull;
            for (const char c : s.name) {
                h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
            }
            const int roll = static_cast<int>((h >> 13) % 100);
            s.door = roll < 18 ? Style::Door::None : roll < 55 ? Style::Door::Plank : roll < 75 ? Style::Door::Double : roll < 90 ? Style::Door::Slide : Style::Door::Lift;
            s.door_stone = s.door == Style::Door::Lift && s.wall == Style::Stuff::Stone;
            s.door_material = s.door_stone ? s.wall_material : (h >> 21) % 3 == 0 ? material::charred : material::timber;
        }
    }
    if (styles.empty()) {
        errors_.push_back("assets/world/styles.world not found");
    }
    if (!errors_.empty() || styles_.empty()) {
        std::vector<std::string> ignored;
        styles_ = parse_styles(kPlainStyle, ignored);
    }
    styles_.reserve(16384); // settlements hold on to theirs by index, buildings by index: nothing is moved anyway
    make_continent();
    make_climate();
    make_water();
    classify_biomes();
    make_bounds();
    road_near_ = index_boxes({}); // the plans are drawn before the roads
    make_history();
    make_roads();
}

std::shared_ptr<const WorldMap> WorldMap::of(const TerrainConfig& config)
{
    static std::mutex mutex;
    static std::map<std::tuple<std::uint64_t, int, int, int>, std::shared_ptr<const WorldMap>> maps;
    const auto key = std::make_tuple(config.seed, config.sea_level, config.nz, config.chunk);
    const std::lock_guard<std::mutex> lock(mutex);
    std::shared_ptr<const WorldMap>& map = maps[key];
    if (!map) {
        map = std::make_shared<const WorldMap>(config.seed, config.sea_level, config.nz, config.chunk, read_styles_file());
    }
    return map;
}

// --- continent -----------------------------------------------------------------------------------------------

namespace {

// A smooth noise over the whole map: worked out every `stride` nodes and filled in between, since it could not
// show any finer anyway.
std::vector<float> noise_field(std::uint64_t seed, double scale, int octaves, int stride)
{
    const int side = kN / stride + 2;
    std::vector<float> coarse(static_cast<std::size_t>(side) * side);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            coarse[static_cast<std::size_t>(y * side + x)] = static_cast<float>(fbm(seed, x * stride, y * stride, scale, octaves));
        }
    }
    std::vector<float> out(static_cast<std::size_t>(kN) * kN);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < kN; ++y) {
        const int iy = y / stride;
        const float ty = static_cast<float>(y % stride) / stride;
        for (int x = 0; x < kN; ++x) {
            const int ix = x / stride;
            const float tx = static_cast<float>(x % stride) / stride;
            const float a = coarse[static_cast<std::size_t>(iy * side + ix)];
            const float b = coarse[static_cast<std::size_t>(iy * side + ix + 1)];
            const float c = coarse[static_cast<std::size_t>((iy + 1) * side + ix)];
            const float d = coarse[static_cast<std::size_t>((iy + 1) * side + ix + 1)];
            out[static_cast<std::size_t>(at(x, y))] = (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
        }
    }
    return out;
}

} // namespace

void WorldMap::make_continent()
{
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    elevation_.assign(count, 0.0f);
    rough_.assign(count, 0.0f);
    // Plates, one to a cell of a coarse grid, off its centre: where two drive into each other, a range stands.
    constexpr int kPlateCell = 150;
    const int plates_side = kN / kPlateCell + 8;
    struct Plate {
        float x, y, dx, dy;
    };
    std::vector<Plate> plates(static_cast<std::size_t>(plates_side) * plates_side);
    for (int gy = 0; gy < plates_side; ++gy) {
        for (int gx = 0; gx < plates_side; ++gx) {
            Rng rng{hash(seed_, 0x101, gx, gy)};
            const double angle = rng.next() * 6.283185307;
            const float px = static_cast<float>((gx - 4 + rng.next()) * kPlateCell);
            const float py = static_cast<float>((gy - 4 + rng.next()) * kPlateCell);
            plates[static_cast<std::size_t>(gy * plates_side + gx)] = Plate{px, py, static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
        }
    }
    const std::vector<float> warp_x = noise_field(seed_ ^ 0x21, 240.0, 4, 8);
    const std::vector<float> warp_y = noise_field(seed_ ^ 0x22, 240.0, 4, 8);
    const std::vector<float> wander_x = noise_field(seed_ ^ 0x26, 48.0, 5, 2);
    const std::vector<float> wander_y = noise_field(seed_ ^ 0x27, 48.0, 5, 2);
    const std::vector<float> masses = noise_field(seed_ ^ 0x23, 460.0, 6, 4);   // continents
    const std::vector<float> chains = noise_field(seed_ ^ 0x28, 64.0, 4, 4);    // island chains
    const std::vector<float> chain_belt = noise_field(seed_ ^ 0x29, 300.0, 3, 8); // and where they lie
    const std::vector<float> ridges = noise_field(seed_ ^ 0x24, 34.0, 4, 2);
    const std::vector<float> rolling = noise_field(seed_ ^ 0x25, 26.0, 3, 2);
    const std::vector<float> rifts = noise_field(seed_ ^ 0x2a, 130.0, 3, 2);      // where the land has split
    const std::vector<float> rift_belt = noise_field(seed_ ^ 0x2b, 340.0, 2, 8);  // and where it does so
    const std::vector<float> massifs = noise_field(seed_ ^ 0x2c, 170.0, 3, 4);    // great lumps that are no plate's doing
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < kN; ++y) {
        for (int x = 0; x < kN; ++x) {
            const std::size_t n = static_cast<std::size_t>(at(x, y));
            const double wx = x + 300.0 * (warp_x[n] - 0.5);
            const double wy = y + 300.0 * (warp_y[n] - 0.5);
            // The rim of the map is sea, whatever else is true.
            const double u = std::max(std::abs((x + 0.5) / kN * 2.0 - 1.0), std::abs((y + 0.5) / kN * 2.0 - 1.0));
            const double rim = smoothstep(0.86, 0.985, u);
            double land = (masses[n] - 0.5) * 1.5 - rim * 0.5;
            // Island chains rise out of the sea where the belt is.
            const double chain = (chains[n] - 0.61) * 1.2 * smoothstep(0.42, 0.62, chain_belt[n]) * (1.0 - rim);
            land = std::max(land, chain);
            // The two plates nearest: their border, and whether they meet head on.
            const double px = wx + 70.0 * (wander_x[n] - 0.5);
            const double py = wy + 70.0 * (wander_y[n] - 0.5);
            const int gx = std::clamp(static_cast<int>(std::floor(px / kPlateCell)) + 4, 2, plates_side - 3);
            const int gy = std::clamp(static_cast<int>(std::floor(py / kPlateCell)) + 4, 2, plates_side - 3);
            const Plate* first = nullptr;
            const Plate* second = nullptr;
            double d1 = 1e30;
            double d2 = 1e30;
            for (int oy = -2; oy <= 2; ++oy) {
                for (int ox = -2; ox <= 2; ++ox) {
                    const Plate& p = plates[static_cast<std::size_t>((gy + oy) * plates_side + gx + ox)];
                    const double d = std::hypot(px - p.x, py - p.y);
                    if (d < d1) {
                        d2 = d1;
                        second = first;
                        d1 = d;
                        first = &p;
                    } else if (d < d2) {
                        d2 = d;
                        second = &p;
                    }
                }
            }
            const double ab = std::max(1e-6, std::hypot(static_cast<double>(second->x - first->x), static_cast<double>(second->y - first->y)));
            const double closing = ((first->dx - second->dx) * (second->x - first->x) + (first->dy - second->dy) * (second->y - first->y)) / ab;
            const double border = saturate(1.0 - (d2 - d1) / 16.0);
            const double ridged = 1.0 - std::abs(2.0 * ridges[n] - 1.0);
            const double range = border * border * saturate(closing) * ridged * saturate(land * 7.0);
            const double hills = saturate((rolling[n] - 0.46) * 3.5) * saturate(land * 8.0);
            const double massif = smoothstep(0.60, 0.80, massifs[n]) * saturate(land * 5.0);
            // A rift: narrow, where the noise crosses its middle, and only in its belt. It stops short of the sea.
            const double rift_line = 1.0 - std::abs(2.0 * rifts[n] - 1.0);
            const double rift = std::pow(smoothstep(0.90, 1.0, rift_line), 2.0) * smoothstep(0.48, 0.64, rift_belt[n]);
            double e;
            if (land > 0.0) {
                e = sea_ + 8.0 + land * 150.0 + hills * 110.0 + range * 1300.0 + massif * 520.0 * (0.55 + 0.45 * ridged);
                e -= std::min(e - sea_ - 10.0, 360.0 * rift * saturate(land * 6.0));
            } else {
                e = sea_ + land * 520.0;
            }
            elevation_[n] = static_cast<float>(std::clamp(e, 30.0, height_ - 110.0));
            rough_[n] = static_cast<float>((land > 0.0 ? 3.5 : 2.0) + 15.0 * hills + 75.0 * range + 28.0 * massif + 14.0 * rift);
        }
    }
}

// --- climate: wind carries the sea inland, the ranges wring it out ---------------------------------------------

void WorldMap::make_climate()
{
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    temperature_.assign(count, 0.0f);
    rain_.assign(count, 0.0f);
    const bool from_west = (hash(seed_, 0x31, 0) & 1u) == 0;
    const bool north_cold = (hash(seed_, 0x32, 0) & 1u) == 0;
    const std::vector<float> weather = noise_field(seed_ ^ 0x33, 70.0, 3, 8);
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < kN; ++y) {
        double carried = 1.0;
        double before = elevation_[static_cast<std::size_t>(at(from_west ? 0 : kN - 1, y))];
        for (int i = 0; i < kN; ++i) {
            const int x = from_west ? i : kN - 1 - i;
            const std::size_t n = static_cast<std::size_t>(at(x, y));
            const double e = elevation_[n];
            if (e < sea_) {
                carried = std::min(1.0, carried + 0.06);
                rain_[n] = static_cast<float>(carried * 0.5);
            } else {
                const double wrung = carried * (0.014 + std::max(0.0, e - before) * 0.006);
                const double fell = std::min(carried, wrung);
                carried -= fell * 0.3; // the rest comes back up, off the land
                rain_[n] = static_cast<float>(saturate(fell * 35.0));
            }
            before = e;
            const double latitude = (north_cold ? y : kN - 1 - y) / static_cast<double>(kN - 1);
            const double height_m = std::max(0.0, e - sea_) * 0.5;
            const double t = -9.0 + 44.0 * latitude + 12.0 * (weather[n] - 0.5) - height_m * 0.0065 * 4.0;
            temperature_[n] = static_cast<float>(t);
        }
    }
    // Rain drifts between rows, and weather has moods of its own.
    std::vector<float> next = rain_;
    for (int pass = 0; pass < 3; ++pass) {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < kN; ++y) {
            for (int x = 0; x < kN; ++x) {
                double sum = 0.0;
                for (int k = -4; k <= 4; ++k) {
                    sum += rain_[static_cast<std::size_t>(at(std::clamp(x + k, 0, kN - 1), y))];
                }
                next[static_cast<std::size_t>(at(x, y))] = static_cast<float>(sum / 9.0);
            }
        }
#pragma omp parallel for schedule(static)
        for (int y = 0; y < kN; ++y) {
            for (int x = 0; x < kN; ++x) {
                double sum = 0.0;
                for (int k = -4; k <= 4; ++k) {
                    sum += next[static_cast<std::size_t>(at(x, std::clamp(y + k, 0, kN - 1)))];
                }
                rain_[static_cast<std::size_t>(at(x, y))] = static_cast<float>(sum / 9.0);
            }
        }
    }
    const std::vector<float> moods = noise_field(seed_ ^ 0x34, 60.0, 3, 8);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < static_cast<int>(count); ++n) {
        rain_[static_cast<std::size_t>(n)] = static_cast<float>(saturate(rain_[static_cast<std::size_t>(n)] * (0.45 + 1.1 * moods[static_cast<std::size_t>(n)])));
    }
}

// --- water: every pit fills to its spill, every drop has a way to the sea ---------------------------------------

void WorldMap::make_water()
{
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    filled_.assign(count, 0.0f);
    down_.assign(count, -1);
    flow_.assign(count, 0.0f);
    level_.assign(count, 0.0f);
    lake_.assign(count, 0);
    order_.clear();
    static constexpr int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static constexpr int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    auto flood = [&] {
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
        std::vector<char> seen(count, 0);
        order_.clear();
        order_.reserve(count);
        for (int y = 0; y < kN; ++y) {
            for (int x = 0; x < kN; ++x) {
                const int n = at(x, y);
                const bool border = x == 0 || y == 0 || x == kN - 1 || y == kN - 1;
                if (border || elevation_[static_cast<std::size_t>(n)] < sea_) {
                    seen[static_cast<std::size_t>(n)] = 1;
                    filled_[static_cast<std::size_t>(n)] = std::max(elevation_[static_cast<std::size_t>(n)], static_cast<float>(sea_));
                    down_[static_cast<std::size_t>(n)] = -1;
                    // Only the sea's edge needs to be looked at from.
                    bool coast = false;
                    for (int k = 0; k < 8 && !coast; ++k) {
                        const int ax = x + kDx[k];
                        const int ay = y + kDy[k];
                        coast = ax >= 0 && ay >= 0 && ax < kN && ay < kN && elevation_[static_cast<std::size_t>(at(ax, ay))] >= sea_;
                    }
                    if (coast || border) {
                        open.emplace(filled_[static_cast<std::size_t>(n)], n);
                    } else {
                        order_.push_back(n);
                    }
                }
            }
        }
        while (!open.empty()) {
            const auto [level, n] = open.top();
            open.pop();
            order_.push_back(n);
            const int x = n % kN;
            const int y = n / kN;
            for (int k = 0; k < 8; ++k) {
                const int ax = x + kDx[k];
                const int ay = y + kDy[k];
                if (ax < 0 || ay < 0 || ax >= kN || ay >= kN) {
                    continue;
                }
                const std::size_t m = static_cast<std::size_t>(at(ax, ay));
                if (seen[m]) {
                    continue;
                }
                seen[m] = 1;
                filled_[m] = std::max(elevation_[m], level + 0.02f);
                down_[m] = n;
                open.emplace(filled_[m], static_cast<int>(m));
            }
        }
    };
    auto gather = [&] {
        for (std::size_t i = 0; i < count; ++i) {
            flow_[i] = elevation_[i] >= sea_ ? rain_[i] : 0.0f;
        }
        for (std::size_t i = order_.size(); i-- > 0;) {
            const int n = order_[i];
            const int d = down_[static_cast<std::size_t>(n)];
            if (d >= 0) {
                flow_[static_cast<std::size_t>(d)] += flow_[static_cast<std::size_t>(n)];
            }
        }
    };
    constexpr float kRiver = 18.0f;
    flood();
    gather();
    // Rivers cut their valleys; then the land is drained again, over what they cut.
    for (std::size_t i = 0; i < count; ++i) {
        if (flow_[i] > kRiver && elevation_[i] > sea_ && filled_[i] - elevation_[i] < 1.0f) {
            const float cut = std::min(1.5f + 0.5f * std::sqrt(flow_[i]), 0.4f * (elevation_[i] - sea_));
            elevation_[i] -= cut;
        }
    }
    flood();
    gather();
    for (const int n : order_) {
        const std::size_t i = static_cast<std::size_t>(n);
        const int d = down_[i];
        if (d < 0) {
            level_[i] = static_cast<float>(sea_);
            continue;
        }
        lake_[i] = filled_[i] - elevation_[i] > 2.0f ? 1 : 0;
        // Water only ever runs down.
        const float wanted = lake_[i] ? filled_[i] : filled_[i] - 0.5f;
        level_[i] = std::max(level_[static_cast<std::size_t>(d)], wanted);
    }
    // The channels, node to node, each node a little off its grid point so nothing runs straight.
    auto spot = [&](int n, float& px, float& py) {
        const int x = n % kN;
        const int y = n / kN;
        const std::uint64_t h = hash(seed_ ^ 0x41, x, y);
        const double jx = (unit(h) - 0.5) * 0.6;
        const double jy = (unit(mix64(h)) - 0.5) * 0.6;
        px = static_cast<float>((x + 0.5 + jx) * kNodeCells - kHalfCells);
        py = static_cast<float>((y + 0.5 + jy) * kNodeCells - kHalfCells);
    };
    segments_.clear();
    std::vector<std::array<int, 4>> boxes;
    for (std::size_t i = 0; i < count; ++i) {
        const int d = down_[i];
        if (d < 0 || flow_[i] <= kRiver || elevation_[i] < sea_ || (lake_[i] && lake_[static_cast<std::size_t>(d)])) {
            continue;
        }
        Segment s{};
        spot(static_cast<int>(i), s.ax, s.ay);
        spot(d, s.bx, s.by);
        s.level_a = level_[i];
        s.level_b = level_[static_cast<std::size_t>(d)];
        s.width = static_cast<float>(std::clamp(2.5 + 1.3 * std::sqrt(flow_[i] - kRiver), 3.0, 34.0));
        segments_.push_back(s);
        const int reach = 170;
        boxes.push_back({static_cast<int>(std::min(s.ax, s.bx)) - reach, static_cast<int>(std::min(s.ay, s.by)) - reach,
            static_cast<int>(std::max(s.ax, s.bx)) + reach, static_cast<int>(std::max(s.ay, s.by)) + reach});
    }
    river_near_ = index_boxes(boxes);
}

WorldMap::Index WorldMap::index_boxes(const std::vector<std::array<int, 4>>& boxes) const
{
    Index index;
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    std::vector<std::uint32_t> sizes(count, 0);
    auto each_node = [&](const std::array<int, 4>& b, auto&& fn) {
        const int x0 = std::clamp((b[0] + kHalfCells) / kNodeCells, 0, kN - 1);
        const int y0 = std::clamp((b[1] + kHalfCells) / kNodeCells, 0, kN - 1);
        const int x1 = std::clamp((b[2] + kHalfCells) / kNodeCells, 0, kN - 1);
        const int y1 = std::clamp((b[3] + kHalfCells) / kNodeCells, 0, kN - 1);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                fn(at(x, y));
            }
        }
    };
    for (const auto& b : boxes) {
        each_node(b, [&](int n) { ++sizes[static_cast<std::size_t>(n)]; });
    }
    index.first.assign(count + 1, 0);
    for (std::size_t i = 0; i < count; ++i) {
        index.first[i + 1] = index.first[i] + sizes[i];
    }
    index.items.resize(index.first[count]);
    std::vector<std::uint32_t> filled(count, 0);
    for (std::uint32_t k = 0; k < boxes.size(); ++k) {
        each_node(boxes[k], [&](int n) {
            const std::size_t i = static_cast<std::size_t>(n);
            index.items[index.first[i] + filled[i]++] = k;
        });
    }
    return index;
}

void WorldMap::classify_biomes()
{
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    biome_.assign(count, Biome::Ocean);
    const auto near_sea = [&](int n) {
        const int x = n % kN;
        const int y = n / kN;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (elevation_[static_cast<std::size_t>(at(std::clamp(x + dx, 0, kN - 1), std::clamp(y + dy, 0, kN - 1)))] < sea_) {
                    return true;
                }
            }
        }
        return false;
    };
#pragma omp parallel for schedule(static)
    for (int n = 0; n < static_cast<int>(count); ++n) {
        const std::size_t i = static_cast<std::size_t>(n);
        const double e = elevation_[i] - sea_;
        const double t = temperature_[i];
        const double r = rain_[i];
        Biome b = Biome::Grassland;
        if (e < 0.0) {
            b = Biome::Ocean;
        } else if (e < 7.0 && near_sea(n)) {
            b = Biome::Beach;
        } else if (t < -6.0) {
            b = Biome::Snow;
        } else if (rough_[i] > 26.0 && e > 150.0) {
            b = Biome::Mountain;
        } else if (t < 1.0) {
            b = Biome::Tundra;
        } else if (t < 8.0) {
            b = r > 0.18 ? Biome::Taiga : Biome::Tundra;
        } else if (r > 0.5 && e < 30.0 && (flow_[i] > 9.0f || lake_[i])) {
            b = Biome::Swamp;
        } else if (t > 20.0) {
            b = r > 0.5 ? Biome::Jungle : (r > 0.16 ? Biome::Savanna : Biome::Desert);
        } else if (r > 0.3) {
            b = Biome::Forest;
        } else if (r > 0.11) {
            b = Biome::Grassland;
        } else {
            b = t > 12.0 ? Biome::Desert : Biome::Grassland;
        }
        biome_[i] = b;
    }
}

void WorldMap::make_bounds()
{
    constexpr double kFeature = 170.0; // what can stand on the ground, in cells
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    std::vector<float> low(count);
    std::vector<float> high(count);
    for (int y = 0; y < kN; ++y) {
        for (int x = 0; x < kN; ++x) {
            float lo = 1e9f;
            float hi = -1e9f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t m = static_cast<std::size_t>(at(std::clamp(x + dx, 0, kN - 1), std::clamp(y + dy, 0, kN - 1)));
                    const float margin = rough_[m] * 1.7f;
                    lo = std::min(lo, elevation_[m] - margin);
                    hi = std::max(hi, elevation_[m] + margin);
                }
            }
            low[static_cast<std::size_t>(at(x, y))] = lo - 12.0f;
            high[static_cast<std::size_t>(at(x, y))] = std::max(hi, static_cast<float>(sea_)) + static_cast<float>(kFeature);
        }
    }
    low_ = {std::move(low)};
    high_ = {std::move(high)};
    for (int size = kN / 2; size >= 1; size /= 2) {
        const std::vector<float>& fine_low = low_.back();
        const std::vector<float>& fine_high = high_.back();
        const int fine = size * 2;
        std::vector<float> cl(static_cast<std::size_t>(size) * size);
        std::vector<float> ch(static_cast<std::size_t>(size) * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float lo = 1e9f;
                float hi = -1e9f;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const std::size_t m = static_cast<std::size_t>((y * 2 + dy) * fine + x * 2 + dx);
                        lo = std::min(lo, fine_low[m]);
                        hi = std::max(hi, fine_high[m]);
                    }
                }
                cl[static_cast<std::size_t>(y * size + x)] = lo;
                ch[static_cast<std::size_t>(y * size + x)] = hi;
            }
        }
        low_.push_back(std::move(cl));
        high_.push_back(std::move(ch));
    }
}

void WorldMap::height_bounds(int x0, int y0, int x1, int y1, int& lo, int& hi) const
{
    const auto clamp_node = [](int cell) { return std::clamp((cell + kHalfCells) / kNodeCells, 0, kN - 1); };
    const int nx0 = clamp_node(x0);
    const int ny0 = clamp_node(y0);
    const int nx1 = clamp_node(x1);
    const int ny1 = clamp_node(y1);
    int level = 0;
    while ((nx1 >> level) - (nx0 >> level) > 1 || (ny1 >> level) - (ny0 >> level) > 1) {
        ++level;
    }
    const int size = kN >> level;
    float low = 1e9f;
    float high = -1e9f;
    for (int y = ny0 >> level; y <= ny1 >> level; ++y) {
        for (int x = nx0 >> level; x <= nx1 >> level; ++x) {
            low = std::min(low, low_[static_cast<std::size_t>(level)][static_cast<std::size_t>(y * size + x)]);
            high = std::max(high, high_[static_cast<std::size_t>(level)][static_cast<std::size_t>(y * size + x)]);
        }
    }
    // Beyond the map, the sea.
    if (x0 < -kHalfCells || y0 < -kHalfCells || x1 > kHalfCells || y1 > kHalfCells) {
        low = std::min(low, static_cast<float>(sea_ - 70));
        high = std::max(high, static_cast<float>(sea_ + 2));
    }
    lo = std::max(0, static_cast<int>(std::floor(low)));
    hi = std::min(height_, static_cast<int>(std::ceil(high)));
}

// --- history -------------------------------------------------------------------------------------------------

namespace {

std::string make_name(Rng& rng, int culture)
{
    static const char* const kFirst[6][6] = {{"Bel", "Har", "Tor", "Wen", "Eld", "Mar"}, {"Ka", "Su", "Ra", "Ne", "Tu", "Az"},
        {"Skar", "Vor", "Ulf", "Hja", "Brun", "Dag"}, {"Ly", "Ae", "Sil", "Ora", "Thi", "Ela"}, {"Mok", "Zul", "Gra", "Ush", "Ogh", "Tar"},
        {"Ver", "Cas", "Dom", "Lia", "Ser", "Ton"}};
    static const char* const kLast[6][6] = {{"ford", "ham", "wick", "ton", "mere", "stead"}, {"bar", "mun", "ris", "tep", "kesh", "lun"},
        {"heim", "vik", "gard", "by", "fell", "stad"}, {"thas", "wyn", "mira", "lis", "dor", "ien"}, {"gul", "dun", "rok", "mar", "zan", "ash"},
        {"ona", "ia", "enza", "ar", "ium", "ello"}};
    const int c = culture % 6;
    return std::string(kFirst[c][rng.range(0, 5)]) + kLast[c][rng.range(0, 5)];
}

constexpr int kAges = 16;

int tier_of(int people) { return people < 70 ? 0 : (people < 350 ? 1 : (people < 1800 ? 2 : 3)); }
constexpr int kRadius[4] = {50, 115, 170, 280};

} // namespace

// HOW A PEOPLE COMES TO BUILD AS IT DOES: from the land it began in. What grows there, what can be quarried, how much
// it rains and how cold it gets narrow what makes sense; within that, a people has its own mind.
Style WorldMap::architect(std::uint64_t seed, Biome land, double temperature, double rain, const std::string& name) const
{
    Rng r{seed};
    Style s;
    s.name = name;
    s.biomes = {land};
    const auto pick = [&](std::initializer_list<int> options) { return *(options.begin() + r.range(0, static_cast<int>(options.size()) - 1)); };
    const bool forest = land == Biome::Forest || land == Biome::Taiga || land == Biome::Jungle || land == Biome::Swamp;
    const bool dry = land == Biome::Desert || land == Biome::Savanna || rain < 0.18;
    const bool cold = temperature < 6.0;
    const bool hot = temperature > 20.0;
    // Walls: what is at hand.
    if (land == Biome::Swamp || (land == Biome::Jungle && r.next() < 0.6)) {
        s.wall = Style::Stuff::Wood;
        s.wall_material = pick({material::reed, material::timber});
    } else if (forest && r.next() < 0.7) {
        s.wall = Style::Stuff::Wood;
        s.wall_material = pick({material::timber, material::timber, material::reed});
    } else if (land == Biome::Desert) {
        s.wall_material = pick({material::sandstone, material::clay, material::plaster});
    } else if (land == Biome::Mountain || land == Biome::Tundra || land == Biome::Snow) {
        s.wall_material = pick({material::granite, material::slate, material::basalt});
    } else {
        s.wall_material = pick({material::brick, material::plaster, material::clay, material::granite, material::sandstone});
    }
    s.base_material = s.wall == Style::Stuff::Wood ? pick({material::granite, material::slate, material::basalt})
                                                   : pick({material::basalt, material::slate, s.wall_material});
    // Roofs: how the weather falls on them.
    if (cold) {
        s.roof = pick({0, 1, 1}) == 0 ? Style::Roof::Gable : Style::Roof::Hip;
        s.pitch = r.range(2, 3); // snow must slide off
        s.roof_stuff = r.next() < 0.5 ? Style::Stuff::Wood : Style::Stuff::Stone;
        s.roof_material = s.roof_stuff == Style::Stuff::Wood ? material::timber : material::slate;
    } else if (dry && hot) {
        const bool can_dome = s.wall == Style::Stuff::Stone;
        s.roof = can_dome && r.next() < 0.5 ? Style::Roof::Dome : Style::Roof::Flat;
        s.pitch = 1;
        s.roof_stuff = Style::Stuff::Stone;
        s.roof_material = pick({material::clay, material::sandstone, material::plaster, material::chalk});
    } else if (rain > 0.4) {
        s.roof = pick({0, 1}) == 0 ? Style::Roof::Gable : Style::Roof::Hip;
        s.pitch = r.range(1, 2);
        s.roof_stuff = r.next() < 0.5 ? Style::Stuff::Leaf : Style::Stuff::Stone;
        s.roof_material = s.roof_stuff == Style::Stuff::Leaf ? pick({material::thatch, material::reed}) : pick({material::tile, material::slate});
    } else {
        s.roof = pick({0, 0, 1, 2}) == 0 ? Style::Roof::Gable : (r.next() < 0.5 ? Style::Roof::Hip : Style::Roof::Flat);
        s.pitch = 1;
        s.roof_stuff = r.next() < 0.3 ? Style::Stuff::Wood : Style::Stuff::Stone;
        s.roof_material = s.roof_stuff == Style::Stuff::Wood ? pick({material::thatch, material::timber}) : pick({material::tile, material::slate, material::clay});
    }
    // A roof of the same stuff as the wall reads as one lump, and nobody builds to look like that.
    if (s.roof_material == s.wall_material) {
        s.roof_material = s.roof_stuff == Style::Stuff::Stone ? (s.wall_material == material::slate ? material::tile : material::slate)
                                                              : (s.wall_material == material::thatch ? material::reed : material::thatch);
    }
    s.eaves = s.roof == Style::Roof::Flat || s.roof == Style::Roof::Dome ? 0 : r.range(hot ? 1 : 0, 2); // shade, in the sun
    // How a house is shut: where it is hot a people may not shut it at all; where wood is short, stone slides or lifts.
    // (From its name, not from `r`: how a door works changes nothing else about how a people builds.)
    s.door_set = true;
    {
        std::uint64_t h = 1469598103934665603ull;
        for (const char c : name) {
            h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
        }
        h ^= h >> 29;
        const double roll = static_cast<double>(h % 1000u) / 1000.0;
        const double second = static_cast<double>((h >> 12) % 1000u) / 1000.0;
        if (hot && dry && roll < 0.5) {
            s.door = Style::Door::None;
        } else if (s.wall == Style::Stuff::Stone && !forest && roll < 0.65) {
            s.door = second < 0.5 ? Style::Door::Slide : Style::Door::Lift;
            s.door_stone = (h >> 24) % 2u == 0u;
            s.door_material = s.door_stone ? s.wall_material : material::timber;
        } else {
            s.door = second < 0.3 ? Style::Door::Double : Style::Door::Plank;
            static const int kWoods[3] = {material::timber, material::charred, material::reed};
            s.door_material = kWoods[(h >> 24) % 3u];
        }
    }
    // Off the ground where the ground is wet.
    if (land == Biome::Swamp) {
        s.piles = true;
        s.raise = r.range(3, 5);
        s.base_material = material::timber;
    } else if (land == Biome::Jungle && r.next() < 0.6) {
        s.piles = true;
        s.raise = r.range(2, 4);
        s.base_material = material::timber;
    } else {
        s.raise = r.range(0, 2);
    }
    s.storey = cold ? r.range(5, 6) : (hot ? r.range(6, 8) : r.range(5, 7)); // high rooms stay cool, low ones warm
    s.window_every = cold ? pick({0, 6, 7}) : (dry && hot ? r.range(5, 8) : r.range(3, 5));
    s.floors_min = 1;
    s.floors_max = r.range(1, 3);
    s.width_min = r.range(12, 15);
    s.width_max = s.width_min + r.range(2, 8);
    s.depth_min = r.range(12, 15);
    s.depth_max = std::max(s.depth_min + r.range(2, 6), s.storey + 7);
    if (s.wall == Style::Stuff::Stone && forest && s.floors_max > 1 && r.next() < 0.6) {
        s.upper = Style::Stuff::Wood; // stone below, timber above
        s.upper_material = pick({material::timber, material::plaster, material::reed});
    }
    s.jetty = s.floors_max > 1 && (s.upper == Style::Stuff::Wood || s.wall == Style::Stuff::Wood) && r.next() < 0.5;
    if (s.wall == Style::Stuff::Wood || r.next() < 0.3) {
        s.trim_material = pick({material::timber, material::basalt, material::chalk, material::slate});
        if (s.trim_material == s.wall_material) {
            s.trim_material = -1;
        }
    }
    s.wing = r.next() < 0.3;
    s.yard = (dry || hot) && r.next() < 0.6; // shade and a wall against the wind
    s.chimney = cold || r.next() < 0.2;
    s.round_towers = r.next() < 0.45;
    s.ring_material = r.next() < 0.75 ? pick({material::granite, material::basalt, s.wall == Style::Stuff::Stone ? s.wall_material : material::sandstone}) : -1;
    s.ring_round = r.next() < 0.5;
    const auto layouts = [&](std::initializer_list<Style::Layout> options) { return *(options.begin() + r.range(0, static_cast<int>(options.size()) - 1)); };
    if (land == Biome::Mountain || land == Biome::Taiga || land == Biome::Tundra) {
        s.layout = layouts({Style::Layout::Organic, Style::Layout::Organic, Style::Layout::Ribbon, Style::Layout::Clusters});
    } else if (dry) {
        s.layout = layouts({Style::Layout::Clusters, Style::Layout::Ring, Style::Layout::Grid});
    } else if (land == Biome::Swamp || land == Biome::Beach) {
        s.layout = layouts({Style::Layout::Ribbon, Style::Layout::Ribbon, Style::Layout::Organic});
    } else {
        s.layout = layouts({Style::Layout::Grid, Style::Layout::Ring, Style::Layout::Organic, Style::Layout::Ribbon});
    }
    if (s.wall == Style::Stuff::Stone && (dry || hot)) {
        s.temple = r.next() < 0.5 ? Style::Temple::Stepped : Style::Temple::Dome;
    } else if (cold || s.roof == Style::Roof::Hip) {
        s.temple = r.next() < 0.5 ? Style::Temple::Spire : Style::Temple::Hall;
    } else {
        s.temple = r.next() < 0.3 ? Style::Temple::Spire : Style::Temple::Hall;
    }
    if (s.temple == Style::Temple::Dome && s.wall != Style::Stuff::Stone) {
        s.temple = Style::Temple::Hall;
    }
    return s;
}

void WorldMap::make_history()
{
    Rng rng{hash(seed_, 0x51, 0)};
    // Where people would live.
    const std::size_t count = static_cast<std::size_t>(kN) * kN;
    site_.assign(count, -1.0f);
    const std::vector<float> taste = noise_field(seed_ ^ 0x52, 3.0, 1, 2);
#pragma omp parallel for schedule(dynamic)
    for (int y = 4; y < kN - 4; y += 2) {
        for (int x = 4; x < kN - 4; x += 2) {
            const std::size_t n = static_cast<std::size_t>(at(x, y));
            const double e = elevation_[n] - sea_;
            if (e < 6.0 || e > 280.0 || rough_[n] > 14.0 || lake_[n]) {
                continue;
            }
            double water = 0.0;
            double slope = 0.0;
            bool in_the_way = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t m = static_cast<std::size_t>(at(x + dx, y + dy));
                    in_the_way = in_the_way || flow_[m] > 15.0f || lake_[m] || elevation_[m] < sea_;
                }
            }
            if (in_the_way) {
                continue; // near a river, not in one: the channel would take the town's lots
            }
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const std::size_t m = static_cast<std::size_t>(at(x + dx, y + dy));
                    water = std::max(water, std::min(1.0, static_cast<double>(flow_[m]) / 60.0));
                    if (elevation_[m] < sea_) {
                        water = std::max(water, 0.55);
                    }
                    slope = std::max(slope, static_cast<double>(std::abs(elevation_[m] - elevation_[n])));
                }
            }
            if (slope > 70.0) {
                continue; // no flat ground to be had
            }
            const double t = temperature_[n];
            const double warm = 1.0 - saturate(std::abs(t - 15.0) / 25.0);
            const double score = water * 1.1 + warm * 0.9 + rain_[n] * 0.4 - rough_[n] / 30.0 + taste[n] * 0.25;
            if (score > 0.2) {
                site_[n] = static_cast<float>(score);
            }
        }
    }
    struct Site {
        int node;
        float score;
    };
    std::vector<Site> sites;
    for (std::size_t n = 0; n < count; ++n) {
        if (site_[n] > 0.0f) {
            sites.push_back(Site{static_cast<int>(n), site_[n]});
        }
    }
    std::sort(sites.begin(), sites.end(), [](const Site& a, const Site& b) { return a.score != b.score ? a.score > b.score : a.node < b.node; });

    const auto node_x = [](int n) { return n % kN; };
    const auto node_y = [](int n) { return n / kN; };
    // Who lives near: a coarse grid of settlements.
    constexpr int kGrid = 16;
    const int grid_side = kN / kGrid + 1;
    std::vector<std::vector<int>> grid(static_cast<std::size_t>(grid_side) * grid_side);
    const auto near_each = [&](int nx, int ny, int reach_nodes, auto&& fn) {
        const int g0x = std::max(0, (nx - reach_nodes) / kGrid);
        const int g1x = std::min(grid_side - 1, (nx + reach_nodes) / kGrid);
        const int g0y = std::max(0, (ny - reach_nodes) / kGrid);
        const int g1y = std::min(grid_side - 1, (ny + reach_nodes) / kGrid);
        for (int gy = g0y; gy <= g1y; ++gy) {
            for (int gx = g0x; gx <= g1x; ++gx) {
                for (const int id : grid[static_cast<std::size_t>(gy * grid_side + gx)]) {
                    fn(id);
                }
            }
        }
    };
    const auto far_enough = [&](int node, int min_nodes) {
        bool ok = true;
        near_each(node_x(node), node_y(node), min_nodes, [&](int id) {
            const Settlement& s = settlements_[static_cast<std::size_t>(id)];
            ok = ok && std::hypot((s.x + kHalfCells) / kNodeCells - node_x(node), (s.y + kHalfCells) / kNodeCells - node_y(node)) >= min_nodes;
        });
        return ok;
    };
    std::vector<double> people;
    const auto found = [&](int node, int culture, int age) {
        const std::size_t n = static_cast<std::size_t>(node);
        Settlement s;
        s.id = static_cast<int>(settlements_.size());
        s.x = node_x(node) * kNodeCells + kNodeCells / 2 - kHalfCells;
        s.y = node_y(node) * kNodeCells + kNodeCells / 2 - kHalfCells;
        s.ground = static_cast<int>(std::floor(elevation_[n]));
        s.culture = culture;
        s.style = cultures_[static_cast<std::size_t>(culture)].style;
        s.founded = age;
        s.name = make_name(rng, culture);
        grid[static_cast<std::size_t>(node_y(node) / kGrid * grid_side + node_x(node) / kGrid)].push_back(s.id);
        settlements_.push_back(std::move(s));
        people.push_back(30.0);
        chronicle_.push_back(Chronicle{age, settlements_.back().id,
            cultures_[static_cast<std::size_t>(culture)].name + " founded " + settlements_.back().name + "."});
    };
    // The first peoples, far apart, each with its own way of building: what the land asks, or now and then what a
    // style in the file says for that land.
    const int file_styles = static_cast<int>(styles_.size());
    constexpr int kMaxCultures = 60;
    // Every kind of land has its peoples first, however hard the living; then the best of what is left.
    const auto settle_people = [&](const Site& site) {
        const std::size_t n = static_cast<std::size_t>(site.node);
        const Biome land = biome_[n];
        Culture culture;
        const int index = static_cast<int>(cultures_.size());
        Rng names{hash(seed_, 0x54, index)};
        const std::string word = make_name(names, index);
        culture.name = "The " + word + " people";
        int chosen = -1;
        if (names.next() < 0.2) {
            for (int k = 0; k < file_styles && chosen < 0; ++k) {
                const Style& fixed = styles_[static_cast<std::size_t>(k)];
                const bool snowed_under = temperature_[n] < 6.0 && (fixed.roof == Style::Roof::Flat || fixed.roof == Style::Roof::Dome); // snow lies on a flat roof
                if (!snowed_under && std::find(fixed.biomes.begin(), fixed.biomes.end(), land) != fixed.biomes.end() && names.next() < 0.5) {
                    chosen = k;
                }
            }
        }
        if (chosen < 0) {
            styles_.push_back(architect(hash(seed_, 0x55, index), land, temperature_[n], rain_[n], word));
            chosen = static_cast<int>(styles_.size()) - 1;
        }
        culture.style = chosen;
        culture.home = static_cast<int>(settlements_.size());
        cultures_.push_back(culture);
        found(site.node, index, 0);
    };
    constexpr int kPerLand = 5;
    std::map<Biome, int> peoples_of;
    for (const Site& site : sites) {
        int& here = peoples_of[biome_[static_cast<std::size_t>(site.node)]];
        if (here < kPerLand && far_enough(site.node, 80)) {
            settle_people(site);
            ++here;
        }
    }
    for (const Site& site : sites) {
        if (static_cast<int>(cultures_.size()) >= kMaxCultures) {
            break;
        }
        if (far_enough(site.node, 110)) {
            settle_people(site);
        }
    }
    // Ages: they grow, spread, fight, and are struck by what the land does. What happens to a place stays on it.
    for (int age = 1; age <= kAges; ++age) {
        const std::size_t living_now = settlements_.size();
        for (std::size_t i = 0; i < living_now; ++i) {
            Settlement& s = settlements_[i];
            if (s.fell >= 0) {
                continue;
            }
            people[i] *= 1.28 + 0.2 * rng.next();
            if (people[i] > 70.0 && rng.next() < 0.40 && settlements_.size() < 4000) {
                // A new place, the best near enough to reach.
                const int sx = (s.x + kHalfCells) / kNodeCells;
                const int sy = (s.y + kHalfCells) / kNodeCells;
                const int culture = s.culture;
                int best = -1;
                float best_score = 0.0f;
                for (int y = std::max(4, sy - 30) & ~1; y <= std::min(kN - 5, sy + 30); y += 2) {
                    for (int x = std::max(4, sx - 30) & ~1; x <= std::min(kN - 5, sx + 30); x += 2) {
                        const int node = at(x, y);
                        const float score = site_[static_cast<std::size_t>(node)];
                        const double d = std::hypot(x - sx, y - sy);
                        if (score > best_score && d >= 8.0 && d <= 30.0 && far_enough(node, 8)) {
                            best_score = score;
                            best = node;
                        }
                    }
                }
                if (best >= 0) {
                    found(best, culture, age);
                    people[i] -= 25.0;
                }
            }
        }
        for (std::size_t i = 0; i < settlements_.size(); ++i) {
            if (settlements_[i].fell >= 0 || settlements_[i].founded == age) {
                continue;
            }
            Settlement& s = settlements_[i];
            const int sx = (s.x + kHalfCells) / kNodeCells;
            const int sy = (s.y + kHalfCells) / kNodeCells;
            double river = 0.0;
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    river = std::max(river, static_cast<double>(flow_[static_cast<std::size_t>(at(std::clamp(sx + dx, 0, kN - 1), std::clamp(sy + dy, 0, kN - 1)))]));
                }
            }
            const double roll = rng.next();
            if (roll < 0.035) {
                people[i] *= 0.45;
                ++s.plagues;
                chronicle_.push_back(Chronicle{age, s.id, "Plague in " + s.name + ": a quarter of it was never lived in again."});
            } else if (roll < 0.07 && river > 25.0) {
                people[i] *= 0.6;
                ++s.floods;
                chronicle_.push_back(Chronicle{age, s.id, "The river rose over " + s.name + "; they built higher after that."});
            } else if (roll < 0.09) {
                people[i] *= 0.7;
                ++s.burned_and_rebuilt;
                chronicle_.push_back(Chronicle{age, s.id, "A great fire in " + s.name + ". The burnt quarter stands as it was left."});
            } else if (roll < 0.13) {
                people[i] *= 1.6;
                chronicle_.push_back(Chronicle{age, s.id, "Good years in " + s.name + "."});
            }
            // War: a neighbour of another people, stronger.
            if (rng.next() < 0.1) {
                int enemy = -1;
                near_each(sx, sy, 110, [&](int id) {
                    const Settlement& o = settlements_[static_cast<std::size_t>(id)];
                    if (enemy < 0 && o.fell < 0 && o.culture != s.culture && people[static_cast<std::size_t>(id)] > people[i] &&
                        std::hypot(o.x - s.x, o.y - s.y) <= 110.0 * kNodeCells) {
                        enemy = id;
                    }
                });
                if (enemy >= 0) {
                    Settlement& o = settlements_[static_cast<std::size_t>(enemy)];
                    const double outcome = rng.next();
                    ++o.wars_won;
                    if (outcome < 0.3) {
                        s.fell = age;
                        s.burned = true;
                        chronicle_.push_back(Chronicle{age, s.id, o.name + " burned " + s.name + " to the ground."});
                    } else if (outcome < 0.7) {
                        s.conquerors = cultures_[static_cast<std::size_t>(o.culture)].style;
                        s.culture = o.culture;
                        people[i] *= 0.7;
                        chronicle_.push_back(Chronicle{age, s.id, o.name + " took " + s.name + " and built a quarter of its own there."});
                    } else {
                        ++s.wars_won;
                        o.wars_won = std::max(0, o.wars_won - 1);
                        chronicle_.push_back(Chronicle{age, s.id, s.name + " held against " + o.name + ", and raised a stone to it."});
                    }
                }
            }
            if (s.fell < 0 && people[i] < 22.0) {
                s.fell = age;
                chronicle_.push_back(Chronicle{age, s.id, s.name + " was left empty."});
            }
        }
    }
    for (std::size_t i = 0; i < settlements_.size(); ++i) {
        Settlement& s = settlements_[i];
        s.people = s.fell >= 0 ? 0 : static_cast<int>(people[i]);
        s.tier = tier_of(static_cast<int>(people[i]));
        s.radius = kRadius[s.tier];
    }
    std::vector<std::array<int, 4>> boxes;
    for (const Settlement& s : settlements_) {
        const int reach = static_cast<int>(s.radius * 2.7) + 2;
        boxes.push_back({s.x - reach, s.y - reach, s.x + reach, s.y + reach});
    }
    settlement_near_ = index_boxes(boxes);
    make_local_styles();
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(settlements_.size()); ++k) {
        plan_settlement(settlements_[static_cast<std::size_t>(k)]);
    }
    // A place with a house or two is not a village, whatever its people call it: a family's own place. No streets, no
    // square, its ground levelled only as far as it needs. (make_roads leaves it alone.)
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(settlements_.size()); ++k) {
        Settlement& s = settlements_[static_cast<std::size_t>(k)];
        if (s.fell >= 0) {
            continue;
        }
        const auto houses = std::count_if(s.buildings.begin(), s.buildings.end(), [](const Building& b) { return b.kind == Building::Kind::House; });
        if (houses > (s.tier == 0 ? 2 : 4)) {
            continue;
        }
        s.lone = true;
        s.tier = 0;
        s.streets.clear();
        double reach = 6.0;
        for (const Building& b : s.buildings) {
            int bx0, by0, bx1, by1, top;
            b.bounds(styles_, bx0, by0, bx1, by1, top);
            reach = std::max({reach, std::hypot(static_cast<double>(bx0 - s.x), static_cast<double>(by0 - s.y)), std::hypot(static_cast<double>(bx1 - s.x), static_cast<double>(by0 - s.y)),
                std::hypot(static_cast<double>(bx0 - s.x), static_cast<double>(by1 - s.y)), std::hypot(static_cast<double>(bx1 - s.x), static_cast<double>(by1 - s.y))});
        }
        s.radius = std::min(s.radius, static_cast<int>(std::ceil(reach / 0.78)) + 4);
    }
}

// NO TWO TOWNS ALIKE. Each takes its people's way and changes it by what happened to it.
void WorldMap::make_local_styles()
{
    for (Settlement& s : settlements_) {
        Rng r{hash(seed_, 0x56, s.id)};
        Style local = styles_[static_cast<std::size_t>(s.style)];
        local.name = s.name;
        // Its own hands: sizes and details wander from town to town.
        local.width_min = std::max(12, local.width_min + r.range(-1, 2));
        local.width_max = std::max(local.width_min, local.width_max + r.range(-2, 3));
        local.depth_min = std::max(12, local.depth_min + r.range(-1, 2));
        local.depth_max = std::max({local.depth_min, local.depth_max + r.range(-1, 3), local.storey + 7});
        if (local.window_every > 0) {
            local.window_every = std::max(2, local.window_every + r.range(-1, 1));
        }
        local.eaves = local.roof == Style::Roof::Flat || local.roof == Style::Roof::Dome ? local.eaves : std::clamp(local.eaves + r.range(-1, 1), 0, 2);
        local.wing = local.wing != (r.next() < 0.2);
        local.chimney = local.chimney != (r.next() < 0.15);
        if (r.next() < 0.3 && local.roof_stuff == Style::Stuff::Stone) {
            static constexpr int kRoofs[] = {material::tile, material::slate, material::clay, material::basalt, material::brick};
            local.roof_material = kRoofs[r.range(0, 4)];
        }
        if (r.next() < 0.35) {
            local.layout = static_cast<Style::Layout>(r.range(0, 4));
        }
        // A rich place builds high; a small one does not.
        local.floors_max = std::clamp(local.floors_max + (s.tier >= 3 ? 2 : (s.tier == 2 ? 1 : (s.tier == 0 ? -1 : 0))), 1, 4);
        local.floors_min = std::min(local.floors_min, local.floors_max);
        if (s.tier == 0) {
            local.layout = local.layout == Style::Layout::Grid || local.layout == Style::Layout::Ring ? Style::Layout::Clusters : local.layout;
        }
        // A place that burned does not build in wood again.
        if (s.burned_and_rebuilt > 0) {
            if (local.wall != Style::Stuff::Stone) {
                local.wall = Style::Stuff::Stone;
                local.wall_material = local.base_material == material::timber ? material::granite : local.base_material;
            }
            local.upper = Style::Stuff::Stone;
            local.upper_material = -1;
            local.jetty = false;
            if (local.roof_stuff != Style::Stuff::Stone) {
                local.roof_stuff = Style::Stuff::Stone;
                local.roof_material = material::slate;
            }
        }
        // A place that flooded stands up out of the way.
        if (s.floods > 0) {
            local.raise = std::min(local.raise + 1 + s.floods, 6);
            if (local.base_material == material::soil) {
                local.base_material = material::granite;
            }
        }
        // A place that has been fought over has a wall.
        if ((s.wars_won > 0 || s.conquerors >= 0) && local.ring_material < 0) {
            local.ring_material = material::granite;
        }
        styles_.push_back(std::move(local));
        s.style = static_cast<int>(styles_.size()) - 1;
    }
}

// --- plan: streets first, then the lots along them ----------------------------------------------------------

void WorldMap::plan_settlement(Settlement& s)
{
    Rng rng{hash(seed_, 0x61, s.id)};
    const Style& home_style = styles_[static_cast<std::size_t>(s.style)];
    const bool along_x = rng.next() < 0.5;
    const bool walled = s.tier >= 1 && home_style.ring_material >= 0 && (s.tier >= 2 || s.wars_won > 0 || s.conquerors >= 0);
    const int reach = static_cast<int>(s.radius * 0.76);
    const int plaza = s.tier >= 2 ? 16 : (s.tier == 1 ? 11 : 7);
    s.streets.push_back(Settlement::Street{s.x - plaza, s.y - plaza, s.x + plaza, s.y + plaza, 0}); // the square
    const auto street = [&](int x0, int y0, int x1, int y1, int half) { s.streets.push_back(Settlement::Street{x0, y0, x1, y1, half}); };
    struct Run {
        bool x_axis;
        int fixed;
        int from;
        int to;
    };
    std::vector<Run> runs;
    const auto straight = [&](bool x_axis, int fixed, int from, int to, int half) {
        if (x_axis) {
            street(from, fixed, to, fixed, half);
        } else {
            street(fixed, from, fixed, to, half);
        }
        runs.push_back(Run{x_axis, fixed, from, to});
    };
    const auto main_coord = [&](bool x_axis) { return x_axis ? s.y : s.x; };
    const auto main_from = [&](bool x_axis) { return x_axis ? s.x : s.y; };
    switch (home_style.layout) {
    case Style::Layout::Grid: {
        straight(along_x, main_coord(along_x), main_from(along_x) - reach, main_from(along_x) + reach, 2);
        const int every = 28 + rng.range(0, 14);
        if (s.tier >= 1) {
            for (int k = -reach / every; k <= reach / every; ++k) {
                if (k != 0) {
                    const int half_len = static_cast<int>(reach * (s.tier >= 2 ? 0.85 : 0.5));
                    straight(!along_x, main_from(along_x) + k * every, main_coord(along_x) - half_len, main_coord(along_x) + half_len, 1);
                }
            }
        }
        if (s.tier >= 2) {
            for (const int side : {-1, 1}) {
                const int at_k = side * (every + every / 2);
                straight(along_x, main_coord(along_x) + at_k, main_from(along_x) - reach * 7 / 10, main_from(along_x) + reach * 7 / 10, 1);
            }
        }
        break;
    }
    case Style::Layout::Organic: {
        // Streets that wander off from the square and turn when they feel like it.
        const int walkers = 3 + s.tier * 3;
        for (int k = 0; k < walkers; ++k) {
            int x = s.x;
            int y = s.y;
            bool x_axis = k % 2 == 0;
            int dir = k % 4 < 2 ? 1 : -1;
            for (int leg = 0; leg < 3 + s.tier; ++leg) {
                const int length = rng.range(16, 34);
                const int to = (x_axis ? x : y) + dir * length;
                const int from = x_axis ? x : y;
                const int low = std::min(from, to);
                const int high = std::max(from, to);
                if (std::hypot(static_cast<double>((x_axis ? to : x) - s.x), static_cast<double>((x_axis ? y : to) - s.y)) > reach) {
                    break;
                }
                straight(x_axis, x_axis ? y : x, low, high, leg == 0 ? 2 : 1);
                (x_axis ? x : y) = to;
                x_axis = !x_axis;
                dir = rng.next() < 0.5 ? 1 : -1;
            }
        }
        break;
    }
    case Style::Layout::Ring: {
        // Rings of houses around the square, with streets out through them.
        straight(true, s.y, s.x - reach, s.x + reach, 2);
        straight(false, s.x, s.y - reach, s.y + reach, 2);
        for (int r = plaza + 22; r < reach - 6; r += 34) {
            s.streets.push_back(Settlement::Street{s.x, s.y, s.x, s.y, 2, r});
        }
        break;
    }
    case Style::Layout::Ribbon: {
        // One long street, and alleys off it.
        straight(along_x, main_coord(along_x), main_from(along_x) - reach, main_from(along_x) + reach, 3);
        for (int k = -2; k <= 2; ++k) {
            if (k != 0 && s.tier >= 2) {
                const int side = rng.next() < 0.5 ? 1 : -1;
                straight(!along_x, main_from(along_x) + k * (reach / 3), main_coord(along_x), main_coord(along_x) + side * reach / 2, 1);
            }
        }
        for (Run& run : runs) {
            if (run.from > run.to) {
                std::swap(run.from, run.to);
            }
        }
        break;
    }
    case Style::Layout::Clusters:
        break; // farmsteads, each on its own ground, with a square to meet on
    }
    std::vector<std::array<int, 4>> taken;
    std::vector<std::array<int, 4>> paved;
    for (const Settlement::Street& t : s.streets) {
        if (t.circle == 0) {
            paved.push_back({std::min(t.x0, t.x1) - t.half, std::min(t.y0, t.y1) - t.half, std::max(t.x0, t.x1) + t.half, std::max(t.y0, t.y1) + t.half});
        }
    }
    const auto free_for = [&](int x0, int y0, int x1, int y1) {
        if (std::hypot(std::max(std::abs(x0 - s.x), std::abs(x1 - s.x)), std::max(std::abs(y0 - s.y), std::abs(y1 - s.y))) > s.radius * 0.78) {
            return false;
        }
        for (const auto& t : taken) {
            if (x0 - 3 <= t[2] && x1 + 3 >= t[0] && y0 - 3 <= t[3] && y1 + 3 >= t[1]) {
                return false;
            }
        }
        for (const auto& t : paved) {
            if (x0 <= t[2] && x1 >= t[0] && y0 <= t[3] && y1 >= t[1]) {
                return false;
            }
        }
        for (const Settlement::Street& t : s.streets) {
            if (t.circle > 0) {
                const double near = std::hypot(std::max({0, x0 - s.x, s.x - x1}), std::max({0, y0 - s.y, s.y - y1}));
                const double far = std::hypot(std::max(std::abs(x0 - s.x), std::abs(x1 - s.x)), std::max(std::abs(y0 - s.y), std::abs(y1 - s.y)));
                if (near <= t.circle + t.half + 1 && far >= t.circle - t.half - 1) {
                    return false;
                }
            }
        }
        // The ground under all of it, not only under the corners: level, and dry.
        for (int y = y0; ; y = std::min(y + 4, y1)) {
            for (int x = x0; ; x = std::min(x + 4, x1)) {
                const Base b = base(x, y, 1);
                if (b.water >= 0 || std::abs(b.ground - s.ground) > 0.5) {
                    return false;
                }
                if (x == x1) {
                    break;
                }
            }
            if (y == y1) {
                break;
            }
        }
        return true;
    };
    // What happened here decides who built what, and what is left of it.
    const double burnt_angle = rng.next() * 6.283185307;
    const double plague_angle = burnt_angle + 2.0 + rng.next() * 2.0;
    const double conquered_angle = burnt_angle + 1.0 + rng.next() * 3.0;
    const auto in_sector = [&](int x, int y, double angle, double half) {
        double a = std::atan2(static_cast<double>(y - s.y), static_cast<double>(x - s.x)) - angle;
        a = std::remainder(a, 6.283185307);
        return std::abs(a) < half;
    };
    int placed_houses = 0;
    const auto place = [&](int x0, int y0, int w, int d, int facing, Building::Kind kind, int floors, int style_id) {
        const Style& st = styles_[static_cast<std::size_t>(style_id)];
        const int x1 = x0 + w - 1;
        const int y1 = y0 + d - 1;
        // The steps need the ground in front of a raised door, and the door something to open on.
        const int front = st.raise + 2;
        const int fx0 = x0 - (facing == 1 ? front : 0);
        const int fx1 = x1 + (facing == 0 ? front : 0);
        const int fy0 = y0 - (facing == 3 ? front : 0);
        const int fy1 = y1 + (facing == 2 ? front : 0);
        // Houses with their upper floors over the lot, roofs and all.
        const int over = st.eaves + (st.jetty ? 1 : 0);
        if (!free_for(fx0 - over, fy0 - over, fx1 + over, fy1 + over)) {
            return false;
        }
        Building b;
        b.kind = kind;
        b.x0 = x0;
        b.y0 = y0;
        b.x1 = x1;
        b.y1 = y1;
        b.ground = s.ground;
        b.style = style_id;
        b.facing = facing;
        b.seed = hash(seed_ ^ 0x62, s.id, x0, y0);
        Rng local{b.seed};
        b.floors = d >= st.storey + 7 ? floors : 1;
        const int cx = (x0 + x1) / 2;
        const int cy = (y0 + y1) / 2;
        if (kind == Building::Kind::Round) {
            b.floors = 1;
            b.roof = static_cast<int>(st.roof == Style::Roof::Dome ? Style::Roof::Dome : Style::Roof::Spire);
            b.door = (facing < 2 ? cy : cx);
        } else {
            b.wing = kind == Building::Kind::House && st.wing && floors == 1 && w >= 14 && d >= 14 && st.roof != Style::Roof::Dome;
            if (b.wing) {
                b.floors = 1;
            }
            const int cut_y = y0 + d * 6 / 10;
            // Inner walls leave a passage of two cells past the stair, and two beyond each.
            const int room = 5 + local.range(0, 2);
            if (w >= 11 && room + 4 < w) {
                b.part[0] = x0 + room;
                const int last_gap = b.wing ? cut_y - 3 : y1 - 3;
                const int first_gap = b.floors > 1 ? y0 + st.storey + 2 : y0 + 2;
                b.gap[0] = std::min(first_gap + local.range(0, std::max(0, last_gap - first_gap)), last_gap);
                if (w >= room + 9) {
                    b.part[1] = b.part[0] + 4 + local.range(0, std::max(0, w - room - 8));
                    if (b.part[1] > x1 - 3) {
                        b.part[1] = -1;
                    } else {
                        b.gap[1] = std::min(y0 + 2 + local.range(0, std::max(0, last_gap - y0 - 2)), last_gap);
                    }
                }
            }
            // A door where nothing stands behind it.
            const int rise = b.floors > 1 ? st.storey + 2 : 0;
            if (facing < 2) {
                const int first = facing == 1 ? y0 + 1 + rise : y0 + 1;
                const int last = b.wing && facing == 0 ? cut_y - 3 : y1 - 2;
                b.door = std::min(first + local.range(0, std::max(0, last - first)), last);
            } else {
                b.door = x0 + 3; // the corridor past the stair is always clear
            }
        }
        if (s.fell >= 0) {
            b.burned = s.burned;
            b.ruin = static_cast<float>(0.35 + 0.5 * local.next());
        } else if (kind == Building::Kind::House) {
            // The burnt quarter stands as it was left; so does the one nobody came back to after the plague.
            if (s.burned_and_rebuilt > 0 && in_sector(cx, cy, burnt_angle, 0.5 + 0.2 * s.burned_and_rebuilt) && local.next() < 0.75) {
                b.burned = true;
                b.ruin = static_cast<float>(0.4 + 0.5 * local.next());
            } else if (s.plagues > 0 && in_sector(cx, cy, plague_angle, 0.4 + 0.2 * s.plagues) && local.next() < 0.7) {
                b.ruin = static_cast<float>(0.3 + 0.5 * local.next());
            }
        }
        taken.push_back({fx0 - over, fy0 - over, fx1 + over, fy1 + over});
        s.buildings.push_back(b);
        if (kind == Building::Kind::House) {
            ++placed_houses;
        }
        return true;
    };
    const auto simple = [&](Building::Kind kind, int x0, int y0, int x1, int y1, int height, int facing = 0) {
        Building b;
        b.kind = kind;
        b.x0 = x0;
        b.y0 = y0;
        b.x1 = x1;
        b.y1 = y1;
        b.ground = s.ground;
        b.style = s.style;
        b.floors = height;
        b.facing = facing;
        b.seed = hash(seed_ ^ 0x63, s.id, x0, y0);
        if (s.fell >= 0) {
            b.ruin = 0.4f;
        }
        s.buildings.push_back(b);
    };
    // Who builds this lot.
    const auto style_at = [&](int x, int y) {
        return s.conquerors >= 0 && in_sector(x, y, conquered_angle, 0.9) ? s.conquerors : s.style;
    };
    // The well in the square, and what it remembers: a stone for every war it won.
    simple(Building::Kind::Well, s.x - 1, s.y - 1, s.x + 2, s.y + 2, 1);
    if (s.wars_won > 0) {
        const int mx = s.x + plaza / 2;
        simple(Building::Kind::Monument, mx, s.y - 1, mx + 2, s.y + 1, 6 + std::min(s.wars_won, 6) * 2);
    }
    // The people's own house of what they believe in, facing the square.
    if (s.tier >= 1) {
        const Style& st = home_style;
        const int w = s.tier >= 2 ? std::max(st.width_max + 4, 18) : 16;
        const int d = s.tier >= 2 ? std::max(st.depth_max + 3, st.storey + 8) : st.storey + 8;
        // On a corner of the square, on whichever side the streets leave room.
        const std::array<std::pair<int, int>, 6> corners{std::pair{s.x + 5, s.y + plaza + 3}, std::pair{s.x - 5 - w, s.y + plaza + 3},
            std::pair{s.x + plaza + 3, s.y + 5}, std::pair{s.x + plaza + 3, s.y - 5 - d}, std::pair{s.x + 5 + w, s.y + plaza + 3}, std::pair{s.x - 5 - 2 * w, s.y + plaza + 3}};
        for (const auto& [x0, y0] : corners) {
            bool stands = false;
            switch (st.temple) {
            case Style::Temple::Stepped:
                if (free_for(x0 - 3, y0 - 3, x0 + w + 3, y0 + w + 3)) {
                    simple(Building::Kind::Stepped, x0, y0, x0 + w, y0 + w, 1);
                    taken.push_back({x0 - 3, y0 - 3, x0 + w + 3, y0 + w + 3});
                    stands = true;
                }
                break;
            case Style::Temple::Dome:
            case Style::Temple::Spire:
                stands = place(x0, y0, w, d, 3, Building::Kind::Hall, 1, s.style);
                if (stands) {
                    s.buildings.back().roof = static_cast<int>(st.temple == Style::Temple::Dome ? Style::Roof::Dome : Style::Roof::Spire);
                }
                break;
            case Style::Temple::Hall:
                stands = place(x0, y0, w, d, 3, Building::Kind::Hall, std::min(st.floors_max + 1, 4), s.style);
                break;
            }
            if (stands) {
                break;
            }
        }
    }
    if (s.tier >= 2) {
        const int side = std::max(14, home_style.storey + 8);
        const int tx = s.x - plaza - side - 3;
        const int ty = s.y - plaza - side - 3;
        place(tx, ty, side, side, 0, home_style.round_towers ? Building::Kind::Round : Building::Kind::Tower, std::min(home_style.floors_max + 3, 6), s.style);
    }
    // Houses along every street, both sides, doors to the street.
    for (const Run& run : runs) {
        for (const int side : {-1, 1}) {
            int pos = run.from + rng.range(0, 4);
            while (pos < run.to) {
                const int mid_x = run.x_axis ? pos : run.fixed + side * 12;
                const int mid_y = run.x_axis ? run.fixed + side * 12 : pos;
                const int style_id = style_at(mid_x, mid_y);
                const Style& st = styles_[static_cast<std::size_t>(style_id)];
                const int w = rng.range(st.width_min, st.width_max);
                const int d = rng.range(st.depth_min, st.depth_max);
                const int floors = rng.range(st.floors_min, st.floors_max);
                const int off = 3 + st.raise + st.eaves + (st.jetty ? 1 : 0) + 2 + rng.range(0, 2);
                bool placed;
                int bx0;
                int by0;
                int bx1;
                int by1;
                if (run.x_axis) {
                    by0 = side > 0 ? run.fixed + off : run.fixed - off - d + 1;
                    bx0 = pos;
                    bx1 = bx0 + w - 1;
                    by1 = by0 + d - 1;
                    placed = place(bx0, by0, w, d, side > 0 ? 3 : 2, Building::Kind::House, floors, style_id);
                } else {
                    // Street along y: the lot's long side runs along it, so the stair fits.
                    bx0 = side > 0 ? run.fixed + off : run.fixed - off - d + 1;
                    by0 = pos;
                    bx1 = bx0 + d - 1;
                    by1 = by0 + w - 1;
                    placed = place(bx0, by0, d, w, side > 0 ? 1 : 0, Building::Kind::House, floors, style_id);
                }
                if (placed && st.yard) {
                    // A walled yard behind, open to nothing but the house.
                    const int yard = 5 + rng.range(0, 3);
                    const int yx0 = run.x_axis ? bx0 : (side > 0 ? bx1 + 1 : bx0 - yard);
                    const int yx1 = run.x_axis ? bx1 : (side > 0 ? bx1 + yard : bx0 - 1);
                    const int yy0 = run.x_axis ? (side > 0 ? by1 + 1 : by0 - yard) : by0;
                    const int yy1 = run.x_axis ? (side > 0 ? by1 + yard : by0 - 1) : by1;
                    if (free_for(yx0 + 1, yy0 + 1, yx1 - 1, yy1 - 1)) {
                        simple(Building::Kind::Yard, yx0, yy0, yx1, yy1, 1, run.x_axis ? (side > 0 ? 3 : 2) : (side > 0 ? 1 : 0));
                        taken.push_back({yx0, yy0, yx1, yy1});
                    }
                }
                pos += placed ? w + rng.range(2, 6) : 6;
            }
        }
    }
    // Around the rings of a ring town, and over the ground of a scattered one: houses wherever there is room, doors to
    // the middle (or anywhere), as many as the town has people for.
    const int ring_rows = home_style.layout == Style::Layout::Ring ? 1 : 0;
    if (ring_rows > 0) {
        for (const Settlement::Street& t : s.streets) {
            if (t.circle == 0) {
                continue;
            }
            for (const int outward : {-1, 1}) {
                const double middle = t.circle + outward * (t.half + 15.0);
                const int steps = std::max(6, static_cast<int>(6.283 * std::max(middle, 8.0) / 21.0));
                for (int k = 0; k < steps; ++k) {
                    const double a = 6.283185307 * (k + 0.5) / steps;
                    const int style_id = style_at(s.x + static_cast<int>(middle * std::cos(a)), s.y + static_cast<int>(middle * std::sin(a)));
                    const Style& st = styles_[static_cast<std::size_t>(style_id)];
                    const int w = rng.range(st.width_min, st.width_max);
                    const int d = rng.range(st.depth_min, st.depth_max);
                    // Far enough from the street for its size and the steps at its door.
                    const int over = st.eaves + (st.jetty ? 1 : 0) + st.raise + 2;
                    const double r = t.circle + outward * (t.half + 2.0 + over + std::hypot(w, d) * 0.5);
                    if (r < plaza + 8.0) {
                        continue;
                    }
                    const int cx = s.x + static_cast<int>(std::lround(r * std::cos(a)));
                    const int cy = s.y + static_cast<int>(std::lround(r * std::sin(a)));
                    // The door to whichever side of the ring street it is on.
                    const bool mostly_x = std::abs(std::cos(a)) > std::abs(std::sin(a));
                    const int toward = outward > 0 ? -1 : 1; // back toward the street
                    int facing;
                    if (mostly_x) {
                        facing = (std::cos(a) > 0) == (toward > 0) ? 0 : 1;
                    } else {
                        facing = (std::sin(a) > 0) == (toward > 0) ? 2 : 3;
                    }
                    place(cx - w / 2, cy - d / 2, w, d, facing, Building::Kind::House, rng.range(st.floors_min, st.floors_max), style_id);
                }
            }
        }
    }
    if (home_style.layout == Style::Layout::Clusters) {
        const int wanted = 4 + s.tier * 12;
        for (int tries = 0; tries < wanted * 6 && placed_houses < wanted; ++tries) {
            const double a = rng.next() * 6.283185307;
            const double r = plaza + 6 + rng.next() * (reach - plaza - 6);
            const int cx = s.x + static_cast<int>(r * std::cos(a));
            const int cy = s.y + static_cast<int>(r * std::sin(a));
            const int style_id = style_at(cx, cy);
            const Style& st = styles_[static_cast<std::size_t>(style_id)];
            const int w = rng.range(st.width_min, st.width_max);
            const int d = rng.range(st.depth_min, st.depth_max);
            const int facing = rng.range(0, 3);
            if (place(cx - w / 2, cy - d / 2, w, d, facing, Building::Kind::House, rng.range(st.floors_min, st.floors_max), style_id) && st.yard) {
                const int yx0 = cx - w / 2;
                const int yy0 = cy - d / 2;
                const int yard = 6;
                int ax0 = yx0;
                int ay0 = yy0;
                int ax1 = yx0 + w - 1;
                int ay1 = yy0 + d - 1;
                // The side away from the door.
                switch (facing) {
                case 0: ax1 = ax0 - 1; ax0 = ax1 - yard; break;
                case 1: ax0 = ax1 + 1; ax1 = ax0 + yard; break;
                case 2: ay1 = ay0 - 1; ay0 = ay1 - yard; break;
                default: ay0 = ay1 + 1; ay1 = ay0 + yard; break;
                }
                if (free_for(ax0 + 1, ay0 + 1, ax1 - 1, ay1 - 1)) {
                    simple(Building::Kind::Yard, ax0, ay0, ax1, ay1, 1, facing == 0 ? 0 : (facing == 1 ? 1 : (facing == 2 ? 2 : 3)));
                    taken.push_back({ax0, ay0, ax1, ay1});
                }
            }
        }
    }
    // However its streets run, a town has as many houses as it has people for: what the streets did not reach is
    // built on anyway, up against whatever is there.
    {
        static constexpr int kWanted[4] = {6, 22, 70, 150};
        const int wanted = kWanted[s.tier];
        for (int tries = 0; tries < wanted * 8 && placed_houses < wanted; ++tries) {
            const double a = rng.next() * 6.283185307;
            const double r = plaza + 8 + std::sqrt(rng.next()) * (reach - plaza - 8);
            const int cx = s.x + static_cast<int>(r * std::cos(a));
            const int cy = s.y + static_cast<int>(r * std::sin(a));
            const int style_id = style_at(cx, cy);
            const Style& st = styles_[static_cast<std::size_t>(style_id)];
            const int w = rng.range(st.width_min, st.width_max);
            const int d = rng.range(st.depth_min, st.depth_max);
            // The door toward the middle of the town.
            const bool mostly_x = std::abs(std::cos(a)) > std::abs(std::sin(a));
            const int facing = mostly_x ? (std::cos(a) > 0 ? 1 : 0) : (std::sin(a) > 0 ? 3 : 2);
            place(cx - w / 2, cy - d / 2, w, d, facing, Building::Kind::House, rng.range(st.floors_min, st.floors_max), style_id);
        }
    }
    // A plague leaves a field of stones, outside the houses.
    if (s.plagues > 0) {
        const double a = plague_angle;
        const int cx = s.x + static_cast<int>(reach * 0.8 * std::cos(a));
        const int cy = s.y + static_cast<int>(reach * 0.8 * std::sin(a));
        const int half = 5 + 2 * std::min(s.plagues, 3);
        if (free_for(cx - half, cy - half, cx + half, cy + half)) {
            simple(Building::Kind::Graves, cx - half, cy - half, cx + half, cy + half, 1);
            taken.push_back({cx - half, cy - half, cx + half, cy + half});
        }
    }
    // A town wall, inside the levelled ground, with gates where the main streets run out and a turret by each.
    if (walled) {
        const int r = home_style.ring_round ? static_cast<int>(s.radius * 0.84) : static_cast<int>(s.radius * 0.6);
        Building ring;
        ring.kind = Building::Kind::Ring;
        ring.x0 = s.x - r;
        ring.y0 = s.y - r;
        ring.x1 = s.x + r;
        ring.y1 = s.y + r;
        ring.ground = s.ground;
        ring.style = s.style;
        ring.part[0] = home_style.ring_round ? 1 : 0;
        ring.seed = hash(seed_ ^ 0x64, s.id, 0);
        if (s.fell >= 0) {
            ring.ruin = 0.5f;
        }
        // Whatever stands where the wall goes makes way for the wall.
        const auto on_wall = [&](const Building& b) {
            if (!b.has_rooms() && b.kind != Building::Kind::Yard && b.kind != Building::Kind::Graves) {
                return false;
            }
            const int px0 = b.x0 - 2;
            const int px1 = b.x1 + 2;
            const int py0 = b.y0 - 2;
            const int py1 = b.y1 + 2;
            if (home_style.ring_round) {
                const double near = std::hypot(std::max({0, px0 - s.x, s.x - px1}), std::max({0, py0 - s.y, s.y - py1}));
                const double far = std::hypot(std::max(std::abs(px0 - s.x), std::abs(px1 - s.x)), std::max(std::abs(py0 - s.y), std::abs(py1 - s.y)));
                return near <= r && far >= r - 2;
            }
            const bool touches = px0 <= s.x + r && px1 >= s.x - r && py0 <= s.y + r && py1 >= s.y - r;
            const bool inside = px0 >= s.x - r + 2 && px1 <= s.x + r - 2 && py0 >= s.y - r + 2 && py1 <= s.y + r - 2;
            return touches && !inside;
        };
        s.buildings.erase(std::remove_if(s.buildings.begin(), s.buildings.end(), on_wall), s.buildings.end());
        s.buildings.push_back(ring);
        const int gate = 7;
        for (const auto& [gx, gy] : {std::pair{gate, r - 1}, std::pair{-gate, r - 1}, std::pair{gate, -r + 1}, std::pair{-gate, -r + 1}, std::pair{r - 1, gate}, std::pair{r - 1, -gate},
                 std::pair{-r + 1, gate}, std::pair{-r + 1, -gate}}) {
            const double far = std::hypot(gx, gy);
            const double scale = home_style.ring_round ? (r - 1.0) / far : 1.0;
            const int tx = s.x + static_cast<int>(gx * scale);
            const int ty = s.y + static_cast<int>(gy * scale);
            simple(Building::Kind::Monument, tx - 2, ty - 2, tx + 2, ty + 2, 15);
        }
    }
}

// --- roads: between neighbours, the easy way over the land -----------------------------------------------------

void WorldMap::make_roads()
{
    roads_.clear();
    // Who joins whom: each living place to its two nearest living neighbours.
    constexpr int kCell = 64 * kNodeCells;
    std::map<std::pair<int, int>, std::vector<int>> cells;
    for (const Settlement& s : settlements_) {
        if (s.fell < 0 && !s.lone) {
            cells[{(s.x + kHalfCells) / kCell, (s.y + kHalfCells) / kCell}].push_back(s.id);
        }
    }
    std::vector<std::pair<int, int>> pairs;
    for (const Settlement& s : settlements_) {
        if (s.fell >= 0 || s.lone) {
            continue;
        }
        std::vector<std::pair<double, int>> near;
        const int gx = (s.x + kHalfCells) / kCell;
        const int gy = (s.y + kHalfCells) / kCell;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                const auto it = cells.find({gx + ox, gy + oy});
                if (it == cells.end()) {
                    continue;
                }
                for (const int id : it->second) {
                    const Settlement& o = settlements_[static_cast<std::size_t>(id)];
                    const double d = std::hypot(o.x - s.x, o.y - s.y);
                    if (id != s.id && d <= 50.0 * kNodeCells) {
                        near.emplace_back(d, id);
                    }
                }
            }
        }
        std::sort(near.begin(), near.end());
        for (std::size_t k = 0; k < near.size() && k < 2; ++k) {
            pairs.emplace_back(std::min(s.id, near[k].second), std::max(s.id, near[k].second));
        }
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    std::vector<std::vector<Segment>> found(pairs.size());
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < static_cast<int>(pairs.size()); ++p) {
        const Settlement& a = settlements_[static_cast<std::size_t>(pairs[static_cast<std::size_t>(p)].first)];
        const Settlement& b = settlements_[static_cast<std::size_t>(pairs[static_cast<std::size_t>(p)].second)];
        const int sx = (a.x + kHalfCells) / kNodeCells;
        const int sy = (a.y + kHalfCells) / kNodeCells;
        const int tx = (b.x + kHalfCells) / kNodeCells;
        const int ty = (b.y + kHalfCells) / kNodeCells;
        const int margin = 10;
        const int x0 = std::max(0, std::min(sx, tx) - margin);
        const int y0 = std::max(0, std::min(sy, ty) - margin);
        const int x1 = std::min(kN - 1, std::max(sx, tx) + margin);
        const int y1 = std::min(kN - 1, std::max(sy, ty) + margin);
        const int w = x1 - x0 + 1;
        const int h = y1 - y0 + 1;
        const auto local = [&](int x, int y) { return static_cast<std::size_t>((y - y0) * w + x - x0); };
        std::vector<float> best(static_cast<std::size_t>(w) * h, 1e30f);
        std::vector<int> came(static_cast<std::size_t>(w) * h, -1);
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
        best[local(sx, sy)] = 0.0f;
        open.emplace(0.0f, at(sx, sy));
        while (!open.empty()) {
            const auto [cost, n] = open.top();
            open.pop();
            const int x = n % kN;
            const int y = n / kN;
            if (x == tx && y == ty) {
                break;
            }
            if (cost > best[local(x, y)]) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int ax = x + dx;
                    const int ay = y + dy;
                    if ((dx == 0 && dy == 0) || ax < x0 || ay < y0 || ax > x1 || ay > y1) {
                        continue;
                    }
                    const std::size_t m = static_cast<std::size_t>(at(ax, ay));
                    const float step = dx != 0 && dy != 0 ? 1.41f : 1.0f;
                    const float climb = std::abs(elevation_[m] - elevation_[static_cast<std::size_t>(n)]);
                    const float wet = elevation_[m] < sea_ ? 60.0f : (lake_[m] ? 20.0f : (flow_[m] > 18.0f ? 2.5f : 0.0f));
                    const float next = cost + step * (1.0f + climb * 0.08f + wet);
                    if (next < best[local(ax, ay)]) {
                        best[local(ax, ay)] = next;
                        came[local(ax, ay)] = n;
                        open.emplace(next, static_cast<int>(m));
                    }
                }
            }
        }
        // No way but across the sea: no road.
        if (came[local(tx, ty)] < 0 || best[local(tx, ty)] > 50.0f * static_cast<float>(1 + std::max(std::abs(tx - sx), std::abs(ty - sy)))) {
            continue;
        }
        int n = at(tx, ty);
        while (came[local(n % kN, n / kN)] >= 0) {
            const int from = came[local(n % kN, n / kN)];
            const auto spot = [&](int node, float& px, float& py) {
                const std::uint64_t jh = hash(seed_ ^ 0x71, node % kN, node / kN);
                const bool end = node == at(sx, sy) || node == at(tx, ty);
                px = static_cast<float>((node % kN + 0.5 + (end ? 0.0 : (unit(jh) - 0.5) * 0.5)) * kNodeCells - kHalfCells);
                py = static_cast<float>((node / kN + 0.5 + (end ? 0.0 : (unit(mix64(jh)) - 0.5) * 0.5)) * kNodeCells - kHalfCells);
            };
            Segment r{};
            spot(from, r.ax, r.ay);
            spot(n, r.bx, r.by);
            r.width = 3.0f;
            found[static_cast<std::size_t>(p)].push_back(r);
            n = from;
        }
    }
    std::vector<std::array<int, 4>> boxes;
    for (const std::vector<Segment>& way : found) {
        for (const Segment& r : way) {
            roads_.push_back(r);
            boxes.push_back({static_cast<int>(std::min(r.ax, r.bx)) - 12, static_cast<int>(std::min(r.ay, r.by)) - 12,
                static_cast<int>(std::max(r.ax, r.bx)) + 12, static_cast<int>(std::max(r.ay, r.by)) + 12});
        }
    }
    road_near_ = index_boxes(boxes);
}

// --- a column ------------------------------------------------------------------------------------------------

int WorldMap::node_of(int wx, int wy) const
{
    const int x = (wx + kHalfCells) / kNodeCells;
    const int y = (wy + kHalfCells) / kNodeCells;
    if (wx < -kHalfCells || wy < -kHalfCells || x >= kN || y >= kN) {
        return -1;
    }
    return at(x, y);
}

double WorldMap::smooth_at(const std::vector<float>& values, double wx, double wy) const
{
    const double fx = std::clamp((wx + kHalfCells) / kNodeCells - 0.5, 0.0, kN - 1.001);
    const double fy = std::clamp((wy + kHalfCells) / kNodeCells - 0.5, 0.0, kN - 1.001);
    const int ix = static_cast<int>(fx);
    const int iy = static_cast<int>(fy);
    const double tx = smooth(fx - ix);
    const double ty = smooth(fy - iy);
    const double a = values[static_cast<std::size_t>(at(ix, iy))];
    const double b = values[static_cast<std::size_t>(at(ix + 1, iy))];
    const double c = values[static_cast<std::size_t>(at(ix, iy + 1))];
    const double d = values[static_cast<std::size_t>(at(ix + 1, iy + 1))];
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
}

WorldMap::Base WorldMap::base(int wx, int wy, int factor) const
{
    Base out;
    const double x = wx;
    const double y = wy;
    const int node = node_of(wx, wy);
    if (node < 0) {
        out.ground = sea_ - 60;
        out.water = sea_;
        return out;
    }
    double e = smooth_at(elevation_, x, y);
    const double r = smooth_at(rough_, x, y);
    e += r * 2.0 * (fbm(seed_ ^ 0x11, x, y, 96.0, 4, factor) - 0.5);
    e += r * 0.9 * (fbm(seed_ ^ 0x12, x, y, 26.0, 3, factor) - 0.5);
    e += 1.6 * (fbm(seed_ ^ 0x13, x, y, 11.0, 2, factor) - 0.5);

    // People level the ground they build on.
    bool on_lot = false;
    for (std::size_t k = settlement_near_.begin(node); k < settlement_near_.end(node); ++k) {
        const Settlement& s = settlements_[settlement_near_.items[k]];
        const double d = std::hypot(x - s.x, y - s.y);
        // A steep slope gets a longer way down to the town, so no hill stands over its wall.
        const double way = std::clamp(1.3 * std::abs(e - s.ground), s.radius * 0.4, s.radius * 1.7);
        if (d < s.radius * 0.9 + way) {
            e = s.ground + (e - s.ground) * smoothstep(s.radius * 0.9, s.radius * 0.9 + way, d);
            on_lot = on_lot || d < s.radius * 0.9;
        }
    }
    int water = -1;
    double ground = e;
    // Rivers: the nearest channel, on a warped map so no reach runs straight.
    if (river_near_.begin(node) < river_near_.end(node)) {
        const double px = x + 44.0 * (fbm(seed_ ^ 0x14, x, y, 120.0, 2, factor) - 0.5);
        const double py = y + 44.0 * (fbm(seed_ ^ 0x15, x, y, 120.0, 2, factor) - 0.5);
        double best = 1e30;
        double best_level = 0.0;
        double best_width = 0.0;
        for (std::size_t k = river_near_.begin(node); k < river_near_.end(node); ++k) {
            const Segment& s = segments_[river_near_.items[k]];
            // A river too thin to show at this coarseness is left out, a big one stays a voxel wide.
            if (s.width * 10.0 < factor) {
                continue;
            }
            const double width = std::max(static_cast<double>(s.width), factor * 0.55);
            double t = 0.0;
            const double d = segment_distance(px, py, s.ax, s.ay, s.bx, s.by, t) - width;
            if (d < best) {
                best = d;
                best_level = s.level_a + (s.level_b - s.level_a) * t;
                best_width = width;
            }
        }
        if (best < 150.0) {
            const int level = static_cast<int>(std::floor(best_level));
            if (best < 0.0) {
                ground = level - 2.0 - best_width / 7.0;
                water = level;
                out.bed = true;
            } else {
                const double valley = 12.0 + best_width * 1.5 + 0.2 * std::abs(e - level);
                ground = (level + 1.0) + (e - (level + 1.0)) * smoothstep(0.0, std::min(valley, 140.0), best);
            }
        }
    }
    // Lakes stand as high as their spill.
    {
        const double fx = (x + kHalfCells) / kNodeCells - 0.5;
        const double fy = (y + kHalfCells) / kNodeCells - 0.5;
        const int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, kN - 2);
        const int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, kN - 2);
        for (const int n : {at(ix, iy), at(ix + 1, iy), at(ix, iy + 1), at(ix + 1, iy + 1)}) {
            if (lake_[static_cast<std::size_t>(n)]) {
                const int level = static_cast<int>(std::floor(level_[static_cast<std::size_t>(n)]));
                if (ground < level && level > water) {
                    water = level;
                }
            }
        }
    }
    if (ground < sea_ && sea_ > water) {
        water = sea_;
    }
    // Flats at the level of the sea are land and water by the cell; from far off that is only flicker, and they read as sea.
    if (factor >= 8 && water < 0 && !on_lot && ground < sea_ + 2.0) {
        water = sea_;
        ground = sea_ - 1.0;
    }
    // Roads go over everything: a causeway where they meet water.
    if (road_near_.begin(node) < road_near_.end(node)) {
        const double half = std::max(1.5, factor * 0.3);
        for (std::size_t k = road_near_.begin(node); k < road_near_.end(node); ++k) {
            const Segment& s = roads_[road_near_.items[k]];
            double t = 0.0;
            if (factor <= 16 && segment_distance(x, y, s.ax, s.ay, s.bx, s.by, t) <= half) {
                out.street = true;
                out.clear = true;
                break;
            }
        }
        if (out.street && water > sea_ - 1) {
            ground = std::max(ground, static_cast<double>(water) + 1.0);
            water = -1;
            out.bed = false;
        } else if (out.street && water >= 0) {
            out.street = false;
            out.clear = false;
        }
    }
    // Streets and lots.
    if (on_lot) {
        out.clear = true;
        for (std::size_t k = settlement_near_.begin(node); k < settlement_near_.end(node); ++k) {
            const Settlement& s = settlements_[settlement_near_.items[k]];
            if (std::hypot(x - s.x, y - s.y) >= s.radius) {
                continue;
            }
            for (const Settlement::Street& t : s.streets) {
                if (t.circle > 0) {
                    if (std::abs(std::hypot(x - t.x0, y - t.y0) - t.circle) <= t.half) {
                        out.street = true;
                    }
                } else if (wx >= std::min(t.x0, t.x1) - t.half && wx <= std::max(t.x0, t.x1) + t.half && wy >= std::min(t.y0, t.y1) - t.half &&
                    wy <= std::max(t.y0, t.y1) + t.half) {
                    out.street = true;
                }
            }
        }
    }
    out.ground = ground;
    out.water = water;
    return out;
}

Biome WorldMap::biome_at(int wx, int wy, int factor) const
{
    // Borders wander: the nearest node, looked up a little off where you stand. (Seen from far off, they wander no finer than the samples.)
    const double wave = std::max(90.0, factor * 3.0);
    const double jx = 150.0 * (value_noise(seed_ ^ 0x16, wx / wave, wy / wave) - 0.5);
    const double jy = 150.0 * (value_noise(seed_ ^ 0x17, wx / wave, wy / wave) - 0.5);
    const int node = node_of(static_cast<int>(wx + jx), static_cast<int>(wy + jy));
    return node < 0 ? Biome::Ocean : biome_[static_cast<std::size_t>(node)];
}

double WorldMap::temperature_at(int wx, int wy) const { return smooth_at(temperature_, wx, wy); }
double WorldMap::rain_at(int wx, int wy) const { return smooth_at(rain_, wx, wy); }

double WorldMap::capacity_at_node(int nx, int ny) const
{
    const std::size_t n = static_cast<std::size_t>(at(nx, ny));
    switch (biome_[n]) {
    case Biome::Grassland: return 1.0;
    case Biome::Savanna: return 0.8;
    case Biome::Forest: return 0.6;
    case Biome::Swamp: return 0.4;
    case Biome::Jungle: return 0.5;
    case Biome::Taiga: return 0.35;
    case Biome::Tundra: return 0.15;
    case Biome::Beach: return 0.1;
    default: return 0.0;
    }
}

Column WorldMap::finish(const Base& b, int wx, int wy, int factor) const
{
    Column out;
    out.top = std::clamp(static_cast<int>(std::floor(b.ground)), 2, height_ - 40);
    out.water = b.water > out.top ? b.water : -1;
    out.clear = b.clear;
    out.biome = biome_at(wx, wy, factor);
    out.bend = static_cast<float>(6.0 * fbm(seed_ ^ 0x44, wx, wy, 40.0, 2));
    if (out.water >= 0) {
        out.surface = static_cast<std::uint8_t>(out.water - out.top > 3 ? material::clay : material::sand);
        return out;
    }
    if (b.street) {
        out.surface = material::gravel;
        return out;
    }
    const double above = temperature_at(wx, wy) - std::max(0.0, b.ground - smooth_at(elevation_, wx, wy)) * 0.5 * 0.026;
    if (above < -3.5 || out.top >= height_ - 60) {
        out.surface = material::chalk; // snow
        return out;
    }
    switch (out.biome) {
    case Biome::Ocean:
    case Biome::Beach:
    case Biome::Desert:
        out.surface = material::sand;
        break;
    case Biome::Snow:
        out.surface = material::chalk;
        break;
    case Biome::Tundra:
        out.surface = material::slate;
        break;
    case Biome::Mountain:
        out.surface = material::granite;
        break;
    case Biome::Savanna:
        out.surface = factor < 8 && (hash(seed_ ^ 0x18, wx / 3, wy / 3) & 3u) == 0 ? material::sand : material::soil; // patches too small to see from far off
        out.turf = out.surface == material::soil;
        break;
    default:
        out.surface = material::soil;
        out.turf = true;
        break;
    }
    return out;
}

// WATER STAYS WHERE IT IS PUT. The law moves water freely, so every level must be held in by what stands next to
// it: a column of water beside anything lower is itself a bank, standing as high as its water would. Rivers are
// therefore chains of still pools, each spilling over a sill, and they do not run out in the law.
Column WorldMap::contained(int wx, int wy) const
{
    Base b = base(wx, wy, 1);
    if (b.water >= 0 && b.ground < b.water) {
        for (const auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
            const Base n = base(wx + dx, wy + dy, 1);
            const int level = n.ground < n.water ? n.water : -1;
            if (level < b.water && std::floor(n.ground) < b.water) {
                b.ground = b.water;
                b.water = -1;
                break;
            }
        }
    } else {
        b.water = -1;
    }
    return finish(b, wx, wy);
}

Column WorldMap::column(int wx, int wy, int factor) const
{
    return factor > 1 ? finish(base(wx, wy, factor), wx, wy, factor) : contained(wx, wy);
}

bool WorldMap::tree_at(int gx, int gy, Tree& out, int factor) const
{
    const std::uint64_t h = hash(seed_ ^ 0x77, gx, gy);
    const int x = gx * kTreeGrid + static_cast<int>(h % kTreeGrid);
    const int y = gy * kTreeGrid + static_cast<int>((h >> 8) % kTreeGrid);
    const Biome biome = biome_at(x, y);
    const LandGrowth& land = growth_[static_cast<std::size_t>(biome)];
    const bool big = ((gx % kBigEvery) + kBigEvery) % kBigEvery == 0 && ((gy % kBigEvery) + kBigEvery) % kBigEvery == 0;
    const std::vector<int>& kinds = (big ? big_of_ : small_of_)[static_cast<std::size_t>(biome)];
    if (kinds.empty()) {
        return false;
    }
    // Woods and clearings.
    const double density = (big ? land.big : land.small) * std::clamp((fbm(seed_ ^ 0x55, x, y, 70.0, 3, factor) - 0.28) * 3.0, 0.1, 1.3);
    if (unit(hash(seed_ ^ 0x66, gx, gy)) >= density) {
        return false;
    }
    const Column col = column(x, y, factor);
    if (col.clear || col.water >= 0 || (big && settlement_at(x, y) != nullptr)) {
        return false;
    }
    // A wood is mostly of one kind, that changes slowly over the land, with others in it.
    const double stand = fbm(seed_ ^ 0x5a, x, y, 260.0, 2, factor);
    const std::uint64_t r = h >> 16;
    const std::size_t pick = (r & 3u) != 0u ? static_cast<std::size_t>(std::clamp((stand - 0.25) * 2.0, 0.0, 0.999) * static_cast<double>(kinds.size())) : static_cast<std::size_t>((r >> 2) % kinds.size());
    out.species = kinds[pick];
    const TreeSpecies& sp = species_[static_cast<std::size_t>(out.species)];
    const double metres = sp.height_min + (sp.height_max - sp.height_min) * unit(hash(seed_ ^ 0x67, gx, gy));
    out.x = x;
    out.y = y;
    out.ground = col.top;
    out.trunk = std::max(1, static_cast<int>(metres * 2.0));
    out.crown = sp.leaf * 2.0 + 3.0;
    switch (sp.form) {
    case TreeSpecies::Form::Conifer:
        out.kind = Tree::Pine;
        out.spread = out.trunk * 0.24 + 1.0;
        break;
    case TreeSpecies::Form::Column:
        out.kind = Tree::Cactus;
        out.spread = 3.0;
        break;
    case TreeSpecies::Form::Bush:
        out.kind = Tree::Shrub;
        out.spread = sp.leaf + 1.0;
        break;
    case TreeSpecies::Form::Palm:
        out.kind = Tree::Oak;
        out.spread = 3.0 + out.trunk * 0.28;
        break;
    default:
        out.kind = Tree::Oak;
        out.spread = std::min(static_cast<double>(kTreeReach), out.trunk * 0.42 + sp.leaf);
        break;
    }
    return out.top() + 2 < height_;
}

std::shared_ptr<const std::vector<TreeVoxel>> WorldMap::grown(const Tree& tree, int gx, int gy) const
{
    const std::int64_t key = chunk_key(gx, gy, 0);
    {
        const std::lock_guard<std::mutex> lock(grown_mutex_);
        const auto found = grown_.find(key);
        if (found != grown_.end()) {
            return found->second;
        }
    }
    auto made = std::make_shared<const std::vector<TreeVoxel>>(grow_tree(species_[static_cast<std::size_t>(tree.species)], hash(seed_ ^ 0x78, gx, gy), tree.trunk, 0.5));
    const std::lock_guard<std::mutex> lock(grown_mutex_);
    if (grown_.size() > 3000) {
        grown_.clear(); // whoever holds one keeps it; the rest grow again, the same
    }
    grown_[key] = made;
    return made;
}

const FloraKind* WorldMap::flora_at(int wx, int wy, const Column& col) const
{
    if (col.clear || col.water >= 0) {
        return nullptr;
    }
    const std::size_t land = static_cast<std::size_t>(col.biome);
    const std::vector<int>& kinds = flora_of_[land];
    if (kinds.empty()) {
        return nullptr;
    }
    const std::uint64_t h = hash(seed_ ^ 0x79, wx, wy);
    // In patches: a meadow here, bare there.
    const double thick = growth_[land].flora * std::clamp((fbm(seed_ ^ 0x7a, wx, wy, 22.0, 2) - 0.32) * 4.0, 0.0, 2.0);
    if (unit(h) >= thick) {
        return nullptr;
    }
    // One kind to a patch, mostly.
    const double which = fbm(seed_ ^ 0x7b, wx, wy, 18.0, 1);
    const std::size_t pick = (h >> 40 & 3u) != 0u ? static_cast<std::size_t>(std::clamp((which - 0.2) * 1.7, 0.0, 0.999) * static_cast<double>(kinds.size())) : static_cast<std::size_t>((h >> 42) % kinds.size());
    return &flora_[static_cast<std::size_t>(kinds[pick])];
}

const Settlement* WorldMap::settlement_at(int wx, int wy) const
{
    const int node = node_of(wx, wy);
    if (node < 0) {
        return nullptr;
    }
    for (std::size_t k = settlement_near_.begin(node); k < settlement_near_.end(node); ++k) {
        const Settlement& s = settlements_[settlement_near_.items[k]];
        if (std::hypot(static_cast<double>(wx - s.x), static_cast<double>(wy - s.y)) < s.radius) {
            return &s;
        }
    }
    return nullptr;
}

const Settlement* WorldMap::nearest_settlement(int wx, int wy) const
{
    const Settlement* best = nullptr;
    double best_d = 1e30;
    for (const Settlement& s : settlements_) {
        const double d = std::hypot(static_cast<double>(wx - s.x), static_cast<double>(wy - s.y));
        if (d < best_d) {
            best_d = d;
            best = &s;
        }
    }
    return best;
}

const Building* WorldMap::building_over(const Settlement& s, int wx, int wy) const
{
    for (const Building& b : s.buildings) {
        if (wx < b.x0 - 1 || wx > b.x1 + 1 || wy < b.y0 - 1 || wy > b.y1 + 1) {
            continue;
        }
        if (b.kind == Building::Kind::Ring || b.kind == Building::Kind::Yard || b.kind == Building::Kind::Graves) {
            // Only its band is there.
            const Style& st = styles_[static_cast<std::size_t>(b.style)];
            if (building_cell(b, st, wx, wy, b.ground + 2).kind == BuildingCell::Outside) {
                continue;
            }
        }
        return &b;
    }
    return nullptr;
}

FarColumn WorldMap::far_column(int wx, int wy, int factor) const
{
    const Column col = column(wx, wy, std::max(factor, 2));
    FarColumn out{.top = col.top, .water = col.water, .surface = col.surface, .turf = col.turf};
    if (const Settlement* s = settlement_at(wx, wy)) {
        const Style& town = styles_[static_cast<std::size_t>(s->style)];
        if (factor >= 64) {
            // A town from the sky: a patch of its roofs.
            if (!s->buildings.empty() && std::hypot(static_cast<double>(wx - s->x), static_cast<double>(wy - s->y)) < s->radius * 0.8) {
                out.roof_top = s->ground + (s->fell >= 0 ? 2 : town.storey + 2);
                out.roof_material = s->burned ? material::charred : town.roof_material;
            }
            return out;
        }
        if (const Building* b = building_over(*s, wx, wy)) {
            const Style& st = styles_[static_cast<std::size_t>(b->style)];
            int bx0, by0, bx1, by1, top;
            b->bounds(styles_, bx0, by0, bx1, by1, top);
            const int walls = b->has_rooms() ? b->ground + st.raise + 1 + b->floors * st.storey + std::min(b->x1 - b->x0, b->y1 - b->y0) / 4 : top - 2;
            out.roof_top = b->ground + static_cast<int>((walls - b->ground) * (1.0f - b->ruin * 0.8f));
            out.roof_material = b->burned ? material::charred
                : (b->has_rooms() ? (b->ruin > 0.25f ? st.wall_material : st.roof_material) : (b->kind == Building::Kind::Ring ? std::max(st.ring_material, 0) : st.base_material));
            return out;
        }
    }
    if (col.water >= 0 || col.clear) {
        return out;
    }
    if (factor < 16) {
        const int gx = static_cast<int>(std::floor(static_cast<double>(wx) / kTreeGrid));
        const int gy = static_cast<int>(std::floor(static_cast<double>(wy) / kTreeGrid));
        Tree tree;
        if (tree_at(gx, gy, tree, factor) && tree.kind != Tree::Cactus && std::hypot(static_cast<double>(wx - tree.x), static_cast<double>(wy - tree.y)) <= tree.spread) {
            out.crown_top = tree.ground + tree.trunk + static_cast<int>(tree.crown);
        }
        // And the crowns of the big ones around.
        const int bx = static_cast<int>(std::floor(static_cast<double>(gx) / kBigEvery)) * kBigEvery;
        const int by = static_cast<int>(std::floor(static_cast<double>(gy) / kBigEvery)) * kBigEvery;
        for (int oy = -kBigEvery; oy <= kBigEvery; oy += kBigEvery) {
            for (int ox = -kBigEvery; ox <= kBigEvery; ox += kBigEvery) {
                if (tree_at(bx + ox, by + oy, tree, factor) && tree.kind != Tree::Cactus) {
                    const double d = std::hypot(static_cast<double>(wx - tree.x), static_cast<double>(wy - tree.y));
                    if (d <= tree.spread) {
                        // A crown is lower at its rim.
                        const int crown = tree.ground + tree.trunk + static_cast<int>(tree.crown) - static_cast<int>(tree.trunk * 0.3 * d / tree.spread);
                        out.crown_top = std::max(out.crown_top, crown);
                    }
                }
            }
        }
    } else {
        // Trees no voxel can hold, but a forest still stands: as much canopy as it has trees.
        const LandGrowth& land = growth_[static_cast<std::size_t>(col.biome)];
        double density = std::min(1.0, land.big + 0.5 * land.small);
        const std::vector<int>& kinds = big_of_[static_cast<std::size_t>(col.biome)];
        const double height = kinds.empty() ? 6.0 : species_[static_cast<std::size_t>(kinds.front())].height_max * 1.4;
        density *= std::clamp((fbm(seed_ ^ 0x55, wx, wy, 70.0, 3, factor) - 0.28) * 3.0, 0.1, 1.3);
        // From high up a forest is its colour, not its single trees.
        if (factor >= 128 ? density > 0.3 : unit(hash(seed_ ^ 0x19, wx / factor, wy / factor)) < density) {
            out.crown_top = col.top + static_cast<int>(height);
        }
    }
    return out;
}

std::shared_ptr<const ColumnTile> WorldMap::tile(int cx, int cy) const
{
    const std::int64_t key = chunk_key(cx, cy, 0);
    {
        const std::lock_guard<std::mutex> lock(tiles_mutex_);
        const auto found = tiles_.find(key);
        if (found != tiles_.end()) {
            return found->second;
        }
    }
    const int n = chunk_;
    const int x0 = cx * n;
    const int y0 = cy * n;
    auto made = std::make_shared<ColumnTile>();
    // The columns, each held in by its neighbours: one ring wider than the tile.
    const int side = n + 2;
    std::vector<Base> bases(static_cast<std::size_t>(side) * side);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            bases[static_cast<std::size_t>(y * side + x)] = base(x0 + x - 1, y0 + y - 1, 1);
        }
    }
    made->min_top = height_;
    made->max_top = 0;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            Base b = bases[static_cast<std::size_t>((y + 1) * side + x + 1)];
            if (b.water >= 0 && b.ground < b.water) {
                for (const auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                    const Base& o = bases[static_cast<std::size_t>((y + 1 + dy) * side + x + 1 + dx)];
                    const int level = o.ground < o.water ? o.water : -1;
                    if (level < b.water && std::floor(o.ground) < b.water) {
                        b.ground = b.water;
                        b.water = -1;
                        break;
                    }
                }
            } else {
                b.water = -1;
            }
            const Column c = finish(b, x0 + x, y0 + y);
            made->columns[static_cast<std::size_t>(y * n + x)] = c;
            made->min_top = std::min(made->min_top, c.top);
            made->max_top = std::max({made->max_top, c.top, c.water});
        }
    }
    made->feature_top = made->max_top + 3; // what grows on the ground
    const int g0x = static_cast<int>(std::floor(static_cast<double>(x0 - kTreeReach) / kTreeGrid));
    const int g1x = static_cast<int>(std::floor(static_cast<double>(x0 + n + kTreeReach) / kTreeGrid));
    const int g0y = static_cast<int>(std::floor(static_cast<double>(y0 - kTreeReach) / kTreeGrid));
    const int g1y = static_cast<int>(std::floor(static_cast<double>(y0 + n + kTreeReach) / kTreeGrid));
    for (int gy = g0y; gy <= g1y; ++gy) {
        for (int gx = g0x; gx <= g1x; ++gx) {
            Tree tree;
            if (tree_at(gx, gy, tree, 1)) {
                // Near enough to reach this tile?
                if (tree.x + kTreeReach < x0 || tree.x - kTreeReach >= x0 + n || tree.y + kTreeReach < y0 || tree.y - kTreeReach >= y0 + n) {
                    continue;
                }
                tree.shape = grown(tree, gx, gy);
                made->trees.push_back(tree);
                made->feature_top = std::max(made->feature_top, tree.top());
            }
        }
    }
    // What stands here.
    const int node = node_of(x0 + n / 2, y0 + n / 2);
    if (node >= 0) {
        for (std::size_t k = settlement_near_.begin(node); k < settlement_near_.end(node); ++k) {
            const Settlement& s = settlements_[settlement_near_.items[k]];
            for (const Building& b : s.buildings) {
                int bx0, by0, bx1, by1, top;
                b.bounds(styles_, bx0, by0, bx1, by1, top);
                if (bx1 >= x0 && bx0 < x0 + n && by1 >= y0 && by0 < y0 + n) {
                    made->buildings.push_back(&b);
                    made->feature_top = std::max(made->feature_top, top);
                    made->min_top = std::min(made->min_top, b.lowest());
                }
            }
        }
    }
    const std::lock_guard<std::mutex> lock(tiles_mutex_);
    if (tiles_.size() > 6000) {
        tiles_.clear(); // whoever holds one keeps it; the rest are made again, the same
    }
    tiles_[key] = made;
    return made;
}

void WorldMap::home(int& wx, int& wy) const
{
    wx = 0;
    wy = 0;
    // Living places, nearest the middle first; the first with grass by its gate is home.
    std::vector<const Settlement*> places;
    for (const Settlement& s : settlements_) {
        if (s.fell < 0 && s.tier >= 1) {
            places.push_back(&s);
        }
    }
    std::sort(places.begin(), places.end(), [](const Settlement* a, const Settlement* b) {
        return std::hypot(static_cast<double>(a->x), static_cast<double>(a->y)) < std::hypot(static_cast<double>(b->x), static_cast<double>(b->y));
    });
    const auto look_around = [&](int cx, int cy, int from, int to) {
        for (int r = from; r < to; r += 4) {
            for (int k = 0; k < 32; ++k) {
                const double angle = 6.283185307 * k / 32.0;
                const int x = cx + static_cast<int>(std::lround(r * std::cos(angle)));
                const int y = cy + static_cast<int>(std::lround(r * std::sin(angle)));
                const Column c = column(x, y);
                if (c.water < 0 && c.turf && c.top > sea_ + 2) {
                    wx = x;
                    wy = y;
                    return true;
                }
            }
        }
        return false;
    };
    for (const Settlement* s : places) {
        const int r = static_cast<int>(s->radius * 0.95);
        if (look_around(s->x, s->y, r, r + 120)) {
            return;
        }
    }
    look_around(0, 0, 0, 40000);
}

bool WorldMap::rivers_run_down() const
{
    return std::all_of(segments_.begin(), segments_.end(), [](const Segment& s) { return s.level_b <= s.level_a; });
}

bool WorldMap::river_point(double min_width, double max_width, int& wx, int& wy) const
{
    // Land rivers only, nearest the middle of the map first, so a gate finds one quickly.
    std::vector<const Segment*> order;
    for (const Segment& s : segments_) {
        if (s.width >= min_width && s.width <= max_width && s.level_b > sea_ + 4.0f) {
            order.push_back(&s);
        }
    }
    std::sort(order.begin(), order.end(), [](const Segment* a, const Segment* b) {
        return std::hypot(a->ax, a->ay) < std::hypot(b->ax, b->ay);
    });
    for (const Segment* s : order) {
        const double mx = (s->ax + s->bx) * 0.5;
        const double my = (s->ay + s->by) * 0.5;
        for (int r = 0; r < 80; r += 2) {
            for (int k = 0; k < 16; ++k) {
                const double angle = 6.283185307 * k / 16.0;
                const int x = static_cast<int>(mx + r * std::cos(angle));
                const int y = static_cast<int>(my + r * std::sin(angle));
                const Column c = column(x, y);
                if (c.water > sea_ + 2) {
                    wx = x;
                    wy = y;
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace fieldengine::field

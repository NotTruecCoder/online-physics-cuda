#include "field/Looks.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fieldengine::field {

const char* const kHairs[] = {"bald", "cropped", "short", "side part", "curls", "long", "ponytail", "bun", "braids", "mohawk"};
const int kHairCount = 10;
const char* const kHairColours[] = {"black", "dark brown", "brown", "auburn", "red", "blond", "ash"};
const int kHairColourCount = 7;
const char* const kEyes[] = {"brown", "dark", "hazel", "green", "blue", "grey"};
const int kEyeCount = 6;
const char* const kFaces[] = {"round", "oval", "square", "long"};
const int kFaceCount = 4;

namespace {

struct Rgb {
    double r, g, b;
};

Rgb mix(Rgb a, Rgb b, double t)
{
    return Rgb{a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

Rgb scale(Rgb a, double s)
{
    return Rgb{a.r * s, a.g * s, a.b * s};
}

Rgb skin_of(double m)
{
    const Rgb pale{244, 208, 182};
    const Rgb light{226, 176, 138};
    const Rgb tan{184, 128, 88};
    const Rgb brown{126, 82, 54};
    const Rgb dark{74, 48, 36};
    m = std::clamp(m, 0.0, 1.0) * 4.0;
    const int k = std::min(static_cast<int>(m), 3);
    const Rgb stops[5] = {pale, light, tan, brown, dark};
    return mix(stops[k], stops[k + 1], m - k);
}

Rgb hair_of(int colour, double grey)
{
    static const Rgb kColours[] = {{28, 24, 26}, {58, 38, 28}, {96, 62, 38}, {120, 58, 36}, {176, 82, 42}, {214, 176, 104}, {166, 158, 148}};
    return mix(kColours[std::clamp(colour, 0, kHairColourCount - 1)], Rgb{196, 194, 192}, grey);
}

Rgb eyes_of(int eyes)
{
    static const Rgb kColours[] = {{92, 58, 30}, {36, 24, 20}, {116, 90, 44}, {66, 120, 74}, {66, 112, 178}, {120, 134, 146}};
    return kColours[std::clamp(eyes, 0, kEyeCount - 1)];
}

struct Writer {
    std::ostringstream o;
    void colour(const char* name, Rgb c)
    {
        o << "colour " << name << " " << static_cast<int>(std::clamp(c.r, 0.0, 255.0)) << " " << static_cast<int>(std::clamp(c.g, 0.0, 255.0)) << " " << static_cast<int>(std::clamp(c.b, 0.0, 255.0)) << "\n";
    }
    void part(const char* bone, int x, int y, int z) { o << "part " << bone << " pivot " << x << " " << y << " " << z << "\n"; }
    void box(int x0, int y0, int z0, int x1, int y1, int z1, const char* c) { o << "box " << x0 << " " << y0 << " " << z0 << " " << x1 << " " << y1 << " " << z1 << " " << c << "\n"; }
    void dot(int x, int y, int z, const char* c) { o << "dot " << x << " " << y << " " << z << " " << c << "\n"; }
};

std::uint64_t mix64(std::uint64_t a, std::uint64_t b)
{
    std::uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x51ED27ull) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return x;
}

double unit(std::uint64_t a, std::uint64_t b)
{
    return static_cast<double>(mix64(a, b) >> 11) / 9007199254740992.0;
}

} // namespace

std::string body_model(const Looks& l)
{
    Writer w;
    const bool woman = l.sex == 0;
    const Rgb skin = skin_of(l.melanin);
    const Rgb hair = hair_of(l.hair_colour, l.grey);
    w.o << "voxel 0.07\n";
    w.colour("skin", skin);
    w.colour("shade", scale(skin, 0.86));
    w.colour("blush", mix(skin, Rgb{214, 96, 90}, 0.28));
    w.colour("lip", mix(scale(skin, 0.8), Rgb{176, 66, 72}, woman ? 0.6 : 0.3));
    w.colour("hair", hair);
    w.colour("hair2", scale(hair, 0.78));
    w.colour("brow", scale(hair, l.grey > 0.5 ? 0.8 : 0.72));
    w.colour("eye", eyes_of(l.eyes));
    w.colour("white", Rgb{240, 238, 232});
    w.colour("linen", Rgb{214, 206, 186});
    w.colour("fleck", scale(skin, 0.74));

    // Legs: a hair thinner than they look clothed, so that cloth has somewhere to be.
    for (const int side : {1, -1}) {
        const int y0 = side > 0 ? 1 : -3;
        const int y1 = side > 0 ? 3 : -1;
        w.part(side > 0 ? "leg_l" : "leg_r", 0, side * 2, 10);
        w.box(-1, y0, 2, 1, y1, 9, "skin");
        w.box(-1, y0, 1, 2, y1, 1, "shade");
        w.box(-1, y0, 8, 1, y1, 9, "linen");
    }
    // What a playing piece stands on.
    w.colour("stand", Rgb{74, 62, 50});
    w.colour("stand2", Rgb{96, 82, 66});
    w.part("leg_l", 0, 2, 10);
    w.box(-3, -5, 0, 3, 5, 0, "stand");
    w.box(-4, -4, 0, 4, 4, 0, "stand2");
    w.part("torso", 0, 0, 10);
    if (woman) {
        w.box(-2, -4, 10, 2, 4, 11, "linen");
        w.box(-2, -3, 12, 1, 3, 14, "skin");
        w.box(-2, -4, 15, 2, 4, 17, "linen");
        w.box(3, -2, 16, 3, 2, 17, "linen");
        w.box(-2, -3, 18, 1, 3, 18, "skin");
    } else {
        w.box(-2, -4, 10, 2, 4, 11, "linen");
        w.box(-2, -4, 12, 2, 4, 18, "skin");
        w.box(2, -3, 15, 2, -1, 16, "shade");
        w.box(2, 1, 15, 2, 3, 16, "shade");
    }
    for (const int side : {1, -1}) {
        const int y0 = side > 0 ? 5 : -6;
        const int y1 = side > 0 ? 6 : -5;
        // No arm: a hand in the air, going where an arm would have taken it.
        w.part(side > 0 ? "arm_l" : "arm_r", 0, side * 5, 18);
        w.box(-1, y0, 8, 1, y1, 10, "skin");
        w.box(-1, y0, 8, 1, y1, 8, "shade");
        w.o << "socket " << (side > 0 ? "hand_l" : "hand_r") << " 0 " << side * 5.5 << " 9\n";
    }
    w.part("head", 0, 0, 19);
    w.box(-1, -1, 19, 1, 1, 19, "shade");
    // The shape of a face is the shape of its jaw.
    const int top = 26;
    switch (l.face) {
    case 1: // oval
        w.box(-3, -3, 21, 3, 3, top, "skin");
        w.box(-2, -2, 20, 3, 2, 20, "skin");
        break;
    case 2: // square
        w.box(-3, -3, 20, 3, 3, top, "skin");
        break;
    case 3: // long
        w.box(-3, -3, 21, 3, 3, top, "skin");
        w.box(-2, -2, 19, 3, 2, 20, "skin");
        break;
    default: // round
        w.box(-3, -3, 21, 3, 3, top - 1, "skin");
        w.box(-2, -2, 20, 3, 2, 20, "skin");
        w.box(-2, -2, top, 2, 2, top, "skin");
        break;
    }
    // Ears, under the hair if the hair says so.
    w.dot(0, 4, 23, "shade");
    w.dot(0, -4, 23, "shade");
    // Eyes: the white, and what looks out of it. Brows over them. A woman's have lashes in the brow's colour.
    for (const int side : {1, -1}) {
        w.dot(3, side * 2, 23, "eye");
        w.dot(3, side * 1, 23, "white");
        w.dot(3, side * 3, 23, "white");
        w.dot(3, side * 2, 24, l.brows ? "brow" : "shade");
        if (l.brows) {
            w.dot(3, side * 1, 24, "brow");
            w.dot(3, side * 3, 24, "brow");
        }
        if (woman) {
            w.dot(3, side * 3, 24, "brow");
            w.dot(3, side * 3, 22, "blush");
        }
    }
    if (l.nose == 1) {
        w.box(4, 0, 22, 4, 0, 23, "skin");
    } else if (l.nose == 2) {
        w.box(4, -1, 22, 4, 1, 22, "skin");
    } else {
        w.dot(4, 0, 22, "skin");
    }
    w.box(3, -1, 21, 3, 1, 21, "lip");
    if (l.freckles) {
        w.dot(3, 2, 22, "fleck");
        w.dot(3, -2, 22, "fleck");
        w.dot(3, 1, 22, "fleck");
    }
    if (!woman && l.beard > 0) {
        if (l.beard == 3) {
            w.box(3, -3, 20, 4, 3, 21, "hair");
            w.dot(3, 0, 21, "lip");
            w.box(2, -3, 19, 3, 3, 19, "hair2");
        } else if (l.beard == 2) {
            w.box(3, -1, 20, 4, 1, 20, "hair");
            w.box(4, -1, 19, 4, 1, 19, "hair2");
        } else {
            w.box(3, -3, 20, 3, 3, 20, "shade");
        }
    }
    // Hair. A hat takes everything above the ears with it.
    const int style = l.covered ? std::min(l.hair, 1) : l.hair;
    const auto cap = [&]() {
        w.box(-4, -4, 25, 3, 4, 27, "hair");
        w.box(-4, -4, 21, -3, 4, 25, "hair");
        w.box(-3, -4, 24, 2, -4, 25, "hair");
        w.box(-3, 4, 24, 2, 4, 25, "hair");
        w.box(3, -3, 26, 3, 3, 26, "hair");
    };
    switch (style) {
    case 1: // cropped
        w.box(-3, -3, 26, 3, 3, 27, "hair");
        w.box(-3, -4, 23, -3, 4, 26, "hair2");
        w.box(-2, -4, 25, 2, -4, 26, "hair2");
        w.box(-2, 4, 25, 2, 4, 26, "hair2");
        break;
    case 2: // short
        cap();
        break;
    case 3: // side part
        cap();
        w.box(-1, 2, 27, 3, 4, 28, "hair");
        w.box(3, 1, 25, 4, 3, 26, "hair");
        w.box(-2, -4, 28, 1, -2, 28, "hair2");
        break;
    case 4: // curls
        cap();
        for (int y = -4; y <= 4; y += 2) {
            for (int x = -4; x <= 2; x += 2) {
                w.dot(x, y, 28, "hair2");
            }
            w.dot(-5, y, 25, "hair2");
            w.dot(-5, y, 22, "hair");
        }
        w.dot(3, -4, 24, "hair2");
        w.dot(3, 4, 24, "hair2");
        break;
    case 5: // long
        cap();
        w.box(-4, -4, 14, -3, 4, 21, "hair");
        w.box(-2, -4, 18, 1, -4, 24, "hair");
        w.box(-2, 4, 18, 1, 4, 24, "hair");
        w.box(-4, -2, 13, -4, 2, 13, "hair2");
        break;
    case 6: // ponytail
        cap();
        w.box(-5, -1, 24, -5, 1, 26, "hair2");
        w.box(-6, -1, 15, -5, 1, 24, "hair");
        w.box(-6, 0, 13, -6, 0, 14, "hair2");
        break;
    case 7: // bun
        cap();
        w.box(-3, -2, 28, 1, 2, 30, "hair");
        w.box(-2, -1, 31, 0, 1, 31, "hair2");
        break;
    case 8: // braids
        cap();
        for (const int side : {1, -1}) {
            w.box(-1, side * 4, 17, 0, side * 4, 24, "hair");
            w.dot(-1, side * 4, 16, "hair2");
            for (int z = 18; z < 24; z += 2) {
                w.dot(0, side * 4, z, "hair2");
            }
        }
        break;
    case 9: // mohawk
        w.box(-4, -1, 27, 3, 1, 29, "hair");
        w.box(-3, 0, 30, 2, 0, 31, "hair2");
        w.box(-4, -1, 23, -4, 1, 26, "hair");
        break;
    default:
        break;
    }
    return w.o.str();
}

Looks looks_of_person(std::uint64_t pid, std::uint64_t family, int culture, double temperature)
{
    // Two parents, each of a people that has lived a long time where it lives: the sun in the skin, and a little of their own.
    const std::uint64_t mother = mix64(family, 101);
    const std::uint64_t father = mix64(family, 102);
    const double sun = std::clamp((temperature + 6.0) / 34.0, 0.0, 1.0);
    const double people = 0.25 * (unit(static_cast<std::uint64_t>(culture), 7) - 0.5);
    const auto parent_melanin = [&](std::uint64_t parent) { return std::clamp(0.08 + 0.75 * sun + people + 0.2 * (unit(parent, 1) - 0.5), 0.0, 1.0); };
    Looks l;
    l.sex = unit(pid, 1) < 0.5 ? 0 : 1;
    l.melanin = std::clamp(0.5 * (parent_melanin(mother) + parent_melanin(father)) + 0.06 * (unit(pid, 2) - 0.5), 0.0, 1.0);
    // Dark hair and dark eyes hold over the fair ones, and where the sun is strong the fair ones are few.
    const auto hair_gene = [&](std::uint64_t parent) {
        const double fair = unit(parent, 3) * (1.2 - sun) + 0.15 * unit(static_cast<std::uint64_t>(culture), 8);
        return fair < 0.28 ? 0 : fair < 0.45 ? 1 : fair < 0.62 ? 2 : fair < 0.72 ? 3 : fair < 0.8 ? 4 : 5;
    };
    l.hair_colour = std::min(hair_gene(mother), hair_gene(father));
    if (unit(pid, 4) < 0.25) {
        l.hair_colour = std::max(hair_gene(mother), hair_gene(father));
    }
    const auto eye_gene = [&](std::uint64_t parent) {
        const double light = unit(parent, 5) * (1.25 - sun);
        return light < 0.35 ? 0 : light < 0.5 ? 1 : light < 0.62 ? 2 : light < 0.75 ? 3 : light < 0.9 ? 4 : 5;
    };
    l.eyes = std::min(eye_gene(mother), eye_gene(father));
    l.face = unit(pid, 6) < 0.6 ? static_cast<int>(mix64(unit(pid, 7) < 0.5 ? mother : father, 9) % kFaceCount) : static_cast<int>(mix64(pid, 10) % kFaceCount);
    l.nose = static_cast<int>(mix64(unit(pid, 11) < 0.5 ? mother : father, 12) % 3);
    l.brows = unit(pid, 13) < (l.sex ? 0.6 : 0.2) ? 1 : 0;
    l.freckles = l.melanin < 0.35 && unit(pid, 14) < 0.2;
    l.size = 0.94 + 0.12 * 0.5 * (unit(mother, 15) + unit(father, 15)) + 0.03 * (unit(pid, 16) - 0.5) - (l.sex == 0 ? 0.03 : 0.0);
    // How a people wears its hair is theirs; how long is partly who you are.
    const std::uint64_t fashion = mix64(static_cast<std::uint64_t>(culture), 17);
    static const int kWomen[] = {5, 6, 7, 8, 3, 4, 5, 6};
    static const int kMen[] = {1, 2, 3, 4, 2, 1, 9, 0};
    const int fashion_pick = static_cast<int>(fashion % 8);
    const int own_pick = static_cast<int>(mix64(pid, 18) % 8);
    l.hair = l.sex == 0 ? kWomen[unit(pid, 19) < 0.5 ? fashion_pick : own_pick] : kMen[unit(pid, 19) < 0.5 ? fashion_pick : own_pick];
    if (l.sex == 1) {
        const double beards = unit(static_cast<std::uint64_t>(culture), 20);
        const double roll = unit(pid, 21);
        l.beard = roll < beards * 0.5 ? 3 : roll < beards * 0.8 ? 2 : roll < beards ? 1 : 0;
    }
    l.grey = std::max(0.0, unit(pid, 22) - 0.75) * 3.0;
    return l;
}

std::string for_sex(const std::string& text, int sex)
{
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    const char* mine = sex == 0 ? "f:" : "m:";
    const char* other = sex == 0 ? "m:" : "f:";
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line.compare(first, 2, other) == 0) {
            continue;
        }
        if (first != std::string::npos && line.compare(first, 2, mine) == 0) {
            line = line.substr(first + 2);
        }
        out << line << "\n";
    }
    return out.str();
}

std::string looks_text(const Looks& l)
{
    std::ostringstream o;
    o << "sex " << l.sex << "\nmelanin " << l.melanin << "\nhair " << l.hair << "\nhair_colour " << l.hair_colour << "\neyes " << l.eyes << "\nface " << l.face
      << "\nnose " << l.nose << "\nbeard " << l.beard << "\nbrows " << l.brows << "\nsize " << l.size << "\ngrey " << l.grey << "\nfreckles " << (l.freckles ? 1 : 0) << "\n";
    return o.str();
}

Looks parse_looks(const std::string& text)
{
    Looks l;
    std::istringstream in(text);
    std::string key;
    double value = 0.0;
    while (in >> key >> value) {
        if (key == "sex") l.sex = static_cast<int>(value);
        else if (key == "melanin") l.melanin = value;
        else if (key == "hair") l.hair = static_cast<int>(value);
        else if (key == "hair_colour") l.hair_colour = static_cast<int>(value);
        else if (key == "eyes") l.eyes = static_cast<int>(value);
        else if (key == "face") l.face = static_cast<int>(value);
        else if (key == "nose") l.nose = static_cast<int>(value);
        else if (key == "beard") l.beard = static_cast<int>(value);
        else if (key == "brows") l.brows = static_cast<int>(value);
        else if (key == "size") l.size = value;
        else if (key == "grey") l.grey = value;
        else if (key == "freckles") l.freckles = value != 0.0;
    }
    return l;
}

} // namespace fieldengine::field

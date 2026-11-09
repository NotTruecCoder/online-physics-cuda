#pragma once

#include <cstdint>
#include <string>
#include <vector>

// WHAT SOMEONE IS BORN WITH: skin, sex, a face. Nothing they put on. A body is worked out from these as the same text a figure
// file holds (VoxelFigure::parse), so it hangs on the same bones and any clip and any clothes fit. Clothes are files with
// lines for each shape: `f:` and `m:` at the head of a line (for_sex).
namespace fieldengine::field {

struct Looks {
    int sex = 1;          // 0 female, 1 male
    double melanin = 0.3; // 0 pale .. 1 very dark
    int hair = 2;         // kHairs
    int hair_colour = 1;  // kHairColours
    int eyes = 0;         // kEyes
    int face = 0;         // kFaces
    int nose = 0;         // 0 small 1 long 2 broad
    int beard = 0;        // 0 none 1 stubble 2 goatee 3 full
    int brows = 0;        // 0 thin 1 thick
    double size = 1.0;    // how tall, about 1
    double grey = 0.0;    // of the hair, with age
    bool freckles = false;
    bool covered = false; // a hat is on: only what shows under it
};

extern const char* const kHairs[];
extern const int kHairCount;
extern const char* const kHairColours[];
extern const int kHairColourCount;
extern const char* const kEyes[];
extern const int kEyeCount;
extern const char* const kFaces[];
extern const int kFaceCount;

// As figure text.
std::string body_model(const Looks& looks);
// What someone is like, given who their parents are and how far from the sun their people live.
Looks looks_of_person(std::uint64_t pid, std::uint64_t family, int culture, double temperature);
// A figure file for this body's shape.
std::string for_sex(const std::string& text, int sex);

std::string looks_text(const Looks& looks);
Looks parse_looks(const std::string& text);

} // namespace fieldengine::field

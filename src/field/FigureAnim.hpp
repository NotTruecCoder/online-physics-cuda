#pragma once

#include "field/VoxelFigure.hpp"

#include <map>
#include <string>
#include <vector>

// HOW A BODY MOVES, WITH NO BODY IN IT. A clip turns bones and knows nothing of what hangs on them, so any clip fits any
// skin. Read from assets/world/anim. Nothing in the simulation reads this.
namespace fieldengine::field {

struct FigureClip {
    struct Key {
        double t = 0.0;
        BonePose bone;
    };
    std::string name;
    bool cycle = false;
    std::array<std::vector<Key>, kBones> tracks;
    std::vector<std::pair<double, double>> lift; // t, in heights of the figure
};

class ClipBook {
public:
    // Errors name the line.
    static ClipBook parse(const std::string& text, std::vector<std::string>& errors);
    static ClipBook load(const std::string& path, std::vector<std::string>& errors);

    // More clips, from more text: a name already there is replaced.
    void add(const std::string& text, std::vector<std::string>& errors);
    [[nodiscard]] const FigureClip* find(const std::string& name) const;
    [[nodiscard]] std::size_t size() const { return clips_.size(); }

    // Plays a clip over a pose, `weight` of the way: bones it does not name stay as they are.
    void play(const std::string& name, double t, double weight, FigurePose& pose) const;

private:
    std::map<std::string, FigureClip> clips_;
    bool frames_ = false; // keys hold instead of easing into each other
    std::array<bool, kBones> still_{}; // bones nothing moves
};

} // namespace fieldengine::field

#include "world/FigureAnim.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

ClipBook ClipBook::parse(const std::string& text, std::vector<std::string>& errors)
{
    ClipBook book;
    FigureClip* clip = nullptr;
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
        const std::string where = "anim:" + std::to_string(number) + ": ";
        if (w[0] == "frames" && w.size() == 1) {
            book.frames_ = true;
            continue;
        }
        if (w[0] == "still") {
            for (std::size_t k = 1; k < w.size(); ++k) {
                const int bone = bone_of(w[k]);
                if (bone >= 0) {
                    book.still_[static_cast<std::size_t>(bone)] = true;
                }
            }
            continue;
        }
        if (w[0] == "clip" && (w.size() == 2 || (w.size() == 3 && w[2] == "cycle"))) {
            clip = &book.clips_[w[1]];
            clip->name = w[1];
            clip->cycle = w.size() == 3;
            continue;
        }
        if (w[0] != "key" || clip == nullptr || w.size() < 4 || w.size() % 2 != 0) {
            errors.push_back(where + "expected clip NAME [cycle], or key TIME and then pairs of a bone and its angles");
            continue;
        }
        try {
            const double t = std::stod(w[1]);
            for (std::size_t k = 2; k + 1 < w.size(); k += 2) {
                std::vector<double> values;
                std::istringstream parts(w[k + 1]);
                for (std::string part; std::getline(parts, part, ',');) {
                    values.push_back(std::stod(part));
                }
                if (w[k] == "lift") {
                    clip->lift.emplace_back(t, values.at(0));
                    continue;
                }
                const int bone = bone_of(w[k]);
                if (bone < 0) {
                    errors.push_back(where + "no such bone: " + w[k]);
                    continue;
                }
                clip->tracks[static_cast<std::size_t>(bone)].push_back(FigureClip::Key{t, BonePose{values.at(0), values.size() > 1 ? values[1] : 0.0, values.size() > 2 ? values[2] : 0.0}});
            }
        } catch (...) {
            errors.push_back(where + "a number was expected");
        }
    }
    return book;
}

ClipBook ClipBook::load(const std::string& path, std::vector<std::string>& errors)
{
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + path, std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            return parse(text.str(), errors);
        }
    }
    errors.push_back(path + " not found");
    return ClipBook{};
}

void ClipBook::add(const std::string& text, std::vector<std::string>& errors)
{
    ClipBook more = parse(text, errors);
    for (auto& [name, clip] : more.clips_) {
        clips_[name] = std::move(clip);
    }
    frames_ = frames_ || more.frames_;
    for (std::size_t b = 0; b < still_.size(); ++b) {
        still_[b] = still_[b] || more.still_[b];
    }
}

const FigureClip* ClipBook::find(const std::string& name) const
{
    const auto found = clips_.find(name);
    return found == clips_.end() ? nullptr : &found->second;
}

void ClipBook::play(const std::string& name, double t, double weight, FigurePose& pose) const
{
    const FigureClip* clip = find(name);
    if (clip == nullptr || weight <= 0.0) {
        return;
    }
    const double at = clip->cycle ? t - std::floor(t) : t;
    const auto ease = [](double x) { return x * x * (3.0 - 2.0 * x); };
    // Between the keys either side, easing in and out.
    const auto between = [&](const auto& keys, const auto& time_of, const auto& blend) {
        if (keys.empty()) {
            return;
        }
        std::size_t next = 0;
        while (next < keys.size() && time_of(keys[next]) <= at) {
            ++next;
        }
        if (next == 0) {
            blend(keys.front(), keys.front(), 0.0);
        } else if (next == keys.size()) {
            blend(keys.back(), keys.back(), 0.0);
        } else {
            const auto& a = keys[next - 1];
            const auto& b = keys[next];
            // Frames hold until the next one comes; otherwise it eases from one to the next.
            blend(a, b, frames_ ? 0.0 : ease((at - time_of(a)) / std::max(time_of(b) - time_of(a), 1e-6)));
        }
    };
    for (int b = 0; b < kBones; ++b) {
        if (still_[static_cast<std::size_t>(b)]) {
            continue;
        }
        BonePose& out = pose.bones[static_cast<std::size_t>(b)];
        between(clip->tracks[static_cast<std::size_t>(b)], [](const FigureClip::Key& k) { return k.t; },
            [&](const FigureClip::Key& x, const FigureClip::Key& y, double f) {
                out.swing += ((x.bone.swing + (y.bone.swing - x.bone.swing) * f) - out.swing) * weight;
                out.out += ((x.bone.out + (y.bone.out - x.bone.out) * f) - out.out) * weight;
                out.twist += ((x.bone.twist + (y.bone.twist - x.bone.twist) * f) - out.twist) * weight;
            });
    }
    between(clip->lift, [](const std::pair<double, double>& k) { return k.first; }, [&](const std::pair<double, double>& x, const std::pair<double, double>& y, double f) {
        pose.lift += ((x.second + (y.second - x.second) * f) - pose.lift) * weight;
    });
}

} // namespace fieldengine::field

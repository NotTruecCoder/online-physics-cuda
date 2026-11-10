#include "world/VoxelFigure.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace fieldengine::field {

namespace {

constexpr int kNormals[6][3] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
// The corners of each side, on a unit cube.
constexpr int kCorners[6][4][3] = {
    {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
    {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {0, 0, 0}},
    {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}},
    {{0, 1, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 1}},
    {{1, 1, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}},
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
};

// Each cube its own shade, as in the world.
float jitter(int x, int y, int z)
{
    std::uint32_t h = static_cast<std::uint32_t>(x + 64) * 73856093u ^ static_cast<std::uint32_t>(y + 64) * 19349663u ^ static_cast<std::uint32_t>(z + 64) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return 0.90f + 0.20f * static_cast<float>(h & 1023u) / 1023.0f;
}

std::uint32_t abgr_of(float r, float g, float b)
{
    const auto channel = [](float v) { return static_cast<std::uint32_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
    return 0xff000000u | (channel(b) << 16) | (channel(g) << 8) | channel(r);
}

} // namespace

int bone_of(const std::string& name)
{
    static const char* const kNames[kBones] = {"root", "torso", "head", "cape", "arm_l", "arm_r", "leg_l", "leg_r", "grip"};
    for (int b = 0; b < kBones; ++b) {
        if (name == kNames[b]) {
            return b;
        }
    }
    return -1;
}

VoxelFigure VoxelFigure::parse(const std::string& text, std::vector<std::string>& errors)
{
    VoxelFigure figure;
    struct Colour {
        float rgb[3];
        bool glow;
    };
    std::map<std::string, Colour> colours;
    struct Voxel {
        int part;
        int x, y, z;
    };
    std::map<std::tuple<int, int, int, int>, std::string> cubes; // part, x, y, z -> colour
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
        const std::string where = "player.model:" + std::to_string(number) + ": ";
        try {
            if (w[0] == "voxel" && w.size() == 2) {
                figure.voxel_ = std::stod(w[1]);
            } else if (w[0] == "colour" && w.size() == 5) {
                colours[w[1]] = Colour{{std::stof(w[2]) / 255.0f, std::stof(w[3]) / 255.0f, std::stof(w[4]) / 255.0f}, w[1] == "glow"};
            } else if (w[0] == "part" && w.size() == 6 && w[2] == "pivot") {
                Part part;
                part.name = w[1];
                // A thing that is held hangs on nothing: it goes where its socket says.
                part.bone = w[1] == "item" ? kRoot : bone_of(w[1]);
                if (part.bone < 0 || (part.bone == kRoot && w[1] != "item")) {
                    errors.push_back(where + "no such bone: " + w[1] + " (torso head cape arm_l arm_r leg_l leg_r, or item)");
                    part.bone = kRoot;
                }
                for (int k = 0; k < 3; ++k) {
                    part.pivot[k] = std::stoi(w[3 + static_cast<std::size_t>(k)]);
                    figure.pivots_[part.bone][k] = part.pivot[k];
                }
                figure.parts_.push_back(std::move(part));
            } else if (w[0] == "socket" && w.size() == 5) {
                if (figure.parts_.empty()) {
                    errors.push_back(where + "a socket before any part");
                    continue;
                }
                Socket socket;
                socket.name = w[1];
                socket.bone = figure.parts_.back().bone;
                for (int k = 0; k < 3; ++k) {
                    socket.at[k] = std::stod(w[2 + static_cast<std::size_t>(k)]);
                }
                figure.sockets_.push_back(std::move(socket));
            } else if ((w[0] == "box" && w.size() == 8) || (w[0] == "dot" && w.size() == 5)) {
                if (figure.parts_.empty()) {
                    errors.push_back(where + "a cube before any part");
                    continue;
                }
                const bool dot = w[0] == "dot";
                const std::string& colour = dot ? w[4] : w[7];
                if (colours.find(colour) == colours.end()) {
                    errors.push_back(where + "no colour '" + colour + "'");
                    continue;
                }
                int lo[3];
                int hi[3];
                for (int k = 0; k < 3; ++k) {
                    lo[k] = std::stoi(w[1 + static_cast<std::size_t>(k)]);
                    hi[k] = dot ? lo[k] : std::stoi(w[4 + static_cast<std::size_t>(k)]);
                    if (hi[k] < lo[k] || lo[k] < -60 || hi[k] > 60) {
                        errors.push_back(where + "a box inside out, or too far from the middle");
                        hi[k] = lo[k] - 1;
                    }
                }
                const int part = static_cast<int>(figure.parts_.size()) - 1;
                for (int z = lo[2]; z <= hi[2]; ++z) {
                    for (int y = lo[1]; y <= hi[1]; ++y) {
                        for (int x = lo[0]; x <= hi[0]; ++x) {
                            cubes[{part, x, y, z}] = colour; // the later line paints over the earlier one
                        }
                    }
                }
            } else {
                errors.push_back(where + "expected voxel, colour, part, socket, box or dot");
            }
        } catch (...) {
            errors.push_back(where + "a number was expected");
        }
    }
    // Only what looks out of its part.
    figure.voxels_ = cubes.size();
    for (const auto& [key, colour_name] : cubes) {
        const auto [part, x, y, z] = key;
        const Colour& colour = colours[colour_name];
        const float own = colour.glow ? 1.0f : jitter(x, y, z);
        for (int side = 0; side < 6; ++side) {
            const int* n = kNormals[side];
            if (cubes.find({part, x + n[0], y + n[1], z + n[2]}) != cubes.end()) {
                continue;
            }
            figure.parts_[static_cast<std::size_t>(part)].faces.push_back(Face{static_cast<std::int8_t>(x), static_cast<std::int8_t>(y), static_cast<std::int8_t>(z),
                static_cast<std::uint8_t>(side), {colour.rgb[0] * own, colour.rgb[1] * own, colour.rgb[2] * own}, colour.glow});
            ++figure.faces_;
        }
    }
    return figure;
}

VoxelFigure VoxelFigure::load(const std::string& path, std::vector<std::string>& errors)
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
    return VoxelFigure{};
}

bool VoxelFigure::has_part(const std::string& name) const
{
    return std::any_of(parts_.begin(), parts_.end(), [&](const Part& p) { return p.name == name; });
}

double VoxelFigure::height() const
{
    int top = -1000;
    for (const Part& part : parts_) {
        for (const Face& face : part.faces) {
            top = std::max(top, static_cast<int>(face.z) + 1);
        }
    }
    return top * voxel_;
}

double VoxelFigure::lowest() const
{
    int low = 1000;
    for (const Part& part : parts_) {
        for (const Face& face : part.faces) {
            low = std::min(low, static_cast<int>(face.z));
        }
    }
    return low * voxel_;
}

double VoxelFigure::width() const
{
    int lo = 1000;
    int hi = -1000;
    for (const Part& part : parts_) {
        for (const Face& face : part.faces) {
            lo = std::min(lo, static_cast<int>(face.y));
            hi = std::max(hi, static_cast<int>(face.y) + 1);
        }
    }
    return (hi - lo) * voxel_;
}

namespace {

using M3 = std::array<std::array<double, 3>, 3>;

M3 times(const M3& a, const M3& b)
{
    M3 r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] * b[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
            }
        }
    }
    return r;
}

// About the axis that runs side to side: a limb hanging down (-z) goes toward +x.
M3 swing_by(double v)
{
    const double c = std::cos(v);
    const double s = std::sin(v);
    return M3{{{c, 0.0, -s}, {0.0, 1.0, 0.0}, {s, 0.0, c}}};
}

// About the axis that runs front to back: a limb hanging down goes toward +y.
M3 out_by(double v)
{
    const double c = std::cos(v);
    const double s = std::sin(v);
    return M3{{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}};
}

// About the upright: to its left.
M3 twist_by(double v)
{
    const double c = std::cos(v);
    const double s = std::sin(v);
    return M3{{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
}

} // namespace

// The whole leans and turns about the feet; arms, head and cape go with the torso, legs do not.
void VoxelFigure::frames(const FigurePose& pose, std::array<Frame, kBones>& out) const
{
    const auto own = [&](int b) {
        const BonePose& p = pose.bones[static_cast<std::size_t>(b)];
        const double side = b == kArmR || b == kLegR ? -1.0 : 1.0;
        return times(times(swing_by(p.swing), out_by(p.out * side)), twist_by(p.twist));
    };
    const BonePose& root = pose.bones[kRoot];
    // Squat: shorter and, to keep its volume, wider. All of it, in metres.
    const double spread = voxel_ * pose.size / std::sqrt(std::max(pose.squash, 0.2));
    const M3 scale{{{spread, 0.0, 0.0}, {0.0, spread, 0.0}, {0.0, 0.0, voxel_ * pose.size * pose.squash}}};
    // Forward from the feet is a swing the other way round; over to its left is the out of a limb the other way round.
    const M3 whole = times(twist_by(pose.yaw + root.twist), times(out_by(-root.out), swing_by(-root.swing)));
    const M3 world = times(whole, scale);
    const double lift = pose.lift * height() / voxel_;
    const M3 torso = own(kTorso);
    const auto pivot_of = [&](int b, int k) { return static_cast<double>(pivots_[b][k]); };
    for (int b = 0; b < kBones; ++b) {
        // p -> R (p - pivot) + pivot, and then the torso the same way if it hangs on the torso.
        M3 rot = b == kRoot ? M3{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}} : own(b);
        double off[3] = {0.0, 0.0, 0.0};
        if (b != kRoot) {
            for (int i = 0; i < 3; ++i) {
                off[i] = pivot_of(b, i);
                for (int j = 0; j < 3; ++j) {
                    off[i] -= rot[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * pivot_of(b, j);
                }
            }
        }
        if (b == kHead || b == kCape || b == kArmL || b == kArmR) {
            double moved[3];
            for (int i = 0; i < 3; ++i) {
                moved[i] = pivot_of(kTorso, i);
                for (int j = 0; j < 3; ++j) {
                    moved[i] += torso[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * (off[j] - pivot_of(kTorso, j));
                }
            }
            off[0] = moved[0];
            off[1] = moved[1];
            off[2] = moved[2];
            rot = times(torso, rot);
        }
        if (b != kLegL && b != kLegR) {
            off[2] += lift;
        }
        const M3 all = times(world, rot);
        Frame& f = out[static_cast<std::size_t>(b)];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                f.a[i][j] = all[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
        }
        f.t[0] = pose.feet.x + world[0][0] * off[0] + world[0][1] * off[1] + world[0][2] * off[2];
        f.t[1] = pose.feet.y + world[1][0] * off[0] + world[1][1] * off[1] + world[1][2] * off[2];
        f.t[2] = pose.feet.z + world[2][0] * off[0] + world[2][1] * off[1] + world[2][2] * off[2];
    }
}

void VoxelFigure::emit(const Face& face, const FigurePose& pose, const Frame& frame, double scale, double x0, double y0, double z0, float tinted, std::vector<MeshVertex>& out, double grip)
{
    const int* n = kNormals[face.side];
    Vec3 corner[4];
    const double gc = std::cos(grip);
    const double gs = std::sin(grip);
    const auto at = [&](double x, double y, double z) { return frame.of(x0 + (gc * x - gs * z) * scale, y0 + y * scale, z0 + (gs * x + gc * z) * scale); };
    for (int k = 0; k < 4; ++k) {
        const int* c = kCorners[face.side][k];
        corner[k] = at(face.x + c[0], face.y + c[1], face.z + c[2]);
    }
    // The side's light from where it now looks, not where it looked at rest.
    const Vec3 normal = normalize(at(face.x + 0.5 + n[0], face.y + 0.5 + n[1], face.z + 0.5 + n[2]) - at(face.x + 0.5, face.y + 0.5, face.z + 0.5));
    const double light = std::max(0.0, normal.z) + 0.30 * std::max(0.0, 0.8 * normal.x + 0.6 * normal.y) + 0.12 * std::max(0.0, -normal.x);
    const float shade = face.glow ? 1.0f : static_cast<float>(std::clamp(0.46 + 0.54 * light, 0.42, 1.0));
    float r = (face.glow ? pose.glow[0] : face.colour[0]) * shade;
    float g = (face.glow ? pose.glow[1] : face.colour[1]) * shade;
    float b = (face.glow ? pose.glow[2] : face.colour[2]) * shade;
    r += (pose.held_tint[0] - r) * tinted;
    g += (pose.held_tint[1] - g) * tinted;
    b += (pose.held_tint[2] - b) * tinted;
    r += (1.0f - r) * pose.hurt;
    g -= g * pose.hurt * 0.8f;
    b -= b * pose.hurt * 0.8f;
    const std::uint32_t abgr = abgr_of(r, g, b);
    const auto vertex = [&](const Vec3& p) {
        out.push_back(MeshVertex{static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z), abgr});
    };
    vertex(corner[0]);
    vertex(corner[1]);
    vertex(corner[2]);
    vertex(corner[0]);
    vertex(corner[2]);
    vertex(corner[3]);
}

void VoxelFigure::mesh(const FigurePose& pose, std::vector<MeshVertex>& out, bool no_arms) const
{
    out.reserve(out.size() + faces_ * 6);
    std::array<Frame, kBones> bones;
    frames(pose, bones);
    for (const Part& part : parts_) {
        if (no_arms && (part.bone == kArmL || part.bone == kArmR)) {
            continue;
        }
        for (const Face& face : part.faces) {
            emit(face, pose, bones[static_cast<std::size_t>(part.bone)], 1.0, 0.0, 0.0, 0.0, 0.0f, out);
        }
    }
}

bool VoxelFigure::mesh_held(const FigurePose& pose, const VoxelFigure& item, const std::string& socket, std::vector<MeshVertex>& out) const
{
    const auto found = std::find_if(sockets_.begin(), sockets_.end(), [&](const Socket& s) { return s.name == socket; });
    if (found == sockets_.end()) {
        return false;
    }
    std::array<Frame, kBones> bones;
    frames(pose, bones);
    const Frame& frame = bones[static_cast<std::size_t>(found->bone)];
    const double scale = item.voxel_ / voxel_;
    for (const Part& part : item.parts_) {
        for (const Face& face : part.faces) {
            emit(face, pose, frame, scale, found->at[0], found->at[1], found->at[2], pose.held_tinted, out, pose.bones[kGrip].swing);
        }
    }
    return true;
}

bool VoxelFigure::held_ends(const FigurePose& pose, const VoxelFigure& item, const std::string& socket, Vec3& grip, Vec3& tip) const
{
    const auto found = std::find_if(sockets_.begin(), sockets_.end(), [&](const Socket& s) { return s.name == socket; });
    if (found == sockets_.end()) {
        return false;
    }
    std::array<Frame, kBones> bones;
    frames(pose, bones);
    const Frame& frame = bones[static_cast<std::size_t>(found->bone)];
    int far_end = 8;
    for (const Part& part : item.parts_) {
        for (const Face& face : part.faces) {
            far_end = std::max(far_end, static_cast<int>(face.x));
        }
    }
    const double scale = item.voxel_ / voxel_;
    grip = frame.of(found->at[0], found->at[1], found->at[2]);
    tip = frame.of(found->at[0] + far_end * scale * std::cos(pose.bones[kGrip].swing), found->at[1], found->at[2] + far_end * scale * std::sin(pose.bones[kGrip].swing));
    return true;
}

bool VoxelFigure::socket_at(const FigurePose& pose, const std::string& socket, Vec3& where) const
{
    const auto found = std::find_if(sockets_.begin(), sockets_.end(), [&](const Socket& s) { return s.name == socket; });
    if (found == sockets_.end()) {
        return false;
    }
    std::array<Frame, kBones> bones;
    frames(pose, bones);
    where = bones[static_cast<std::size_t>(found->bone)].of(found->at[0], found->at[1], found->at[2]);
    return true;
}

void VoxelFigure::sample(const FigurePose& pose, int every, std::vector<Bit>& out) const
{
    int k = 0;
    std::array<Frame, kBones> bones;
    frames(pose, bones);
    for (const Part& part : parts_) {
        for (const Face& face : part.faces) {
            if (++k % every != 0) {
                continue;
            }
            const std::array<float, 3> rgb = face.glow ? pose.glow : std::array<float, 3>{face.colour[0], face.colour[1], face.colour[2]};
            out.push_back(Bit{bones[static_cast<std::size_t>(part.bone)].of(face.x + 0.5, face.y + 0.5, face.z + 0.5), abgr_of(rgb[0], rgb[1], rgb[2])});
        }
    }
}

} // namespace fieldengine::field

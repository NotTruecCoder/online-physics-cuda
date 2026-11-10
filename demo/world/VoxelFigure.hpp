#pragma once

#include "world/Body.hpp"
#include "world/VoxelMesh.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// A FIGURE MADE OF VOXELS. Read from text (assets/world/skins): cubes hanging on the bones of a skeleton that is the same for
// everyone. How the bones move is not here (FigureAnim.hpp), and knows nothing of the cubes, so a skin can be changed with
// no other change. Only the faces that look out of a part are kept, so a figure of a thousand cubes is a few thousand
// triangles. Everything here is picture: nothing in the simulation reads it.
namespace fieldengine::field {

// `grip` hangs nothing of a body: it is how what is held turns in the hand, so a hand can be forward and a spear level.
enum Bone { kRoot, kTorso, kHead, kCape, kArmL, kArmR, kLegL, kLegR, kGrip, kBones };

// By name; -1 if there is no such bone.
int bone_of(const std::string& name);

// Radians. Swing > 0 is forward, out is away from the body, twist turns it to its left.
struct BonePose {
    double swing = 0.0;
    double out = 0.0;
    double twist = 0.0;
};

// Where a figure is and how it holds itself.
struct FigurePose {
    Vec3 feet{};       // world metres
    double yaw = 0.0;  // where +x of the figure points
    double squash = 1.0; // of its height (landing), and it spreads to keep its volume
    double size = 1.0;   // how tall this one is
    double lift = 0.0;   // of everything above the legs, in heights of the figure
    std::array<BonePose, kBones> bones{};
    std::array<float, 3> glow{1.0f, 1.0f, 1.0f};
    float hurt = 0.0f;  // 0..1: how far toward red
    std::array<float, 3> held_tint{1.0f, 1.0f, 1.0f}; // what is held, toward this colour
    float held_tinted = 0.0f;
};

class VoxelFigure {
public:
    // Errors name the line. An empty figure draws nothing.
    static VoxelFigure parse(const std::string& text, std::vector<std::string>& errors);
    // From the first root that has it.
    static VoxelFigure load(const std::string& path, std::vector<std::string>& errors);

    [[nodiscard]] bool empty() const { return parts_.empty(); }
    [[nodiscard]] double voxel() const { return voxel_; }
    [[nodiscard]] std::size_t voxels() const { return voxels_; }
    [[nodiscard]] std::size_t faces() const { return faces_; }
    [[nodiscard]] bool has_part(const std::string& name) const;
    // In the figure's own frame, at rest: metres.
    [[nodiscard]] double height() const;
    [[nodiscard]] double lowest() const;
    [[nodiscard]] double width() const; // across, side to side

    // Three vertices a triangle, in metres, appended to `out`.
    // `no_arms`: what hangs on the arm bones stays out of it (cloth, on a figure whose hands float).
    void mesh(const FigurePose& pose, std::vector<MeshVertex>& out, bool no_arms = false) const;
    // Something held: another figure, its origin at this socket, turning with the bone the socket is on. False if there is none.
    bool mesh_held(const FigurePose& pose, const VoxelFigure& item, const std::string& socket, std::vector<MeshVertex>& out) const;
    // Where the two ends of it are now: its grip, and as far as it goes.
    bool held_ends(const FigurePose& pose, const VoxelFigure& item, const std::string& socket, Vec3& grip, Vec3& tip) const;
    // Where a socket is now.
    [[nodiscard]] bool socket_at(const FigurePose& pose, const std::string& socket, Vec3& where) const;
    // A scatter of its cubes as they stand now: where each is and its colour, for what comes off it.
    struct Bit {
        Vec3 at{};
        std::uint32_t abgr = 0;
    };
    void sample(const FigurePose& pose, int every, std::vector<Bit>& out) const;

private:
    struct Face {
        std::int8_t x, y, z; // the voxel
        std::uint8_t side;   // +z -z +x -x +y -y
        float colour[3];
        bool glow;
    };
    struct Part {
        std::string name;
        int bone = kRoot;
        int pivot[3] = {0, 0, 0};
        std::vector<Face> faces;
    };
    struct Socket {
        std::string name;
        int bone = kRoot;
        double at[3] = {0.0, 0.0, 0.0};
    };
    // From a bone's own voxels to the world, in metres.
    struct Frame {
        double a[3][3];
        double t[3];
        [[nodiscard]] Vec3 of(double x, double y, double z) const
        {
            return Vec3{a[0][0] * x + a[0][1] * y + a[0][2] * z + t[0], a[1][0] * x + a[1][1] * y + a[1][2] * z + t[1], a[2][0] * x + a[2][1] * y + a[2][2] * z + t[2]};
        }
    };
    void frames(const FigurePose& pose, std::array<Frame, kBones>& out) const;
    static void emit(const Face& face, const FigurePose& pose, const Frame& frame, double scale, double x0, double y0, double z0, float tinted, std::vector<MeshVertex>& out, double grip = 0.0);

    double voxel_ = 0.07;
    std::size_t voxels_ = 0;
    std::size_t faces_ = 0;
    std::vector<Part> parts_;
    std::vector<Socket> sockets_;
    int pivots_[kBones][3] = {};
};

} // namespace fieldengine::field

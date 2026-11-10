#pragma once

#include "field/Grid.hpp"
#include "world/Terrain.hpp"

#include <array>
#include <cstdint>
#include <vector>

// THE PICTURE OF THE VOXELS, AS CUBES. Every voxel the law says is there becomes a cube; only faces
// that look at something not solid are emitted. A cube is a cube you can count: each has its own
// shade (a fixed hash of where it is), its corners darken where other cubes crowd them (ambient
// occlusion), its top shows the turf the law grew on it. Colours come from the same fields the law runs
// on (material, rubble, ice, wood, leaves, lava, water, turf, temperature), so a wall going red before
// it slumps is the law being drawn, not an effect.
//
// Two steps: classify (fields -> one small VoxelCell per voxel, from the live lattice or a stored /
// generated chunk alike) and mesh (cells -> indexed quads). A zoomed-out view is meshed from a coarser box
// (downsample): see WorldMesher.hpp.
namespace fieldengine::field {

inline constexpr float kGlowsFrom = 700.0f; // kelvin: cooler than this looks the same

struct MeshVertex {
    float x = 0.0f; // metres (world coordinates when meshed from a chunk)
    float y = 0.0f;
    float z = 0.0f;
    std::uint32_t abgr = 0xffffffffu;
};

struct VoxelPalette {
    // By `material` index (Terrain.hpp: granite, basalt, slate, sandstone, soil, sand, clay, chalk, gravel, plaster,
    // brick, thatch, tile, timber, char, reed). Wood and leaves take theirs from 8 on, so a roof burns like a tree.
    std::vector<std::array<float, 3>> stone{{0.55f, 0.53f, 0.50f}, {0.36f, 0.35f, 0.37f}, {0.47f, 0.50f, 0.56f},
        {0.66f, 0.52f, 0.38f}, {0.45f, 0.33f, 0.22f}, {0.86f, 0.78f, 0.55f}, {0.55f, 0.45f, 0.40f},
        {0.88f, 0.88f, 0.84f}, {0.58f, 0.54f, 0.47f}, {0.86f, 0.82f, 0.72f}, {0.62f, 0.30f, 0.22f},
        {0.72f, 0.60f, 0.30f}, {0.60f, 0.27f, 0.17f}, {0.48f, 0.33f, 0.20f}, {0.13f, 0.12f, 0.12f},
        {0.60f, 0.56f, 0.36f},
        // barks: birch, dark, red, palm, and two more
        {0.84f, 0.82f, 0.74f}, {0.24f, 0.17f, 0.12f}, {0.50f, 0.26f, 0.16f}, {0.56f, 0.47f, 0.33f},
        // leaves: deep, bright, yellow-green, orange, red, pink, blue-green, silver
        {0.13f, 0.32f, 0.15f}, {0.27f, 0.47f, 0.18f}, {0.42f, 0.50f, 0.20f}, {0.52f, 0.40f, 0.17f},
        {0.46f, 0.26f, 0.18f}, {0.70f, 0.54f, 0.58f}, {0.17f, 0.36f, 0.27f}, {0.42f, 0.50f, 0.40f},
        // flowers: yellow, white, purple, scarlet
        {0.72f, 0.64f, 0.26f}, {0.74f, 0.74f, 0.66f}, {0.46f, 0.34f, 0.56f}, {0.58f, 0.28f, 0.22f}};
    std::array<float, 3> rubble{0.42f, 0.36f, 0.30f};
    std::array<float, 3> ice{0.70f, 0.86f, 0.95f};
    std::array<float, 3> water{0.18f, 0.42f, 0.75f};
    std::array<float, 3> lava{1.00f, 0.42f, 0.08f};
    std::array<float, 3> wood{0.40f, 0.27f, 0.15f};
    std::array<float, 3> leaf{0.22f, 0.50f, 0.20f};
    std::array<float, 3> grass{0.33f, 0.62f, 0.22f};
};

// One voxel as the mesher sees it.
struct VoxelCell {
    enum Kind : std::uint8_t { Air = 0, Opaque = 1, Lava = 2, Water = 3 };
    std::uint8_t kind = Air;
    std::uint8_t fill = 0;     // liquid level, 0..255
    std::uint8_t trunk = 0;    // wood: thin, but kept when the box is coarsened
    std::uint8_t tree = 0;     // leaves or a tree's wood: steps aside for the view
    std::uint8_t r = 0, g = 0, b = 0;    // sides
    std::uint8_t tr = 0, tg = 0, tb = 0; // top (turf shows here)
};

// A box of cells, x fastest. (wx, wy, wz) is the world cell (at this box's resolution) of cell (0,0,0).
// Below the box is rock, above it air; beyond its x/y sides, air.
struct VoxelBox {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int wx = 0;
    int wy = 0;
    int wz = 0;
    std::vector<VoxelCell> cells;

    void resize(int x, int y, int z);
    [[nodiscard]] std::size_t index(int x, int y, int z) const
    {
        return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(x);
    }
    [[nodiscard]] VoxelCell& at(int x, int y, int z) { return cells[index(x, y, z)]; }
    [[nodiscard]] const VoxelCell& at(int x, int y, int z) const { return cells[index(x, y, z)]; }
};

// The matter fields of some storage — the live lattice, or one chunk's arrays — addressed through
// strides. Absent fields are null (a law without water draws no water).
struct MatterView {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    std::size_t sy = 0; // index strides
    std::size_t sz = 0;
    const float* solid = nullptr;
    const float* rubble = nullptr;
    const float* ice = nullptr;
    const float* wood = nullptr;
    const float* leaf = nullptr;
    const float* lava = nullptr;
    const float* water = nullptr;
    const float* grass = nullptr;
    const float* material = nullptr;
    const float* temperature = nullptr;

    static MatterView of(const FieldSet& fields);
    static MatterView of(const ChunkData& chunk, int side, int nz);
    // The same storage, seen from cell (x0, y0, z0) on.
    [[nodiscard]] MatterView offset(int x0, int y0, int z0 = 0) const;
};

VoxelCell classify(const MatterView& view, int x, int y, int z, const VoxelPalette& palette);

struct ChunkMesh {
    std::vector<MeshVertex> vertices;
    std::vector<std::uint32_t> indices; // 6 per face
    std::size_t faces = 0;
    void clear()
    {
        vertices.clear();
        indices.clear();
        faces = 0;
    }
};

// Meshes the cells of `box` that are `border` cells in from its x/y sides (all of z). Border cells only
// hide faces and darken corners. `cell` is the box's voxel edge in metres.
//
// `skirt` > 0: this mesh meets a neighbour drawn at another resolution, whose surface does not line up
// with what the border cells say, so a side face on the mesh's edge that lies within `skirt` cells of
// open air is kept even when a border cell would hide it — otherwise the seam is a slit you see through.
//
// `cell_z` > 0: voxels of another height than width. The camera looks down, so no face looks down.
void mesh_box(const VoxelBox& box, int border, double cell, ChunkMesh& out, int skirt = 0, double cell_z = 0.0);

// `fine` at 1/factor resolution: a coarse voxel is matter where 3/8 of it is (a trunk counts four
// times, so trees keep their trunks), liquid where a quarter is (filled as deep as the liquid in it),
// its colour the mean of what it holds. `factor_z` > 0: coarsened less in height than in width.
void downsample(const VoxelBox& fine, int factor, VoxelBox& coarse, int factor_z = 0);

struct VoxelMeshStats {
    std::size_t solid_voxels = 0;
    std::size_t liquid_voxels = 0;
    std::size_t faces = 0;
};

// The whole lattice as a plain triangle list (6 vertices per face), lattice coordinates.
VoxelMeshStats build_voxel_mesh(const FieldSet& fields, std::vector<MeshVertex>& out,
    const VoxelPalette& palette = VoxelPalette{});

} // namespace fieldengine::field

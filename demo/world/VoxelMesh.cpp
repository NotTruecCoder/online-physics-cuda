#include "world/VoxelMesh.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::field {

namespace {

std::uint8_t to_byte(float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

std::uint32_t pack(float r, float g, float b)
{
    return (0xffu << 24) | (static_cast<std::uint32_t>(to_byte(b)) << 16) |
        (static_cast<std::uint32_t>(to_byte(g)) << 8) | static_cast<std::uint32_t>(to_byte(r));
}

// Six faces: outward normal, how much light it takes, and its four corners (unit cube),
// counter-clockwise seen from outside.
struct Face {
    int n[3];
    float shade;
    int corners[4][3];
};

constexpr Face kFaces[6] = {
    {{0, 0, 1}, 1.00f, {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}},  // top
    {{0, 0, -1}, 0.45f, {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {0, 0, 0}}}, // bottom
    {{1, 0, 0}, 0.80f, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}},  // +x
    {{-1, 0, 0}, 0.62f, {{0, 1, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 1}}}, // -x
    {{0, 1, 0}, 0.72f, {{1, 1, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}}},  // +y
    {{0, -1, 0}, 0.56f, {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}}, // -y
};

// Each voxel's own shade: a fixed hash of where it is, so the same stone looks the same every frame
// and two neighbouring stones never look the same.
float jitter(int x, int y, int z)
{
    std::uint32_t h = static_cast<std::uint32_t>(x) * 73856093u ^ static_cast<std::uint32_t>(y) * 19349663u ^
        static_cast<std::uint32_t>(z) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return 0.92f + 0.16f * static_cast<float>(h & 1023u) / 1023.0f;
}

constexpr float kOcclusion[4] = {1.0f, 0.80f, 0.66f, 0.52f};

} // namespace

void VoxelBox::resize(int x, int y, int z)
{
    nx = x;
    ny = y;
    nz = z;
    cells.assign(static_cast<std::size_t>(x) * static_cast<std::size_t>(y) * static_cast<std::size_t>(z), VoxelCell{});
}

MatterView MatterView::of(const FieldSet& fields)
{
    const GridShape& g = fields.shape();
    MatterView v;
    v.nx = g.nx;
    v.ny = g.ny;
    v.nz = g.nz;
    v.sy = static_cast<std::size_t>(g.nx);
    v.sz = static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny);
    auto ptr = [&](const char* name) -> const float* {
        const int id = fields.find(name);
        return id < 0 ? nullptr : fields.values(id).data();
    };
    v.solid = ptr("solid");
    v.rubble = ptr("rubble");
    v.ice = ptr("ice");
    v.wood = ptr("wood");
    v.leaf = ptr("leaf");
    v.lava = ptr("lava");
    v.water = ptr("water");
    v.grass = ptr("grass");
    v.material = ptr("material");
    v.temperature = ptr("T");
    return v;
}

MatterView MatterView::of(const ChunkData& chunk, int side, int nz)
{
    MatterView v;
    v.nx = side;
    v.ny = side;
    v.nz = nz;
    v.sy = static_cast<std::size_t>(side);
    v.sz = static_cast<std::size_t>(side) * static_cast<std::size_t>(side);
    auto ptr = [&](const char* name) -> const float* {
        const std::vector<float>* values = chunk.find(name);
        return values == nullptr ? nullptr : values->data();
    };
    v.solid = ptr("solid");
    v.rubble = ptr("rubble");
    v.ice = ptr("ice");
    v.wood = ptr("wood");
    v.leaf = ptr("leaf");
    v.lava = ptr("lava");
    v.water = ptr("water");
    v.grass = ptr("grass");
    v.material = ptr("material");
    v.temperature = ptr("T");
    return v;
}

MatterView MatterView::offset(int x0, int y0, int z0) const
{
    MatterView v = *this;
    const std::size_t shift = static_cast<std::size_t>(z0) * sz + static_cast<std::size_t>(y0) * sy + static_cast<std::size_t>(x0);
    for (const float** p : {&v.solid, &v.rubble, &v.ice, &v.wood, &v.leaf, &v.lava, &v.water, &v.grass, &v.material,
             &v.temperature}) {
        if (*p != nullptr) {
            *p += shift;
        }
    }
    v.nx = nx - x0;
    v.ny = ny - y0;
    v.nz = nz - z0;
    return v;
}

VoxelCell classify(const MatterView& view, int x, int y, int z, const VoxelPalette& palette)
{
    const std::size_t i = static_cast<std::size_t>(z) * view.sz + static_cast<std::size_t>(y) * view.sy +
        static_cast<std::size_t>(x);
    auto value = [&](const float* field) { return field == nullptr ? 0.0f : field[i]; };
    VoxelCell cell;
    const float s = value(view.solid);
    const float r = value(view.rubble);
    const float ice = value(view.ice);
    const float wood = value(view.wood);
    const float leaf = value(view.leaf);
    std::array<float, 3> side{};
    std::array<float, 3> top{};
    if (s + r + ice + wood + leaf >= 0.5f) {
        cell.kind = VoxelCell::Opaque;
        const float most = std::max({s, r, ice, wood, leaf});
        const long built = view.material == nullptr ? 0 : std::lround(value(view.material));
        const auto stone_of = [&](long m) { return palette.stone[static_cast<std::size_t>(std::clamp<long>(m, 0, static_cast<long>(palette.stone.size()) - 1))]; };
        if (most == ice) {
            side = palette.ice;
        } else if (most == wood) {
            side = built >= 8 ? stone_of(built) : palette.wood;
            cell.trunk = built >= 8 && built < 16 ? 0 : 1; // walls are of 8 to 15; barks from 16
            cell.tree = cell.trunk;
        } else if (most == leaf) {
            side = built >= 8 ? stone_of(built) : palette.leaf;
            cell.tree = built >= 8 && built < 16 ? 0 : 1; // a thatch roof is not a tree
        } else if (most == r) {
            side = palette.rubble;
        } else {
            side = stone_of(built);
        }
        top = side;
        // Turf lives in the open cell on top of the soil: it is drawn as the soil's top.
        if (view.grass != nullptr && z + 1 < view.nz) {
            const float turf = std::clamp(view.grass[i + view.sz] * 1.2f, 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                top[static_cast<std::size_t>(c)] += (palette.grass[static_cast<std::size_t>(c)] -
                                                        top[static_cast<std::size_t>(c)]) *
                    turf;
            }
        }
    } else {
        const float hot = value(view.lava);
        const float wet = value(view.water);
        // A liquid is drawn down to a thin film: a puddle thins out toward its edge, and hiding the
        // thin cells cut holes in it.
        constexpr float film = 0.04f;
        if (hot >= film && hot >= wet) {
            cell.kind = VoxelCell::Lava;
            cell.fill = to_byte(std::clamp(hot, film, 1.0f));
            side = palette.lava;
        } else if (wet >= film) {
            cell.kind = VoxelCell::Water;
            cell.fill = to_byte(std::clamp(wet, film, 1.0f));
            side = palette.water;
        } else {
            return cell;
        }
        top = side;
    }
    // Heat shows: stone reddens long before it melts; lava is its own light.
    const float t = std::isfinite(value(view.temperature)) ? value(view.temperature) : 0.0f;
    const float glow = std::clamp((t - kGlowsFrom) / 900.0f, 0.0f, 1.0f);
    auto heat = [&](std::array<float, 3>& c) {
        c[0] += glow * 0.9f;
        c[1] += glow * glow * 0.45f;
        c[2] += glow * glow * glow * 0.15f;
    };
    heat(side);
    heat(top);
    cell.r = to_byte(side[0]);
    cell.g = to_byte(side[1]);
    cell.b = to_byte(side[2]);
    cell.tr = to_byte(top[0]);
    cell.tg = to_byte(top[1]);
    cell.tb = to_byte(top[2]);
    return cell;
}

void mesh_box(const VoxelBox& box, int border, double cell_size, ChunkMesh& out, int skirt, double cell_z)
{
    out.clear();
    const float c = static_cast<float>(cell_size);
    const float cz = static_cast<float>(cell_z > 0.0 ? cell_z : cell_size);
    auto kind_at = [&](int x, int y, int z) -> std::uint8_t {
        if (z < 0) {
            return VoxelCell::Opaque; // below the world is rock: its underside is never seen
        }
        if (z >= box.nz || x < 0 || y < 0 || x >= box.nx || y >= box.ny) {
            return VoxelCell::Air;
        }
        return box.at(x, y, z).kind;
    };
    auto opaque = [&](int x, int y, int z) { return kind_at(x, y, z) == VoxelCell::Opaque ? 1 : 0; };
    // How high a liquid stands in its cell: full under more of itself, otherwise its fill level.
    auto liquid_top = [&](int x, int y, int z) -> float {
        const std::uint8_t k = kind_at(x, y, z);
        return kind_at(x, y, z + 1) == k ? 1.0f : static_cast<float>(box.at(x, y, z).fill) / 255.0f;
    };

    for (int z = 0; z < box.nz; ++z) {
        for (int y = border; y < box.ny - border; ++y) {
            for (int x = border; x < box.nx - border; ++x) {
                const VoxelCell& v = box.at(x, y, z);
                if (v.kind == VoxelCell::Air) {
                    continue;
                }
                const bool solid = v.kind == VoxelCell::Opaque;
                const float own = solid ? jitter(box.wx + x, box.wy + y, box.wz + z) : 1.0f;
                const float height = solid ? 1.0f : liquid_top(x, y, z);
                for (const Face& face : kFaces) {
                    if (face.n[2] < 0) {
                        continue;
                    }
                    const std::uint8_t neighbour = kind_at(x + face.n[0], y + face.n[1], z + face.n[2]);
                    // Matter hides only behind other matter. A liquid is hidden by matter and by more of
                    // itself — except beside a SHALLOWER cell of the same liquid, where the step between
                    // the two surfaces shows (hiding it left a gap you saw the floor through).
                    float low = 0.0f;
                    if (neighbour == VoxelCell::Opaque) {
                        const int fx = x + face.n[0];
                        const int fy = y + face.n[1];
                        const bool on_edge = fx < border || fy < border || fx >= box.nx - border || fy >= box.ny - border;
                        bool keep = false;
                        if (solid && on_edge && skirt > 0) {
                            for (int up = 1; up <= skirt && !keep; ++up) {
                                keep = kind_at(x, y, z + up) != VoxelCell::Opaque;
                            }
                        }
                        if (!keep) {
                            continue;
                        }
                    }
                    if (!solid && neighbour == v.kind) {
                        if (face.n[2] != 0) {
                            continue;
                        }
                        low = liquid_top(x + face.n[0], y + face.n[1], z);
                        if (low >= height - 1e-3f) {
                            continue;
                        }
                    }
                    const bool is_top = face.n[2] == 1;
                    const float base[3] = {(is_top ? v.tr : v.r) / 255.0f, (is_top ? v.tg : v.g) / 255.0f,
                        (is_top ? v.tb : v.b) / 255.0f};
                    const float lit = (v.kind == VoxelCell::Lava ? 1.0f : face.shade) * own;
                    // Ambient occlusion: a corner crowded by matter on the far side of the face is darker.
                    int ao[4] = {0, 0, 0, 0};
                    if (solid) {
                        const int ax = face.n[0] != 0 ? 0 : (face.n[1] != 0 ? 1 : 2); // normal axis
                        const int t1 = ax == 0 ? 1 : 0;
                        const int t2 = ax == 2 ? 1 : 2;
                        for (int k = 0; k < 4; ++k) {
                            int o[3] = {x + face.n[0], y + face.n[1], z + face.n[2]};
                            int a[3] = {o[0], o[1], o[2]};
                            int b[3] = {o[0], o[1], o[2]};
                            const int s1 = face.corners[k][t1] * 2 - 1;
                            const int s2 = face.corners[k][t2] * 2 - 1;
                            a[t1] += s1;
                            b[t2] += s2;
                            int d[3] = {o[0], o[1], o[2]};
                            d[t1] += s1;
                            d[t2] += s2;
                            const int side1 = opaque(a[0], a[1], a[2]);
                            const int side2 = opaque(b[0], b[1], b[2]);
                            const int corner = opaque(d[0], d[1], d[2]);
                            ao[k] = side1 && side2 ? 3 : side1 + side2 + corner;
                        }
                    }
                    const std::uint32_t first = static_cast<std::uint32_t>(out.vertices.size());
                    for (int k = 0; k < 4; ++k) {
                        const float zc = face.corners[k][2] > 0 ? height : low;
                        const float f = lit * kOcclusion[ao[k]];
                        out.vertices.push_back(MeshVertex{static_cast<float>(box.wx + x + face.corners[k][0]) * c,
                            static_cast<float>(box.wy + y + face.corners[k][1]) * c,
                            (static_cast<float>(box.wz + z) + zc) * cz, v.tree != 0 ? (pack(base[0] * f, base[1] * f, base[2] * f) & 0x00ffffffu) | 0xfe000000u : pack(base[0] * f, base[1] * f, base[2] * f)});
                    }
                    // Split the quad along the diagonal that keeps the occlusion gradient smooth.
                    if (ao[0] + ao[2] > ao[1] + ao[3]) {
                        for (const std::uint32_t k : {1u, 2u, 3u, 1u, 3u, 0u}) {
                            out.indices.push_back(first + k);
                        }
                    } else {
                        for (const std::uint32_t k : {0u, 1u, 2u, 0u, 2u, 3u}) {
                            out.indices.push_back(first + k);
                        }
                    }
                    ++out.faces;
                }
            }
        }
    }
}

void downsample(const VoxelBox& fine, int f, VoxelBox& coarse, int factor_z)
{
    const int fz = factor_z > 0 ? factor_z : f;
    coarse.resize(fine.nx / f, fine.ny / f, fine.nz / fz);
    coarse.wx = fine.wx / f;
    coarse.wy = fine.wy / f;
    coarse.wz = fine.wz / fz;
    const int volume = f * f * fz;
    for (int Z = 0; Z < coarse.nz; ++Z) {
        for (int Y = 0; Y < coarse.ny; ++Y) {
            for (int X = 0; X < coarse.nx; ++X) {
                int matter = 0;
                int trees = 0;
                int lava = 0;
                int water = 0;
                float side[3] = {0, 0, 0};
                float top[3] = {0, 0, 0};
                int sides = 0;
                int tops = 0;
                float wet[3] = {0, 0, 0};
                for (int dy = 0; dy < f; ++dy) {
                    for (int dx = 0; dx < f; ++dx) {
                        bool column_top = false;
                        for (int dz = fz - 1; dz >= 0; --dz) {
                            const VoxelCell& v = fine.at(X * f + dx, Y * f + dy, Z * fz + dz);
                            if (v.kind == VoxelCell::Opaque) {
                                matter += v.trunk ? 4 : 1;
                                trees += v.tree;
                                side[0] += v.r;
                                side[1] += v.g;
                                side[2] += v.b;
                                ++sides;
                                if (!column_top) {
                                    top[0] += v.tr;
                                    top[1] += v.tg;
                                    top[2] += v.tb;
                                    ++tops;
                                    column_top = true;
                                }
                            } else if (v.kind == VoxelCell::Lava || v.kind == VoxelCell::Water) {
                                (v.kind == VoxelCell::Lava ? lava : water) += 1;
                                wet[0] += v.r;
                                wet[1] += v.g;
                                wet[2] += v.b;
                            }
                        }
                    }
                }
                VoxelCell& out = coarse.at(X, Y, Z);
                if (matter * 8 >= volume * 3 && sides > 0) {
                    out.kind = VoxelCell::Opaque;
                    out.tree = trees * 2 >= sides ? 1 : 0;
                    out.r = static_cast<std::uint8_t>(side[0] / sides);
                    out.g = static_cast<std::uint8_t>(side[1] / sides);
                    out.b = static_cast<std::uint8_t>(side[2] / sides);
                    out.tr = static_cast<std::uint8_t>(top[0] / tops);
                    out.tg = static_cast<std::uint8_t>(top[1] / tops);
                    out.tb = static_cast<std::uint8_t>(top[2] / tops);
                } else if ((lava + water) * 4 >= volume) {
                    out.kind = lava > water ? VoxelCell::Lava : VoxelCell::Water;
                    // As deep as the liquid really is, so the sea stands at one height at every level.
                    const int n = lava + water;
                    out.fill = static_cast<std::uint8_t>(std::min(255, n * 255 / volume));
                    out.r = out.tr = static_cast<std::uint8_t>(wet[0] / n);
                    out.g = out.tg = static_cast<std::uint8_t>(wet[1] / n);
                    out.b = out.tb = static_cast<std::uint8_t>(wet[2] / n);
                }
            }
        }
    }
}

VoxelMeshStats build_voxel_mesh(const FieldSet& fields, std::vector<MeshVertex>& out, const VoxelPalette& palette)
{
    out.clear();
    VoxelMeshStats stats;
    const MatterView view = MatterView::of(fields);
    VoxelBox box;
    box.resize(view.nx, view.ny, view.nz);
    for (int z = 0; z < view.nz; ++z) {
        for (int y = 0; y < view.ny; ++y) {
            for (int x = 0; x < view.nx; ++x) {
                const VoxelCell cell = classify(view, x, y, z, palette);
                box.at(x, y, z) = cell;
                stats.solid_voxels += cell.kind == VoxelCell::Opaque ? 1 : 0;
                stats.liquid_voxels += cell.kind == VoxelCell::Lava || cell.kind == VoxelCell::Water ? 1 : 0;
            }
        }
    }
    ChunkMesh mesh;
    mesh_box(box, 0, fields.shape().cell, mesh);
    stats.faces = mesh.faces;
    out.reserve(mesh.indices.size());
    for (const std::uint32_t i : mesh.indices) {
        out.push_back(mesh.vertices[i]);
    }
    return stats;
}

} // namespace fieldengine::field

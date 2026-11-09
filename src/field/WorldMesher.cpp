#include "field/WorldMesher.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

namespace fieldengine::field {

namespace {

// How much the far path is worth: it starts where single trees stop mattering.
constexpr int kFarLevel = 2;

constexpr std::uint64_t kLive = 1ull << 63;

std::uint64_t mix_hash(std::uint64_t h, std::uint64_t v)
{
    return (h ^ v) * 1099511628211ull;
}

std::uint64_t hash_cells(const std::vector<VoxelCell>& cells)
{
    std::uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const unsigned char*>(cells.data());
    const std::size_t bytes = cells.size() * sizeof(VoxelCell);
    std::size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, p + i, 8);
        h = mix_hash(h, word);
    }
    for (; i < bytes; ++i) {
        h = mix_hash(h, p[i]);
    }
    return h;
}

int floor_div(int a, int b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

std::int64_t slot_key(int level, int rx, int ry) { return chunk_key(rx, ry, level); }

void classify_chunk(const MatterView& view, int n, const VoxelPalette& palette, std::vector<VoxelCell>& out)
{
    out.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    std::size_t k = 0;
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                out[k++] = classify(view, x, y, z, palette);
            }
        }
    }
}

// Turf lives in the cell above the soil, which for the top row of a chunk is in the next chunk up: it is drawn all
// the same. (`above`: that chunk's turf as it now is; null: as the generator made it.)
void turf_from_above(std::vector<VoxelCell>& cells, int n, int cz, const ColumnTile& tile, const std::vector<float>* above, const VoxelPalette& palette)
{
    const int top_row = (cz + 1) * n - 1;
    const auto byte = [](float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float turf = 0.0f;
            if (above != nullptr) {
                turf = std::clamp((*above)[static_cast<std::size_t>(y * n + x)] * 1.2f, 0.0f, 1.0f);
            } else {
                const Column& col = tile.columns[static_cast<std::size_t>(y * n + x)];
                turf = col.turf && col.top == top_row ? 1.0f : 0.0f;
            }
            if (turf <= 0.0f) {
                continue;
            }
            VoxelCell& cell = cells[(static_cast<std::size_t>(n - 1) * n + static_cast<std::size_t>(y)) * n + static_cast<std::size_t>(x)];
            if (cell.kind == VoxelCell::Opaque) {
                cell.tr = byte(cell.tr / 255.0f + (palette.grass[0] - cell.tr / 255.0f) * turf);
                cell.tg = byte(cell.tg / 255.0f + (palette.grass[1] - cell.tg / 255.0f) * turf);
                cell.tb = byte(cell.tb / 255.0f + (palette.grass[2] - cell.tb / 255.0f) * turf);
            }
        }
    }
}

// One chunk as the law left it in the store, or as the generator makes it.
void chunk_cells(const TerrainConfig& t, const WorldMap& map, const ChunkStore* store, int cx, int cy, int cz,
    const StoredChunk* stored, const VoxelPalette& palette, std::vector<VoxelCell>& out)
{
    ChunkData data;
    read_chunk_state(t, cx, cy, cz, stored, data);
    classify_chunk(MatterView::of(data, t.chunk, t.chunk), t.chunk, palette, out);
    const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
    std::vector<float> plane;
    if (store != nullptr) {
        if (const StoredChunk* up = store->find(cx, cy, cz + 1)) {
            for (const StoredField& f : up->fields) {
                if (f.name == "grass") {
                    plane.assign(static_cast<std::size_t>(t.chunk) * t.chunk, f.uniform ? f.value : 0.0f);
                    if (!f.uniform) {
                        std::copy_n(f.values.begin(), plane.size(), plane.begin());
                    }
                }
            }
        }
    }
    turf_from_above(out, t.chunk, cz, *tile, plane.empty() ? nullptr : &plane, palette);
}

// The cells of one chunk into a box in world-cell coordinates, where they overlap.
void blit(const std::vector<VoxelCell>& cells, int cx, int cy, int cz, int n, VoxelBox& box)
{
    const int x0 = std::max(cx * n, box.wx);
    const int x1 = std::min((cx + 1) * n, box.wx + box.nx);
    const int y0 = std::max(cy * n, box.wy);
    const int y1 = std::min((cy + 1) * n, box.wy + box.ny);
    const int z0 = std::max(cz * n, box.wz);
    const int z1 = std::min((cz + 1) * n, box.wz + box.nz);
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) {
        return;
    }
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            const std::size_t from = (static_cast<std::size_t>(z - cz * n) * static_cast<std::size_t>(n) + static_cast<std::size_t>(y - cy * n)) *
                    static_cast<std::size_t>(n) + static_cast<std::size_t>(x0 - cx * n);
            std::copy_n(cells.data() + from, x1 - x0, &box.at(x0 - box.wx, y - box.wy, z - box.wz));
        }
    }
}

using CellsFn = std::function<WorldMesher::CellsRef(int cx, int cy, int cz)>;
using LayersFn = std::function<std::vector<int>(int cx, int cy)>; // the layers of a column not to be left out

// A region from whole chunks: as thick as the ground and what stands on it asks, and whatever else `layers` names.
std::uint64_t build_from_chunks(const TerrainConfig& t, const WorldMap& map, const LodLevel& l, int rx, int ry, const LayersFn& layers,
    const CellsFn& cells, VoxelBox& fine, VoxelBox& coarse, ChunkMesh& mesh, int skirt = 0)
{
    const int n = t.chunk;
    int low = t.nz / n;
    int high = 0;
    struct Wanted {
        int cx, cy;
        std::vector<int> layers;
    };
    std::vector<Wanted> columns;
    for (int cy = ry - 1; cy <= ry + l.span; ++cy) {
        for (int cx = rx - 1; cx <= rx + l.span; ++cx) {
            const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
            std::vector<int> named = layers(cx, cy);
            int from = (tile->min_top - 8) / n;
            int to = tile->feature_top / n;
            for (const int cz : named) {
                from = std::min(from, cz);
                to = std::max(to, cz);
            }
            low = std::min(low, std::max(0, from));
            high = std::max(high, to);
            columns.push_back(Wanted{cx, cy, std::move(named)});
        }
    }
    low = std::max(0, low);
    high = std::min(high, t.nz / n - 1);
    fine.resize(l.span * n + 2 * l.factor, l.span * n + 2 * l.factor, (high - low + 1) * n);
    fine.wx = rx * n - l.factor;
    fine.wy = ry * n - l.factor;
    fine.wz = low * n;
    std::uint64_t content = 1469598103934665603ull;
    for (const Wanted& w : columns) {
        const std::shared_ptr<const ColumnTile> tile = map.tile(w.cx, w.cy);
        const int from = std::max(low, (tile->min_top - 8) / n);
        const int to = tile->feature_top / n;
        for (int cz = low; cz <= high; ++cz) {
            const bool named = std::find(w.layers.begin(), w.layers.end(), cz) != w.layers.end();
            if (cz < from && !named) {
                // Rock all the way: no face of it can be seen unless the law opened it, which `layers` would name.
                continue;
            }
            if (cz > to && !named) {
                continue;
            }
            const WorldMesher::CellsRef ref = cells(w.cx, w.cy, cz);
            content = mix_hash(content, ref.hash);
            blit(*ref.cells, w.cx, w.cy, cz, n, fine);
        }
    }
    // What was left out below: rock, so the mesh does not see a hollow.
    for (const Wanted& w : columns) {
        const std::shared_ptr<const ColumnTile> tile = map.tile(w.cx, w.cy);
        const int from = std::max(low, (tile->min_top - 8) / n);
        for (int z = low * n; z < from * n; ++z) {
            for (int y = std::max(w.cy * n, fine.wy); y < std::min((w.cy + 1) * n, fine.wy + fine.ny); ++y) {
                for (int x = std::max(w.cx * n, fine.wx); x < std::min((w.cx + 1) * n, fine.wx + fine.nx); ++x) {
                    VoxelCell& cell = fine.at(x - fine.wx, y - fine.wy, z - fine.wz);
                    if (cell.kind == VoxelCell::Air && std::find(w.layers.begin(), w.layers.end(), z / n) == w.layers.end()) {
                        cell.kind = VoxelCell::Opaque;
                    }
                }
            }
        }
    }
    if (l.factor == 1) {
        mesh_box(fine, 1, t.cell, mesh, skirt);
        return content;
    }
    downsample(fine, l.factor, coarse, l.factor_z);
    // Deep enough to cover the worst mismatch with a neighbour at another level while the zoom settles.
    mesh_box(coarse, 1, t.cell * l.factor, mesh, 2, t.cell * l.factor_z);
    return content;
}

// What one coarse voxel's column looks like from above.
struct Sample {
    int top = 0;
    int liquid = -1;
    std::uint8_t liquid_kind = VoxelCell::Water;
    std::array<std::uint8_t, 3> side{};
    std::array<std::uint8_t, 3> cap{};
    std::array<std::uint8_t, 3> wet{};
    int over = 0; // a roof or a crown over the ground
    std::array<std::uint8_t, 3> over_colour{};
};

// A region as columns of the map, one per voxel, read from what is really there where the law has been.
void build_from_columns(const TerrainConfig& t, const WorldMap& map, const LodLevel& l, const VoxelPalette& palette, int rx, int ry,
    const std::vector<std::pair<int, int>>& real, const LayersFn& layers, const CellsFn& cells, VoxelBox& box, ChunkMesh& mesh)
{
    const int n = t.chunk;
    const int side = l.span * n / l.factor + 2;
    const auto byte = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
    const auto rgb = [&](const std::array<float, 3>& c) { return std::array<std::uint8_t, 3>{byte(c[0]), byte(c[1]), byte(c[2])}; };
    std::vector<Sample> samples(static_cast<std::size_t>(side) * side);
    const int wx0 = rx * n / l.factor - 1;
    const int wy0 = ry * n / l.factor - 1;
    int low = t.nz;
    int high = 0;
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            Sample& s = samples[static_cast<std::size_t>(y * side + x)];
            const int cellx = (wx0 + x) * l.factor + l.factor / 2;
            const int celly = (wy0 + y) * l.factor + l.factor / 2;
            const int cx = floor_div(cellx, n);
            const int cy = floor_div(celly, n);
            if (std::find(real.begin(), real.end(), std::make_pair(cx, cy)) != real.end()) {
                // What is really there, from the top down.
                const std::shared_ptr<const ColumnTile> tile = map.tile(cx, cy);
                int from = tile->feature_top / n;
                for (const int cz : layers(cx, cy)) {
                    from = std::max(from, cz);
                }
                const int lx = cellx - cx * n;
                const int ly = celly - cy * n;
                bool found = false;
                for (int cz = std::min(from, t.nz / n - 1); cz >= 0 && !found; --cz) {
                    const WorldMesher::CellsRef ref = cells(cx, cy, cz);
                    for (int z = n - 1; z >= 0 && !found; --z) {
                        const VoxelCell& v = (*ref.cells)[(static_cast<std::size_t>(z) * n + static_cast<std::size_t>(ly)) * n + static_cast<std::size_t>(lx)];
                        if (v.kind == VoxelCell::Air) {
                            continue;
                        }
                        if (v.kind == VoxelCell::Opaque) {
                            s.top = cz * n + z;
                            s.side = {v.r, v.g, v.b};
                            s.cap = {v.tr, v.tg, v.tb};
                            found = true;
                        } else if (s.liquid < 0) {
                            s.liquid = cz * n + z;
                            s.liquid_kind = v.kind;
                            s.wet = {v.r, v.g, v.b};
                        }
                    }
                }
            } else {
                const FarColumn col = map.far_column(cellx, celly, l.factor);
                const std::array<float, 3>& stone = palette.stone[static_cast<std::size_t>(col.surface)];
                s.top = col.top;
                s.side = rgb(stone);
                s.cap = rgb(col.turf ? palette.grass : stone);
                s.liquid = col.water;
                s.wet = rgb(palette.water);
                if (col.roof_top > col.top) {
                    s.over = col.roof_top;
                    s.over_colour = rgb(palette.stone[static_cast<std::size_t>(col.roof_material)]);
                } else if (col.crown_top > col.top) {
                    s.over = col.crown_top;
                    s.over_colour = rgb(palette.leaf);
                }
            }
            low = std::min(low, s.top);
            high = std::max({high, s.top, s.liquid, s.over});
        }
    }
    box.wx = wx0;
    box.wy = wy0;
    box.wz = std::max(0, floor_div(low, l.factor_z) - 1);
    box.resize(side, side, floor_div(high, l.factor_z) + 2 - box.wz);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            const Sample& s = samples[static_cast<std::size_t>(y * side + x)];
            for (int z = 0; z < box.nz; ++z) {
                const int cell_low = (box.wz + z) * l.factor_z;
                const int cell_high = cell_low + l.factor_z; // cells [low, high)
                const int enough = (l.factor_z * 3 + 7) / 8;
                VoxelCell& cell = box.at(x, y, z);
                if (s.top + 1 - cell_low >= enough) {
                    cell.kind = VoxelCell::Opaque;
                    cell.r = s.side[0];
                    cell.g = s.side[1];
                    cell.b = s.side[2];
                    cell.tr = s.cap[0];
                    cell.tg = s.cap[1];
                    cell.tb = s.cap[2];
                } else if (s.over + 1 - cell_low >= enough && s.over > s.top) {
                    cell.kind = VoxelCell::Opaque;
                    cell.r = cell.tr = s.over_colour[0];
                    cell.g = cell.tg = s.over_colour[1];
                    cell.b = cell.tb = s.over_colour[2];
                } else if (cell_low <= s.liquid) {
                    cell.kind = s.liquid_kind;
                    cell.fill = static_cast<std::uint8_t>(std::min(cell_high, s.liquid + 1) - cell_low) * 255 / l.factor_z;
                    cell.r = cell.tr = s.wet[0];
                    cell.g = cell.tg = s.wet[1];
                    cell.b = cell.tb = s.wet[2];
                }
            }
        }
    }
    mesh_box(box, 1, t.cell * l.factor, mesh, 2, t.cell * l.factor_z);
}

// How far around a region its voxels' sample points can reach, in chunks.
int sample_ring(const LodLevel& l, int n) { return l.factor / (2 * n) + 1; }

} // namespace

LodLevel lod_level(int level)
{
    constexpr LodLevel kLevels[kLodLevels] = {{1, 1, 1}, {2, 1, 2}, {4, 2, 4}, {8, 2, 8}, {16, 4, 16}, {32, 4, 32},
        {64, 8, 64}, {128, 8, 128}, {256, 16, 256}, {512, 16, 512}, {1024, 16, 1024}, {2048, 16, 2048}};
    return kLevels[std::clamp(level, 0, kLodLevels - 1)];
}

int lod_for_zoom(double pixels_per_metre, double cell_metres, double min_pixels, int current)
{
    const auto pixels = [&](int level) { return pixels_per_metre * cell_metres * lod_level(level).factor; };
    int level = 0;
    while (level + 1 < kLodLevels && pixels(level) < min_pixels) {
        ++level;
    }
    // Stay coarse until the finer level clears the bar by a margin.
    if (current > level && current < kLodLevels && pixels(current - 1) < min_pixels * 1.15) {
        return current;
    }
    return level;
}

WorldMesher::WorldMesher() : WorldMesher(Settings{}) {}

WorldMesher::WorldMesher(Settings settings) : settings_(settings)
{
    for (int t = 0; t < settings_.threads; ++t) {
        threads_.emplace_back([this] { worker(); });
    }
}

WorldMesher::~WorldMesher()
{
    {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        stopping_ = true;
    }
    jobs_ready_.notify_all();
    for (std::thread& thread : threads_) {
        thread.join();
    }
}

void WorldMesher::on_fields(const FieldStep& step, const TerrainConfig& terrain)
{
    const FieldSet& fields = step.fields;
    const int n = terrain.chunk;
    const int w = fields.shape().nx / n;
    const int layers = fields.shape().nz / n;
    const std::size_t total = fields.shape().cells();
    const std::size_t sx = static_cast<std::size_t>(fields.shape().nx);
    const std::size_t plane = sx * static_cast<std::size_t>(fields.shape().ny);
    const MatterView lattice = MatterView::of(fields);
    const std::shared_ptr<const WorldMap> map = WorldMap::of(terrain);
    // Only a matter field the step changed can make a chunk look different.
    std::vector<int> ids;
    for (const char* name : {"solid", "rubble", "ice", "wood", "leaf", "lava", "water", "grass", "material", "T"}) {
        if (const int id = fields.find(name); id >= 0) {
            ids.push_back(id);
        }
    }
    const int temperature = fields.find("T");
    const bool first = seen_.size() != fields.count() || seen_[static_cast<std::size_t>(ids[0])].size() != total;
    if (first) {
        seen_.assign(fields.count(), {});
        for (const int id : ids) {
            seen_[static_cast<std::size_t>(id)].assign(total, 0.0f);
        }
    }
    // Chunks that stayed in the window slid across the lattice: so do the copies, and they compare equal.
    const int dx = (step.window_cx - seen_cx_) * n;
    const int dy = (step.window_cy - seen_cy_) * n;
    const int dz = (step.window_cz - seen_cz_) * n;
    const int was_cx = seen_cx_;
    const int was_cy = seen_cy_;
    const int was_cz = seen_cz_;
    if (!first && (dx != 0 || dy != 0 || dz != 0)) {
        const GridShape& g = fields.shape();
        if (std::abs(dx) >= g.nx || std::abs(dy) >= g.ny || std::abs(dz) >= g.nz) {
            for (const int id : ids) {
                std::fill(seen_[static_cast<std::size_t>(id)].begin(), seen_[static_cast<std::size_t>(id)].end(), 0.0f);
            }
        } else {
            const int x0 = std::max(0, -dx);
            const int x1 = std::min(g.nx, g.nx - dx);
            const int y0 = std::max(0, -dy);
            const int y1 = std::min(g.ny, g.ny - dy);
            const int z0 = std::max(0, -dz);
            const int z1 = std::min(g.nz, g.nz - dz);
#pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
                float* v = seen_[static_cast<std::size_t>(ids[static_cast<std::size_t>(k)])].data();
                for (int pz = 0; pz < z1 - z0; ++pz) {
                    const int z = dz >= 0 ? z0 + pz : z1 - 1 - pz;
                    for (int r = 0; r < y1 - y0; ++r) {
                        const int y = dy >= 0 ? y0 + r : y1 - 1 - r;
                        std::memmove(v + g.index(x0, y, z), v + g.index(x0 + dx, y + dy, z + dz), static_cast<std::size_t>(x1 - x0) * sizeof(float));
                    }
                }
            }
        }
    }
    seen_cx_ = step.window_cx;
    seen_cy_ = step.window_cy;
    seen_cz_ = step.window_cz;
#pragma omp parallel
    {
        std::vector<VoxelCell> cells;
#pragma omp for schedule(dynamic) collapse(3)
        for (int l = 0; l < layers; ++l) {
            for (int y = 0; y < w; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int cx = step.window_cx + x;
                    const int cy = step.window_cy + y;
                    const int cz = step.window_cz + l;
                    const std::int64_t key = chunk_key(cx, cy, cz);
                    std::unique_lock<std::mutex> lock(cells_mutex_);
                    Cells& entry = cells_[key];
                    // (Live from an earlier stay in the window is not live: it may have changed since it left.)
                    const bool stayed = !first && cx >= was_cx && cx < was_cx + w && cy >= was_cy && cy < was_cy + w && cz >= was_cz && cz < was_cz + layers;
                    const bool known = entry.source == kLive && stayed;
                    lock.unlock();
                    // A chunk just arrived, or never seen: all of it is news.
                    bool changed = !known;
                    for (const int id : ids) {
                        if (known && !step.changed[static_cast<std::size_t>(id)]) {
                            continue;
                        }
                        const float* now = fields.values(id).data();
                        float* held = seen_[static_cast<std::size_t>(id)].data();
                        for (int z = 0; z < n; ++z) {
                            for (int row = 0; row < n; ++row) {
                                const std::size_t at = static_cast<std::size_t>(l * n + z) * plane + static_cast<std::size_t>(y * n + row) * sx + static_cast<std::size_t>(x * n);
                                if (std::memcmp(now + at, held + at, static_cast<std::size_t>(n) * 4) != 0) {
                                    // Warmth nobody can see is no reason to look again.
                                    bool shows = id != temperature;
                                    for (int k = 0; !shows && k < n; ++k) {
                                        shows = std::max(now[at + k], kGlowsFrom) != std::max(held[at + k], kGlowsFrom);
                                    }
                                    std::memcpy(held + at, now + at, static_cast<std::size_t>(n) * 4);
                                    changed = changed || shows;
                                }
                            }
                        }
                    }
                    if (!changed) {
                        continue;
                    }
                    classify_chunk(lattice.offset(x * n, y * n, l * n), n, palette_, cells);
                    if (l + 1 == layers) {
                        // The lattice ends here; the turf above it is the generator's.
                        turf_from_above(cells, n, cz, *map->tile(cx, cy), nullptr, palette_);
                    }
                    const std::uint64_t hash = hash_cells(cells);
                    lock.lock();
                    if (entry.cells && entry.source == kLive && entry.hash == hash) {
                        continue;
                    }
                    lock.unlock();
                    auto published = std::make_shared<const std::vector<VoxelCell>>(cells);
                    lock.lock();
                    entry.cells = std::move(published);
                    entry.hash = hash;
                    entry.source = kLive;
                    ++live_generation_;
                }
            }
        }
    }
}

WorldMesher::CellsRef WorldMesher::cells_for(const Input& input)
{
    const std::int64_t key = chunk_key(input.cx, input.cy, input.cz);
    {
        const std::lock_guard<std::mutex> lock(cells_mutex_);
        const auto found = cells_.find(key);
        if (found != cells_.end() && found->second.cells && found->second.source == input.source) {
            found->second.used = frame_;
            return CellsRef{found->second.cells, found->second.hash};
        }
    }
    std::vector<VoxelCell> made;
    chunk_cells(terrain_, *map_, nullptr, input.cx, input.cy, input.cz, input.stored.get(), palette_, made);
    const std::uint64_t hash = hash_cells(made);
    auto shared = std::make_shared<const std::vector<VoxelCell>>(std::move(made));
    const std::lock_guard<std::mutex> lock(cells_mutex_);
    Cells& entry = cells_[key];
    // What the law just left stays. Only the law's own step says a chunk is live: a build that was asked for it as
    // live and did not find it had to make it from the store, and must not leave that behind under the law's name.
    if (entry.source != kLive && input.source != kLive) {
        entry.cells = shared;
        entry.hash = hash;
        entry.source = input.source;
        entry.used = frame_;
    }
    return CellsRef{std::move(shared), hash};
}

void WorldMesher::run_job(const Job& job, VoxelBox& fine, VoxelBox& coarse, Done& out)
{
    const LodLevel l = lod_level(job.level);
    out.generation = job.generation;
    out.slot = job.slot;
    out.signature = job.signature;
    out.plan = job.plan;
    const LayersFn layers = [&](int cx, int cy) {
        std::vector<int> named;
        for (const Input& input : job.inputs) {
            if (input.cx == cx && input.cy == cy) {
                named.push_back(input.cz);
            }
        }
        return named;
    };
    const CellsFn cells = [&](int cx, int cy, int cz) {
        for (const Input& input : job.inputs) {
            if (input.cx == cx && input.cy == cy && input.cz == cz) {
                return cells_for(input);
            }
        }
        return cells_for(Input{cx, cy, cz, nullptr, 0});
    };
    if (job.level >= kFarLevel) {
        std::vector<std::pair<int, int>> real;
        for (const Input& input : job.inputs) {
            if (real.empty() || real.back() != std::make_pair(input.cx, input.cy)) {
                real.emplace_back(input.cx, input.cy);
            }
        }
        build_from_columns(terrain_, *map_, l, palette_, job.rx, job.ry, real, layers, cells, coarse, out.mesh);
        out.content = job.signature;
        return;
    }
    // The live window over coarser regions keeps its side faces where it meets them: their surface is not its surface.
    out.content = build_from_chunks(terrain_, *map_, l, job.rx, job.ry, layers, cells, fine, coarse, out.mesh, job.overlay ? 3 : 0);
}

void WorldMesher::worker()
{
    VoxelBox fine;
    VoxelBox coarse;
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(jobs_mutex_);
            jobs_ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
            if (stopping_) {
                return;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        Done done;
        run_job(job, fine, coarse, done);
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        done_.push_back(std::move(done));
    }
}

// What the view wants, and the queue of what is not there yet. The queue is replaced whole: what the view no
// longer wants never gets built. The map says how high the ground is anywhere, at any scale, so the view is looked
// through from the top of a tree down: a tall world costs what is in sight and not what is under or behind it.
void WorldMesher::plan(const Playground& world, const LodView& view)
{
    const TerrainConfig& t = terrain_;
    const int n = t.chunk;
    const double chunk_m = n * t.cell;
    const LodLevel l = lod_level(level_);
    ++plan_id_;
    {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (const Job& job : jobs_) {
            slots_[job.slot].building = false;
        }
        jobs_.clear();
    }
    for (auto& [key, slot] : slots_) {
        slot.wanted = false;
    }
    const int ccx = floor_div(static_cast<int>(std::floor(view.centre_x / chunk_m)), l.span);
    const int ccy = floor_div(static_cast<int>(std::floor(view.centre_y / chunk_m)), l.span);
    const int live_low = world.window_cz() * n;
    const int live_high = (world.window_cz() + world.window_layers()) * n;
    std::vector<Job> asked;
    const std::lock_guard<std::mutex> cells_lock(cells_mutex_);
    // What is really there in one column: what the law changed, and the live layers.
    const auto add_column = [&](Job& job, std::uint64_t& signature, int cx, int cy) {
        signature = mix_hash(signature, static_cast<std::uint64_t>(chunk_key(cx, cy, 0)));
        std::vector<int> changed = world.store().changed_layers(cx, cy);
        if (world.column_in_window(cx, cy)) {
            for (int cz = world.window_cz(); cz < world.window_cz() + world.window_layers(); ++cz) {
                if (std::find(changed.begin(), changed.end(), cz) == changed.end()) {
                    changed.push_back(cz);
                }
            }
            std::sort(changed.begin(), changed.end());
        }
        for (const int cz : changed) {
            const auto found = world.in_window(cx, cy, cz) ? cells_.find(chunk_key(cx, cy, cz)) : cells_.end();
            const bool live = found != cells_.end() && found->second.source == kLive;
            const std::uint64_t version = world.store().version(cx, cy, cz);
            signature = mix_hash(signature, live ? found->second.hash : version + 1);
            const std::vector<int>& kept = world.store().changed_layers(cx, cy);
            const bool has_fields = std::find(kept.begin(), kept.end(), cz) != kept.end();
            job.inputs.push_back(Input{.cx = cx, .cy = cy, .cz = cz, .stored = live || !has_fields ? nullptr : world.store().share(cx, cy, cz),
                .source = live ? kLive : (has_fields ? version : 0)});
        }
    };
    const auto leaf = [&](int level, int rx, int ry, int low, int high) {
        const LodLevel ll = lod_level(level);
        const bool overlay = level != level_;
        const std::int64_t key = slot_key(level, rx, ry);
        std::uint64_t signature = overlay ? 7809847782465536322ull : 1469598103934665603ull;
        Job job;
        if (level < kFarLevel) {
            for (int cy = ry - 1; cy <= ry + ll.span; ++cy) {
                for (int cx = rx - 1; cx <= rx + ll.span; ++cx) {
                    add_column(job, signature, cx, cy);
                }
            }
        } else {
            const int ring = sample_ring(ll, n);
            const int x0 = rx - ring;
            const int y0 = ry - ring;
            const int x1 = rx + ll.span + ring - 1;
            const int y1 = ry + ll.span + ring - 1;
            std::vector<std::pair<int, int>> real;
            world.store().changed_columns(x0, y0, x1, y1, real);
            for (int cy = std::max(y0, world.window_cy()); cy <= std::min(y1, world.window_cy() + world.window_chunks() - 1); ++cy) {
                for (int cx = std::max(x0, world.window_cx()); cx <= std::min(x1, world.window_cx() + world.window_chunks() - 1); ++cx) {
                    real.emplace_back(cx, cy);
                }
            }
            std::sort(real.begin(), real.end());
            real.erase(std::unique(real.begin(), real.end()), real.end());
            for (const auto& [cx, cy] : real) {
                add_column(job, signature, cx, cy);
            }
        }
        Slot& slot = slots_[key];
        slot.level = level;
        slot.rx = rx;
        slot.ry = ry;
        slot.wanted = true;
        slot.low = low;
        slot.high = high;
        if ((slot.handle < 0 || slot.signature != signature) && !slot.building) {
            job.slot = key;
            job.level = level;
            job.overlay = overlay;
            job.rx = rx;
            job.ry = ry;
            job.plan = plan_id_;
            job.generation = generation_;
            job.distance = overlay ? 0 : std::max(std::abs(floor_div(rx, l.span) - ccx), std::abs(floor_div(ry, l.span) - ccy));
            job.hole = slot.handle < 0;
            job.signature = signature;
            asked.push_back(std::move(job));
        }
    };
    const std::function<void(int, int, int)> visit = [&](int span, int rx, int ry) {
        int low = 0;
        int high = t.nz;
        map_->height_bounds(rx * n, ry * n, (rx + span) * n, (ry + span) * n, low, high);
        if (rx + span > world.window_cx() && rx < world.window_cx() + world.window_chunks() && ry + span > world.window_cy() &&
            ry < world.window_cy() + world.window_chunks()) {
            low = std::min(low, live_low);
            high = std::max(high, live_high);
        }
        const double lo[3] = {rx * chunk_m, ry * chunk_m, low * t.cell};
        const double hi[3] = {(rx + span) * chunk_m, (ry + span) * chunk_m, high * t.cell};
        if (view.visible && !view.visible(lo, hi)) {
            return;
        }
        if (span == l.span) {
            leaf(level_, rx, ry, low, high);
            return;
        }
        const int half = span / 2;
        for (const int y : {ry, ry + half}) {
            for (const int x : {rx, rx + half}) {
                visit(half, x, y);
            }
        }
    };
    int top = l.span;
    while (top < 2048) {
        top *= 2;
    }
    const int reach = static_cast<int>(std::ceil(view.radius_metres / (chunk_m * top))) + 1;
    const int tcx = floor_div(static_cast<int>(std::floor(view.centre_x / chunk_m)), top);
    const int tcy = floor_div(static_cast<int>(std::floor(view.centre_y / chunk_m)), top);
    for (int gy = tcy - reach; gy <= tcy + reach; ++gy) {
        for (int gx = tcx - reach; gx <= tcx + reach; ++gx) {
            visit(top, gx * top, gy * top);
        }
    }
    // THE LIVE WINDOW IS ALWAYS DRAWN WHOLE. Zoomed out a little, the regions around it are coarse; what the law is running stays
    // at full detail over them (the coarse ones are clipped where it is: see window_box), so what is seen there is what bodies
    // stand on and pieces hit. Further out the window is a few pixels and this is not worth its meshes.
    overlay_ = level_ > 0 && level_ <= kOverlayMost;
    window_box_ = {world.window_cx(), world.window_cy(), world.window_cx() + world.window_chunks(), world.window_cy() + world.window_chunks()};
    if (overlay_) {
        for (int cy = world.window_cy(); cy < world.window_cy() + world.window_chunks(); ++cy) {
            for (int cx = world.window_cx(); cx < world.window_cx() + world.window_chunks(); ++cx) {
                int low = 0;
                int high = t.nz;
                map_->height_bounds(cx * n, cy * n, (cx + 1) * n, (cy + 1) * n, low, high);
                leaf(0, cx, cy, std::min(low, live_low), std::max(high, live_high));
            }
        }
    }
    std::sort(asked.begin(), asked.end(), [](const Job& a, const Job& b) {
        return a.hole != b.hole ? a.hole : a.distance < b.distance;
    });
    asked_ = asked.size();
    {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (Job& job : asked) {
            slots_[job.slot].building = true;
            jobs_.push_back(std::move(job));
        }
    }
    jobs_ready_.notify_all();
}

void WorldMesher::update(const Playground& world, const LodView& view, const Upload& upload, const Destroy& destroy)
{
    const auto start = std::chrono::steady_clock::now();
    ++frame_;
    stats_ = Stats{};
    if (!world.open_world()) {
        return;
    }
    if (slots_.empty()) {
        terrain_ = world.config().terrain; // nothing is being built from it
        map_ = WorldMap::of(terrain_);
    }
    const TerrainConfig& t = terrain_;
    const int level = lod_for_zoom(view.pixels_per_metre, t.cell, settings_.min_pixels, level_);
    const std::uint64_t live = live_generation_.load();
    // A build that was asked for before the last plan may have come back stale, and nothing else would ask again.
    // Going through the view costs milliseconds. What the view asks for now goes now; news of a build or of a chunk can wait a few frames.
    const bool news = replan_needed_ || world.store().counter() != planned_store_ || live != planned_live_;
    bool replan = level != level_ || view.stamp != planned_stamp_ || slots_.empty() || (news && frame_ >= planned_frame_ + kFramesBetweenPlans);
    level_ = level;
    const LodLevel l = lod_level(level_);
    stats_.level = level_;
    std::vector<Done> finished;
    {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        finished.swap(done_);
    }
    if (replan) {
        replan_needed_ = false;
        planned_stamp_ = view.stamp;
        planned_store_ = world.store().counter();
        planned_live_ = live;
        planned_frame_ = frame_;
        plan(world, view);
    }
    stats_.pending = asked_;
    const auto taking_up = std::chrono::steady_clock::now();
    const bool holes = std::any_of(slots_.begin(), slots_.end(), [](const auto& entry) { return entry.second.wanted && entry.second.handle < 0; });
    // While there is nothing at all to show somewhere, showing it comes before a smooth frame.
    const double budget = holes ? settings_.budget_ms * 2.0 : settings_.budget_ms;

    // --- take up what is ready, within the budget --------------------------------------------------------
    std::vector<Done> later;
    for (Done& done : finished) {
        const auto found = slots_.find(done.slot);
        if (done.generation != generation_ || found == slots_.end()) {
            continue;
        }
        Slot& slot = found->second;
        const double spent = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - taking_up).count();
        if (stats_.built_this_frame > 0 && spent > budget) {
            later.push_back(std::move(done));
            continue;
        }
        slot.building = false;
        replan_needed_ = replan_needed_ || done.plan != plan_id_;
        if (!slot.wanted) {
            continue; // the view moved on while it was being built
        }
        slot.signature = done.signature;
        if (slot.handle < 0 || slot.content != done.content) {
            slot.handle = upload(slot.handle, done.mesh);
            slot.content = done.content;
            slot.faces = done.mesh.faces;
            ++stats_.built_this_frame;
        }
    }
    if (!later.empty()) {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (Done& done : later) {
            done_.push_back(std::move(done));
        }
    }

    // --- a mesh nobody wants goes when what replaces it has come, or when it is out of sight --------------
    auto covered = [&](const Slot& old) {
        const int span = lod_level(old.level).span;
        for (int cy = old.ry; cy < old.ry + span; ++cy) {
            for (int cx = old.rx; cx < old.rx + span; ++cx) {
                const int rx = floor_div(cx, l.span) * l.span;
                const int ry = floor_div(cy, l.span) * l.span;
                const auto found = slots_.find(slot_key(level_, rx, ry));
                if (found != slots_.end() && found->second.wanted && found->second.handle < 0) {
                    return false;
                }
            }
        }
        return true;
    };
    for (auto it = slots_.begin(); it != slots_.end();) {
        Slot& slot = it->second;
        if (slot.wanted) {
            ++it;
            continue;
        }
        if (slot.handle < 0 && !slot.building) {
            it = slots_.erase(it);
            continue;
        }
        if (slot.handle >= 0 && covered(slot)) {
            destroy(slot.handle);
            slot.handle = -1;
            slot.faces = 0;
        }
        ++it;
    }
    bool settled = !replan_needed_ && world.store().counter() == planned_store_ && live_generation_.load() == planned_live_;
    for (const auto& [key, slot] : slots_) {
        stats_.regions_drawn += slot.handle >= 0 ? 1 : 0;
        stats_.faces += slot.handle >= 0 ? slot.faces : 0;
        stats_.wanted += slot.wanted ? 1 : 0;
        settled = settled && !slot.building && (slot.wanted ? slot.handle >= 0 : slot.handle < 0);
    }
    settled_ = settled;
    // Built a quarter of a screen around, so walking finds it there; drawn only what is in sight.
    if (view.in_sight) {
        const double chunk_m = t.chunk * t.cell;
        for (auto& [key, slot] : slots_) {
            if (slot.handle >= 0 && slot.wanted) {
                const int span = lod_level(slot.level).span;
                const double lo[3] = {slot.rx * chunk_m, slot.ry * chunk_m, slot.low * t.cell};
                const double hi[3] = {(slot.rx + span) * chunk_m, (slot.ry + span) * chunk_m, slot.high * t.cell};
                slot.in_sight = view.in_sight(lo, hi);
            } else {
                slot.in_sight = true;
            }
        }
    }

    // --- generated chunks are kept while they are used; the live window always is -------------------------
    {
        const std::lock_guard<std::mutex> lock(cells_mutex_);
        if (cells_.size() > settings_.cached_chunks) {
            std::vector<std::pair<std::uint64_t, std::int64_t>> order;
            for (const auto& [key, entry] : cells_) {
                if (entry.source != kLive) {
                    order.emplace_back(entry.used, key);
                }
            }
            std::sort(order.begin(), order.end());
            const std::size_t drop = std::min(order.size(), cells_.size() - settings_.cached_chunks);
            for (std::size_t i = 0; i < drop; ++i) {
                cells_.erase(order[i].second);
            }
        }
        // A chunk that has left the window is no longer the law's: the store says what it is now.
        for (auto& [key, entry] : cells_) {
            const int cx = key_cx(key);
            const int cy = key_cy(key);
            const int cz = key_cz(key);
            if (entry.source == kLive && !world.in_window(cx, cy, cz)) {
                const std::vector<int> kept = world.store().changed_layers(cx, cy);
                entry.source = std::find(kept.begin(), kept.end(), cz) != kept.end() ? world.store().version(cx, cy, cz) : 0;
                // What was last seen of it may be older than what left: the store says what it is now.
                entry.cells.reset();
            }
        }
        stats_.cached = cells_.size();
    }
    stats_.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void WorldMesher::for_each_drawn(const std::function<void(int handle, int level, int rx, int ry)>& fn) const
{
    for (const auto& [key, slot] : slots_) {
        if (slot.handle >= 0 && slot.in_sight) {
            fn(slot.handle, slot.level, slot.rx, slot.ry);
        }
    }
}

void WorldMesher::clear(const Destroy& destroy)
{
    {
        const std::lock_guard<std::mutex> lock(jobs_mutex_);
        jobs_.clear();
        done_.clear();
    }
    ++generation_;
    for (auto& [key, slot] : slots_) {
        if (slot.handle >= 0) {
            destroy(slot.handle);
        }
    }
    slots_.clear();
    const std::lock_guard<std::mutex> lock(cells_mutex_);
    cells_.clear();
}

namespace {

// Cells of a world's current state, as a worker would have them.
WorldMesher::CellsRef cells_of(const Playground& world, int cx, int cy, int cz, const VoxelPalette& palette)
{
    const TerrainConfig& t = world.config().terrain;
    std::vector<VoxelCell> cells;
    if (world.in_window(cx, cy, cz)) {
        const int n = t.chunk;
        classify_chunk(MatterView::of(world.fields()).offset((cx - world.window_cx()) * n, (cy - world.window_cy()) * n, (cz - world.window_cz()) * n), n, palette, cells);
        if (cz + 1 == world.window_cz() + world.window_layers()) {
            turf_from_above(cells, n, cz, *world.map()->tile(cx, cy), nullptr, palette);
        }
    } else {
        chunk_cells(t, *world.map(), &world.store(), cx, cy, cz, world.store().find(cx, cy, cz), palette, cells);
    }
    return WorldMesher::CellsRef{std::make_shared<const std::vector<VoxelCell>>(std::move(cells)), 0};
}

std::vector<int> layers_of(const Playground& world, int cx, int cy)
{
    std::vector<int> named = world.store().changed_layers(cx, cy);
    if (world.column_in_window(cx, cy)) {
        for (int cz = world.window_cz(); cz < world.window_cz() + world.window_layers(); ++cz) {
            if (std::find(named.begin(), named.end(), cz) == named.end()) {
                named.push_back(cz);
            }
        }
    }
    return named;
}

} // namespace

void WorldMesher::build_region_from_chunks(const Playground& world, int level, int rx, int ry, ChunkMesh& out)
{
    const TerrainConfig& t = world.config().terrain;
    const VoxelPalette palette;
    VoxelBox fine;
    VoxelBox coarse;
    build_from_chunks(t, *world.map(), lod_level(level), rx, ry, [&](int cx, int cy) { return layers_of(world, cx, cy); },
        [&](int cx, int cy, int cz) { return cells_of(world, cx, cy, cz, palette); }, fine, coarse, out);
}

void WorldMesher::build_region(const Playground& world, int level, int rx, int ry, ChunkMesh& out)
{
    if (level < kFarLevel) {
        build_region_from_chunks(world, level, rx, ry, out);
        return;
    }
    const TerrainConfig& t = world.config().terrain;
    const LodLevel l = lod_level(level);
    const VoxelPalette palette;
    const int ring = sample_ring(l, t.chunk);
    const int x0 = rx - ring;
    const int y0 = ry - ring;
    const int x1 = rx + l.span + ring - 1;
    const int y1 = ry + l.span + ring - 1;
    std::vector<std::pair<int, int>> real;
    world.store().changed_columns(x0, y0, x1, y1, real);
    for (int cy = std::max(y0, world.window_cy()); cy <= std::min(y1, world.window_cy() + world.window_chunks() - 1); ++cy) {
        for (int cx = std::max(x0, world.window_cx()); cx <= std::min(x1, world.window_cx() + world.window_chunks() - 1); ++cx) {
            real.emplace_back(cx, cy);
        }
    }
    std::sort(real.begin(), real.end());
    real.erase(std::unique(real.begin(), real.end()), real.end());
    VoxelBox box;
    build_from_columns(t, *world.map(), l, palette, rx, ry, real, [&](int cx, int cy) { return layers_of(world, cx, cy); },
        [&](int cx, int cy, int cz) { return cells_of(world, cx, cy, cz, palette); }, box, out);
}

} // namespace fieldengine::field

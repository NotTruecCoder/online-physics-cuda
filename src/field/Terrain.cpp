#include "field/Terrain.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::field {

namespace {

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

int stone_at(int z, double bend)
{
    // Strata: bands of stone that bend with a slow noise, basalt at the bottom of the world.
    if (z < 3) {
        return material::basalt;
    }
    const int band = static_cast<int>(std::floor((z + bend) / 4.0));
    static constexpr int kBands[] = {material::granite, material::slate, material::sandstone, material::granite,
        material::slate, material::basalt};
    return kBands[static_cast<std::size_t>(((band % 6) + 6) % 6)];
}

} // namespace

std::vector<float>& ChunkData::add(const std::string& name, std::size_t cells, float value)
{
    if (std::vector<float>* existing = find(name)) {
        existing->assign(cells, value);
        return *existing;
    }
    fields.emplace_back(name, std::vector<float>(cells, value));
    return fields.back().second;
}

std::vector<float>* ChunkData::find(std::string_view name)
{
    for (auto& [n, values] : fields) {
        if (n == name) {
            return &values;
        }
    }
    return nullptr;
}

const std::vector<float>* ChunkData::find(std::string_view name) const
{
    for (const auto& [n, values] : fields) {
        if (n == name) {
            return &values;
        }
    }
    return nullptr;
}

int terrain_top(const TerrainConfig& config, int wx, int wy) { return WorldMap::of(config)->column(wx, wy).top; }

void generate_chunk(const TerrainConfig& c, int cx, int cy, int cz, ChunkData& out, std::vector<CreatureSpawn>* spawns)
{
    const int n = c.chunk;
    const std::size_t cells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    out.fields.clear();
    std::vector<float>& solid = out.add("solid", cells);
    std::vector<float>& mat = out.add("material", cells);
    std::vector<float>& water = out.add("water", cells);
    std::vector<float>& wood = out.add("wood", cells);
    std::vector<float>& leaf = out.add("leaf", cells);
    std::vector<float>& grass = out.add("grass", cells);
    std::vector<float>& rubble = out.add("rubble", cells);
    std::vector<float>& built = out.add("built", cells);
    const int z0 = cz * n;
    const int z1 = z0 + n; // cells [z0, z1)
    auto index = [&](int x, int y, int z) {
        return (static_cast<std::size_t>(z - z0) * static_cast<std::size_t>(n) + static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(n) +
            static_cast<std::size_t>(x);
    };
    const int x0 = cx * n;
    const int y0 = cy * n;
    const std::shared_ptr<const WorldMap> map = WorldMap::of(c);
    const std::shared_ptr<const ColumnTile> tile = map->tile(cx, cy);
    const auto col_at = [&](int x, int y) -> const Column& { return tile->columns[static_cast<std::size_t>(y * n + x)]; };

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const Column& col = col_at(x, y);
            for (int z = z0; z <= std::min(col.top, z1 - 1); ++z) {
                const std::size_t i = index(x, y, z);
                solid[i] = 1.0f;
                const int depth = col.top - z;
                int m = stone_at(z, col.bend);
                if ((col.surface == material::soil && depth < 3) || (depth < 2 && (col.surface == material::sand ||
                        col.surface == material::clay || col.surface == material::chalk || col.surface == material::gravel)) ||
                    (depth < 1 && (col.surface == material::granite || col.surface == material::slate))) {
                    m = col.surface;
                }
                mat[i] = static_cast<float>(m);
            }
            for (int z = std::max(col.top + 1, z0); z <= col.water && z < z1; ++z) {
                water[index(x, y, z)] = 1.0f;
            }
            if (col.turf && col.top + 1 >= z0 && col.top + 1 < z1) {
                grass[index(x, y, col.top + 1)] = 1.0f;
            }
        }
    }

    // Boulders: a few rounded stones lying on the ground (whole, inside one chunk).
    if (tile->feature_top >= z0 && tile->min_top < z1) {
        const std::uint64_t h = hash(c.seed ^ 0x88, cx, cy);
        const int bx = 3 + static_cast<int>((h >> 12) % static_cast<std::uint64_t>(n - 6));
        const int by = 3 + static_cast<int>((h >> 24) % static_cast<std::uint64_t>(n - 6));
        const Column& rest = col_at(bx, by);
        if (unit(h) < 0.2 && !rest.clear && rest.water < 0) {
            const double r = 1.3 + unit(hash(c.seed ^ 0x89, cx, cy)) * 0.9;
            const int ground = rest.top;
            const int stone = (h >> 40) % 2 == 0 ? material::granite : material::basalt;
            const int reach = static_cast<int>(std::ceil(r));
            for (int z = std::max(ground - 1, z0); z <= ground + reach + 1 && z < z1; ++z) {
                for (int y = by - reach; y <= by + reach; ++y) {
                    for (int x = bx - reach; x <= bx + reach; ++x) {
                        const double dx = x - bx;
                        const double dy = y - by;
                        const double dz = (z - ground - 0.5) * 1.25;
                        if (x < 0 || y < 0 || x >= n || y >= n || dx * dx + dy * dy + dz * dz > r * r) {
                            continue;
                        }
                        const std::size_t i = index(x, y, z);
                        solid[i] = 1.0f;
                        mat[i] = static_cast<float>(stone);
                        water[i] = 0.0f;
                        grass[i] = 0.0f;
                    }
                }
            }
        }
    }

    // What grows on the ground: a plant to a cell, at most. (Leaves to the law: they burn.)
    if (tile->max_top + 3 >= z0 && tile->min_top < z1) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const Column& col = col_at(x, y);
                if (col.top + 3 < z0 || col.top >= z1) {
                    continue;
                }
                const FloraKind* plant = map->flora_at(x0 + x, y0 + y, col);
                if (plant == nullptr) {
                    continue;
                }
                for (int z = col.top + 1; z <= col.top + plant->height; ++z) {
                    if (z < z0 || z >= z1) {
                        continue;
                    }
                    const std::size_t i = index(x, y, z);
                    if (solid[i] == 0.0f) {
                        leaf[i] = 1.0f;
                        mat[i] = static_cast<float>(plant->colour);
                    }
                }
            }
        }
    }

    // Trees: what grew, cell by cell (WorldMap.cpp, grow_tree).
    for (const Tree& tree : tile->trees) {
        if (tree.top() < z0 || tree.ground >= z1 || tree.shape == nullptr) {
            continue;
        }
        for (const TreeVoxel& v : *tree.shape) {
            const int z = tree.ground + v.dz;
            const int x = tree.x + v.dx - x0;
            const int y = tree.y + v.dy - y0;
            if (z < z0 || z >= z1 || x < 0 || y < 0 || x >= n || y >= n) {
                continue;
            }
            const std::size_t i = index(x, y, z);
            if (v.wood != 0) {
                wood[i] = 1.0f;
                leaf[i] = 0.0f;
                grass[i] = 0.0f;
                water[i] = 0.0f;
                solid[i] = 0.0f;
                mat[i] = static_cast<float>(v.material);
            } else if (solid[i] == 0.0f && wood[i] == 0.0f) {
                leaf[i] = 1.0f;
                water[i] = 0.0f;
                mat[i] = static_cast<float>(v.material);
            }
        }
    }

    // What people built.
    const std::vector<Style>& styles = map->styles();
    for (const Building* b : tile->buildings) {
        const Style& style = styles[static_cast<std::size_t>(b->style)];
        int bx0, by0, bx1, by1, top;
        b->bounds(styles, bx0, by0, bx1, by1, top);
        const bool ring = b->kind == Building::Kind::Ring;
        if (ring ? (top + 60 < z0 || b->lowest() >= z1) : (top < z0 || b->lowest() >= z1)) {
            continue;
        }
        for (int z = std::max(b->lowest(), z0); z < z1 && (ring || z <= top); ++z) {
            for (int wy = std::max(by0, y0); wy <= by1 && wy < y0 + n; ++wy) {
                for (int wx = std::max(bx0, x0); wx <= bx1 && wx < x0 + n; ++wx) {
                    // A town wall stops at the water it runs into, and goes over a bank it meets.
                    const int lift = ring ? std::max(0, col_at(wx - x0, wy - y0).top - b->ground) : 0;
                    if (ring && (col_at(wx - x0, wy - y0).water >= 0 || lift > 40)) {
                        continue;
                    }
                    const BuildingCell cell = building_cell(*b, style, wx, wy, z - lift);
                    if (cell.kind == BuildingCell::Outside) {
                        continue;
                    }
                    const std::size_t i = index(wx - x0, wy - y0, z);
                    built[i] = 1.0f;
                    solid[i] = 0.0f;
                    wood[i] = 0.0f;
                    leaf[i] = 0.0f;
                    water[i] = 0.0f;
                    grass[i] = 0.0f;
                    rubble[i] = 0.0f;
                    mat[i] = 0.0f;
                    switch (cell.kind) {
                    case BuildingCell::Stone:
                        solid[i] = 1.0f;
                        mat[i] = static_cast<float>(cell.material);
                        break;
                    case BuildingCell::Wood:
                        wood[i] = 1.0f;
                        mat[i] = static_cast<float>(cell.material);
                        break;
                    case BuildingCell::Leaf:
                        leaf[i] = 1.0f;
                        mat[i] = static_cast<float>(cell.material);
                        break;
                    case BuildingCell::Rubble:
                        rubble[i] = 1.0f;
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }

    if (spawns == nullptr) {
        return;
    }
    // Creatures start on the ground of their column, in the chunk that holds it: every candidate there could be.
    // How many of them the land feeds now is the map's law, read where they are taken up (Playground).
    const auto stand = [&](int x, int y, std::size_t& above) {
        const Column& col = col_at(x, y);
        if (!col.turf || col.clear || col.water >= 0 || col.top + 1 < z0 || col.top + 1 >= z1) {
            return false;
        }
        above = index(x, y, col.top + 1);
        return solid[above] == 0.0f && wood[above] == 0.0f && leaf[above] == 0.0f;
    };
    {
        const std::uint64_t hh = hash(c.seed ^ 0xac, cx, cy);
        const int x = 2 + static_cast<int>((hh >> 8) % static_cast<std::uint64_t>(n - 4));
        const int y = 2 + static_cast<int>((hh >> 24) % static_cast<std::uint64_t>(n - 4));
        std::size_t above = 0;
        if (unit(hh) < 0.10 && stand(x, y, above)) {
            spawns->push_back(CreatureSpawn{.kind = "hunter", .x = (x0 + x + 0.5) * c.cell, .y = (y0 + y + 0.5) * c.cell,
                .z = (col_at(x, y).top + 1) * c.cell + 0.42});
        }
    }
    // What there is to hunt: alone, and not on every hill.
    {
        const std::uint64_t hh = hash(c.seed ^ 0xad, cx, cy);
        const double roll = unit(hh);
        const char* kind = roll < 0.05 ? "deer" : roll < 0.08 ? "boar" : roll < 0.10 ? "wolf" : roll < 0.108 ? "bear" : nullptr;
        const int x = 2 + static_cast<int>((hh >> 8) % static_cast<std::uint64_t>(n - 4));
        const int y = 2 + static_cast<int>((hh >> 24) % static_cast<std::uint64_t>(n - 4));
        std::size_t above = 0;
        if (kind != nullptr && stand(x, y, above)) {
            spawns->push_back(CreatureSpawn{.kind = kind, .x = (x0 + x + 0.5) * c.cell, .y = (y0 + y + 0.5) * c.cell,
                .z = (col_at(x, y).top + 1) * c.cell + 0.55});
        }
    }
    const std::uint64_t h = hash(c.seed ^ 0xaa, cx, cy);
    if (unit(h) < 0.45) {
        const int count = 2 + static_cast<int>((h >> 20) % 3);
        for (int k = 0; k < count; ++k) {
            const std::uint64_t hk = hash(c.seed ^ 0xab, cx, cy, k);
            const int x = 2 + static_cast<int>(hk % static_cast<std::uint64_t>(n - 4));
            const int y = 2 + static_cast<int>((hk >> 16) % static_cast<std::uint64_t>(n - 4));
            std::size_t above = 0;
            if (stand(x, y, above)) {
                spawns->push_back(CreatureSpawn{.kind = "grazer", .x = (x0 + x + 0.5) * c.cell, .y = (y0 + y + 0.5) * c.cell,
                    .z = (col_at(x, y).top + 1) * c.cell + 0.35});
            }
        }
    }
}

void read_chunk_state(const TerrainConfig& config, int cx, int cy, int cz, const StoredChunk* stored, ChunkData& out)
{
    generate_chunk(config, cx, cy, cz, out, nullptr);
    if (stored == nullptr) {
        return;
    }
    const std::size_t cells = static_cast<std::size_t>(config.chunk) * static_cast<std::size_t>(config.chunk) * static_cast<std::size_t>(config.chunk);
    for (const StoredField& field : stored->fields) {
        std::vector<float>& values = out.add(field.name, cells);
        if (field.uniform) {
            std::fill(values.begin(), values.end(), field.value);
        } else {
            values = field.values;
        }
    }
}

const StoredChunk* ChunkStore::find(int cx, int cy, int cz) const
{
    const auto it = chunks_.find(chunk_key(cx, cy, cz));
    return it == chunks_.end() ? nullptr : it->second.get();
}

std::shared_ptr<const StoredChunk> ChunkStore::share(int cx, int cy, int cz) const
{
    const auto it = chunks_.find(chunk_key(cx, cy, cz));
    return it == chunks_.end() ? nullptr : it->second;
}

void ChunkStore::put(int cx, int cy, int cz, StoredChunk chunk)
{
    const std::int64_t key = chunk_key(cx, cy, cz);
    const std::int64_t column = chunk_key(cx, cy, 0);
    versions_[key] = ++counter_;
    column_versions_[column] = counter_;
    auto shared = std::make_shared<const StoredChunk>(std::move(chunk));
    std::vector<int>& layers = changed_[column];
    const auto found = std::lower_bound(layers.begin(), layers.end(), cz);
    const bool had = found != layers.end() && *found == cz;
    if (shared->fields.empty() && had) {
        layers.erase(found);
    } else if (!shared->fields.empty() && !had) {
        layers.insert(found, cz);
    }
    if (layers.empty()) {
        changed_.erase(column);
    }
    chunks_[key] = std::move(shared);
}

void ChunkStore::changed_columns(int cx0, int cy0, int cx1, int cy1, std::vector<std::pair<int, int>>& out) const
{
    const auto take = [&](std::int64_t key) {
        const int cy = key_cy(key);
        if (cy >= cy0 && cy <= cy1) {
            out.emplace_back(key_cx(key), cy);
        }
    };
    if (cx1 - cx0 < 64) {
        // Keys sort by x first: a narrow box only looks at its own stretches of the map.
        for (int cx = cx0; cx <= cx1; ++cx) {
            const std::int64_t low = chunk_key(cx, 0, 0);
            for (auto it = changed_.lower_bound(low); it != changed_.end() && key_cx(it->first) == cx; ++it) {
                take(it->first);
            }
        }
        return;
    }
    for (const auto& [key, layers] : changed_) {
        const int cx = key_cx(key);
        if (cx >= cx0 && cx <= cx1) {
            take(key);
        }
    }
}

bool ChunkStore::changed_in(int cx0, int cy0, int cx1, int cy1) const
{
    std::vector<std::pair<int, int>> found;
    changed_columns(cx0, cy0, cx1, cy1, found);
    return !found.empty();
}

std::vector<int> ChunkStore::changed_layers(int cx, int cy) const
{
    const auto it = changed_.find(chunk_key(cx, cy, 0));
    return it == changed_.end() ? std::vector<int>{} : it->second;
}

std::size_t ChunkStore::bytes() const
{
    std::size_t total = 0;
    for (const auto& [key, chunk] : chunks_) {
        for (const StoredField& f : chunk->fields) {
            total += f.values.size() * sizeof(float);
        }
    }
    return total;
}

std::uint64_t ChunkStore::version(int cx, int cy, int cz) const
{
    const auto it = versions_.find(chunk_key(cx, cy, cz));
    return it == versions_.end() ? 0 : it->second;
}

std::uint64_t ChunkStore::column_version(int cx, int cy) const
{
    const auto it = column_versions_.find(chunk_key(cx, cy, 0));
    return it == column_versions_.end() ? 0 : it->second;
}

} // namespace fieldengine::field

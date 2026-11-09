#pragma once

#include "field/WorldMap.hpp"

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// THE WORLD AS A FUNCTION. Terrain, trees, buildings and where creatures start are a pure function of (seed,
// chunk coordinates), by way of the layers of the map (WorldMap.hpp): the same chunk comes out the same however
// often and in whatever order it is generated, so a chunk nobody changed is never stored — it is generated again.
// A chunk is a cube of `chunk` cells; the world is `nz` cells tall, so a column of them holds nz / chunk chunks,
// and only those near the ground ever hold anything but air or rock. What the generator writes are ordinary
// fields of the law (solid, material, water, wood, leaf, grass, rubble); from the first step on, the law owns them
// like any other cell.
namespace fieldengine::field {

// Palette indices the generator writes into `material` (VoxelPalette::stone).
namespace material {
inline constexpr int granite = 0;
inline constexpr int basalt = 1;
inline constexpr int slate = 2;
inline constexpr int sandstone = 3;
inline constexpr int soil = 4; // life.law grows turf on it
inline constexpr int sand = 5;
inline constexpr int clay = 6;
inline constexpr int chalk = 7;
inline constexpr int gravel = 8;
inline constexpr int plaster = 9;
inline constexpr int brick = 10;
inline constexpr int thatch = 11;
inline constexpr int tile = 12;
inline constexpr int timber = 13;
inline constexpr int charred = 14;
inline constexpr int reed = 15;
} // namespace material

struct TerrainConfig {
    std::uint64_t seed = 20261025;
    int chunk = 16;      // cells per chunk edge
    int nz = 2048;       // cells tall (the whole height of the world)
    double cell = 0.5;   // metres
    int sea_level = 256; // cells: every open cell at or below it is water
};

// One chunk (chunk^3 cells, x fastest, then y, then z) as named arrays. Only fields the generator has an opinion
// about are present; every other field starts at its law's init.
struct ChunkData {
    // A deque: add() hands out references, and adding the next field must not move the last one.
    std::deque<std::pair<std::string, std::vector<float>>> fields;

    std::vector<float>& add(const std::string& name, std::size_t cells, float value = 0.0f);
    [[nodiscard]] std::vector<float>* find(std::string_view name);
    [[nodiscard]] const std::vector<float>* find(std::string_view name) const;
};

struct CreatureSpawn {
    std::string kind;
    double x = 0.0; // world metres
    double y = 0.0;
    double z = 0.0;
};

// Fills `out` with chunk (cx, cy, cz); `spawns` (optional) gets the creatures that start in it.
void generate_chunk(const TerrainConfig& config, int cx, int cy, int cz, ChunkData& out, std::vector<CreatureSpawn>* spawns);

// A chunk outside the simulated window as it now is: what the store kept (may be null) over what the generator
// makes. Fields neither knows about are absent (they hold their law's init).
struct StoredChunk;
void read_chunk_state(const TerrainConfig& config, int cx, int cy, int cz, const StoredChunk* stored, ChunkData& out);

// The cell index of the highest ground cell of a world column (what the generator builds on).
int terrain_top(const TerrainConfig& config, int wx, int wy);

// A chunk coordinate triple as one map key: 26 bits each way, 12 up.
inline std::int64_t chunk_key(int cx, int cy, int cz)
{
    const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) & 0x3ffffffull;
    const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cy)) & 0x3ffffffull;
    const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cz)) & 0xfffull;
    return static_cast<std::int64_t>((x << 38) | (y << 12) | z);
}
inline int key_cx(std::int64_t key) { return static_cast<int>(key >> 38); }
inline int key_cy(std::int64_t key)
{
    return static_cast<int>(static_cast<std::int64_t>(static_cast<std::uint64_t>(key) << 26) >> 38);
}
inline int key_cz(std::int64_t key) { return static_cast<int>(static_cast<std::uint64_t>(key) & 0xfffull); }

// Every chunk that has been in the simulated window: the fields that differ from what the generator makes (most
// fields of most chunks do not, and are not kept).
struct StoredField {
    std::string name;
    bool uniform = false;
    float value = 0.0f;
    std::vector<float> values;
};

struct StoredChunk {
    std::vector<StoredField> fields;
};

class ChunkStore {
public:
    [[nodiscard]] const StoredChunk* find(int cx, int cy, int cz) const;
    // The same, to read on another thread: a stored chunk is replaced whole, never changed in place.
    [[nodiscard]] std::shared_ptr<const StoredChunk> share(int cx, int cy, int cz) const;
    void put(int cx, int cy, int cz, StoredChunk chunk);
    // Has anything been written that differs from the generator, in this box of columns?
    [[nodiscard]] bool changed_in(int cx0, int cy0, int cx1, int cy1) const;
    // The columns of this box that do.
    void changed_columns(int cx0, int cy0, int cx1, int cy1, std::vector<std::pair<int, int>>& out) const;
    // Which layers of a column differ from the generator, lowest first.
    [[nodiscard]] std::vector<int> changed_layers(int cx, int cy) const;
    [[nodiscard]] std::size_t size() const { return chunks_.size(); }
    // Bytes held by field data (for the overlay).
    [[nodiscard]] std::size_t bytes() const;
    // Bumped whenever a chunk's stored state changes (a mesh built from the old state is stale).
    [[nodiscard]] std::uint64_t version(int cx, int cy, int cz) const;
    // The same for anything in a whole column.
    [[nodiscard]] std::uint64_t column_version(int cx, int cy) const;
    // Bumped whenever anything is put.
    [[nodiscard]] std::uint64_t counter() const { return counter_; }

private:
    std::map<std::int64_t, std::shared_ptr<const StoredChunk>> chunks_;
    std::map<std::int64_t, std::uint64_t> versions_;
    std::map<std::int64_t, std::uint64_t> column_versions_;
    std::map<std::int64_t, std::vector<int>> changed_; // column -> layers with fields
    std::uint64_t counter_ = 0;
};

} // namespace fieldengine::field

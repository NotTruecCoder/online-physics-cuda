#pragma once

#include "field/Playground.hpp"
#include "field/VoxelMesh.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// THE OPEN WORLD AS MESHES, BUILT OFF THE FRAME. The camera is orthographic, so every voxel on the screen is
// the same size, however far from you it is: level of detail is a matter of how many pixels a voxel covers at
// this zoom, not of distance. Each level doubles the voxel's side and the region it is drawn in, so a region
// is always 16 voxels wide and the screen always holds about the same number of them, at any zoom. Only what
// the camera can see is built, nearest first, on worker threads; a frame only takes up what is ready, and an
// old mesh stays until its replacement has come.
//
// The two nearest levels are made from real chunks, as thick as the ground and what stands on it asks. From level 2
// on a voxel is one column of the map, asked for at that coarseness (WorldMap::far_column): each layer of the map has
// its own way of being far away, so the whole continent can be in view without generating a chunk. A column whose
// chunk the law has touched, or that is in the live window, is read from what is really there.
//
// Live chunks are classified where the field step runs (on_fields), so the frame does not pay for that either.
namespace fieldengine::field {

// One level: a voxel `factor` cells wide and `factor_z` cells tall (height stays finer than width, so hills
// keep their shape), in regions `span` chunks wide.
struct LodLevel {
    int factor = 1;
    int factor_z = 1;
    int span = 1;
};
inline constexpr int kLodLevels = 12;
LodLevel lod_level(int level);
// The coarsest detail that still spends no less than `min_pixels` on a voxel side; `current` is kept while it
// is still about right, so a zoom held at a boundary does not rebuild the world back and forth.
int lod_for_zoom(double pixels_per_metre, double cell_metres, double min_pixels, int current);

// What the camera sees.
struct LodView {
    double pixels_per_metre = 20.0;
    double centre_x = 0.0;     // world metres
    double centre_y = 0.0;
    double radius_metres = 50.0; // nothing farther than this from the centre can be on screen
    std::uint64_t stamp = 0;     // differs whenever what `visible` says may have changed
    // Whether a world-metre box is on screen (with a margin, so walking finds it built). Null = everything.
    std::function<bool(const double lo[3], const double hi[3])> visible;
    std::function<bool(const double lo[3], const double hi[3])> in_sight; // and of that, what is drawn; none: all of it
};

class WorldMesher {
public:
    struct Settings {
        double min_pixels = 5.0; // a voxel's side on screen
        double budget_ms = 3.0;  // meshes taken up per frame
        int threads = 3;
        std::size_t cached_chunks = 4000; // generated chunks kept for their neighbours (37 KB each)
    };

    struct Stats {
        int level = 0;
        std::size_t regions_drawn = 0;
        std::size_t faces = 0;
        std::size_t wanted = 0;
        std::size_t pending = 0;
        std::size_t built_this_frame = 0;
        std::size_t cached = 0;
        double ms = 0.0;
    };

    using Upload = std::function<int(int handle, const ChunkMesh& mesh)>; // old handle or -1 -> handle
    using Destroy = std::function<void(int handle)>;

    WorldMesher();
    explicit WorldMesher(Settings settings);
    ~WorldMesher();
    WorldMesher(const WorldMesher&) = delete;
    WorldMesher& operator=(const WorldMesher&) = delete;

    // The live window as the field step left it. Called where the step ran.
    void on_fields(const FieldStep& step, const TerrainConfig& terrain);

    // Brings the meshes the view wants up to date, within the budget. Never waits for a build.
    void update(const Playground& world, const LodView& view, const Upload& upload, const Destroy& destroy);
    void for_each_drawn(const std::function<void(int handle, int level, int rx, int ry)>& fn) const;
    void clear(const Destroy& destroy);

    // Zoomed out a little, the live window is drawn at full detail over coarser regions: those are to be clipped (by the renderer)
    // inside window_box(), chunk coordinates [x0, x1) x [y0, y1).
    static constexpr int kOverlayMost = 3;
    [[nodiscard]] bool live_overlay() const { return overlay_; }
    [[nodiscard]] std::array<int, 4> window_box() const { return window_box_; }

    // Is the part of the world the view wants fully drawn, at the wanted level, from current voxels?
    [[nodiscard]] bool settled() const { return settled_; }
    [[nodiscard]] const Stats& stats() const { return stats_; }

    // One region's mesh from a world's current state, as a worker would build it (for gates).
    static void build_region(const Playground& world, int level, int rx, int ry, ChunkMesh& out);
    // The same place from whole chunks, however coarse: what the far columns are compared with.
    static void build_region_from_chunks(const Playground& world, int level, int rx, int ry, ChunkMesh& out);

    struct CellsRef {
        std::shared_ptr<const std::vector<VoxelCell>> cells;
        std::uint64_t hash = 0;
    };

private:
    struct Cells {
        std::shared_ptr<const std::vector<VoxelCell>> cells;
        std::uint64_t hash = 0;
        std::uint64_t source = 0; // the store version it came from; kLive for what the law just left
        std::uint64_t used = 0;
    };
    // A chunk that is not simply what the generator makes.
    struct Input {
        int cx = 0;
        int cy = 0;
        int cz = 0;
        std::shared_ptr<const StoredChunk> stored;
        std::uint64_t source = 0;
    };
    struct Job {
        std::uint64_t generation = 0;
        std::int64_t slot = 0;
        int level = 0;
        int rx = 0;
        int ry = 0;
        int distance = 0;
        bool hole = false;
        bool overlay = false; // the live window at full detail over a coarser level
        std::uint64_t signature = 0;
        std::uint64_t plan = 0;
        std::vector<Input> inputs;
    };
    struct Done {
        std::uint64_t generation = 0;
        std::int64_t slot = 0;
        std::uint64_t signature = 0;
        std::uint64_t content = 0;
        std::uint64_t plan = 0;
        ChunkMesh mesh;
    };
    struct Slot {
        int level = 0;
        int rx = 0;
        int ry = 0;
        int handle = -1;
        std::uint64_t signature = 0; // of the chunks it depends on, when it was last asked for
        std::uint64_t content = 0;   // of the voxels the drawn mesh came from
        std::size_t faces = 0;
        bool building = false;
        bool wanted = false;
        bool in_sight = true;
        int low = 0;  // cells: what it can hold, top to bottom
        int high = 0;
    };

    void worker();
    void plan(const Playground& world, const LodView& view);
    CellsRef cells_for(const Input& input);
    void run_job(const Job& job, VoxelBox& fine, VoxelBox& coarse, Done& out);

    Settings settings_;
    TerrainConfig terrain_;
    std::shared_ptr<const WorldMap> map_;
    VoxelPalette palette_;
    Stats stats_;
    bool settled_ = false;
    int level_ = 0;
    bool overlay_ = false;
    std::array<int, 4> window_box_{};
    std::atomic<std::uint64_t> frame_ = 0;
    std::atomic<std::uint64_t> live_generation_ = 0;
    std::uint64_t planned_stamp_ = 0;
    std::uint64_t planned_store_ = 0;
    std::uint64_t planned_live_ = 0;
    std::uint64_t planned_frame_ = 0;
    static constexpr std::uint64_t kFramesBetweenPlans = 8;
    std::uint64_t plan_id_ = 0;
    bool replan_needed_ = false;
    std::size_t asked_ = 0;
    std::uint64_t generation_ = 1;
    std::map<std::int64_t, Slot> slots_;

    mutable std::mutex cells_mutex_;
    std::map<std::int64_t, Cells> cells_;
    // The matter fields as on_fields last saw them, to tell what changed.
    std::vector<std::vector<float>> seen_;
    int seen_cx_ = 0;
    int seen_cy_ = 0;
    int seen_cz_ = 0;

    std::mutex jobs_mutex_;
    std::condition_variable jobs_ready_;
    std::deque<Job> jobs_;
    std::vector<Done> done_;
    bool stopping_ = false;
    std::vector<std::thread> threads_;
};

} // namespace fieldengine::field

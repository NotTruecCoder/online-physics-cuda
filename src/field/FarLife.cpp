// BEYOND THE WINDOW LIFE GOES ON, MORE SIMPLY. Three grains, each cheaper than the last, none of them touching a
// frame, all of them taken up at fixed ticks so the world stays a function of its inputs:
//
//   busy chunks    A chunk that leaves the window hot, burning, molten or full of mana goes on under the same law:
//                  on the CPU, a few at a time, with a ring of its neighbours around it, as far as the turns allow.
//                  What it does to that ring goes to the neighbours, which wake if it matters, so a fire walks on.
//                  It is behind real time when too many are busy, and the far_lag says by how much.
//   far creatures  A creature that leaves walks over the map's columns, once a second: its amble, the ground it can
//                  stand on, nothing else. Far enough away it stops being a body and becomes what the land holds.
//   the land       assets/law/map.law, on the map's nodes: how much life each can feed, how it spreads. New chunks
//                  are peopled from it, so a herd you drove off is missing where you left it.
#include "field/FieldWorker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

namespace fieldengine::field {

namespace {

constexpr int kFarEvery = 15;      // ticks between turns of a busy chunk (a turn is taken up a whole turn later)
constexpr double kFarSlice = 0.34; // law seconds a chunk gets in a turn: a flame splits badly into longer ones, and goes out
constexpr int kFarAtOnce = 6;      // chunks in a turn, none next to another
constexpr double kFarMaxLag = 90.0; // a chunk further behind than this has simply missed that time
constexpr std::size_t kFarMaxBusy = 256;
constexpr int kWalkEvery = 60;     // ticks between far creature steps
constexpr int kMapEvery = 600;     // ticks between steps of the land's law
constexpr int kMapStride = 4;      // map nodes to a cell of it
constexpr double kFarBodyReach = 1500.0; // metres from you; beyond this a creature is only numbers on the map
constexpr std::size_t kFarBodyCap = 3000;

constexpr double kPi = 3.14159265358979323846;

double amble(std::uint32_t id, std::uint64_t epoch)
{
    std::uint64_t x = (static_cast<std::uint64_t>(id) << 32) ^ epoch ^ 0x5a5a5a5aull;
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) * 2.0 * kPi;
}

} // namespace

// A job holds everything it reads, so nothing it does can see the store change.
struct Playground::FarJob {
    int cx = 0;
    int cy = 0;
    int cz = 0;
    double dt = 0.0;
    double until = 0.0;
    std::uint64_t self_version = 0;
    bool map_step = false;
    double map_dt = 0.0;
    std::shared_ptr<const StoredChunk> self;
    struct Side {
        int cx, cy, cz;
        std::shared_ptr<const StoredChunk> stored;
        std::uint64_t version = 0;
        std::optional<StoredChunk> result;
        bool woke = false;
    };
    std::array<Side, 6> sides{};
    StoredChunk chunk;
    bool busy = false;
    bool ran = false;
};

void Playground::start_far_life()
{
    if (!config_.far_life) {
        return;
    }
    // The land's law.
    std::optional<LawSource> source;
    for (const std::string root : {"", "../", "../../"}) {
        std::ifstream in(root + "assets/law/map.law", std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            source = LawSource{root + "assets/law/map.law", text.str()};
            break;
        }
    }
    if (!source) {
        far_errors_.push_back("assets/law/map.law not found: the land has no law of its own");
        return;
    }
    CompileResult compiled = compile_law({*source});
    far_errors_.insert(far_errors_.end(), compiled.errors.begin(), compiled.errors.end());
    if (!compiled.law) {
        return;
    }
    // One cell for every few nodes: herds do not care for a hundred metres.
    const int nodes = WorldMap::kNodes / kMapStride;
    map_solver_ = std::make_unique<Solver>(*compiled.law, GridShape{.nx = nodes, .ny = nodes, .nz = 1, .cell = kMapStride * WorldMap::kNodeCells * config_.terrain.cell});
    FieldSet& f = map_solver_->fields();
    const int capacity = f.find("capacity");
    const int grazers = f.find("grazers");
    const int hunters = f.find("hunters");
    for (int y = 0; y < nodes; ++y) {
        for (int x = 0; x < nodes; ++x) {
            const float c = static_cast<float>(map_->capacity_at_node(x * kMapStride + kMapStride / 2, y * kMapStride + kMapStride / 2));
            const std::size_t i = f.shape().index(x, y, 0);
            f.values(capacity)[i] = c;
            f.values(grazers)[i] = 0.7f * c;
            f.values(hunters)[i] = 0.12f * c;
        }
    }
    map_solver_->refresh_derived();
    map_grazers_ = f.values(grazers);
    map_hunters_ = f.values(hunters);
}

double Playground::map_total(const std::string& field) const
{
    if (!map_solver_) {
        return 0.0;
    }
    const FieldSet& f = std::as_const(*map_solver_).fields();
    const int id = f.find(field.c_str());
    double sum = 0.0;
    if (id >= 0) {
        const std::vector<float>& values = field == "grazers" ? map_grazers_ : (field == "hunters" ? map_hunters_ : f.values(id));
        for (const float v : values) {
            sum += v;
        }
    }
    return sum;
}

double Playground::map_at(const std::string& field, int nx, int ny) const
{
    if (!map_solver_ || nx < 0 || ny < 0 || nx >= WorldMap::kNodes || ny >= WorldMap::kNodes) {
        return 0.0;
    }
    const std::size_t i = static_cast<std::size_t>(ny / kMapStride) * (WorldMap::kNodes / kMapStride) + static_cast<std::size_t>(nx / kMapStride);
    if (field == "grazers") {
        return map_grazers_[i];
    }
    if (field == "hunters") {
        return map_hunters_[i];
    }
    return 0.0;
}

// How likely the land is to have fed a creature of this kind that could start here.
double Playground::feed_chance(const std::string& kind, double wx, double wy) const
{
    if (!map_solver_) {
        return 1.0;
    }
    const double cell = config_.terrain.cell;
    const int nx = static_cast<int>(std::floor((wx / cell + WorldMap::kHalfCells) / WorldMap::kNodeCells));
    const int ny = static_cast<int>(std::floor((wy / cell + WorldMap::kHalfCells) / WorldMap::kNodeCells));
    if (kind == "grazer") {
        return std::clamp(map_at("grazers", nx, ny) * 1.2, 0.0, 1.0);
    }
    if (kind == "hunter") {
        return std::clamp(map_at("hunters", nx, ny) * 6.0, 0.0, 1.0);
    }
    return 1.0;
}

double Playground::far_lag() const
{
    const double now = static_cast<double>(tick_) * kTick;
    double worst = 0.0;
    for (const auto& [key, time] : far_busy_) {
        worst = std::max(worst, now - time);
    }
    return worst;
}

// Is what the law left in this chunk still going to change by itself? What a far chunk is run for: heat, burning,
// melting and magic. Turf, scent and the like settle on their own, and nobody needs them run.
bool Playground::still_busy(const StoredChunk& chunk) const
{
    for (const StoredField& f : chunk.fields) {
        const bool heat = f.name == "T";
        const bool energetic = heat || f.name == "lava" || f.name == "fuel" || f.name == "smoke" || f.name == "alight" || f.name.rfind("mana", 0) == 0;
        if (!energetic) {
            continue;
        }
        const float at_rest = heat ? 290.0f : 0.0f;
        const float enough = heat ? 8.0f : 0.02f;
        if (f.uniform) {
            if (std::abs(f.value - at_rest) > enough) {
                return true;
            }
            continue;
        }
        for (const float v : f.values) {
            if (std::abs(v - at_rest) > enough) {
                return true;
            }
        }
    }
    return false;
}

void Playground::note_busy(int cx, int cy, int cz, const StoredChunk& chunk)
{
    if (config_.far_life && far_busy_.size() < kFarMaxBusy && still_busy(chunk)) {
        far_busy_.try_emplace(chunk_key(cx, cy, cz), static_cast<double>(tick_) * kTick);
    }
}

// One chunk under the law, with a ring of its neighbours around it. Runs off the frame; reads only what the job holds.
void Playground::run_far_job(FarJob& job, Solver& solver)
{
    const TerrainConfig& t = config_.terrain;
    const int n = t.chunk;
    const Law& law = solver_.law();
    FieldSet& f = solver.fields();
    const GridShape& g = f.shape();
    const auto put = [&](const ChunkData& data, int lx0, int ly0, int lz0) {
        // Only the part of the chunk that falls in the lattice.
        for (std::size_t id = 0; id < law.fields.size(); ++id) {
            const FieldDecl& decl = law.fields[id];
            if (decl.derived) {
                continue;
            }
            const std::vector<float>* from = data.find(decl.name);
            std::vector<float>& to = f.values(static_cast<int>(id));
            for (int z = 0; z < n; ++z) {
                for (int y = 0; y < n; ++y) {
                    for (int x = 0; x < n; ++x) {
                        const int ax = lx0 + x;
                        const int ay = ly0 + y;
                        const int az = lz0 + z;
                        if (g.inside(ax, ay, az)) {
                            const std::size_t k = (static_cast<std::size_t>(z) * n + static_cast<std::size_t>(y)) * n + static_cast<std::size_t>(x);
                            to[g.index(ax, ay, az)] = from != nullptr ? (*from)[k] : static_cast<float>(decl.init);
                        }
                    }
                }
            }
        }
    };
    for (std::size_t id = 0; id < law.fields.size(); ++id) {
        if (!law.fields[id].derived) {
            std::vector<float>& to = f.values(static_cast<int>(id));
            std::fill(to.begin(), to.end(), static_cast<float>(law.fields[id].init));
        }
    }
    ChunkData data;
    read_chunk_state(t, job.cx, job.cy, job.cz, job.self.get(), data);
    put(data, 1, 1, 1);
    for (const FarJob::Side& side : job.sides) {
        ChunkData other;
        read_chunk_state(t, side.cx, side.cy, side.cz, side.stored.get(), other);
        put(other, 1 + (side.cx - job.cx) * n, 1 + (side.cy - job.cy) * n, 1 + (side.cz - job.cz) * n);
    }
    solver.refresh_derived();
    const FieldSet before = std::as_const(solver).fields();
    solver.step(job.dt);
    const FieldSet& after = std::as_const(solver).fields();
    job.chunk = capture_chunk(after, 1, 1, 1, job.cx, job.cy, job.cz);
    job.busy = still_busy(job.chunk);
    const auto is_matter = [](const std::string& name) {
        return name == "solid" || name == "rubble" || name == "lava" || name == "water" || name == "ice" || name == "wood" || name == "leaf";
    };
    std::vector<int> matter_ids;
    for (std::size_t id = 0; id < law.fields.size(); ++id) {
        if (is_matter(law.fields[id].name)) {
            matter_ids.push_back(static_cast<int>(id));
        }
    }
    // What it passed to the ring is the neighbours' now: heat and matter that arrived there.
    for (FarJob::Side& side : job.sides) {
        const int dx = side.cx - job.cx;
        const int dy = side.cy - job.cy;
        const int dz = side.cz - job.cz;
        const int gx = dx > 0 ? n + 1 : (dx < 0 ? 0 : -1);
        const int gy = dy > 0 ? n + 1 : (dy < 0 ? 0 : -1);
        const int gz = dz > 0 ? n + 1 : (dz < 0 ? 0 : -1);
        ChunkData state;
        bool touched = false;
        for (std::size_t id = 0; id < law.fields.size(); ++id) {
            const FieldDecl& decl = law.fields[id];
            if (decl.derived) {
                continue;
            }
            const std::vector<float>& a = before.values(static_cast<int>(id));
            const std::vector<float>& b = after.values(static_cast<int>(id));
            const float tolerance = decl.name == "T" ? 2.0f : 0.005f;
            std::vector<float>* into = nullptr;
            for (int z = 1; z <= n; ++z) {
                for (int y = 1; y <= n; ++y) {
                    for (int x = 1; x <= n; ++x) {
                        // The face of the ring on this side, one cell from the chunk.
                        const int ax = gx >= 0 ? gx : x;
                        const int ay = gy >= 0 ? gy : y;
                        const int az = gz >= 0 ? gz : z;
                        if ((gx >= 0 && x != 1) || (gy >= 0 && y != 1) || (gz >= 0 && z != 1)) {
                            continue;
                        }
                        const std::size_t i = g.index(ax, ay, az);
                        // Only what arrived: a loss here is the ring's own life going on, which the neighbour
                        // lives through in its own turn, and taking it too would burn its edge twice.
                        const float delta = b[i] - a[i];
                        if (delta <= tolerance) {
                            continue;
                        }
                        // Stone melting where it stands has not arrived from anywhere: the neighbour melts it in its own
                        // turn, and passing on the lava and not the loss of the stone would make matter out of nothing.
                        if (is_matter(decl.name)) {
                            float was = 0.0f;
                            float now = 0.0f;
                            for (const int m : matter_ids) {
                                was += before.values(m)[i];
                                now += after.values(m)[i];
                            }
                            if (now - was <= tolerance) {
                                continue;
                            }
                        }
                        if (!touched) {
                            read_chunk_state(t, side.cx, side.cy, side.cz, side.stored.get(), state);
                            touched = true;
                        }
                        if (into == nullptr) {
                            const std::size_t cells = static_cast<std::size_t>(n) * n * n;
                            into = state.find(decl.name);
                            if (into == nullptr) {
                                into = &state.add(decl.name, cells, static_cast<float>(decl.init));
                            }
                        }
                        // In the neighbour's own cells.
                        const int ox = dx > 0 ? 0 : (dx < 0 ? n - 1 : x - 1);
                        const int oy = dy > 0 ? 0 : (dy < 0 ? n - 1 : y - 1);
                        const int oz = dz > 0 ? 0 : (dz < 0 ? n - 1 : z - 1);
                        const std::size_t k = (static_cast<std::size_t>(oz) * n + static_cast<std::size_t>(oy)) * n + static_cast<std::size_t>(ox);
                        (*into)[k] = b[i];
                    }
                }
            }
        }
        if (!touched) {
            continue;
        }
        // As the store keeps it: what differs from the generator.
        ChunkData generated;
        generate_chunk(t, side.cx, side.cy, side.cz, generated, nullptr);
        StoredChunk out;
        for (auto& [name, values] : state.fields) {
            const int id = law.field_id(name);
            if (id < 0 || law.fields[static_cast<std::size_t>(id)].derived) {
                continue;
            }
            const std::vector<float>* made = generated.find(name);
            const float init = static_cast<float>(law.fields[static_cast<std::size_t>(id)].init);
            const bool same = made != nullptr ? values == *made
                                              : std::all_of(values.begin(), values.end(), [&](float v) { return v == init; });
            if (!same) {
                out.fields.push_back(StoredField{.name = name, .values = values});
            }
        }
        side.woke = still_busy(out);
        side.result = std::move(out);
    }
    job.ran = true;
}

// One turn: the chunks side by side (each runs alone, so inner loops stay on its thread).
void Playground::run_far_turn()
{
    const auto started = std::chrono::steady_clock::now();
    const TerrainConfig& t = config_.terrain;
    while (far_solvers_.size() < far_jobs_.size()) {
        far_solvers_.push_back(std::make_unique<Solver>(solver_.law(), GridShape{.nx = t.chunk + 2, .ny = t.chunk + 2, .nz = t.chunk + 2, .cell = t.cell}));
    }
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(far_jobs_.size()); ++k) {
        FarJob& job = *far_jobs_[static_cast<std::size_t>(k)];
        if (job.map_step) {
            if (map_solver_) {
                map_solver_->step(job.map_dt);
            }
        } else {
            run_far_job(job, *far_solvers_[static_cast<std::size_t>(k)]);
        }
    }
    far_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

// A finished turn, at its fixed tick. What the window took in the meantime is the window's; a neighbour that changed
// since the job read it keeps its newer state.
void Playground::finish_far_job()
{
    if (far_jobs_.empty()) {
        return;
    }
    if (config_.async_fields && far_worker_) {
        far_worker_->wait();
    } else {
        run_far_turn();
    }
    for (const std::shared_ptr<FarJob>& running : far_jobs_) {
        FarJob& job = *running;
        if (job.map_step) {
            if (map_solver_) {
                const FieldSet& f = std::as_const(*map_solver_).fields();
                map_grazers_ = f.values(f.find("grazers"));
                map_hunters_ = f.values(f.find("hunters"));
                for (const Violation& v : map_solver_->violations()) {
                    const bool seen = std::any_of(violations_.begin(), violations_.end(), [&](const Violation& old) {
                        return old.kind == v.kind && old.term == v.term && old.field == v.field;
                    });
                    if (!seen && violations_.size() < 64) {
                        violations_.push_back(v);
                    }
                }
            }
            continue;
        }
        if (!job.ran) {
            continue;
        }
        ++far_runs_;
        const std::int64_t key = chunk_key(job.cx, job.cy, job.cz);
        if (!in_window(job.cx, job.cy, job.cz) && store_.version(job.cx, job.cy, job.cz) == job.self_version) {
            store_.put(job.cx, job.cy, job.cz, std::move(job.chunk));
            const auto found = far_busy_.find(key);
            if (found != far_busy_.end()) {
                if (job.busy) {
                    found->second = job.until;
                } else {
                    far_busy_.erase(found);
                }
            }
        } else {
            far_busy_.erase(key);
        }
        for (FarJob::Side& side : job.sides) {
            if (!side.result || in_window(side.cx, side.cy, side.cz) || store_.version(side.cx, side.cy, side.cz) != side.version) {
                continue;
            }
            store_.put(side.cx, side.cy, side.cz, std::move(*side.result));
            if (side.woke && far_busy_.size() < kFarMaxBusy) {
                far_busy_.try_emplace(chunk_key(side.cx, side.cy, side.cz), job.until - job.dt);
            }
        }
    }
    far_jobs_.clear();
}

// Over the map's columns: its amble, the ground it can stand on. Anything else about it waits until it is back.
void Playground::walk_far_bodies()
{
    if (far_bodies_.empty()) {
        return;
    }
    const TerrainConfig& t = config_.terrain;
    const Vec3 me = player().position + origin();
    const Law& law = solver_.law();
    std::vector<FarBody> kept;
    kept.reserve(far_bodies_.size());
    const double dt = kWalkEvery * kTick;
    for (FarBody& b : far_bodies_) {
        const int def = law.creature_id(b.kind);
        if (def < 0) {
            continue; // a kind the law no longer has
        }
        const double away = std::hypot(b.x - me.x, b.y - me.y);
        if (away > kFarBodyReach || kept.size() >= kFarBodyCap) {
            continue; // only numbers on the map from here on
        }
        const CreatureDef& d = law.creatures[static_cast<std::size_t>(def)];
        const std::uint64_t epoch = tick_ / static_cast<std::uint64_t>(kWalkEvery * 8);
        b.heading = amble(b.id, epoch);
        const double stride = d.speed * d.wander * dt;
        for (int attempt = 0; attempt < 4; ++attempt) {
            const double a = b.heading + attempt * 1.7;
            const double nx = b.x + std::cos(a) * stride;
            const double ny = b.y + std::sin(a) * stride;
            const int wx = static_cast<int>(std::floor(nx / t.cell));
            const int wy = static_cast<int>(std::floor(ny / t.cell));
            const Column c = map_->column(wx, wy);
            const double z = (c.top + 1) * t.cell + d.radius;
            if (c.water >= 0 || std::abs(z - b.z) > stride * 1.5 + 0.6) {
                continue; // water, or a cliff: it turns
            }
            b.x = nx;
            b.y = ny;
            b.z = z;
            break;
        }
        kept.push_back(std::move(b));
    }
    far_bodies_ = std::move(kept);
}

// The ones the window has come to are bodies again.
void Playground::bring_far_bodies()
{
    const TerrainConfig& t = config_.terrain;
    const double chunk_m = t.chunk * t.cell;
    const Law& law = solver_.law();
    const Vec3 o = origin();
    std::vector<FarBody> kept;
    for (FarBody& b : far_bodies_) {
        const int cx = static_cast<int>(std::floor(b.x / chunk_m));
        const int cy = static_cast<int>(std::floor(b.y / chunk_m));
        const int cz = static_cast<int>(std::floor(b.z / chunk_m));
        const int def = law.creature_id(b.kind);
        if (def < 0) {
            continue;
        }
        if (!in_window(cx, cy, cz) || creatures_.size() >= static_cast<std::size_t>(config_.max_creatures)) {
            kept.push_back(std::move(b));
            continue;
        }
        Creature c = make_creature(def, Vec3{b.x - o.x, b.y - o.y, b.z - o.z});
        c.id = b.id;
        c.heading = b.heading;
        c.brood = b.brood;
        c.body.hp = b.hp;
        const CreatureDef& d = law.creatures[static_cast<std::size_t>(def)];
        for (const auto& [name, value] : b.locals) {
            const auto it = std::find(d.locals.begin(), d.locals.end(), name);
            if (it != d.locals.end()) {
                c.locals[static_cast<std::size_t>(it - d.locals.begin())] = value;
            }
        }
        creatures_.push_back(std::move(c));
    }
    far_bodies_ = std::move(kept);
}

// Called at field ticks, after the field step has been taken up: nothing else is touching the store now.
void Playground::far_tick()
{
    if (!config_.open_world || !config_.far_life) {
        return;
    }
    if (tick_ >= far_next_walk_) {
        far_next_walk_ = tick_ + kWalkEvery;
        walk_far_bodies();
        bring_far_bodies();
    }
    if (!far_jobs_.empty() && tick_ < far_due_) {
        return;
    }
    finish_far_job();
    if (tick_ < far_next_turn_) {
        return;
    }
    far_next_turn_ = tick_ + kFarEvery;
    if (tick_ >= far_next_map_ && map_solver_) {
        far_next_map_ = tick_ + kMapEvery;
        auto job = std::make_shared<FarJob>();
        job->map_step = true;
        job->map_dt = kMapEvery * kTick;
        far_jobs_.push_back(std::move(job));
    } else {
        // The busy chunks furthest behind, none beside another: what each does to its ring must reach the next as is.
        const double now = static_cast<double>(tick_) * kTick;
        std::vector<std::pair<double, std::int64_t>> behind;
        for (auto& [key, time] : far_busy_) {
            time = std::max(time, now - kFarMaxLag);
            if (now - time >= 0.2) {
                behind.emplace_back(time, key);
            }
        }
        std::sort(behind.begin(), behind.end());
        const TerrainConfig& t = config_.terrain;
        const int layers = t.nz / t.chunk;
        for (const auto& [time, key] : behind) {
            if (static_cast<int>(far_jobs_.size()) >= kFarAtOnce) {
                break;
            }
            const int cx = key_cx(key);
            const int cy = key_cy(key);
            const int cz = key_cz(key);
            const bool beside = std::any_of(far_jobs_.begin(), far_jobs_.end(), [&](const std::shared_ptr<FarJob>& other) {
                return std::abs(other->cx - cx) + std::abs(other->cy - cy) + std::abs(other->cz - cz) <= 2;
            });
            if (beside) {
                continue;
            }
            auto job = std::make_shared<FarJob>();
            job->cx = cx;
            job->cy = cy;
            job->cz = cz;
            job->dt = std::min(now - time, kFarSlice);
            job->until = time + job->dt;
            job->self = store_.share(cx, cy, cz);
            job->self_version = store_.version(cx, cy, cz);
            static constexpr int kSides[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            for (int k = 0; k < 6; ++k) {
                FarJob::Side& side = job->sides[static_cast<std::size_t>(k)];
                side.cx = cx + kSides[k][0];
                side.cy = cy + kSides[k][1];
                side.cz = std::clamp(cz + kSides[k][2], 0, layers - 1);
                side.stored = store_.share(side.cx, side.cy, side.cz);
                side.version = store_.version(side.cx, side.cy, side.cz);
            }
            far_jobs_.push_back(std::move(job));
        }
    }
    if (far_jobs_.empty()) {
        return;
    }
    far_due_ = tick_ + kFarEvery;
    if (config_.async_fields) {
        if (far_worker_ == nullptr) {
            far_worker_ = std::make_shared<FieldWorker>();
        }
        far_worker_->submit([this] { run_far_turn(); });
    }
}

} // namespace fieldengine::field

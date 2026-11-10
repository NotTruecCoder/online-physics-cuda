#include "world/Playground.hpp"

#include "world/FieldWorker.hpp"
#include "field/mind/WorldModel.hpp"
#include "field/Parallel.hpp"
#include "world/Ray.hpp"

#include <algorithm>
#include <iostream>
#include <cstring>
#include <utility>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

namespace fieldengine::field {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<std::string> words(const std::string& line)
{
    std::vector<std::string> out;
    std::istringstream in(line.substr(0, line.find('#')));
    std::string word;
    while (in >> word) {
        out.push_back(word);
    }
    return out;
}

std::optional<std::string> read_text(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

void mix(std::uint64_t& hash, const void* data, std::size_t bytes)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash = (hash ^ p[i]) * 1099511628211ull;
    }
}

void mix(std::uint64_t& hash, double value) { mix(hash, &value, sizeof(value)); }
void mix(std::uint64_t& hash, Vec3 v)
{
    mix(hash, v.x);
    mix(hash, v.y);
    mix(hash, v.z);
}

Vec3 horizontal(Vec3 v) { return Vec3{v.x, v.y, 0.0}; }

// A fixed, well-mixed number from a creature and a moment: where it decides to amble next.
double wander_angle(std::uint32_t id, std::uint64_t epoch)
{
    std::uint64_t x = (static_cast<std::uint64_t>(id) << 32) ^ epoch;
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) * 2.0 * kPi;
}

} // namespace

std::vector<Pour> parse_pours(const std::string& text, std::vector<std::string>& errors)
{
    std::vector<Pour> pours;
    std::istringstream stream(text);
    std::string line;
    int number = 0;
    while (std::getline(stream, line)) {
        ++number;
        const std::vector<std::string> w = words(line);
        if (w.empty()) {
            continue;
        }
        const std::string where = "pours.law:" + std::to_string(number) + ": ";
        if (w[0] != "pour" || w.size() < 2) {
            errors.push_back(where + "expected 'pour NAME key value ...'");
            continue;
        }
        Pour pour{.name = w[1]};
        // `about` takes the rest of the line.
        std::size_t keys_end = w.size();
        for (std::size_t i = 2; i < w.size(); ++i) {
            if (w[i] == "about") {
                keys_end = i;
                for (std::size_t k = i + 1; k < w.size(); ++k) {
                    pour.about += (k > i + 1 ? " " : "") + w[k];
                }
                break;
            }
        }
        bool bad = false;
        for (std::size_t i = 2; i < keys_end && !bad;) {
            const std::string& key = w[i];
            if (i + 1 >= keys_end) {
                bad = true;
                break;
            }
            const std::string& value = w[i + 1];
            std::size_t step = 2; // a colour takes three words, everything else one
            try {
                if (key == "field") {
                    pour.field = value;
                } else if (key == "also") {
                    pour.also = value;
                } else if (key == "also_amount") {
                    pour.also_amount = std::stod(value);
                } else if (key == "shape") {
                    static const std::pair<const char*, Pour::Shape> kShapes[] = {{"ball", Pour::Shape::Ball}, {"wall", Pour::Shape::Wall},
                        {"beam", Pour::Shape::Beam}, {"ring", Pour::Shape::Ring}, {"pillar", Pour::Shape::Pillar}, {"cut", Pour::Shape::Cut}};
                    bad = true;
                    for (const auto& [name, shape] : kShapes) {
                        if (value == name) {
                            pour.shape = shape;
                            bad = false;
                        }
                    }
                } else if (key == "colour") {
                    if (i + 3 >= keys_end) {
                        bad = true;
                    } else {
                        for (int c = 0; c < 3; ++c) {
                            pour.colour[c] = std::stof(w[i + 1 + static_cast<std::size_t>(c)]);
                        }
                        step = 4;
                    }
                } else if (key == "amount") {
                    pour.amount = std::stod(value);
                } else if (key == "radius") {
                    pour.radius = std::stod(value);
                } else if (key == "width") {
                    pour.width = std::stod(value);
                } else if (key == "height") {
                    pour.height = std::stod(value);
                } else if (key == "thickness") {
                    pour.thickness = std::stod(value);
                } else if (key == "range") {
                    pour.range = std::stod(value);
                } else if (key == "flat") {
                    pour.flat = value == "1";
                } else {
                    bad = true;
                }
            } catch (...) {
                bad = true;
            }
            i += step;
        }
        if (bad || pour.field.empty() || (!pour.also.empty() && pour.also_amount <= 0.0)) {
            errors.push_back(where + "malformed pour '" + pour.name + "' (needs field FIELD and known keys)");
            continue;
        }
        pours.push_back(pour);
    }
    return pours;
}

Playground::Playground(Solver solver, std::vector<Pour> pours, Config config)
    : solver_(std::move(solver)), pours_(std::move(pours)), config_(config)
{
    bind_law();
    if (config_.gpu_fields) {
        std::string why;
        solver_.use_gpu(why);
        solver_.check_contracts_every(config_.law_checks_every);
    }
    if (config_.async_fields) {
        sync_front();
    }
    bodies_.push_back(PlayBody{.name = "You", .radius = 0.4, .hp = 100.0, .hp_max = 100.0, .controlled = true});
    bodies_.push_back(PlayBody{.name = "Target", .radius = 0.45, .hp = 300.0, .hp_max = 300.0});
}

Solver& Playground::solver()
{
    settle();
    front_stale_ = config_.async_fields;
    return solver_;
}

void Playground::set_param(const std::string& name, double value)
{
    if (config_.async_fields && stepping_) {
        pending_params_.emplace_back(name, value);
        return;
    }
    solver_.set_param(name, value);
}

void Playground::set_field_listener(FieldListener listener)
{
    settle();
    listener_ = std::move(listener);
    if (listener_) {
        const std::vector<char> everything(fields().count(), 1);
        listener_(FieldStep{std::as_const(solver_).fields(), everything, tick_, origin_cx_, origin_cy_, origin_cz_});
    }
}

void Playground::sync_front()
{
    if (!config_.async_fields) {
        return;
    }
    const FieldSet& state = std::as_const(solver_).fields();
    if (front_.count() != state.count()) {
        front_ = state;
        spare_ = state;
    } else {
        // The spare is only ever written whole by the worker, so only the front needs the copy.
        for (std::size_t id = 0; id < state.count(); ++id) {
            const std::vector<float>& from = state.values(static_cast<int>(id));
            std::vector<float>& to = front_.values(static_cast<int>(id));
            parallel_blocks(from.size(), [&](std::size_t first, std::size_t end) {
                std::copy(from.begin() + static_cast<std::ptrdiff_t>(first), from.begin() + static_cast<std::ptrdiff_t>(end),
                    to.begin() + static_cast<std::ptrdiff_t>(first));
            });
        }
    }
    front_stale_ = false;
    changed_now_.assign(state.count(), 1); // the spare has none of it
    changed_before_.assign(state.count(), 0);
}

void Playground::add_to_cell(int field, std::size_t cell, float delta)
{
    if (config_.async_fields) {
        edits_.push_back(CellEdit{field, cell, delta});
        front_.values(field)[cell] += delta;
    } else {
        solver_.edit_cell(field, cell) += delta;
    }
}

void Playground::note_violations()
{
    for (const Violation& v : solver_.violations()) {
        const bool seen = std::any_of(violations_.begin(), violations_.end(), [&](const Violation& old) {
            return old.kind == v.kind && old.term == v.term && old.field == v.field;
        });
        if (!seen && violations_.size() < 64) {
            violations_.push_back(v);
            std::cerr << "Law broke a contract: [" << v.term << "] " << v.field << ": " << v.detail << "\n"; // for whoever is working on the law: F1 counts them
        }
    }
}

void Playground::settle()
{
    if (!config_.async_fields) {
        return;
    }
    const bool arrived = stepping_;
    int moved_x = 0; // what was written while the window moved was written in the old lattice's cells
    int moved_y = 0;
    int moved_z = 0;
    if (stepping_) {
        worker_->wait();
        stepping_ = false;
        std::swap(front_, spare_);
        note_violations();
        if (moving_) {
            moved_x = (moving_->x - moving_->old_x) * config_.terrain.chunk;
            moved_y = (moving_->y - moving_->old_y) * config_.terrain.chunk;
            moved_z = (moving_->z - moving_->old_z) * config_.terrain.chunk;
            finish_move(*moving_);
            moving_.reset();
        }
    }
    // What was written while the step ran lands on top of its result. (A taking that the step has since
    // made impossible stops at zero.)
    auto land = [](float& v, float delta) {
        v = delta < 0.0f ? std::max(0.0f, v + delta) : v + delta;
    };
    const GridShape& g = front_.shape();
    for (const CellEdit& edit : edits_) {
        std::size_t cell = edit.cell;
        if (moved_x != 0 || moved_y != 0 || moved_z != 0) {
            const std::size_t plane = static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny);
            const int z = static_cast<int>(cell / plane) - moved_z;
            const int x = static_cast<int>(cell % plane % static_cast<std::size_t>(g.nx)) - moved_x;
            const int y = static_cast<int>(cell % plane / static_cast<std::size_t>(g.nx)) - moved_y;
            if (!g.inside(x, y, z)) {
                continue; // it left with its chunk
            }
            cell = g.index(x, y, z);
        }
        land(solver_.edit_cell(edit.field, cell), edit.delta);
        changed_now_[static_cast<std::size_t>(edit.field)] = 1;
        if (arrived) {
            land(front_.values(edit.field)[cell], edit.delta);
        }
    }
    edits_.clear();
    for (const auto& [name, value] : pending_params_) {
        solver_.set_param(name, value);
    }
    pending_params_.clear();
}

void Playground::bind_law()
{
    // A kind with a plan line gets a world model of its own (what it learned is lost with a changed law).
    minds_.assign(solver_.law().creatures.size(), nullptr);
    for (std::size_t d = 0; d < solver_.law().creatures.size(); ++d) {
        if (solver_.law().creatures[d].plan) {
            minds_[d] = std::make_shared<WorldModel>(solver_.law(), static_cast<int>(d));
        }
    }
    occ_ = solver_.field("occ");
    sdf_ = solver_.field("sdf");
    harm_ = solver_.field("harm");
    push_[0] = solver_.field("push_x");
    push_[1] = solver_.field("push_y");
    push_[2] = solver_.field("push_z");
    const Law& law = solver_.law();
    // How a body moves is the law's too, if it says.
    const auto from_law = [&](const char* name, double& value) {
        for (std::size_t i = 0; i < law.param_names.size(); ++i) {
            if (law.param_names[i] == name) {
                value = law.param_values[i];
            }
        }
    };
    from_law("body_gravity", config_.gravity);
    from_law("body_fall_limit", config_.fall_limit);
    from_law("body_walk", config_.walk_speed);
    from_law("body_jump", config_.jump_speed);
    from_law("body_air_control", config_.air_control);
    const int player = law.creature_id("player");
    if (player != player_def_ || player_locals_.empty()) {
        // The player's own states, kept by name across a reload.
        std::vector<double> locals;
        if (player >= 0) {
            const CreatureDef& def = law.creatures[static_cast<std::size_t>(player)];
            locals = def.init;
        }
        player_locals_ = std::move(locals);
    }
    player_def_ = player;
}

std::optional<Playground> Playground::create(const std::vector<LawSource>& laws, const std::string& pours_text,
    std::vector<std::string>& errors, Config config)
{
    CompileResult compiled = compile_law(laws);
    errors.insert(errors.end(), compiled.errors.begin(), compiled.errors.end());
    std::vector<Pour> pours = parse_pours(pours_text, errors);
    if (!compiled.law) {
        return std::nullopt;
    }
    // The playground needs these from the law; a law without them is a law for something else.
    for (const char* needed : {"occ", "sdf", "T"}) {
        if (compiled.law->field_id(needed) < 0) {
            errors.push_back(std::string("the law has no '") + needed + "' field; the playground stands on it");
        }
    }
    for (const Pour& pour : pours) {
        for (const std::string& field : {pour.field, pour.also}) {
            if (!field.empty() && compiled.law->field_id(field) < 0) {
                errors.push_back("pour '" + pour.name + "' pours into '" + field + "', which no law declares");
            }
        }
    }
    if (!errors.empty()) {
        return std::nullopt;
    }
    if (config.open_world) {
        const int side = config.window_chunks * config.terrain.chunk;
        config.shape = GridShape{.nx = side, .ny = side, .nz = config.window_layers * config.terrain.chunk, .cell = config.terrain.cell};
    }
    std::shared_ptr<const WorldMap> map;
    if (config.open_world) {
        map = WorldMap::of(config.terrain);
        errors.insert(errors.end(), map->errors().begin(), map->errors().end());
        if (!errors.empty()) {
            return std::nullopt;
        }
    }
    Playground ground(Solver(*compiled.law, config.shape), std::move(pours), config);
    ground.map_ = std::move(map);
    ground.sources_ = laws;
    ground.pours_text_ = pours_text;
    if (config.open_world) {
        ground.start_far_life();
    }
    return ground;
}

bool Playground::read_default_law(std::vector<LawSource>& laws, std::string& pours_text, std::string& pours_path,
    std::vector<std::string>& errors)
{
    for (const std::string root : {"", "../", "../../"}) {
        const std::optional<LawSource> world = read_law_file(root + "assets/law/world.law");
        const std::optional<LawSource> magic = read_law_file(root + "assets/law/magic.law");
        const std::optional<std::string> pours = read_text(root + "assets/law/pours.law");
        if (world && magic && pours) {
            laws = {*world, *magic};
            if (std::optional<LawSource> life = read_law_file(root + "assets/law/life.law")) {
                laws.push_back(*life);
            }
            if (std::optional<LawSource> people = read_law_file(root + "assets/law/people.law")) {
                laws.push_back(*people);
            }
            pours_text = *pours;
            pours_path = root + "assets/law/pours.law";
            return true;
        }
    }
    errors.push_back("assets/law/{world,magic,pours}.law not found");
    return false;
}

std::optional<Playground> Playground::load_default(std::vector<std::string>& errors, bool open_world)
{
    std::vector<LawSource> laws;
    std::string spells;
    std::string pours_path;
    if (!read_default_law(laws, spells, pours_path, errors)) {
        return std::nullopt;
    }
    Config config;
    config.open_world = open_world;
    std::optional<Playground> ground = create(laws, spells, errors, config);
    if (ground) {
        if (open_world) {
            ground->build_open_world();
        } else {
            ground->build_arena();
        }
    }
    return ground;
}

void Playground::build_arena()
{
    FieldSet& f = solver_.fields();
    const GridShape& g = f.shape();
    const int solid = solver_.field("solid");
    const int material = solver_.field("material");
    const int water = solver_.field("water");
    auto fill = [&](int id, int x0, int x1, int y0, int y1, int z0, int z1, float value) {
        if (id < 0) {
            return;
        }
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (g.inside(x, y, z)) {
                        f.values(id)[g.index(x, y, z)] = value;
                    }
                }
            }
        }
    };
    auto rock = [&](int x0, int x1, int y0, int y1, int z0, int z1, float colour) {
        fill(solid, x0, x1, y0, y1, z0, z1, 1.0f);
        fill(material, x0, x1, y0, y1, z0, z1, colour);
    };
    const int top = g.nx - 1;
    rock(0, top, 0, g.ny - 1, 0, 1, 1.0f);   // the floor: its top is at z = 1 m
    rock(30, 31, 8, 39, 2, 7, 3.0f);          // a long wall, 1 m thick, 3 m tall, at x = 15..16 m
    rock(12, 14, 30, 32, 2, 9, 2.0f);         // a pillar
    for (int s = 0; s < 3; ++s) {             // steps, one voxel each
        rock(6 + s, 6 + s, 36, 44, 2, 2 + s, 2.0f);
    }
    rock(18, 26, 36, 44, 2, 2, 2.0f);         // a basin: a one-voxel rim...
    fill(solid, 19, 25, 37, 43, 2, 2, 0.0f);
    fill(water, 19, 25, 37, 43, 2, 2, 1.0f);  // ...full of water
    solver_.refresh_derived();
    sync_front();

    const double floor_top = 2.0 * g.cell;
    player() = PlayBody{.name = "You", .position = Vec3{5.0, 5.0, floor_top + 0.42}, .radius = 0.4, .hp = 100.0,
        .hp_max = 100.0, .controlled = true};
    target() = PlayBody{.name = "Target", .position = Vec3{11.0, 9.0, floor_top + 0.47}, .radius = 0.45, .hp = 300.0,
        .hp_max = 300.0};
    creatures_.clear();
    note("Proving ground: stone, water, a wall to break, a target to burn. Everything here is fields.");
    start_goods();
}

void Playground::build_ring()
{
    FieldSet& f = solver_.fields();
    const GridShape& g = f.shape();
    const int solid = solver_.field("solid");
    const int material = solver_.field("material");
    const double mid_x = 0.5 * (g.nx - 1);
    const double mid_y = 0.5 * (g.ny - 1);
    const double rim = 0.5 * std::min(g.nx, g.ny) - 1.5;
    for (int z = 0; z < g.nz; ++z) {
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x) {
                const double d = std::hypot(x - mid_x, y - mid_y);
                const double pa = std::hypot(x - mid_x - rim * 0.45, y - mid_y);
                const double pb = std::hypot(x - mid_x + rim * 0.45, y - mid_y);
                const bool floor = z <= 1;
                const bool wall = d >= rim && z <= 7;
                const bool pillar = (pa < 2.0 || pb < 2.0) && z <= 8;
                if (floor || wall || pillar) {
                    const std::size_t i = g.index(x, y, z);
                    if (solid >= 0) {
                        f.values(solid)[i] = 1.0f;
                    }
                    if (material >= 0) {
                        f.values(material)[i] = floor ? ((x / 4 + y / 4) % 2 == 0 ? 2.0f : 3.0f) : 1.0f;
                    }
                }
            }
        }
    }
    solver_.refresh_derived();
    sync_front();
    const double floor_top = 2.0 * g.cell;
    player() = PlayBody{.name = "You", .position = Vec3{(mid_x + 0.5) * g.cell, (mid_y - rim * 0.55) * g.cell, floor_top + 0.42}, .radius = 0.4, .hp = 100.0, .hp_max = 100.0, .controlled = true};
    target() = PlayBody{.name = "Target", .position = Vec3{1.0, 1.0, floor_top + 0.47}, .radius = 0.45, .hp = 300.0, .hp_max = 300.0, .parked = true};
    creatures_.clear();
    start_goods();
}

bool Playground::ring_spawn(const std::string& kind)
{
    const int def = solver_.law().creature_id(kind);
    if (def < 0) {
        return false;
    }
    const GridShape& g = fields().shape();
    const double a = 1.0 + 1.9 * static_cast<double>(creatures_.size()) + static_cast<double>(tick_) * 0.13;
    const double r = 0.35 * std::min(g.nx, g.ny) * g.cell;
    Creature c = make_creature(def, Vec3{0.5 * g.nx * g.cell + r * std::sin(a), 0.5 * g.ny * g.cell + r * std::cos(a), 2.0 * g.cell + solver_.law().creatures[static_cast<std::size_t>(def)].radius + 0.2});
    c.riled_left = 1e9; // it came here for this
    creatures_.push_back(std::move(c));
    return true;
}

void Playground::build_open_world()
{
    const TerrainConfig& t = config_.terrain;
    int home_x = 0;
    int home_y = 0;
    map_->home(home_x, home_y);
    const int home_cx = static_cast<int>(std::floor(static_cast<double>(home_x) / t.chunk));
    const int home_cy = static_cast<int>(std::floor(static_cast<double>(home_y) / t.chunk));
    const int home_cz = (terrain_top(t, home_x, home_y) + 1) / t.chunk;
    origin_cx_ = home_cx - config_.window_chunks / 2;
    origin_cy_ = home_cy - config_.window_chunks / 2;
    origin_cz_ = std::clamp(home_cz - config_.window_layers / 2, 0, t.nz / t.chunk - config_.window_layers);
    store_ = ChunkStore{};
    creatures_.clear();
    far_bodies_.clear();
    far_busy_.clear();
    for (int cz = origin_cz_; cz < origin_cz_ + config_.window_layers; ++cz) {
        for (int cy = origin_cy_; cy < origin_cy_ + config_.window_chunks; ++cy) {
            for (int cx = origin_cx_; cx < origin_cx_ + config_.window_chunks; ++cx) {
                load_chunk(cx, cy, cz);
            }
        }
    }
    solver_.refresh_derived();
    sync_front();
    ++window_version_;

    const Vec3 o = origin();
    const int top = terrain_top(t, home_x, home_y);
    home_ = Vec3{(home_x + 0.5) * t.cell, (home_y + 0.5) * t.cell, (top + 1) * t.cell + 0.45};
    player() = PlayBody{.name = "You", .position = home_ - o, .radius = 0.4, .hp = 100.0, .hp_max = 100.0,
        .controlled = true};
    // The training target stands a few metres off; it is parked whenever the window leaves it behind.
    target() = PlayBody{.name = "Target", .position = player().position + Vec3{4.0, 3.0, 3.0}, .radius = 0.45,
        .hp = 300.0, .hp_max = 300.0};
    target().parked = config_.people; // in a world with people in it nobody needs a dummy
    start_goods();
    start_people();
}

Vec3 Playground::origin() const
{
    if (!config_.open_world) {
        return Vec3{};
    }
    const double chunk_m = config_.terrain.chunk * config_.terrain.cell;
    return Vec3{origin_cx_ * chunk_m, origin_cy_ * chunk_m, origin_cz_ * chunk_m};
}

bool Playground::column_in_window(int cx, int cy) const
{
    return config_.open_world && cx >= origin_cx_ && cy >= origin_cy_ && cx < origin_cx_ + config_.window_chunks &&
        cy < origin_cy_ + config_.window_chunks;
}

bool Playground::in_window(int cx, int cy, int cz) const
{
    return column_in_window(cx, cy) && cz >= origin_cz_ && cz < origin_cz_ + config_.window_layers;
}

void Playground::read_chunk(int cx, int cy, int cz, ChunkData& out) const
{
    read_chunk_state(config_.terrain, cx, cy, cz, store_.find(cx, cy, cz), out);
}

// A chunk of a lattice, from cell (lx0, ly0, lz0), as the store keeps it: only what differs from the generator, so
// a chunk nobody touched costs nothing. Derived fields are recomputed from the rest on load.
StoredChunk Playground::capture_chunk(const FieldSet& fields, int lx0, int ly0, int lz0, int cx, int cy, int cz) const
{
    const TerrainConfig& t = config_.terrain;
    const int n = t.chunk;
    const GridShape& g = fields.shape();
    const std::size_t cells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    ChunkData generated;
    generate_chunk(t, cx, cy, cz, generated, nullptr);
    StoredChunk out;
    const Law& law = solver_.law();
    std::vector<float> current(cells);
    for (std::size_t f = 0; f < law.fields.size(); ++f) {
        const FieldDecl& decl = law.fields[f];
        if (decl.derived) {
            continue;
        }
        const std::vector<float>& values = fields.values(static_cast<int>(f));
        std::size_t k = 0;
        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                const std::size_t row = g.index(lx0, ly0 + y, lz0 + z);
                for (int x = 0; x < n; ++x) {
                    current[k++] = values[row + static_cast<std::size_t>(x)];
                }
            }
        }
        const std::vector<float>* made = generated.find(decl.name);
        const bool same = made != nullptr
            ? current == *made
            : std::all_of(current.begin(), current.end(), [&](float v) { return v == static_cast<float>(decl.init); });
        if (same) {
            continue;
        }
        StoredField stored{.name = decl.name};
        if (std::all_of(current.begin(), current.end(), [&](float v) { return v == current[0]; })) {
            stored.uniform = true;
            stored.value = current[0];
        } else {
            stored.values = current;
        }
        out.fields.push_back(std::move(stored));
    }
    return out;
}

// The creatures standing in a chunk go on beyond the window when it leaves.
void Playground::take_creatures(int cx, int cy, int cz)
{
    const Vec3 o = origin();
    const double chunk_m = config_.terrain.chunk * config_.terrain.cell;
    const auto cell_of = [&](double v, int chunks) { return std::clamp(static_cast<int>(std::floor(v / chunk_m)), 0, chunks - 1); };
    std::vector<Creature> staying;
    for (Creature& c : creatures_) {
        const int ccx = origin_cx_ + cell_of(c.body.position.x, config_.window_chunks);
        const int ccy = origin_cy_ + cell_of(c.body.position.y, config_.window_chunks);
        const int ccz = origin_cz_ + cell_of(c.body.position.z, config_.window_layers);
        if (ccx != cx || ccy != cy || ccz != cz) {
            staying.push_back(std::move(c));
            continue;
        }
        if (c.town >= 0) {
            // Back to being a number. What only it had stays with it.
            society_->record_of(c.who.pid).hp = c.body.hp < c.body.hp_max ? c.body.hp : -1.0;
            continue;
        }
        const CreatureDef& def = solver_.law().creatures[static_cast<std::size_t>(c.def)];
        FarBody s{.kind = def.name, .x = c.body.position.x + o.x, .y = c.body.position.y + o.y,
            .z = c.body.position.z + o.z, .heading = c.heading, .hp = c.body.hp, .brood = c.brood, .id = c.id};
        for (std::size_t i = 0; i < def.locals.size() && i < c.locals.size(); ++i) {
            s.locals.emplace_back(def.locals[i], c.locals[i]);
        }
        far_bodies_.push_back(std::move(s));
    }
    creatures_ = std::move(staying);
}

// A chunk into a lattice at (lx0, ly0, lz0): stored where the law changed it, generated where it never ran.
void Playground::fill_chunk(FieldSet& f, int lx0, int ly0, int lz0, int cx, int cy, int cz, std::vector<CreatureSpawn>& spawns) const
{
    const TerrainConfig& t = config_.terrain;
    const int n = t.chunk;
    const GridShape& g = f.shape();
    const StoredChunk* stored = store_.find(cx, cy, cz);
    ChunkData data;
    generate_chunk(t, cx, cy, cz, data, stored == nullptr ? &spawns : nullptr);
    const Law& law = solver_.law();
    for (std::size_t id = 0; id < law.fields.size(); ++id) {
        const FieldDecl& decl = law.fields[id];
        if (decl.derived) {
            continue;
        }
        const StoredField* kept = nullptr;
        if (stored != nullptr) {
            for (const StoredField& s : stored->fields) {
                if (s.name == decl.name) {
                    kept = &s;
                }
            }
        }
        const std::vector<float>* made = data.find(decl.name);
        std::vector<float>& values = f.values(static_cast<int>(id));
        std::size_t k = 0;
        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                const std::size_t row = g.index(lx0, ly0 + y, lz0 + z);
                for (int x = 0; x < n; ++x, ++k) {
                    float v = static_cast<float>(decl.init);
                    if (kept != nullptr) {
                        v = kept->uniform ? kept->value : kept->values[k];
                    } else if (made != nullptr) {
                        v = (*made)[k];
                    }
                    values[row + static_cast<std::size_t>(x)] = v;
                }
            }
        }
    }
}

// A chunk seen for the first time: the creatures the land can feed start in it. Once seen it never spawns again.
void Playground::bring_creatures(int cx, int cy, int cz, const std::vector<CreatureSpawn>& spawns)
{
    if (store_.find(cx, cy, cz) != nullptr) {
        return;
    }
    const Vec3 o = origin();
    for (const CreatureSpawn& s : spawns) {
        std::uint64_t h = std::hash<std::string>{}(s.kind) ^ static_cast<std::uint64_t>(std::llround(s.x * 4.0)) * 0x9e3779b97f4a7c15ull ^
            static_cast<std::uint64_t>(std::llround(s.y * 4.0)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 29)) * 0x94d049bb133111ebull;
        if (static_cast<double>((h ^ (h >> 31)) >> 11) / 9007199254740992.0 < feed_chance(s.kind, s.x, s.y)) {
            spawn(s.kind, Vec3{s.x - o.x, s.y - o.y, s.z - o.z});
        }
    }
    store_.put(cx, cy, cz, StoredChunk{}); // seen
}

void Playground::load_chunk(int cx, int cy, int cz)
{
    std::vector<CreatureSpawn> spawns;
    const int n = config_.terrain.chunk;
    fill_chunk(solver_.fields(), (cx - origin_cx_) * n, (cy - origin_cy_) * n, (cz - origin_cz_) * n, cx, cy, cz, spawns);
    bring_creatures(cx, cy, cz, spawns);
}

// THE WINDOW MOVES IN TWO PARTS. The lattice part is the expensive one and touches nothing a body reads, so
// with async fields it runs on the worker, next to the field step, and is taken up with that step's result.
// Chunks that leave are captured; chunks that stay slide across the lattice as they are; chunks that arrive are
// read. It reads the store and writes the solver's lattice, nothing else.
void Playground::move_lattice(WindowMove& move)
{
    const int w = config_.window_chunks;
    const int layers = config_.window_layers;
    const int n = config_.terrain.chunk;
    auto inside = [w, layers](int x, int y, int z, int ax, int ay, int az) {
        return x >= ax && y >= ay && z >= az && x < ax + w && y < ay + w && z < az + layers;
    };
    for (int z = move.old_z; z < move.old_z + layers; ++z) {
        for (int y = move.old_y; y < move.old_y + w; ++y) {
            for (int x = move.old_x; x < move.old_x + w; ++x) {
                if (!inside(x, y, z, move.x, move.y, move.z)) {
                    move.leaving.push_back(WindowMove::Leaving{x, y, z, {}});
                }
            }
        }
    }
    // A move of less than the window slides the cells; the GPU's ring moves its origin and takes only what arrives.
    const bool slides = std::abs(move.x - move.old_x) < w && std::abs(move.y - move.old_y) < w && std::abs(move.z - move.old_z) < layers;
    std::vector<CellBox> arriving_boxes;
    for (int z = move.z; z < move.z + layers; ++z) {
        for (int y = move.y; y < move.y + w; ++y) {
            for (int x = move.x; x < move.x + w; ++x) {
                if (!inside(x, y, z, move.old_x, move.old_y, move.old_z)) {
                    const int bx = (x - move.x) * n;
                    const int by = (y - move.y) * n;
                    const int bz = (z - move.z) * n;
                    arriving_boxes.push_back(CellBox{bx, by, bz, bx + n, by + n, bz + n});
                }
            }
        }
    }
    FieldSet& f = slides ? solver_.shift_window((move.x - move.old_x) * n, (move.y - move.old_y) * n, (move.z - move.old_z) * n, arriving_boxes)
                         : solver_.fields();
    const FieldSet& lattice = f;
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(move.leaving.size()); ++k) {
        WindowMove::Leaving& l = move.leaving[static_cast<std::size_t>(k)];
        l.chunk = capture_chunk(lattice, (l.cx - move.old_x) * n, (l.cy - move.old_y) * n, (l.cz - move.old_z) * n, l.cx, l.cy, l.cz);
    }
    // Most fields are zero everywhere most of the time: there is nothing to slide or to copy.
    const Law& fields_law = solver_.law();
    const auto not_all_zero = [&](int id) {
        const std::vector<float>& v = lattice.values(id);
        return !std::all_of(v.begin(), v.end(), [](float x) { return x == 0.0f; });
    };
    move.touched.assign(fields_law.fields.size(), 0);
#pragma omp parallel for schedule(dynamic)
    for (int id = 0; id < static_cast<int>(fields_law.fields.size()); ++id) {
        move.touched[static_cast<std::size_t>(id)] = not_all_zero(id) ? 1 : 0;
    }
    const GridShape& g = f.shape();
    const int cells_x = (move.x - move.old_x) * n;
    const int cells_y = (move.y - move.old_y) * n;
    const int cells_z = (move.z - move.old_z) * n;
    if (slides) {
        const Law& law = solver_.law();
        const int x0 = std::max(0, -cells_x);
        const int x1 = std::min(g.nx, g.nx - cells_x);
        const int y0 = std::max(0, -cells_y);
        const int y1 = std::min(g.ny, g.ny - cells_y);
        const int z0 = std::max(0, -cells_z);
        const int z1 = std::min(g.nz, g.nz - cells_z);
        // Rows are moved in the order that overwrites nothing still needed.
        const int y_first = cells_y >= 0 ? y0 : y1 - 1;
        const int y_step = cells_y >= 0 ? 1 : -1;
        const int z_first = cells_z >= 0 ? z0 : z1 - 1;
        const int z_step = cells_z >= 0 ? 1 : -1;
#pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < static_cast<int>(law.fields.size()); ++id) {
            if (law.fields[static_cast<std::size_t>(id)].derived || !move.touched[static_cast<std::size_t>(id)]) {
                continue;
            }
            float* values = f.values(id).data();
            for (int pz = 0; pz < z1 - z0; ++pz) {
                const int z = z_first + pz * z_step;
                for (int r = 0; r < y1 - y0; ++r) {
                    const int y = y_first + r * y_step;
                    std::memmove(values + g.index(x0, y, z), values + g.index(x0 + cells_x, y + cells_y, z + cells_z),
                        static_cast<std::size_t>(x1 - x0) * sizeof(float));
                }
            }
        }
    }
    for (int z = move.z; z < move.z + layers; ++z) {
        for (int y = move.y; y < move.y + w; ++y) {
            for (int x = move.x; x < move.x + w; ++x) {
                if (!inside(x, y, z, move.old_x, move.old_y, move.old_z)) {
                    move.arriving.push_back(WindowMove::Arriving{x, y, z, {}});
                }
            }
        }
    }
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(move.arriving.size()); ++k) {
        WindowMove::Arriving& a = move.arriving[static_cast<std::size_t>(k)];
        fill_chunk(f, (a.cx - move.x) * n, (a.cy - move.y) * n, (a.cz - move.z) * n, a.cx, a.cy, a.cz, a.spawns);
    }
#pragma omp parallel for schedule(dynamic)
    for (int id = 0; id < static_cast<int>(fields_law.fields.size()); ++id) {
        if (!move.touched[static_cast<std::size_t>(id)] && !fields_law.fields[static_cast<std::size_t>(id)].derived) {
            move.touched[static_cast<std::size_t>(id)] = not_all_zero(id) ? 1 : 0;
        }
    }
    solver_.refresh_derived();
}

// The rest, where bodies live, at one fixed tick.
void Playground::finish_move(WindowMove& move)
{
    for (WindowMove::Leaving& l : move.leaving) {
        take_creatures(l.cx, l.cy, l.cz);
        note_busy(l.cx, l.cy, l.cz, l.chunk);
        store_.put(l.cx, l.cy, l.cz, std::move(l.chunk));
    }
    const GridShape& g = fields().shape();
    const double chunk_m = config_.terrain.chunk * config_.terrain.cell;
    const Vec3 shift{(move.x - move.old_x) * chunk_m, (move.y - move.old_y) * chunk_m, (move.z - move.old_z) * chunk_m};
    origin_cx_ = move.x;
    origin_cy_ = move.y;
    origin_cz_ = move.z;
    for (PlayBody& body : bodies_) {
        body.position = body.position - shift;
    }
    for (Creature& c : creatures_) {
        c.body.position = c.body.position - shift;
    }
    for (const WindowMove::Arriving& a : move.arriving) {
        far_busy_.erase(chunk_key(a.cx, a.cy, a.cz));
        bring_creatures(a.cx, a.cy, a.cz, a.spawns);
    }
    bring_far_bodies();
    // A body left behind by the window (the target) is parked until the window comes back to it.
    for (std::size_t b = 1; b < bodies_.size(); ++b) {
        const Vec3 p = bodies_[b].position;
        bodies_[b].parked = config_.people || p.x < 0.0 || p.y < 0.0 || p.z < 0.0 || p.x > g.nx * g.cell || p.y > g.ny * g.cell || p.z > g.nz * g.cell;
    }
    ++window_version_;
}

void Playground::center_window_on(int cx, int cy, int cz)
{
    const TerrainConfig& t = config_.terrain;
    const int ox = cx - config_.window_chunks / 2;
    const int oy = cy - config_.window_chunks / 2;
    const int oz = std::clamp(cz - config_.window_layers / 2, 0, t.nz / t.chunk - config_.window_layers);
    if (!config_.open_world || (ox == origin_cx_ && oy == origin_cy_ && oz == origin_cz_)) {
        return;
    }
    settle();
    WindowMove move{.x = ox, .y = oy, .z = oz, .old_x = origin_cx_, .old_y = origin_cy_, .old_z = origin_cz_};
    move_lattice(move);
    finish_move(move);
    sync_front();
}

void Playground::center_window_on(int cx, int cy)
{
    const TerrainConfig& t = config_.terrain;
    center_window_on(cx, cy, (terrain_top(t, cx * t.chunk + t.chunk / 2, cy * t.chunk + t.chunk / 2) + 1) / t.chunk);
}

// You keep to the middle chunk; a metre past its edge, the window moves a chunk your way. Up and down it gives you
// more room: a jump is not worth a move.
std::optional<Playground::WindowMove> Playground::wanted_move() const
{
    if (!config_.open_world) {
        return std::nullopt;
    }
    const TerrainConfig& t = config_.terrain;
    const double chunk_m = t.chunk * t.cell;
    const int middle = config_.window_chunks / 2;
    const int middle_z = config_.window_layers / 2;
    constexpr double slack = 1.0;
    constexpr double slack_z = 2.5;
    const Vec3 p = player().position;
    int dx = 0;
    int dy = 0;
    int dz = 0;
    if (p.x < middle * chunk_m - slack) dx = -1;
    if (p.x > (middle + 1) * chunk_m + slack) dx = 1;
    if (p.y < middle * chunk_m - slack) dy = -1;
    if (p.y > (middle + 1) * chunk_m + slack) dy = 1;
    if (p.z < middle_z * chunk_m - slack_z && origin_cz_ > 0) dz = -1;
    if (p.z > (middle_z + 1) * chunk_m + slack_z && origin_cz_ + config_.window_layers < t.nz / t.chunk) dz = 1;
    if (dx == 0 && dy == 0 && dz == 0) {
        return std::nullopt;
    }
    return WindowMove{.x = origin_cx_ + dx, .y = origin_cy_ + dy, .z = origin_cz_ + dz, .old_x = origin_cx_, .old_y = origin_cy_, .old_z = origin_cz_};
}

bool Playground::reload_law(const std::vector<LawSource>& laws, const std::string& pours_text,
    std::vector<std::string>& errors)
{
    CompileResult compiled = compile_law(laws);
    errors.insert(errors.end(), compiled.errors.begin(), compiled.errors.end());
    std::vector<Pour> pours = parse_pours(pours_text, errors);
    if (compiled.law) {
        for (const char* needed : {"occ", "sdf", "T"}) {
            if (compiled.law->field_id(needed) < 0) {
                errors.push_back(std::string("the law has no '") + needed + "' field; the world stands on it");
            }
        }
        for (const Pour& pour : pours) {
            if (compiled.law->field_id(pour.field) < 0 || (!pour.also.empty() && compiled.law->field_id(pour.also) < 0)) {
                errors.push_back("pour '" + pour.name + "' pours into '" + pour.field + "', which no law declares");
            }
        }
    }
    if (!compiled.law || !errors.empty()) {
        return false;
    }
    settle();
    finish_far_job();
    far_solvers_.clear();
    Solver next(*compiled.law, fields().shape());
    if (solver_.on_gpu()) {
        std::string why;
        if (!next.use_gpu(why)) {
            note("The changed law runs on the CPU: " + why + ".");
        }
    }
    const Law& old_law = solver_.law();
    for (std::size_t id = 0; id < next.law().fields.size(); ++id) {
        const int before = old_law.field_id(next.law().fields[id].name);
        if (before >= 0) {
            next.fields().values(static_cast<int>(id)) = std::as_const(solver_).fields().values(before);
        }
    }
    next.refresh_derived();
    // Creatures keep their bodies; their kinds and states are matched by name.
    std::vector<Creature> kept;
    for (Creature& c : creatures_) {
        const CreatureDef& old_def = old_law.creatures[static_cast<std::size_t>(c.def)];
        const int def = next.law().creature_id(old_def.name);
        if (def < 0) {
            continue;
        }
        const CreatureDef& d = next.law().creatures[static_cast<std::size_t>(def)];
        std::vector<double> locals = d.init;
        for (std::size_t i = 0; i < d.locals.size(); ++i) {
            const auto it = std::find(old_def.locals.begin(), old_def.locals.end(), d.locals[i]);
            if (it != old_def.locals.end()) {
                locals[i] = c.locals[static_cast<std::size_t>(it - old_def.locals.begin())];
            }
        }
        c.def = def;
        c.body.radius = d.radius;
        c.body.hp_max = d.hp;
        c.body.hp = std::min(c.body.hp, d.hp);
        c.locals = std::move(locals);
        kept.push_back(std::move(c));
    }
    const int old_player = player_def_;
    std::vector<std::string> old_player_names;
    if (old_player >= 0) {
        old_player_names = old_law.creatures[static_cast<std::size_t>(old_player)].locals;
    }
    const std::vector<double> old_player_locals = player_locals_;

    solver_ = std::move(next);
    creatures_ = std::move(kept);
    pours_ = std::move(pours);
    sources_ = laws;
    pours_text_ = pours_text;
    player_def_ = -1;
    player_locals_.clear();
    bind_law();
    if (config_.falling) {
        set_param("fibre_falls", 0.0);
    }
    if (player_def_ >= 0) {
        const CreatureDef& d = solver_.law().creatures[static_cast<std::size_t>(player_def_)];
        for (std::size_t i = 0; i < d.locals.size(); ++i) {
            const auto it = std::find(old_player_names.begin(), old_player_names.end(), d.locals[i]);
            if (it != old_player_names.end() &&
                static_cast<std::size_t>(it - old_player_names.begin()) < old_player_locals.size()) {
                player_locals_[i] = old_player_locals[static_cast<std::size_t>(it - old_player_names.begin())];
            }
        }
    }
    sync_front();
    if (society_ != nullptr) {
        society_->rebind(solver_.law());
    }
    note("The law changed: " + std::to_string(solver_.law().terms.size()) + " terms, " +
        std::to_string(solver_.law().fields.size()) + " fields, " + std::to_string(solver_.law().creatures.size()) +
        " creature kinds. The world kept running.");
    return true;
}

double Playground::sample(int id, Vec3 p) const
{
    return id < 0 ? 0.0 : static_cast<double>(fields().sample(id, p.x, p.y, p.z));
}

void Playground::note(std::string line)
{
    chat_line(ChatLine::Kind::System, "", line);
    log_.push_back(std::move(line));
    if (log_.size() > 40) {
        log_.erase(log_.begin());
    }
}

std::optional<Vec3> Playground::pick(Vec3 origin, Vec3 direction) const
{
    RayBindings b;
    b.surface = surface();
    const RayHit hit = march_ray(fields(), b, origin, direction, 400.0, false);
    if (!hit.hit) {
        return std::nullopt;
    }
    return hit.position;
}

Creature Playground::make_creature(int def, Vec3 where)
{
    const CreatureDef& d = solver_.law().creatures[static_cast<std::size_t>(def)];
    Creature c;
    c.def = def;
    c.id = next_creature_id_++;
    c.body = PlayBody{.name = d.name, .position = where, .radius = d.radius, .hp = d.hp, .hp_max = d.hp,
        .controlled = true};
    c.body.id = c.id;
    c.locals = d.init;
    c.heading = wander_angle(c.id, 0);
    return c;
}

bool Playground::spawn(const std::string& kind, Vec3 where)
{
    const int def = solver_.law().creature_id(kind);
    if (def < 0 || kind == "player") {
        return false;
    }
    creatures_.push_back(make_creature(def, where));
    return true;
}

void Playground::pour(const Pour& pour, Vec3 where, Vec3 dir)
{
    pour_shape(solver_.field(pour.field), pour.amount, pour, where, dir);
    if (!pour.also.empty()) {
        pour_shape(solver_.field(pour.also), pour.also_amount, pour, where, dir);
    }
}

void Playground::pour_at(int slot, Vec3 where, Vec3 dir)
{
    if (slot < 0 || slot >= static_cast<int>(pours_.size())) {
        return;
    }
    const Pour& p = pours_[static_cast<std::size_t>(slot)];
    pour(p, where, dir);
    const Vec3 end = p.shape == Pour::Shape::Beam ? where + dir : where;
    event(PlayEvent::Kind::Impact, slot, where + origin(), end + origin(), p.radius);
}

int Playground::pour_slot(const std::string& name) const
{
    for (std::size_t i = 0; i < pours_.size(); ++i) {
        if (pours_[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void Playground::pour_shape(int id, double amount, const Pour& spell, Vec3 where, Vec3 dir)
{
    if (id < 0) {
        return;
    }
    const GridShape& g = fields().shape();
    const double c = g.cell;
    // Every cell of a box that is on the lattice.
    const auto each_cell = [&](Vec3 lo, Vec3 hi, const auto& at) {
        const int x0 = std::max(0, static_cast<int>(std::floor(lo.x / c)));
        const int y0 = std::max(0, static_cast<int>(std::floor(lo.y / c)));
        const int z0 = std::max(0, static_cast<int>(std::floor(lo.z / c)));
        const int x1 = std::min(g.nx - 1, static_cast<int>(std::floor(hi.x / c)));
        const int y1 = std::min(g.ny - 1, static_cast<int>(std::floor(hi.y / c)));
        const int z1 = std::min(g.nz - 1, static_cast<int>(std::floor(hi.z / c)));
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    at(g.index(x, y, z), Vec3{(x + 0.5) * c, (y + 0.5) * c, (z + 0.5) * c});
                }
            }
        }
    };
    const auto add = [&](std::size_t cell, double value) { add_to_cell(id, cell, static_cast<float>(value)); };
    switch (spell.shape) {
    case Pour::Shape::Ball: {
        // Densest at the heart, nothing at the rim: a ball of mana has a SLOPE, and the
        // slope is what pushes (a flat ball would push nothing anywhere inside it).
        const Vec3 r{spell.radius, spell.radius, spell.radius};
        each_cell(where - r, where + r, [&](std::size_t cell, Vec3 p) {
            const double d = length(p - where);
            if (d <= spell.radius) {
                add(cell, amount * 2.0 * (1.0 - d / spell.radius));
            }
        });
        return;
    }
    case Pour::Shape::Beam: {
        // Thickest along its line, nothing at its skin.
        const Vec3 end = where + dir;
        const double len2 = std::max(1e-12, dot(dir, dir));
        const Vec3 r{spell.radius, spell.radius, spell.radius};
        const Vec3 lo{std::min(where.x, end.x), std::min(where.y, end.y), std::min(where.z, end.z)};
        const Vec3 hi{std::max(where.x, end.x), std::max(where.y, end.y), std::max(where.z, end.z)};
        each_cell(lo - r, hi + r, [&](std::size_t cell, Vec3 p) {
            const double t = std::clamp(dot(p - where, dir) / len2, 0.0, 1.0);
            const double d = length(p - (where + dir * t));
            if (d <= spell.radius) {
                // A causeway is level and whole; what burns or throws has a slope.
                add(cell, spell.flat ? amount : amount * 1.5 * (1.0 - 0.8 * d / spell.radius));
            }
        });
        return;
    }
    case Pour::Shape::Ring: {
        const double reach = spell.radius + spell.thickness;
        each_cell(where - Vec3{reach, reach, c}, where + Vec3{reach, reach, spell.height}, [&](std::size_t cell, Vec3 p) {
            const double d = length(horizontal(p - where));
            if (std::abs(d - spell.radius) <= spell.thickness * 0.5) {
                add(cell, amount);
            }
        });
        return;
    }
    case Pour::Shape::Pillar: {
        each_cell(where - Vec3{spell.radius, spell.radius, c}, where + Vec3{spell.radius, spell.radius, spell.height}, [&](std::size_t cell, Vec3 p) {
            const double d = length(horizontal(p - where));
            if (d <= spell.radius) {
                add(cell, amount * (1.5 - d / spell.radius));
            }
        });
        return;
    }
    case Pour::Shape::Cut: {
        // A blade's plane: along the way it was cast, as long as `range` and as high as `height`, and thin.
        const Vec3 along = normalize(horizontal(dir));
        const Vec3 across{-along.y, along.x, 0.0};
        const double half = spell.thickness * 0.5;
        const Vec3 tip = where + along * spell.range;
        const Vec3 lo{std::min(where.x, tip.x) - half, std::min(where.y, tip.y) - half, where.z};
        const Vec3 hi{std::max(where.x, tip.x) + half, std::max(where.y, tip.y) + half, where.z + spell.height};
        each_cell(lo, hi, [&](std::size_t cell, Vec3 p) {
            const Vec3 q = p - where;
            const double a = dot(q, along);
            if (a >= 0.0 && a <= spell.range && std::abs(dot(q, across)) <= half) {
                add(cell, amount);
            }
        });
        return;
    }
    case Pour::Shape::Wall:
        break;
    }
    // A wall: a slab standing on `where`, across the direction it was cast in.
    const Vec3 d = normalize(horizontal(dir));
    const Vec3 u{-d.y, d.x, 0.0};
    const Vec3 middle = where + Vec3{0.0, 0.0, spell.height * 0.5};
    const double reach = 0.5 * std::max(spell.width, spell.height) + c;
    const Vec3 r{reach, reach, reach};
    each_cell(middle - r, middle + r, [&](std::size_t cell, Vec3 p) {
        const Vec3 o = p - middle;
        if (std::abs(dot(o, u)) <= spell.width * 0.5 && std::abs(dot(o, d)) <= spell.thickness * 0.5 && std::abs(o.z) <= spell.height * 0.5) {
            add(cell, amount);
        }
    });
}

void Playground::store_state(int cx, int cy, int cz, const ChunkData& state)
{
    // Only what differs from the generator is kept.
    ChunkData generated;
    generate_chunk(config_.terrain, cx, cy, cz, generated, nullptr);
    StoredChunk out;
    for (const FieldDecl& decl : solver_.law().fields) {
        const std::vector<float>* now = state.find(decl.name);
        if (decl.derived || now == nullptr) {
            continue;
        }
        const std::vector<float>* made = generated.find(decl.name);
        if (made != nullptr && *made == *now) {
            continue;
        }
        StoredField stored{.name = decl.name};
        stored.values = *now;
        out.fields.push_back(std::move(stored));
    }
    store_.put(cx, cy, cz, std::move(out));
}

// What came to rest beyond the window goes into the chunks it lies in, as the store keeps them: the first open cell going up from
// where it lies, in the chunk as it now is.
void Playground::set_down_beyond(const std::vector<BeyondCell>& cells)
{
    if (cells.empty() || !config_.open_world) {
        return;
    }
    const TerrainConfig& t = config_.terrain;
    const int n = t.chunk;
    const auto floor_div = [](int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    std::map<std::int64_t, std::vector<const BeyondCell*>> by_chunk;
    for (const BeyondCell& c : cells) {
        by_chunk[chunk_key(floor_div(c.wx, n), floor_div(c.wy, n), floor_div(c.wz, n))].push_back(&c);
    }
    static const char* const kMatter[4] = {"wood", "solid", "ice", "leaf"};
    static const char* const kTaken[] = {"solid", "rubble", "ice", "wood", "leaf", "lava"};
    for (const auto& [key, list] : by_chunk) {
        const int cx = key_cx(key);
        const int cy = key_cy(key);
        const int cz = key_cz(key);
        if (cz < 0 || cz >= t.nz / n || in_window(cx, cy, cz)) {
            continue;
        }
        ChunkData state;
        read_chunk(cx, cy, cz, state);
        const std::size_t cells_in = static_cast<std::size_t>(n) * n * n;
        const auto field = [&](const char* name) -> std::vector<float>& {
            std::vector<float>* found = state.find(name);
            return found != nullptr ? *found : state.add(name, cells_in, 0.0f);
        };
        const auto taken = [&](std::size_t i) {
            float sum = 0.0f;
            for (const char* name : kTaken) {
                if (const std::vector<float>* v = state.find(name)) {
                    sum += (*v)[i];
                }
            }
            return sum > 0.5f;
        };
        bool changed = false;
        for (const BeyondCell* c : list) {
            const int x = c->wx - cx * n;
            const int y = c->wy - cy * n;
            for (int z = c->wz - cz * n; z < n; ++z) {
                const std::size_t i = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * n + static_cast<std::size_t>(z) * n * n;
                if (taken(i)) {
                    continue;
                }
                field(kMatter[std::min<int>(c->matter, 3)])[i] = 1.0f;
                if (c->matter != 2) {
                    field("material")[i] = static_cast<float>(c->material);
                }
                field("held")[i] = 1.0f;
                changed = true;
                break;
            }
        }
        if (changed) {
            store_state(cx, cy, cz, state);
        }
    }
}

double Playground::ground_at(double wx, double wy) const
{
    if (map_ == nullptr || !config_.open_world) {
        return -1e9;
    }
    const TerrainConfig& t = config_.terrain;
    const FarColumn c = map_->far_column(static_cast<int>(std::floor(wx / t.cell)), static_cast<int>(std::floor(wy / t.cell)), 1);
    return (std::max({c.top, c.roof_top, c.crown_top}) + 1) * t.cell;
}

double Playground::terrain_at(double wx, double wy) const
{
    if (map_ == nullptr || !config_.open_world) {
        return -1e9;
    }
    const TerrainConfig& t = config_.terrain;
    return (map_->far_column(static_cast<int>(std::floor(wx / t.cell)), static_cast<int>(std::floor(wy / t.cell)), 1).top + 1) * t.cell;
}

std::optional<Vec3> Playground::pick_world(Vec3 from, Vec3 direction, double max_length) const
{
    const Vec3 dir = normalize(direction);
    const Vec3 o = origin();
    const GridShape& g = fields().shape();
    // The live lattice, marched from where the ray enters it: a ray from over the highest mountain can be a long way
    // from the window before it gets there, and only the stretch inside counts.
    double lattice_t = max_length;
    std::optional<Vec3> lattice_hit;
    double t0 = 0.0;
    double t1 = 0.0;
    if (clip_to_box(g, from - o, dir, t0, t1) && t0 < max_length) {
        RayBindings b;
        b.surface = surface();
        const RayHit hit = march_ray(fields(), b, from - o + dir * t0, dir, std::min(t1, max_length) - t0 + g.cell, false);
        if (hit.hit) {
            lattice_hit = hit.position + o;
            lattice_t = t0 + hit.length;
        }
    }
    if (map_ == nullptr || !config_.open_world) {
        return lattice_hit;
    }
    // The map's heights, up to the lattice's hit: a hill in front of the window hides what is behind it. Inside the
    // lattice the law answers (its own ground, its own holes), so the map is not asked there.
    const auto in_lattice = [&](Vec3 w) {
        const Vec3 p = w - o;
        return p.x >= 0.0 && p.y >= 0.0 && p.z >= 0.0 && p.x <= g.nx * g.cell && p.y <= g.ny * g.cell && p.z <= g.nz * g.cell;
    };
    const auto below = [&](Vec3 w) { return !in_lattice(w) && w.z <= ground_at(w.x, w.y); };
    const double cell = config_.terrain.cell;
    double s = 0.0;
    for (int i = 0; i < 20000 && s < lattice_t; ++i) {
        const Vec3 p = from + dir * s;
        const double gap = in_lattice(p) ? cell : p.z - ground_at(p.x, p.y);
        // A step as long as the height to spare allows, never past what a roof could stand out by.
        const double step = std::min(std::clamp(gap * 0.5, cell * 0.5, 16.0), lattice_t - s + 1e-6);
        if (below(from + dir * (s + step))) {
            double lo = s;
            double hi = s + step;
            for (int k = 0; k < 14; ++k) {
                const double m = 0.5 * (lo + hi);
                (below(from + dir * m) ? hi : lo) = m;
            }
            const Vec3 hit = from + dir * hi;
            return Vec3{hit.x, hit.y, ground_at(hit.x, hit.y)};
        }
        s += step;
    }
    return lattice_hit;
}

void Playground::event(PlayEvent::Kind kind, int pour, Vec3 at, Vec3 to, double power)
{
    if (events_.size() >= 512) {
        events_.erase(events_.begin(), events_.begin() + 256);
    }
    events_.push_back(PlayEvent{.kind = kind, .pour = pour, .at = at, .to = to, .power = power});
}

std::vector<PlayBody*> Playground::all_bodies()
{
    std::vector<PlayBody*> out;
    for (PlayBody& body : bodies_) {
        if (!body.parked) {
            out.push_back(&body);
        }
    }
    for (Creature& c : creatures_) {
        out.push_back(&c.body);
    }
    return out;
}

void Playground::move_body(PlayBody& body, double dt, const Vec3* want, bool jump)
{
    if (body.parked) {
        return;
    }
    const Surface s = surface();
    const FieldSet& f = fields();
    // What the fields do to a body: magic throws it, heat burns it.
    const Vec3 push{sample(push_[0], body.position), sample(push_[1], body.position), sample(push_[2], body.position)};
    body.velocity = body.velocity + push * dt;
    // (What the law gives back is a body mending: never past whole.)
    body.hp = std::clamp(body.hp - sample(harm_, body.position) * dt, 0.0, std::max(body.hp, body.hp_max));
    const bool thrown = body.stagger > 0.0;
    body.stagger = std::max(0.0, body.stagger - dt);

    if (body.surge > 0.0) {
        // A dash: nothing else moves you until it is over.
    } else if (want != nullptr) {
        const double speed = length(*want);
        if (body.grounded) {
            // Firm on the ground: it stops when nothing drives it, unless it was thrown.
            const double grip = std::min(1.0, (thrown ? 1.5 : 16.0) * dt);
            body.velocity.x += (want->x - body.velocity.x) * grip;
            body.velocity.y += (want->y - body.velocity.y) * grip;
        } else if (speed > 1e-9) {
            // In the air you can add to your speed along where you steer, and what you had you keep: a jump out of a
            // dash goes far.
            const Vec3 way = *want * (1.0 / speed);
            const double along = dot(horizontal(body.velocity), way);
            const double add = std::clamp(speed - along, 0.0, config_.air_control * dt);
            body.velocity = body.velocity + way * add;
            const Vec3 side = horizontal(body.velocity) - way * dot(horizontal(body.velocity), way);
            body.velocity = body.velocity - side * std::min(1.0, 2.0 * dt);
        }
        if (jump && body.grounded) {
            body.velocity.z = config_.jump_speed;
        }
    } else if (body.grounded) {
        const double friction = std::max(0.0, 1.0 - (thrown ? 2.0 : 8.0) * dt);
        body.velocity.x *= friction;
        body.velocity.y *= friction;
    }
    if (body.surge <= 0.0) {
        body.velocity.z = std::max(body.velocity.z - config_.gravity * dt, -config_.fall_limit);
    }

    Vec3 remaining = body.velocity * dt;
    for (int pass = 0; pass < 3 && length(remaining) > 1e-9; ++pass) {
        Contact contact;
        const Vec3 aim = body.position + remaining;
        Vec3 reached = sweep_sphere(f, s, body.position, aim, body.radius, &contact);
        if (!contact.touching) {
            body.position = reached;
            break;
        }
        // A ledge one voxel high is a step, not a wall: try the same move from a little higher, then
        // settle back down onto whatever is there.
        const Vec3 n = contact.normal;
        // (A dash runs along the floor, so the first thing it touches may be the floor: any pass will do.)
        if (want != nullptr && (body.grounded || body.surge > 0.0 || (&body == &bodies_[0] && coyote_ > 0.0)) && n.z > -0.5 && (std::abs(n.x) > 0.02 || std::abs(n.y) > 0.02)) {
            const double lift = f.shape().cell * 1.1;
            const Vec3 up = body.position + Vec3{0.0, 0.0, lift};
            if (surface_distance(f, s, up) - body.radius > kSurfaceSkin * f.shape().cell) {
                Contact ahead;
                const Vec3 over = sweep_sphere(f, s, up, up + horizontal(remaining), body.radius, &ahead);
                // A step is something you can stand on: a good way further on is clear at that height. Against a wall a
                // sphere gets a hair further up there, off the rounding of the surface, and that is no step at all.
                const Vec3 flat = horizontal(remaining);
                const double flat_len = length(flat);
                Contact beyond;
                const Vec3 past = up + (flat_len > 1e-9 ? flat * (1.0 / flat_len) : Vec3{}) * (body.radius + f.shape().cell);
                const Vec3 reached_beyond = sweep_sphere(f, s, up, past, body.radius, &beyond);
                const bool a_step = !beyond.touching || length(horizontal(reached_beyond - up)) > 0.3;
                if (a_step && length(horizontal(over - up)) > length(horizontal(reached - body.position)) + 0.01) {
                    Contact down;
                    reached = sweep_sphere(f, s, over, over - Vec3{0.0, 0.0, lift}, body.radius, &down);
                    body.position = reached;
                    remaining = Vec3{};
                    break;
                }
            }
        }
        body.position = reached;
        Vec3 left = aim - reached;
        // The surface is rounded where a wall meets the floor. What you walk into is the wall, and its foot is no ramp: you
        // slide along it, not up it.
        const bool walking = want != nullptr && body.velocity.z < 0.5 && n.z > -0.2 && (std::abs(n.x) > 1e-6 || std::abs(n.y) > 1e-6);
        // Ground that only leans you go on over; it just does not throw you in the air.
        const Vec3 facing = !walking || n.z >= 0.9 ? n : normalize(horizontal(n));
        const double vz_before = body.velocity.z;
        const double into = dot(left, facing);
        if (into < 0.0) {
            left = left - facing * into;
        }
        // And what is left of the move does not go on into the same surface, or you would hang on a wall by a hair.
        const double still_into = dot(left, n);
        if (still_into < 0.0) {
            left = left - n * still_into;
        }
        const double v_into = dot(body.velocity, facing);
        if (v_into < 0.0) {
            body.velocity = body.velocity - facing * v_into;
        }
        if (walking) {
            body.velocity.z = std::min(body.velocity.z, std::max(vz_before, 0.0));
        }
        remaining = left;
    }
    // Something grew into you (a wall rising, rubble landing): you are pushed out of it. It moves you; it does not throw you
    // up in the air, which is what the rounded foot of a wall would otherwise do to anyone who walks into it.
    const double rising = std::max(body.velocity.z, 0.0);
    resolve_sphere(f, s, body.position, body.velocity, body.radius);
    if (want != nullptr && body.surge <= 0.0) {
        body.velocity.z = std::min(body.velocity.z, rising);
    }
    const bool on_piece = !pieces_.empty() && out_of_pieces(body);

    Contact ground;
    sweep_sphere(f, s, body.position, body.position - Vec3{0.0, 0.0, 0.08}, body.radius, &ground);
    // Where a wall meets the floor the surface leans: that is still the floor under you.
    body.grounded = (on_piece || (ground.touching && ground.normal.z > 0.2)) && body.velocity.z <= 0.5;
    if (body.grounded && body.velocity.z < 0.0) {
        body.velocity.z = 0.0;
    }

    // The lattice is the world; nobody leaves it.
    const GridShape& g = f.shape();
    body.position.x = std::clamp(body.position.x, body.radius, g.nx * g.cell - body.radius);
    body.position.y = std::clamp(body.position.y, body.radius, g.ny * g.cell - body.radius);
    if (body.position.z < body.radius) {
        body.position.z = body.radius;
        body.velocity.z = std::max(0.0, body.velocity.z);
    }
}

// What a creature's eat/emit lines do to the cell it stands in. Eating never takes more than is there.
void Playground::apply_flows(const CreatureDef& def, std::vector<double>& locals, const PlayBody& body, double dt)
{
    if (def.flows.empty()) {
        return;
    }
    const FieldSet& f = fields();
    const GridShape& g = f.shape();
    const int x = std::clamp(static_cast<int>(std::floor(body.position.x / g.cell)), 0, g.nx - 1);
    const int y = std::clamp(static_cast<int>(std::floor(body.position.y / g.cell)), 0, g.ny - 1);
    const int z = std::clamp(static_cast<int>(std::floor(body.position.z / g.cell)), 0, g.nz - 1);
    const std::size_t cell = g.index(x, y, z);
    const PointContext at{.fields = &f, .params = &solver_.law().param_values, .locals = &locals,
        .x = body.position.x, .y = body.position.y, .z = body.position.z};
    // Every rate reads the same moment, then all of them act.
    std::vector<double> rates;
    for (const CreatureFlow& flow : def.flows) {
        rates.push_back(flow.rate ? evaluate_at(*flow.rate, at) : 0.0);
    }
    double eaten = 0.0;
    for (std::size_t k = 0; k < def.flows.size(); ++k) {
        const float v = f.values(def.flows[k].field)[cell];
        const double amount = std::max(0.0, rates[k]) * dt;
        if (def.flows[k].eat) {
            const double taken = std::min(amount, static_cast<double>(std::max(0.0f, v)));
            add_to_cell(def.flows[k].field, cell, static_cast<float>(-taken));
            eaten += taken;
        } else {
            add_to_cell(def.flows[k].field, cell, static_cast<float>(amount));
        }
    }
    if (static_cast<std::size_t>(kLocalEaten) < locals.size()) {
        locals[static_cast<std::size_t>(kLocalEaten)] = eaten / dt;
    }
}

// THE KINDS THAT PLAN, every `dt` of their plan: what each sees now goes into what its kind has learned (paired with
// what it saw and did last time), then all of them plan at once on the card, and every few plans the kind learns a
// little more from all it has seen.
void Playground::plan_minds()
{
    const Law& law = solver_.law();
    for (std::size_t d = 0; d < minds_.size(); ++d) {
        WorldModel* mind = minds_[d].get();
        if (mind == nullptr) {
            continue;
        }
        const CreatureDef& def = law.creatures[d];
        const auto every = static_cast<std::uint64_t>(std::max(1.0, std::round(def.plan_dt / kTick)));
        if (tick_ % every != 0) {
            continue;
        }
        std::vector<float> obs;
        std::vector<std::uint32_t> ids;
        std::vector<Creature*> who;
        std::vector<float> one(static_cast<std::size_t>(mind->features()));
        for (Creature& c : creatures_) {
            if (c.def != static_cast<int>(d)) {
                continue;
            }
            const Vec3 p = c.body.position;
            mind->observe(fields(), p.x, p.y, p.z, c.locals, one.data());
            mind->remember(c.id, one.data(), static_cast<double>(every) * kTick);
            obs.insert(obs.end(), one.begin(), one.end());
            ids.push_back(c.id);
            who.push_back(&c);
        }
        std::vector<float> actions;
        mind->plan(obs, ids, tick_, actions);
        for (std::size_t i = 0; i < who.size(); ++i) {
            who[i]->planned = Vec3{actions[2 * i], actions[2 * i + 1], 0.0};
            who[i]->planning = true;
        }
        if ((tick_ / every) % 4 == 0) {
            mind->learn(4);
        }
    }
}

// ONE TICK OF A CREATURE'S MIND, ALL OF IT LAW: its states change by their equations at the point where
// it stands, it walks up the gradient of its seek expression (compared `sense` metres either side), it
// ambles where that is flat, and its eat/emit lines trade with the cell it stands in.
void Playground::think(Creature& c, double dt)
{
    const Law& law = solver_.law();
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(c.def)];
    std::vector<double>& locals = c.locals;
    locals[static_cast<std::size_t>(kLocalHp)] = c.body.hp;
    locals[static_cast<std::size_t>(kLocalAge)] += dt;
    if (c.town >= 0) {
        person_mind(c);
    }
    const Vec3 p = c.body.position;
    PointContext at{.fields = &fields(), .params = &law.param_values, .locals = &locals, .x = p.x, .y = p.y, .z = p.z};
    std::vector<double> next = locals;
    for (std::size_t i = static_cast<std::size_t>(kFirstState); i < def.locals.size(); ++i) {
        if (def.rates[i]) {
            next[i] = std::clamp(locals[i] + evaluate_at(*def.rates[i], at) * dt, def.lo[i], def.hi[i]);
        }
    }
    locals = std::move(next);

    // The pull of the seek field saturates: a whiff barely moves it, a strong gradient moves it at full
    // speed. What the pull leaves of its stride goes to ambling, on a heading of its own that changes
    // every two seconds (chosen by its id and the time).
    if ((tick_ + c.id * 37u) % 120u == 0) {
        c.heading = wander_angle(c.id, (tick_ + c.id * 37u) / 120u);
    }
    Vec3 pull{};
    double strength = 0.0;
    if (def.plan && c.planning && c.town < 0) {
        // Its world model chose the way (plan_minds): as strong as it is sure, ambling for the rest.
        const double sure = std::min(1.0, length(c.planned));
        if (sure > 1e-6) {
            pull = c.planned * (1.0 / sure);
            strength = sure;
        }
    } else if (def.seek) {
        const double d = def.sense;
        auto seek_at = [&](double x, double y) {
            PointContext q = at;
            q.x = x;
            q.y = y;
            return evaluate_at(*def.seek, q);
        };
        const Vec3 grad{(seek_at(p.x + d, p.y) - seek_at(p.x - d, p.y)) / (2.0 * d),
            (seek_at(p.x, p.y + d) - seek_at(p.x, p.y - d)) / (2.0 * d), 0.0};
        const double slope = length(grad);
        if (slope > 1e-9) {
            strength = slope / (slope + def.heed);
            pull = grad * (1.0 / slope);
        }
    }
    const Vec3 amble{std::cos(c.heading), std::sin(c.heading), 0.0};
    const Vec3 stride = pull * strength + amble * ((1.0 - strength) * def.wander);
    const double speed = def.speed;
    const Vec3 dir = length(stride) > 1e-9 ? normalize(stride) : Vec3{};
    c.want = dir;
    Vec3 want = stride * speed;
    bool jump = false;
    if (c.town >= 0) {
        person_turn(c, def, want, jump, pull, strength);
    } else {
        // Hurt, it knows who by; one that only fights when hurt is fighting now.
        if (c.body.hp < c.last_hp - 0.01) {
            c.riled_left = 25.0;
            if (def.riled || def.prey.empty() == false) {
                c.foe = c.body.struck_by;
                c.prey_hold = 8.0;
            }
        }
        c.last_hp = c.body.hp;
        c.riled_left = std::max(0.0, c.riled_left - dt);
        if ((!def.riled || c.riled_left > 0.0) && fight_turn(c, def, want) && length(want) > 1e-6) {
            // After something, it goes round what is in the way, to its own side; a step it can take, it jumps.
            const double pace = length(want);
            const Vec3 way = want * (1.0 / pace);
            const Vec3 feet = c.body.position - Vec3{0.0, 0.0, c.body.radius - 0.25};
            const Vec3 chest = c.body.position + Vec3{0.0, 0.0, 0.3};
            const double look = c.body.radius + 0.55;
            const int side = (c.id & 2u) != 0u ? 1 : -1;
            for (const double turn : {0.0, 0.8, 1.6, 2.4}) {
                const double a = turn * side;
                const Vec3 tried{way.x * std::cos(a) - way.y * std::sin(a), way.x * std::sin(a) + way.y * std::cos(a), 0.0};
                if (occ_ < 0 || sample(occ_, chest + tried * look) <= 0.5) {
                    want = tried * pace;
                    jump = turn == 0.0 && occ_ >= 0 && sample(occ_, feet + tried * look) > 0.5;
                    break;
                }
            }
        }
    }
    move_body(c.body, dt, &want, jump);
    apply_flows(def, locals, c.body, dt);
    // Breeding: its rate adds up; each whole one is a newborn beside it, with a newborn's state.
    if (def.breed) {
        c.brood += std::max(0.0, evaluate_at(*def.breed, at)) * dt;
        if (c.brood >= 1.0) {
            c.brood -= 1.0;
            if (creatures_.size() + born_.size() < static_cast<std::size_t>(config_.max_creatures)) {
                const Vec3 beside = c.body.position + amble * (c.body.radius * 2.5) + Vec3{0.0, 0.0, 0.1};
                born_.push_back(make_creature(c.def, beside));
            }
        }
    }
    if (def.hurt) {
        at.x = c.body.position.x;
        at.y = c.body.position.y;
        at.z = c.body.position.z;
        c.body.hp = std::clamp(c.body.hp - evaluate_at(*def.hurt, at) * dt, 0.0, c.body.hp_max);
    }
}

// BODIES ARE SOLID TO EACH OTHER. Two bodies that overlap are put back to touching. You do not shove
// the target by walking into it — a target stands; only the law moves it (force, air, heat) — so the
// body that walked in is the one that gives way, and it loses the part of its velocity that carried
// it there. Pushed out of one body it may land in stone, so the surface gets the last word.
void Playground::separate_bodies()
{
    const FieldSet& f = fields();
    const Surface s = surface();
    std::vector<PlayBody*> bodies = all_bodies();
    for (std::size_t a = 0; a < bodies.size(); ++a) {
        for (std::size_t b = a + 1; b < bodies.size(); ++b) {
            PlayBody& first = *bodies[a];
            PlayBody& second = *bodies[b];
            const Vec3 d = second.position - first.position;
            const double dist = length(d);
            const double overlap = first.radius + second.radius - dist;
            if (overlap <= 0.0) {
                continue;
            }
            const Vec3 n = dist > 1e-9 ? d * (1.0 / dist) : Vec3{1.0, 0.0, 0.0}; // from first to second
            // Who gives way: the one that walks (controlled). Two of a kind share it.
            const double share_first = first.controlled == second.controlled ? 0.5 : (first.controlled ? 1.0 : 0.0);
            auto give_way = [&](PlayBody& body, Vec3 away, double share) {
                if (share <= 0.0) {
                    return;
                }
                body.position = body.position + away * (overlap * share);
                const double into = dot(body.velocity, away);
                if (into < 0.0) {
                    body.velocity = body.velocity - away * into;
                }
                resolve_sphere(f, s, body.position, body.velocity, body.radius);
            };
            give_way(first, n * -1.0, share_first);
            give_way(second, n, 1.0 - share_first);
        }
    }
}

// The body the window follows, if somebody steers it: walking, a jump. An observer is put where it is and nothing moves it.
void Playground::move_player(const Control& control)
{
    PlayBody& you = player();
    if (config_.observer) {
        you.parked = true; // nothing sees it, hits it, or pushes it; the window still follows it
        you.velocity = Vec3{};
        you.hp = you.hp_max;
        return;
    }
    const Vec3 o = origin();
    Vec3 want{control.move_x, control.move_y, 0.0};
    if (length(want) > 1e-9) {
        want = normalize(want) * config_.walk_speed;
    }
    // A jump asked for just before it lands happens when it lands; one asked for just after it stepped off still counts.
    jump_buffer_ = control.jump ? 0.12 : std::max(0.0, jump_buffer_ - kTick);
    coyote_ = you.grounded ? 0.1 : std::max(0.0, coyote_ - kTick);
    if (jump_buffer_ > 0.0 && (you.grounded || coyote_ > 0.0) && you.surge <= 0.0) {
        you.velocity.z = config_.jump_speed;
        you.grounded = false;
        jump_buffer_ = 0.0;
        coyote_ = 0.0;
        event(PlayEvent::Kind::Jump, -1, you.position + o, you.position + o, config_.jump_speed);
    }
    const double came_down = you.velocity.z;
    const bool was_grounded = you.grounded;
    move_body(you, kTick, &want, false);
    if (you.surge > 0.0) {
        you.surge = std::max(0.0, you.surge - kTick);
    }
    if (you.grounded && !was_grounded && came_down < -13.0) {
        const Vec3 feet = you.position - Vec3{0.0, 0.0, you.radius};
        event(PlayEvent::Kind::Landing, -1, feet + o, feet + o, -came_down);
    }
}

void Playground::step(const Control& control)
{
    ++tick_;
    move_player(control);
    // `creature player` in the law: its own states and what it leaves in the world (scent). An observer leaves nothing.
    if (player_def_ >= 0 && !config_.observer) {
        // `creature player` in the law: your own states and what you leave in the world.
        const CreatureDef& def = solver_.law().creatures[static_cast<std::size_t>(player_def_)];
        player_locals_[static_cast<std::size_t>(kLocalHp)] = player().hp;
        player_locals_[static_cast<std::size_t>(kLocalAge)] += kTick;
        const Vec3 p = player().position;
        const PointContext at{.fields = &fields(), .params = &solver_.law().param_values, .locals = &player_locals_,
            .x = p.x, .y = p.y, .z = p.z};
        std::vector<double> next = player_locals_;
        for (std::size_t i = static_cast<std::size_t>(kFirstState); i < def.locals.size(); ++i) {
            if (def.rates[i]) {
                next[i] = std::clamp(player_locals_[i] + evaluate_at(*def.rates[i], at) * kTick, def.lo[i], def.hi[i]);
            }
        }
        player_locals_ = std::move(next);
        apply_flows(def, player_locals_, player(), kTick);
    }
    for (std::size_t b = 1; b < bodies_.size(); ++b) {
        move_body(bodies_[b], kTick, nullptr, false);
    }
    // A blow begun in the player's body or the target's goes on as any other (creatures' own go on in their turns).
    for (PlayBody& body : bodies_) {
        if (!body.parked) {
            swing_turn(body);
        }
    }
    people_turn();
    local_life_turn();
    scan_falling();
    pieces_turn();
    plan_minds();
    for (Creature& c : creatures_) {
        think(c, kTick);
    }
    for (Creature& newborn : born_) {
        creatures_.push_back(std::move(newborn));
    }
    born_.clear();
    separate_bodies();
    if (target().hp <= 0.0) {
        target().hp = target().hp_max;
        note("The target broke — and is whole again. (It cannot die here.)");
    }
    {
        // Those who saw it are told while everyone is still in place.
        for (const Creature& c : creatures_) {
            if (c.body.hp <= 0.0) {
                note((c.town >= 0 ? "" : "A ") + c.body.name + " died.");
                if (c.town >= 0) {
                    person_died(c);
                } else {
                    beast_died(c);
                }
            }
        }
        std::vector<Creature> alive;
        for (Creature& c : creatures_) {
            if (c.body.hp <= 0.0) {
                continue;
            }
            alive.push_back(std::move(c));
        }
        creatures_ = std::move(alive);
    }
    if (player().hp <= 0.0) {
        ++deaths_;
        player().hp = player().hp_max;
        player().velocity = Vec3{};
        if (config_.open_world) {
            const double chunk_m = config_.terrain.chunk * config_.terrain.cell;
            center_window_on(static_cast<int>(std::floor(home_.x / chunk_m)), static_cast<int>(std::floor(home_.y / chunk_m)),
                static_cast<int>(std::floor(home_.z / chunk_m)));
            player().position = home_ - origin();
        } else {
            player().position = Vec3{5.0, 5.0, fields().shape().cell * 2.0 + 0.42};
        }
        note("You died, and wake at the start.");
    }
    if (config_.async_fields) {
        if (config_.field_every > 0 && tick_ % static_cast<std::uint64_t>(config_.field_every) == 0) {
            if (config_.field_never_waits && stepping_ && worker_->working()) {
                owed_dt_ = std::min(owed_dt_ + kTick * config_.field_every, config_.field_most_owed);
                return;
            }
            // A window move has two turns to itself: it is taken up at a fixed tick, later than a step, so the frame
            // does not wait for it. The law makes up the time, as for the move itself.
            if (moving_ && tick_ < move_due_) {
                owed_dt_ += kTick * config_.field_every;
                return;
            }
            settle();
            if (front_stale_) {
                sync_front();
            }
            far_tick();
            moving_ = wanted_move();
            move_due_ = tick_ + 2 * static_cast<std::uint64_t>(config_.field_every);
            if (worker_ == nullptr) {
                worker_ = std::make_shared<FieldWorker>();
            }
            // A window move takes the worker's whole turn; the law makes up the time with the next step (its
            // max_dt keeps that as fine as any other), so a move fits in a tick and the frame does not wait.
            owed_dt_ += kTick * config_.field_every;
            owed_dt_ = config_.field_never_waits ? std::min(owed_dt_, config_.field_most_owed) : owed_dt_;
            const double dt = moving_ ? 0.0 : owed_dt_;
            if (!moving_) {
                owed_dt_ = 0.0;
            }
            stepping_ = true;
            worker_->submit([this, dt] {
                const auto started = std::chrono::steady_clock::now();
                if (moving_) {
                    move_lattice(*moving_);
                    changed_now_ = moving_->touched;
                } else {
                    solver_.step(dt);
                }
                // The next picture, ready to be swapped in: the worker pays for the copy too.
                const FieldSet& state = std::as_const(solver_).fields();
                // A cell written from outside (a door, a dig) is as changed as one the law moved: the card only says what the shaders did.
                std::vector<char> stepped = solver_.last_changed();
                for (std::size_t id = 0; id < state.count(); ++id) {
                    changed_now_[id] = changed_now_[id] || stepped[id];
                    stepped[id] = changed_now_[id];
                    const bool stale = changed_now_[id] || changed_before_[id];
                    changed_before_[id] = changed_now_[id];
                    changed_now_[id] = 0;
                    if (!stale) {
                        continue;
                    }
                    const std::vector<float>& from = state.values(static_cast<int>(id));
                    std::vector<float>& to = spare_.values(static_cast<int>(id));
                    parallel_blocks(from.size(), [&](std::size_t first, std::size_t end) {
                        std::copy(from.begin() + static_cast<std::ptrdiff_t>(first),
                            from.begin() + static_cast<std::ptrdiff_t>(end),
                            to.begin() + static_cast<std::ptrdiff_t>(first));
                    });
                }
                if (listener_) {
                    listener_(FieldStep{state, stepped, tick_, moving_ ? moving_->x : origin_cx_, moving_ ? moving_->y : origin_cy_,
                        moving_ ? moving_->z : origin_cz_});
                }
                field_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                const std::lock_guard<std::mutex> lock(*profile_mutex_);
                field_profile_ = solver_.profile();
            });
        }
        return;
    }
    if (config_.field_every > 0 && tick_ % static_cast<std::uint64_t>(config_.field_every) == 0) {
        far_tick();
    }
    if (std::optional<WindowMove> move = wanted_move()) {
        center_window_on(move->x + config_.window_chunks / 2, move->y + config_.window_chunks / 2, move->z + config_.window_layers / 2);
    }
    if (config_.field_every > 0 && tick_ % static_cast<std::uint64_t>(config_.field_every) == 0) {
        solver_.step(kTick * config_.field_every);
        {
            const std::lock_guard<std::mutex> lock(*profile_mutex_);
            field_profile_ = solver_.profile();
        }
        note_violations();
        if (listener_) {
            listener_(FieldStep{std::as_const(solver_).fields(), solver_.last_changed(), tick_, origin_cx_, origin_cy_, origin_cz_});
        }
    }
}

double Playground::total(const std::string& field) const
{
    const int id = solver_.field(field);
    if (id < 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (const float v : fields().values(id)) {
        sum += v;
    }
    return sum;
}

std::uint64_t Playground::digest() const
{
    std::uint64_t hash = fields().digest();
    for (const PlayBody& body : bodies_) {
        mix(hash, body.position);
        mix(hash, body.velocity);
        mix(hash, body.hp);
    }
    for (const Creature& c : creatures_) {
        mix(hash, c.body.position);
        mix(hash, c.body.velocity);
        mix(hash, c.body.hp);
        mix(hash, c.brood);
        for (const double v : c.locals) {
            mix(hash, v);
        }
    }
    mix(hash, static_cast<double>(tick_));
    mix(hash, static_cast<double>(origin_cx_));
    mix(hash, static_cast<double>(origin_cy_));
    mix(hash, static_cast<double>(origin_cz_));
    for (const FarBody& f : far_bodies_) {
        mix(hash, f.x);
        mix(hash, f.y);
        mix(hash, f.z);
    }
    for (const auto& [key, time] : far_busy_) {
        mix(hash, static_cast<double>(key));
        mix(hash, time);
    }
    mix(hash, static_cast<double>(store_.counter()));
    mix(hash, map_total("grazers"));
    return hash;
}

} // namespace fieldengine::field

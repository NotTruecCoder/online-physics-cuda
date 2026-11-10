// What the sandbox draws, all of it read off the engine: the world's chunks (WorldMesher, on worker threads, at the detail
// the zoom can show), the law's `show` lines, people and beasts as figures of cubes, pieces in the air, effects.

#include "sandbox/Sandbox.hpp"

#include "world/Looks.hpp"
#include "render/Units.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fieldengine::sandbox {

namespace {

constexpr double kSim = render::units::sim_units_per_metre;
constexpr double kEighth = 0.7853981633974483; // figures face the eight ways, as cubes should

render::Vec3 to_sim(field::Vec3 p) { return render::Vec3{p.x * kSim, p.y * kSim, p.z * kSim}; }

render::DrawItem cube_item(field::Vec3 centre, double half, render::Color colour, render::Layer layer = render::Layer::Body)
{
    render::DrawItem item;
    item.shape = render::Shape::Cube;
    item.layer = layer;
    item.position = to_sim(centre - field::Vec3{0.0, 0.0, half});
    item.half_w = item.half_h = half * kSim;
    item.height = 2.0 * half * kSim;
    item.color = colour;
    return item;
}

} // namespace

void Sandbox::submit_world()
{
    const field::Playground& world = *world_;
    world_view_.update(world, renderer_, camera_);
    world_view_.draw(renderer_, world.config().terrain.chunk * world.config().terrain.cell);
    submit_figures();
    submit_field_visuals();
    effects_.submit(renderer_, kSim);
    // Where the cursor lands: a small marker on the surface.
    if (aim_hit_) {
        const double view_m = camera_.ortho_half_height() * 2.0 / kSim;
        const double size = std::max(0.07, view_m * 0.004);
        const bool can = aim_live_ || tools_[static_cast<std::size_t>(tool_)].kind == Tool::Kind::Throw;
        renderer_.submit(cube_item(aim_world_ + field::Vec3{0.0, 0.0, size}, size, can ? render::Color{255, 240, 150, 200} : render::Color{150, 150, 160, 160}, render::Layer::Field));
    }
}

void Sandbox::submit_figures()
{
    const field::Playground& world = *world_;
    const field::Vec3 o = world.origin();
    figure_mesh_.clear();
    for (const field::Creature& c : world.creatures()) {
        const field::CreatureDef& def = world.law().creatures[static_cast<std::size_t>(c.def)];
        if (c.town >= 0) {
            mesh_person(person_pose(c, def), c.looks, c.bag, c.who.pid + 1, figure_mesh_);
            continue;
        }
        const field::VoxelFigure& beast = beast_figure(def.name);
        if (!beast.empty()) {
            beast.mesh(beast_pose(c, def), figure_mesh_);
        } else if (!def.fights.empty()) {
            // Someone of no town: looks of its own, what it fights with, plain clothes.
            field::Inventory gear;
            gear.worn[field::kWornBody] = world.items().find("wool_tunic");
            gear.worn[field::kWornLegs] = world.items().find("trousers");
            gear.worn[field::kWornFeet] = world.items().find("boots");
            for (std::size_t i = 0; i < world.items().items.size(); ++i) {
                if (world.items().items[i].weapon == def.fights) {
                    gear.worn[field::kWornHand] = static_cast<int>(i);
                    break;
                }
            }
            mesh_person(person_pose(c, def), field::looks_of_person(c.id * 7919u, c.id * 104729u, 0, 15.0), gear, c.id + 1u, figure_mesh_);
        } else {
            // A kind with no figure is drawn as a cube of its colour, as big as its body.
            const auto channel = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
            renderer_.submit(cube_item(c.body.position + o, c.body.radius, render::Color{channel(def.colour[0]), channel(def.colour[1]), channel(def.colour[2]), 255}));
        }
    }
    submit_proxies();
    // What is in the air in one piece: thrown balls, felled trees.
    world.mesh_pieces(figure_mesh_);
    figure_vertices_.resize(figure_mesh_.size());
    for (std::size_t i = 0; i < figure_mesh_.size(); ++i) {
        const field::MeshVertex& v = figure_mesh_[i];
        figure_vertices_[i] = render::ColorVertex{static_cast<float>(v.x * kSim), static_cast<float>(v.y * kSim), static_cast<float>(v.z * kSim), v.abgr};
    }
    renderer_.submit_triangles(figure_vertices_);
}

void Sandbox::update_proxies()
{
    const double view_m = camera_.ortho_half_height() * 2.0 / kSim;
    const double reach = std::max(60.0, view_m * 1.2);
    world_->people_beyond(focus_.x, focus_.y, reach, kMostProxies, proxies_);
    // Nearest first: they are the ones drawn whole.
    std::sort(proxies_.begin(), proxies_.end(), [&](const auto& a, const auto& b) {
        return std::hypot(a.at.x - focus_.x, a.at.y - focus_.y) < std::hypot(b.at.x - focus_.x, b.at.y - focus_.y);
    });
    if (outfits_.size() > 6000) {
        outfits_.clear();
    }
}

const Sandbox::Outfit& Sandbox::outfit_of(const field::Playground::PersonProxy& p)
{
    auto it = outfits_.find(p.pid);
    if (it == outfits_.end()) {
        Outfit o;
        world_->outfit(world_->society()->person(p.town, p.index), p.town, o.looks, o.worn, o.arm);
        it = outfits_.emplace(p.pid, std::move(o)).first;
    }
    return it->second;
}

// The people beyond the window, and the beasts that walk the map out there: figures near, two cubes far.
void Sandbox::submit_proxies()
{
    const field::Playground& world = *world_;
    const double pixels_per_metre = camera_.viewport_height() / std::max(camera_.ortho_half_height() * 2.0 / kSim, 1e-6);
    const bool whole = pixels_per_metre > 7.0; // a figure under about a dozen pixels is not worth its cubes
    std::size_t drawn_whole = 0;
    for (const field::Playground::PersonProxy& p : proxies_) {
        const Outfit& o = outfit_of(p);
        if (whole && drawn_whole < kMostFullProxies) {
            field::FigurePose pose;
            pose.size = o.looks.size;
            pose.feet = p.at;
            clips_.play("idle", figure_time_ * 0.2 + static_cast<double>(p.pid % 97) * 0.13, 1.0, pose);
            clips_.play("run", figure_time_ * (0.35 * p.speed) + static_cast<double>(p.pid % 89) * 0.31, std::min(1.0, p.speed / 4.0), pose);
            if (!o.arm.empty()) {
                clips_.play("hold_" + o.arm, 0.0, 1.0, pose);
            }
            pose.yaw = std::round(p.yaw / kEighth) * kEighth;
            mesh_person(pose, o.looks, o.worn, p.pid + 1, figure_mesh_);
            ++drawn_whole;
            continue;
        }
        // Far off: a body the colour of what it wears, a head the colour of its skin.
        const int body_item = o.worn.worn[field::kWornBody];
        const float* cloth = body_item >= 0 ? world.items().items[static_cast<std::size_t>(body_item)].colour : nullptr;
        const auto channel = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
        const std::uint32_t h = static_cast<std::uint32_t>(p.pid * 2654435761u);
        const render::Color coat = cloth != nullptr ? render::Color{channel(cloth[0]), channel(cloth[1]), channel(cloth[2]), 255}
                                                    : render::Color{static_cast<std::uint8_t>(120 + (h & 63)), static_cast<std::uint8_t>(100 + ((h >> 6) & 63)), 90, 255};
        const double tall = 1.7 * o.looks.size;
        renderer_.submit(cube_item(p.at + field::Vec3{0.0, 0.0, tall * 0.38}, tall * 0.2, coat));
        renderer_.submit(cube_item(p.at + field::Vec3{0.0, 0.0, tall * 0.86}, tall * 0.11, render::Color{205, 160, 125, 255}));
    }
    // Beasts beyond the window walk the map's columns once a second: drawn where they are, ambling.
    for (const field::FarBody& b : world.far_bodies()) {
        if (std::hypot(b.x - focus_.x, b.y - focus_.y) > std::max(60.0, camera_.ortho_half_height() * 2.0 / kSim * 1.2)) {
            continue;
        }
        const field::VoxelFigure& beast = beast_figure(b.kind);
        if (beast.empty() || !whole) {
            const int def = world.law().creature_id(b.kind);
            const float* c = def >= 0 ? world.law().creatures[static_cast<std::size_t>(def)].colour : nullptr;
            const auto channel = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
            renderer_.submit(cube_item(field::Vec3{b.x, b.y, b.z + 0.35}, 0.35, c != nullptr ? render::Color{channel(c[0]), channel(c[1]), channel(c[2]), 255} : render::Color{150, 120, 90, 255}));
            continue;
        }
        field::FigurePose pose;
        pose.feet = field::Vec3{b.x, b.y, b.z};
        clips_.play("run", figure_time_ * 0.4 + static_cast<double>(b.id % 89) * 0.31, 0.5, pose);
        pose.yaw = std::round(b.heading / kEighth) * kEighth;
        beast.mesh(pose, figure_mesh_);
    }
}

// The law's `show` lines: flame, smoke and mana in the open air as translucent cubes. Rebuilt once per field step, on the
// thread that ran it; the flicker is a hash of the cell and the step, so it is the same in every replay.
void Sandbox::build_field_visuals(const field::Playground& world, const field::FieldStep& step)
{
    const field::FieldSet& f = step.fields;
    const field::TerrainConfig& t = world.config().terrain;
    const field::Vec3 o{step.window_cx * t.chunk * t.cell, step.window_cy * t.chunk * t.cell, step.window_cz * t.chunk * t.cell};
    std::vector<render::DrawItem> items;
    const field::GridShape& g = f.shape();
    const int occ = world.law().field_id("occ");
    constexpr std::size_t kCap = 2400;
    const std::vector<field::FieldVisual>& visuals = world.law().visuals;
    visual_shown_.resize(visuals.size(), 1);
    for (std::size_t v = 0; v < visuals.size(); ++v) {
        const field::FieldVisual& visual = visuals[v];
        // A field that showed nothing and has not changed still shows nothing.
        if (!visual_shown_[v] && !step.changed[static_cast<std::size_t>(visual.field)]) {
            continue;
        }
        const std::size_t before = items.size();
        const std::vector<float>& values = f.values(visual.field);
        const float above = static_cast<float>(visual.above);
        for (std::size_t i = 0; i < values.size() && items.size() < kCap; ++i) {
            if (values[i] <= above || (occ >= 0 && f.values(occ)[i] >= 0.5f)) {
                continue;
            }
            const int z = static_cast<int>(i / (static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny)));
            const std::size_t plane = i % (static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny));
            const int y = static_cast<int>(plane / static_cast<std::size_t>(g.nx));
            const int x = static_cast<int>(plane % static_cast<std::size_t>(g.nx));
            const double s = std::clamp((static_cast<double>(values[i]) - visual.above) / (visual.full - visual.above), 0.0, 1.0);
            std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u ^ static_cast<std::uint32_t>(step.step) * 40503u;
            h ^= h >> 13;
            h *= 0x5bd1e995u;
            const double flicker = 0.8 + 0.2 * static_cast<double>(h & 255u) / 255.0;
            const double half = g.cell * 0.5 * (0.35 + 0.65 * s) * flicker;
            const auto channel = [](float c) { return static_cast<std::uint8_t>(std::clamp(c, 0.0f, 1.0f) * 255.0f); };
            const render::Color colour{channel(visual.colour[0]), channel(visual.colour[1]), channel(visual.colour[2]),
                static_cast<std::uint8_t>(std::clamp(visual.alpha * (0.25 + 0.75 * s), 0.02, 0.98) * 255.0)};
            items.push_back(cube_item(field::Vec3{(x + 0.5) * g.cell, (y + 0.5) * g.cell, (z + 0.5) * g.cell} + o, half, colour, render::Layer::Field));
        }
        visual_shown_[v] = items.size() > before ? 1 : 0;
    }
    const std::lock_guard<std::mutex> lock(visual_mutex_);
    visual_next_ = std::move(items);
    visual_fresh_ = true;
}

void Sandbox::submit_field_visuals()
{
    {
        const std::lock_guard<std::mutex> lock(visual_mutex_);
        if (visual_fresh_) {
            visual_items_.swap(visual_next_);
            visual_fresh_ = false;
        }
    }
    for (const render::DrawItem& item : visual_items_) {
        renderer_.submit(item);
    }
}

// ---- figures ----------------------------------------------------------------------------------------------------------

const field::VoxelFigure& Sandbox::body_for(const field::Looks& looks)
{
    const std::string text = field::body_model(looks); // a body is worked out from genes, as text
    auto it = bodies_.find(text);
    if (it == bodies_.end()) {
        std::vector<std::string> errors;
        it = bodies_.emplace(text, field::VoxelFigure::parse(text, errors)).first;
        for (const std::string& e : errors) {
            std::cerr << e << "\n";
        }
    }
    return it->second;
}

const field::VoxelFigure& Sandbox::clothes_for(int item, int sex)
{
    const field::ItemDef& def = world_->items().items[static_cast<std::size_t>(item)];
    const std::string key = def.model + (sex == 0 ? "/f" : "/m");
    auto it = clothes_.find(key);
    if (it == clothes_.end()) {
        std::string text;
        for (const char* root : {"", "../", "../../"}) {
            std::ifstream in(std::string(root) + "assets/world/clothes/" + def.model + ".model", std::ios::binary);
            if (in) {
                std::ostringstream all;
                all << in.rdbuf();
                text = all.str();
                break;
            }
        }
        std::vector<std::string> errors;
        it = clothes_.emplace(key, field::VoxelFigure::parse(field::for_sex(text, sex), errors)).first;
    }
    return it->second;
}

const field::VoxelFigure& Sandbox::beast_figure(const std::string& name)
{
    auto it = beasts_.find(name);
    if (it == beasts_.end()) {
        std::vector<std::string> errors;
        it = beasts_.emplace(name, field::VoxelFigure::load("assets/world/beasts/" + name + ".model", errors)).first;
    }
    return it->second;
}

void Sandbox::mesh_person(const field::FigurePose& pose, const field::Looks& looks, const field::Inventory& wearing, std::uint64_t dye,
    std::vector<field::MeshVertex>& out)
{
    const field::Playground& world = *world_;
    field::Looks shown = looks;
    shown.covered = wearing.worn[field::kWornHead] >= 0;
    body_for(shown).mesh(pose, out);
    for (int slot = 0; slot < field::kWornHand; ++slot) {
        const int item = wearing.worn[static_cast<std::size_t>(slot)];
        if (item < 0) {
            continue;
        }
        const std::size_t first = out.size();
        clothes_for(item, looks.sex).mesh(pose, out, true);
        // The same cloth is dyed by whoever wove it; iron is iron.
        if (dye != 0 && world.items().items[static_cast<std::size_t>(item)].for_job.empty()) {
            const std::uint64_t h = (dye + static_cast<std::uint64_t>(item) * 31u) * 0x9E3779B97F4A7C15ull;
            const std::uint32_t r = 170u + static_cast<std::uint32_t>((h >> 20) & 85u);
            const std::uint32_t g = 170u + static_cast<std::uint32_t>((h >> 30) & 85u);
            const std::uint32_t b = 170u + static_cast<std::uint32_t>((h >> 40) & 85u);
            for (std::size_t i = first; i < out.size(); ++i) {
                std::uint32_t& v = out[i].abgr;
                v = (v & 0xff000000u) | (((v >> 16) & 0xffu) * b / 255u) << 16 | (((v >> 8) & 0xffu) * g / 255u) << 8 | ((v & 0xffu) * r / 255u);
            }
        }
    }
    const int hand = wearing.worn[field::kWornHand];
    if (hand >= 0) {
        const int weapon = world.melee().weapon_slot(world.items().items[static_cast<std::size_t>(hand)].weapon);
        if (weapon >= 0 && weapon < static_cast<int>(weapon_models_.size()) && !weapon_models_[static_cast<std::size_t>(weapon)].empty()) {
            body_for(shown).mesh_held(pose, weapon_models_[static_cast<std::size_t>(weapon)], "hand_r", out);
        }
    }
}

// A beast has no hands: what it does, it does with the whole of itself.
field::FigurePose Sandbox::beast_pose(const field::Creature& c, const field::CreatureDef& def) const
{
    const field::Playground& world = *world_;
    const field::PlayBody& body = c.body;
    field::FigurePose pose;
    pose.feet = body.position + world.origin() - field::Vec3{0.0, 0.0, body.radius};
    const double speed = std::hypot(body.velocity.x, body.velocity.y);
    if (!body.grounded) {
        clips_.play("beast_air", 0.0, 1.0, pose);
    } else {
        clips_.play("run", figure_time_ * (0.3 * speed) + c.id * 0.31, std::min(1.0, speed / 2.0), pose);
    }
    double yaw = def.fights.empty() ? (field::length(c.want) > 1e-6 ? std::atan2(c.want.y, c.want.x) : c.heading) : c.face;
    if (const double phase = world.phase_of(body); phase >= 0.0) {
        clips_.play(world.melee().moves[static_cast<std::size_t>(body.swing.move)].anim, phase, 1.0, pose);
        yaw = std::atan2(body.swing.dir_y, body.swing.dir_x);
    }
    if (body.stagger > 0.0) {
        pose.bones[field::kRoot].swing -= 0.5 * std::min(1.0, body.stagger * 3.0);
    }
    pose.yaw = std::round(yaw / kEighth) * kEighth;
    pose.hurt = static_cast<float>(std::clamp(1.0 - body.hp / std::max(body.hp_max, 1.0), 0.0, 1.0) * 0.5);
    return pose;
}

field::FigurePose Sandbox::person_pose(const field::Creature& c, const field::CreatureDef& def) const
{
    const field::Playground& world = *world_;
    const field::PlayBody& body = c.body;
    field::FigurePose pose;
    pose.size = c.town >= 0 ? c.looks.size : 1.0;
    pose.feet = body.position + world.origin() - field::Vec3{0.0, 0.0, body.radius};
    const double speed = std::hypot(body.velocity.x, body.velocity.y);
    clips_.play("idle", figure_time_ * 0.2 + c.id * 0.13, 1.0, pose);
    if (!body.grounded) {
        clips_.play(body.velocity.z > 1.0 ? "rising" : "falling", 0.0, 1.0, pose);
    } else {
        clips_.play("run", figure_time_ * (0.35 * speed) + c.id * 0.31, std::min(1.0, speed / 4.0), pose);
    }
    clips_.play("hold_" + (c.arm.empty() ? def.fights : c.arm), 0.0, 1.0, pose);
    double yaw = c.face;
    if (body.guard >= 0.0) {
        clips_.play("guard", 0.0, 1.0, pose);
    }
    if (const double phase = world.phase_of(body); phase >= 0.0) {
        const field::MoveDef& move = world.melee().moves[static_cast<std::size_t>(body.swing.move)];
        const double within = phase - std::floor(phase);
        const double snapped = phase < 1.0 ? phase * phase * (3.0 - 2.0 * phase)
            : phase < 2.0                  ? 1.0 + (1.0 - (1.0 - within) * (1.0 - within) * (1.0 - within))
                                           : phase;
        pose.bones[field::kGrip] = field::BonePose{};
        clips_.play(move.anim, snapped, std::min({1.0, phase * 5.0 + 0.2, (3.0 - phase) * 2.0}), pose);
        yaw = std::atan2(body.swing.dir_y, body.swing.dir_x);
    }
    if (body.stagger > 0.0) {
        pose.bones[field::kRoot].swing -= 0.5 * std::min(1.0, body.stagger * 3.0);
        pose.bones[field::kHead].swing -= 0.3;
    }
    pose.yaw = std::round(yaw / kEighth) * kEighth;
    pose.hurt = static_cast<float>(std::clamp(1.0 - body.hp / std::max(body.hp_max, 1.0), 0.0, 1.0) * 0.5);
    pose.bones[field::kCape].swing = -0.3 - speed * 0.06;
    return pose;
}

} // namespace fieldengine::sandbox

#pragma once

#include "world/FigureAnim.hpp"
#include "world/Playground.hpp"
#include "world/VoxelFigure.hpp"
#include "render/BgfxRenderer.hpp"
#include "render/Camera.hpp"
#include "render/ImGuiBgfx.hpp"
#include "sandbox/Effects.hpp"
#include "sandbox/WorldView.hpp"

#include <SDL3/SDL.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct ImDrawList;

// THE SANDBOX: a window onto the field engine with nobody in it but you, as a camera. You fly over the generated
// world and act on it: throw balls of voxels (rigid bodies that become voxels again where they rest), pour mana
// (fire, force, water, lava, stone, as the law in assets/law makes of it), set creatures down, watch the people of
// the towns go about their lives, and change the law's params while it runs.
//
// The engine knows nothing of this file. Everything here goes in through the engine's public surface
// (field::Playground) and everything drawn is read off it.
namespace fieldengine::sandbox {

class Sandbox {
public:
    bool initialize();
    void run();
    void shutdown();

private:
    // A thing the left mouse button does.
    struct Tool {
        enum class Kind { Throw, Pour, Spawn };
        std::string name;
        std::string about;
        Kind kind = Kind::Throw;
        field::BallKind ball = field::BallKind::Stone;
        double radius = 0.8;
        int material = 0;
        std::string pour;     // Pour: by name, from assets/law/pours.law
        std::string creature; // Spawn: by name, from the law
    };

    // --- the shell -------------------------------------------------------------------------------------------------
    void process_events();
    void frame();

    // --- the world (Sandbox.cpp) -----------------------------------------------------------------------------------
    bool load_world();       // the heavy part (law, kernels, continent), on a worker thread while the window shows progress
    bool finish_world();     // the rest, on the main thread, once it is there
    void loading_frame();
    void set_stage(const std::string& stage);
    void make_tools();
    void update_camera(double dt);
    void update_aim();
    void use_tool(double dt);
    void place_observer();
    void step_world(double dt);
    void play_events();
    void go_to_town(int step);
    void demo_turn();
    int demo_frame_ = 0;
    [[nodiscard]] field::Vec3 lattice_of(field::Vec3 world_metres) const;

    // --- what is drawn (Draw.cpp) ----------------------------------------------------------------------------------
    void submit_world();
    void submit_figures();
    void build_field_visuals(const field::Playground& world, const field::FieldStep& step);
    void submit_field_visuals();
    const field::VoxelFigure& body_for(const field::Looks& looks);
    const field::VoxelFigure& clothes_for(int item, int sex);
    const field::VoxelFigure& beast_figure(const std::string& name);
    void mesh_person(const field::FigurePose& pose, const field::Looks& looks, const field::Inventory& wearing,
        std::uint64_t dye, std::vector<field::MeshVertex>& out);
    [[nodiscard]] field::FigurePose beast_pose(const field::Creature& c, const field::CreatureDef& def) const;
    [[nodiscard]] field::FigurePose person_pose(const field::Creature& c, const field::CreatureDef& def) const;
    // The people beyond the live window (no bodies: where their day has them) and how each looks, worked out once.
    struct Outfit {
        field::Looks looks;
        field::Inventory worn;
        std::string arm;
    };
    void update_proxies();
    const Outfit& outfit_of(const field::Playground::PersonProxy& p);
    void submit_proxies();
    std::vector<field::Playground::PersonProxy> proxies_;
    std::map<std::uint64_t, Outfit> outfits_;
    static constexpr std::size_t kMostProxies = 700;
    static constexpr std::size_t kMostFullProxies = 160; // drawn as whole figures; the rest, further off, as two cubes

    // --- the panels (Panels.cpp) -----------------------------------------------------------------------------------
    void render_ui();
    void render_tools_panel();
    void render_world_panel();
    void render_law_panel();
    void render_profiler();
    void render_help();
    void draw_overlay(ImDrawList* overlay);
    [[nodiscard]] render::Vec2 to_overlay(field::Vec3 world_metres) const;

    SDL_Window* window_ = nullptr;
    bool running_ = false;
    bool imgui_ready_ = false;
    bool imgui_platform_ready_ = false;
    bool imgui_bgfx_ready_ = false;
    bool renderer_ready_ = false;
    render::BgfxRenderer renderer_;
    render::ImGuiBgfx imgui_bgfx_;
    render::Camera camera_;
    long frames_left_ = -1; // SANDBOX_FRAMES=<n>: quit after n frames (a launch check)
    std::string shot_path_; // SANDBOX_SHOT=<path.tga>: and capture the screen before that
    std::string gif_dir_;   // SANDBOX_GIF=<dir>: with SANDBOX_DEMO, the law panel open, a param changed live, frames saved

    std::optional<field::Playground> world_;
    std::future<bool> loading_;
    bool loaded_ = false;
    std::mutex stage_mutex_;
    std::string stage_ = "starting";
    std::chrono::steady_clock::time_point load_started_{};
    app::WorldView world_view_;
    app::Effects effects_;
    double accumulator_ = 0.0;
    double step_ms_ = 0.0;
    bool paused_ = false;
    double time_scale_ = 1.0;

    // The camera: a point on the ground it looks at, a turn and a zoom. The engine needs someone for its window of
    // simulated chunks to follow, so its player body is kept, unseen and unhurt, under that point.
    field::Vec3 focus_{};    // world metres
    double focus_z_ = 0.0;   // smoothed ground height under it
    bool focus_set_ = false;
    double yaw_ = 45.0;
    double zoom_ = 0.55;
    double zoom_target_ = 0.55;
    static constexpr double kZoomNear = 4.0;
    static constexpr double kZoomFar = 2e-4;
    int town_ = -1;          // the last town gone to

    // Aim: the cursor's ray, stopped by the first surface (the one bodies stand on).
    field::Vec3 aim_{};      // lattice metres
    field::Vec3 aim_world_{}; // the same, world metres
    bool aim_hit_ = false;
    bool aim_live_ = false;  // in the live window, where the law runs: pouring and setting creatures down only go there
    field::Vec3 to_eye_{0.0, 0.0, 1.0};

    // Tools.
    std::vector<Tool> tools_;
    int tool_ = 0;
    double repeat_left_ = 0.0;
    double throw_speed_ = 26.0;
    bool explode_on_impact_ = false;
    int pour_pick_ = 0;
    int creature_pick_ = 0;
    std::vector<std::string> pour_names_;
    std::vector<std::string> creature_names_;
    int balls_thrown_ = 0;

    // Panels.
    bool show_ui_ = true;
    bool show_law_ = false;
    bool show_profiler_ = false;
    std::vector<float> profile_history_; // the card's (or the CPU's) time for each field step, newest last
    double profile_last_ = -1.0;
    bool show_help_ = true;
    char law_filter_[64] = {};
    std::uint32_t hovered_ = 0;

    // Figures: bodies, clothes and beasts are text in assets/world, built once each.
    field::ClipBook clips_;
    std::map<std::string, field::VoxelFigure> bodies_;
    std::map<std::string, field::VoxelFigure> clothes_;
    std::map<std::string, field::VoxelFigure> beasts_;
    std::vector<field::VoxelFigure> weapon_models_;
    std::vector<field::MeshVertex> figure_mesh_;
    std::vector<render::ColorVertex> figure_vertices_;
    double figure_time_ = 0.0;

    // The law's `show` lines (flame, smoke, mana in the air), rebuilt on the thread that ran each field step.
    std::vector<render::DrawItem> visual_items_;
    std::vector<render::DrawItem> visual_next_;
    std::vector<char> visual_shown_;
    bool visual_fresh_ = false;
    std::mutex visual_mutex_;
};

} // namespace fieldengine::sandbox

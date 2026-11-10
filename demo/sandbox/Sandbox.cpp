// The sandbox's shell and its hands: the window, the frame, the camera, the tools. What is drawn is Draw.cpp, the panels
// Panels.cpp.

#include "sandbox/Sandbox.hpp"

#include "field/GpuSolver.hpp"
#include "world/Melee.hpp"
#include "render/Units.hpp"

#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace fieldengine::sandbox {

namespace {

constexpr double kSim = render::units::sim_units_per_metre; // the renderer works in units of 100 m, the engine in metres
constexpr double kPitch = 35.264;                           // the true isometric angle

render::Vec3 to_sim(field::Vec3 p) { return render::Vec3{p.x * kSim, p.y * kSim, p.z * kSim}; }
field::Vec3 to_metres(render::Vec3 p) { return field::Vec3{p.x / kSim, p.y / kSim, p.z / kSim}; }

bool env_is(const char* name, const char* value)
{
    const char* raw = std::getenv(name);
    return raw != nullptr && std::string(raw) == value;
}

} // namespace

// ---- the shell --------------------------------------------------------------------------------------------------------

bool Sandbox::initialize()
{
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return false;
    }
    window_ = SDL_CreateWindow("Field Engine sandbox", 1600, 900, SDL_WINDOW_RESIZABLE);
    if (window_ == nullptr) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        return false;
    }
    render::RenderConfig config;
    SDL_GetWindowSizeInPixels(window_, &config.width, &config.height);
    const SDL_PropertiesID props = SDL_GetWindowProperties(window_);
#if defined(SDL_PLATFORM_WIN32)
    config.native_window_handle = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
#elif defined(SDL_PLATFORM_LINUX)
    config.native_window_handle = reinterpret_cast<void*>(SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    config.native_display_handle = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
#endif
    config.use_noop_backend = env_is("SANDBOX_NOOP", "1");
    if (!renderer_.init(config)) {
        std::cerr << "bgfx renderer init failed.\n";
        return false;
    }
    renderer_ready_ = true;
    if (const char* frames = std::getenv("SANDBOX_FRAMES")) {
        frames_left_ = std::strtol(frames, nullptr, 10);
    }
    if (const char* shot = std::getenv("SANDBOX_SHOT")) {
        shot_path_ = shot;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    imgui_ready_ = true;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.86f;
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard; // the keys move the camera, not the focus between windows
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    if (!ImGui_ImplSDL3_InitForOther(window_)) {
        std::cerr << "ImGui SDL3 init failed.\n";
        return false;
    }
    imgui_platform_ready_ = true;
    if (!imgui_bgfx_.init()) {
        std::cerr << "ImGui bgfx init failed.\n";
        return false;
    }
    imgui_bgfx_ready_ = true;

    // The world takes a while (the continent's layers, the law compiled to CUDA, the creatures' world models): it is
    // built on a worker thread while the window says what it is doing, so the window is never a black, stuck rectangle.
    load_started_ = std::chrono::steady_clock::now();
    loading_ = std::async(std::launch::async, [this] { return load_world(); });
    running_ = true;
    return true;
}

void Sandbox::set_stage(const std::string& stage)
{
    const std::lock_guard<std::mutex> lock(stage_mutex_);
    stage_ = stage;
    std::cout << "[" << std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started_).count() << " s] " << stage << "\n";
}

void Sandbox::run()
{
    while (running_) {
        process_events();
        if (!loaded_) {
            if (loading_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                loading_frame();
                continue;
            }
            if (!loading_.get() || !finish_world()) {
                running_ = false;
                break;
            }
            loaded_ = true;
        }
        frame();
    }
}

// What the window shows while the world is built: where it is, and how long it has taken.
void Sandbox::loading_frame()
{
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(window_, &width, &height);
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    const ImGuiIO& io = ImGui::GetIO();
    std::string stage;
    {
        const std::lock_guard<std::mutex> lock(stage_mutex_);
        stage = stage_;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started_).count();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin("##loading", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
    ImGui::Text("Field Engine");
    ImGui::Separator();
    ImGui::Text("%s", stage.c_str());
    const char spin[4] = {'|', '/', '-', '\\'};
    ImGui::TextDisabled("%c  %.0f s", spin[static_cast<int>(seconds * 6.0) % 4], seconds);
    ImGui::TextDisabled("The first start compiles the law for this card (NVRTC); later starts take it from the cache.");
    ImGui::End();
    ImGui::Render();
    camera_.set_viewport(static_cast<double>(width), static_cast<double>(height));
    renderer_.set_camera(camera_);
    renderer_.begin_frame();
    renderer_.end_frame();
    imgui_bgfx_.render(ImGui::GetDrawData(), 255);
    renderer_.present();
    SDL_Delay(15);
}

void Sandbox::shutdown()
{
    if (loading_.valid()) {
        loading_.wait(); // a window closed while the world was being built: let the builder finish first
    }
    world_.reset(); // first: its field steps report to the mesher and the visuals below
    world_view_.reset(renderer_);
    if (imgui_bgfx_ready_) {
        imgui_bgfx_.shutdown();
    }
    if (imgui_platform_ready_) {
        ImGui_ImplSDL3_Shutdown();
    }
    if (imgui_ready_) {
        ImGui::DestroyContext();
    }
    if (renderer_ready_) {
        renderer_.shutdown();
    }
    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
    }
    SDL_Quit();
    imgui_bgfx_ready_ = imgui_platform_ready_ = imgui_ready_ = renderer_ready_ = false;
    window_ = nullptr;
}

void Sandbox::process_events()
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);
        if (event.type == SDL_EVENT_QUIT ||
            (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window_))) {
            running_ = false;
        }
    }
}

void Sandbox::frame()
{
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(window_, &width, &height);
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    const ImGuiIO& io = ImGui::GetIO();
    // A scripted launch steps exactly one tick a frame, so a capture shows the same moment however fast the machine is.
    const double dt = frames_left_ > 0 ? field::Playground::kTick : std::min(static_cast<double>(io.DeltaTime), 0.25);
    figure_time_ += dt;

    camera_.set_viewport(static_cast<double>(width), static_cast<double>(height));
    update_camera(dt);
    update_aim();
    use_tool(dt);
    demo_turn();
    step_world(dt);
    update_proxies();
    play_events();
    effects_.update(dt);
    render_ui();

    ImGui::Render();
    renderer_.set_camera(camera_);
    renderer_.begin_frame();
    submit_world();
    renderer_.end_frame();
    imgui_bgfx_.render(ImGui::GetDrawData(), 255);
    renderer_.present();

    if (frames_left_ > 0) {
        if (!shot_path_.empty() && frames_left_ == 3) {
            renderer_.request_screenshot(shot_path_); // bgfx writes it a frame or two later
        }
        if (--frames_left_ == 0) {
            running_ = false;
            std::cout << "frames done; last tick " << step_ms_ << " ms, law " << world_->field_ms() << " ms, creatures "
                      << world_->creatures().size() << ", pieces " << world_->pieces().size() << ", proxies " << proxies_.size()
                      << ", figure vertices " << figure_vertices_.size() << ", show cubes " << visual_items_.size()
                      << ", ImGui vertices " << ImGui::GetDrawData()->TotalVtxCount << "\n";
        }
    }
}

// ---- the world --------------------------------------------------------------------------------------------------------

bool Sandbox::load_world()
{
    std::vector<std::string> errors;
    std::vector<field::LawSource> laws;
    std::string pours;
    std::string pours_path;
    std::optional<field::Playground> world;
    set_stage("Reading the law (assets/law/*.law)");
    if (field::Playground::read_default_law(laws, pours, pours_path, errors)) {
        field::Playground::Config config;
        config.open_world = true;
        config.async_fields = true;      // the law on its own thread: a frame never waits for it
        config.field_never_waits = true;
        config.law_checks_every = 8;
        config.field_every = 4;          // 15 Hz; each term's max_dt keeps it as fine as it asks
        config.gpu_fields = !env_is("SANDBOX_CPU", "1") && field::GpuSolver::unavailable().empty();
        config.window_chunks = config.gpu_fields ? 9 : 5; // 72 m of live world on the GPU, 40 m on the CPU
        config.window_layers = 3;
        config.people = true;            // the towns send out their people
        config.falling = true;           // what nothing holds up comes down in one piece
        config.local_life = true;        // things happen among the people on their own
        config.max_creatures = 110;
        config.observer = true;          // nobody plays: the window follows the camera, and nothing sees or smells it
        set_stage(config.gpu_fields ? "Compiling the law to CUDA kernels, and the creatures' world models (NVRTC)" : "Compiling the law");
        world = field::Playground::create(laws, pours, errors, config);
    }
    if (!world) {
        std::cerr << "The world failed to load (run from the project root, or from the build directory):\n";
        for (const std::string& e : errors) {
            std::cerr << "  " << e << "\n";
        }
        return false;
    }
    world_ = std::move(world);
    set_stage("Building the continent: plates, climate, rivers, ages of history, towns");
    world_->build_open_world();
    set_stage("Loading moves, weapons and animation");

    // People fight with the same moves a player would have: the book is data (assets/world/moves.world).
    std::vector<std::string> load_errors;
    world_->set_melee(field::load_moves(load_errors));
    for (const field::WeaponDef& weapon : world_->melee().weapons) {
        weapon_models_.push_back(weapon.model.empty() ? field::VoxelFigure{} : field::VoxelFigure::load(weapon.model, load_errors));
    }
    clips_ = field::ClipBook::load("assets/world/anim/body.anim", load_errors);
    for (const std::string& e : load_errors) {
        std::cerr << e << "\n";
    }
    set_stage("Starting");
    return true;
}

bool Sandbox::finish_world()
{
    // What the picture needs from each field step is worked out on the thread that ran it.
    world_view_.reset(renderer_);
    world_->set_field_listener([this](const field::FieldStep& step) {
        world_view_.mesher().on_fields(step, world_->config().terrain);
        build_field_visuals(*world_, step);
    });

    const field::Solver& solver = std::as_const(*world_).solver();
    std::cout << "Field engine: law on " << (solver.on_gpu() ? solver.gpu_name() : std::string("the CPU"))
              << ", " << world_->law().fields.size() << " fields, " << world_->law().terms.size() << " terms, "
              << world_->pours().size() << " named pours, " << world_->law().creatures.size() << " kinds of creature\n";

    focus_ = world_->player().position + world_->origin();
    focus_z_ = focus_.z;
    focus_set_ = true;
    make_tools();
    show_profiler_ = env_is("SANDBOX_PROFILER", "1");
    show_ui_ = !env_is("SANDBOX_CLEAN", "1"); // captures without panels, as Tab does
    if (env_is("SANDBOX_LEARNED", "1")) {
        world_->set_learned(true);
    }
    if (const char* zoom = std::getenv("SANDBOX_ZOOM")) {
        zoom_ = zoom_target_ = std::clamp(std::atof(zoom), kZoomFar, kZoomNear);
    }
    if (const char* town = std::getenv("SANDBOX_TOWN")) {
        go_to_town(std::atoi(town) + 1);
    }
    return true;
}

void Sandbox::make_tools()
{
    using K = Tool::Kind;
    tools_ = {
        {"Stone ball", "A ball of granite voxels thrown as one rigid body: it bounces, rolls, knocks people over, and is stone again where it rests.", K::Throw, field::BallKind::Stone, 1.0, 0},
        {"Boulder", "A big ball of basalt. Heavy: what it lands on stays under it.", K::Throw, field::BallKind::Stone, 1.8, 1},
        {"Wooden ball", "Wood: it bounces more, and once at rest it burns like a tree.", K::Throw, field::BallKind::Wood, 1.0},
        {"Ice ball", "Ice: by the law it melts where it is warm and the water runs.", K::Throw, field::BallKind::Ice, 1.0},
        {"Fire", "Fire mana: turf, trees and wooden houses catch and go on burning by themselves.", K::Pour, {}, 0, 0, "fireball"},
        {"Force", "Force mana breaks stone and wood into rubble, and its gradient throws bodies.", K::Pour, {}, 0, 0, "sledge"},
        {"Water", "Water falls, levels out, freezes, boils, and quenches lava into stone.", K::Pour, {}, 0, 0, "flood"},
        {"Lava", "Earth and fire mana at once: molten stone that flows and sets as it cools.", K::Pour, {}, 0, 0, "magma"},
        {"Earth wall", "Earth mana sets into stone where there is room.", K::Pour, {}, 0, 0, "earth"},
        {"Creature", "Sets a creature down. Its mind is lines of the law: what it seeks, eats, leaves behind, flees.", K::Spawn, {}, 0, 0, "", "grazer"},
        {"Any pour", "Any of assets/law/pours.law: mana, water, stone, storm, life, in a ball, a wall, a beam, a ring, a pillar or a cut.", K::Pour, {}, 0, 0, ""},
    };
    pour_names_.clear();
    for (const field::Pour& pour : world_->pours()) {
        pour_names_.push_back(pour.name);
    }
    creature_names_.clear();
    for (const field::CreatureDef& def : world_->law().creatures) {
        if (def.name != "player" && def.name != "town") {
            creature_names_.push_back(def.name);
        }
    }
}

field::Vec3 Sandbox::lattice_of(field::Vec3 world_metres) const
{
    return world_metres - world_->origin();
}

void Sandbox::update_camera(double dt)
{
    const ImGuiIO& io = ImGui::GetIO();
    const bool scripted = frames_left_ >= 0; // a scripted run takes no input from whoever is at the machine meanwhile
    const bool keys = !io.WantTextInput && !scripted && !io.KeyAlt;
    if (!scripted && !io.WantCaptureMouse && io.MouseWheel != 0.0f) {
        zoom_target_ = std::clamp(zoom_target_ * std::pow(1.2, static_cast<double>(io.MouseWheel)), kZoomFar, kZoomNear);
    }
    zoom_ = zoom_target_ + (zoom_ - zoom_target_) * std::exp(-14.0 * dt);
    if (keys) {
        if (ImGui::IsKeyDown(ImGuiKey_Q)) {
            yaw_ -= 90.0 * dt;
        }
        if (ImGui::IsKeyDown(ImGuiKey_E)) {
            yaw_ += 90.0 * dt;
        }
    }
    if (!scripted && !io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f)) {
        yaw_ -= static_cast<double>(io.MouseDelta.x) * 0.35;
    }
    yaw_ = std::fmod(yaw_ + 360.0, 360.0);

    // WASD relative to the screen: two short screen offsets, unprojected onto the ground.
    const double cx = camera_.viewport_width() * 0.5;
    const double cy = camera_.viewport_height() * 0.5;
    const double ground = focus_z_ * kSim;
    const field::Vec3 c0 = to_metres(camera_.screen_to_world(render::Vec2{cx, cy}, ground));
    const field::Vec3 cu = to_metres(camera_.screen_to_world(render::Vec2{cx, cy - 10.0}, ground));
    const field::Vec3 cr = to_metres(camera_.screen_to_world(render::Vec2{cx + 10.0, cy}, ground));
    const field::Vec3 up = field::normalize(field::Vec3{cu.x - c0.x, cu.y - c0.y, 0.0});
    const field::Vec3 right = field::normalize(field::Vec3{cr.x - c0.x, cr.y - c0.y, 0.0});
    if (keys) {
        const double ax = (ImGui::IsKeyDown(ImGuiKey_D) || ImGui::IsKeyDown(ImGuiKey_RightArrow) ? 1.0 : 0.0) -
            (ImGui::IsKeyDown(ImGuiKey_A) || ImGui::IsKeyDown(ImGuiKey_LeftArrow) ? 1.0 : 0.0);
        const double ay = (ImGui::IsKeyDown(ImGuiKey_W) || ImGui::IsKeyDown(ImGuiKey_UpArrow) ? 1.0 : 0.0) -
            (ImGui::IsKeyDown(ImGuiKey_S) || ImGui::IsKeyDown(ImGuiKey_DownArrow) ? 1.0 : 0.0);
        // As fast as the view is wide: a crossing takes the same time at every zoom.
        const double view_m = camera_.ortho_half_height() * 2.0 / kSim;
        const double speed = std::max(6.0, 0.8 * view_m) * (ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0 : 1.0);
        focus_ = focus_ + (right * ax + up * ay) * (speed * dt);
    }
    // The right button drags the ground under the cursor.
    if (!scripted && !io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
        const double sx = io.DisplayFramebufferScale.x;
        const double sy = io.DisplayFramebufferScale.y;
        const render::Vec2 now{io.MousePos.x * sx, io.MousePos.y * sy};
        const render::Vec2 before{(io.MousePos.x - io.MouseDelta.x) * sx, (io.MousePos.y - io.MouseDelta.y) * sy};
        const field::Vec3 a = to_metres(camera_.screen_to_world(before, ground));
        const field::Vec3 b = to_metres(camera_.screen_to_world(now, ground));
        focus_ = focus_ - field::Vec3{b.x - a.x, b.y - a.y, 0.0};
    }
    if (keys && ImGui::IsKeyPressed(ImGuiKey_T, false)) {
        go_to_town(1);
    }

    // The ground under the focus is where the unseen body stands; the view keeps it mid-screen, on a hill as in a valley.
    if (world_) {
        const field::PlayBody& you = world_->player();
        const double ground_z = you.position.z + world_->origin().z - you.radius;
        focus_z_ = ground_z + (focus_z_ - ground_z) * std::exp(-6.0 * dt);
    }
    const field::Vec3 shake = effects_.view_offset() * (1.0 / std::max(zoom_, 1.0));
    camera_.set_pan(render::Vec2{(focus_.x + shake.x) * kSim, (focus_.y + shake.y) * kSim});
    camera_.set_target_height((focus_z_ + shake.z) * kSim);
    camera_.set_vertical_extent(0.20); // 20 m of view at zoom 1
    camera_.set_iso_angles(yaw_, kPitch);
    camera_.set_zoom(zoom_);
    const render::Vec3 eye = camera_.eye();
    const render::Vec3 target = camera_.target();
    to_eye_ = field::normalize(field::Vec3{eye.x - target.x, eye.y - target.y, eye.z - target.z});
}

void Sandbox::update_aim()
{
    const ImGuiIO& io = ImGui::GetIO();
    const field::Vec3 o = world_->origin();
    const field::GridShape& g = world_->fields().shape();
    const field::TerrainConfig& t = world_->config().terrain;
    const render::Vec2 mouse{io.MousePos.x * io.DisplayFramebufferScale.x, io.MousePos.y * io.DisplayFramebufferScale.y};
    // From over the highest the world goes down to the sea floor: the live window's surface first, then the map's.
    const double sky = std::max(o.z + g.nz * g.cell, t.nz * t.cell) + 2.0;
    const field::Vec3 high = to_metres(camera_.screen_to_world(mouse, sky * kSim));
    const field::Vec3 low = to_metres(camera_.screen_to_world(mouse, 0.0));
    const std::optional<field::Vec3> hit = world_->pick_world(high, field::normalize(low - high), field::length(low - high) + 10.0);
    aim_hit_ = hit.has_value();
    aim_world_ = hit.value_or(low);
    aim_ = aim_world_ - o;
    aim_live_ = aim_hit_ && aim_.x > 0.0 && aim_.y > 0.0 && aim_.z > 0.0 && aim_.x < g.nx * g.cell && aim_.y < g.ny * g.cell && aim_.z < g.nz * g.cell;
}

void Sandbox::use_tool(double dt)
{
    const ImGuiIO& io = ImGui::GetIO();
    repeat_left_ = std::max(0.0, repeat_left_ - dt);
    // A scripted run takes no keys (whatever is pressed in other windows meanwhile), and Alt+Tab is not Tab.
    if (!io.WantTextInput && frames_left_ < 0 && !io.KeyAlt) {
        static const ImGuiKey kDigits[10] = {ImGuiKey_1, ImGuiKey_2, ImGuiKey_3, ImGuiKey_4, ImGuiKey_5, ImGuiKey_6, ImGuiKey_7, ImGuiKey_8, ImGuiKey_9, ImGuiKey_0};
        for (int k = 0; k < 10 && k < static_cast<int>(tools_.size()); ++k) {
            if (ImGui::IsKeyPressed(kDigits[k], false)) {
                tool_ = k;
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_P, false)) {
            paused_ = !paused_;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
            show_ui_ = !show_ui_;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) {
            show_law_ = !show_law_;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F2, false)) {
            show_profiler_ = !show_profiler_;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_H, false)) {
            show_help_ = !show_help_;
        }
    }
    if (frames_left_ >= 0 || io.WantCaptureMouse || !aim_hit_ || tools_.empty()) {
        return;
    }
    const Tool& tool = tools_[static_cast<std::size_t>(tool_)];
    const bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    const bool held = ImGui::IsMouseDown(ImGuiMouseButton_Left) && repeat_left_ <= 0.0;
    if (!clicked && !(held && tool.kind != Tool::Kind::Spawn)) {
        return;
    }
    field::Playground& world = *world_;
    const field::Vec3 target = aim_world_;
    if (tool.kind != Tool::Kind::Throw && !aim_live_) {
        return; // the law runs only in the live window: what would be poured beyond it has nothing to run it
    }
    switch (tool.kind) {
    case Tool::Kind::Throw: {
        // From over your shoulder, as the camera sees it, on the arc that lands where you point.
        const field::Vec3 from = target + to_eye_ * 14.0;
        const double flight = field::length(target - from) / std::max(throw_speed_, 1.0);
        const double gravity = world.config().gravity;
        const field::Vec3 velocity = (target - from) * (1.0 / flight) + field::Vec3{0.0, 0.0, 0.5 * gravity * flight};
        if (world.throw_ball(from, velocity, tool.radius, tool.ball, tool.material)) {
            ++balls_thrown_;
        }
        repeat_left_ = 0.12;
        break;
    }
    case Tool::Kind::Pour: {
        const std::string& name = tool.pour.empty() && !pour_names_.empty()
            ? pour_names_[static_cast<std::size_t>(std::clamp(pour_pick_, 0, static_cast<int>(pour_names_.size()) - 1))]
            : tool.pour;
        const int slot = world.pour_slot(name);
        if (slot < 0) {
            break;
        }
        const field::Pour& spell = world.pours()[static_cast<std::size_t>(slot)];
        // Away from the camera, level: the way a wall faces and a beam goes.
        const field::Vec3 away = field::normalize(field::Vec3{-to_eye_.x, -to_eye_.y, 0.0});
        field::Vec3 where = aim_;
        field::Vec3 dir = away;
        if (spell.shape == field::Pour::Shape::Beam) {
            const double reach = spell.range > 0.0 ? spell.range : 20.0;
            where = aim_ - away * (0.5 * reach) + field::Vec3{0.0, 0.0, 0.8};
            dir = away * reach;
        } else if (spell.shape == field::Pour::Shape::Cut) {
            where = aim_ - away * (0.5 * spell.range);
        } else if (spell.shape == field::Pour::Shape::Ball) {
            where = aim_ + field::Vec3{0.0, 0.0, 0.25 * spell.radius};
        }
        world.pour_at(slot, where, dir);
        repeat_left_ = 0.3;
        break;
    }
    case Tool::Kind::Spawn: {
        const std::string& kind = creature_names_.empty()
            ? tool.creature
            : creature_names_[static_cast<std::size_t>(std::clamp(creature_pick_, 0, static_cast<int>(creature_names_.size()) - 1))];
        world.spawn(kind, aim_ + field::Vec3{0.0, 0.0, 0.8});
        break;
    }
    }
}

// The engine's window of simulated chunks follows its player body. Here nobody plays (Config::observer): the body is only the
// point the window follows, kept under the camera's focus, standing on whatever is there.
void Sandbox::place_observer()
{
    field::Playground& world = *world_;
    const field::TerrainConfig& t = world.config().terrain;
    const double chunk_m = t.chunk * t.cell;
    const int fcx = static_cast<int>(std::floor(focus_.x / chunk_m));
    const int fcy = static_cast<int>(std::floor(focus_.y / chunk_m));
    const int mid = world.window_chunks() / 2;
    // A long way at once (a drag at a far zoom, a jump to a town): the window goes there in one move.
    if (std::abs(fcx - (world.window_cx() + mid)) > 2 || std::abs(fcy - (world.window_cy() + mid)) > 2) {
        world.center_window_on(fcx, fcy);
    }
    const field::GridShape& g = world.fields().shape();
    field::PlayBody& you = world.player();
    you.radius = 0.05;
    field::Vec3 p = lattice_of(focus_);
    p.x = std::clamp(p.x, 0.5, g.nx * g.cell - 0.5);
    p.y = std::clamp(p.y, 0.5, g.ny * g.cell - 0.5);
    const double top = g.nz * g.cell - 0.3;
    // Down from the top of the window to the first surface. Inside rock at the top: the ground is higher, the window goes
    // up a layer; nothing down to the floor: it goes down one.
    const std::optional<field::Vec3> below = world.pick(field::Vec3{p.x, p.y, top}, field::Vec3{0.0, 0.0, -1.0});
    p.z = below ? below->z + you.radius + 0.02 : 0.3;
    you.position = p;
    you.velocity = field::Vec3{};
}

void Sandbox::step_world(double dt)
{
    if (paused_) {
        return;
    }
    accumulator_ += dt * time_scale_;
    int steps = 0;
    while (accumulator_ >= field::Playground::kTick && steps < 8) {
        place_observer();
        field::Control control;
        control.aim = aim_;
        const auto start = std::chrono::steady_clock::now();
        world_->step(control);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        step_ms_ = std::max(ms, step_ms_ * 0.97);
        accumulator_ -= field::Playground::kTick;
        ++steps;
    }
    if (steps == 8) {
        accumulator_ = 0.0; // too far behind: the world runs slower than the clock rather than in bursts
    }
}

// What the world just did, as cubes and shaking. Nothing here reaches the simulation.
void Sandbox::play_events()
{
    field::Playground& world = *world_;
    for (const field::PlayEvent& e : world.events()) {
        switch (e.kind) {
        case field::PlayEvent::Kind::Impact: {
            const field::Pour& spell = world.pours()[static_cast<std::size_t>(std::max(e.pour, 0))];
            const app::Rgb colour{spell.colour[0], spell.colour[1], spell.colour[2]};
            effects_.burst(e.at, colour, std::max(0.6, e.power));
            effects_.shake(std::min(0.5, 0.08 * e.power));
            break;
        }
        case field::PlayEvent::Kind::Landing:
            if (e.power > 8.0) {
                effects_.shockwave(e.at, app::Rgb{0.55f, 0.5f, 0.42f}, std::min(3.0, 0.08 * e.power));
                effects_.shake(std::min(0.35, 0.01 * e.power));
                if (explode_on_impact_) {
                    world.pour_at(world.pour_slot("force"), lattice_of(e.at), field::Vec3{1.0, 0.0, 0.0});
                }
            }
            break;
        default:
            break;
        }
    }
    world.clear_events();
}

// SANDBOX_DEMO=1: a scripted minute around the focus, for captures: balls of every kind, a fire, a burst of force.
void Sandbox::demo_turn()
{
    static const bool demo = env_is("SANDBOX_DEMO", "1");
    if (!demo || !world_) {
        return;
    }
    ++demo_frame_;
    field::Playground& world = *world_;
    const field::GridShape& g = world.fields().shape();
    const auto ground = [&](double dx, double dy) {
        const field::Vec3 at = lattice_of(focus_) + field::Vec3{dx, dy, 0.0};
        return world.pick(field::Vec3{at.x, at.y, g.nz * g.cell - 0.3}, field::Vec3{0.0, 0.0, -1.0});
    };
    if (demo_frame_ % 18 == 0 && demo_frame_ < 300) {
        const double a = 0.9 * demo_frame_ / 18.0;
        if (const std::optional<field::Vec3> hit = ground(5.0 * std::cos(a), 5.0 * std::sin(a))) {
            const field::Vec3 target = *hit + world.origin();
            const field::Vec3 from = target + to_eye_ * 14.0;
            const double flight = field::length(target - from) / throw_speed_;
            const field::Vec3 v = (target - from) * (1.0 / flight) + field::Vec3{0.0, 0.0, 0.5 * world.config().gravity * flight};
            const field::BallKind kind = (demo_frame_ / 18) % 3 == 0 ? field::BallKind::Wood : (demo_frame_ / 18) % 3 == 1 ? field::BallKind::Ice : field::BallKind::Stone;
            world.throw_ball(from, v, 1.0, kind, 3);
        }
    }
    if (demo_frame_ == 120) {
        if (const std::optional<field::Vec3> hit = ground(-6.0, 3.0)) {
            world.pour_at(world.pour_slot("fireball"), *hit, field::Vec3{1.0, 0.0, 0.0});
        }
    }
    if (demo_frame_ == 200) {
        if (const std::optional<field::Vec3> hit = ground(3.0, -6.0)) {
            world.pour_at(world.pour_slot("flood"), *hit + field::Vec3{0.0, 0.0, 1.0}, field::Vec3{1.0, 0.0, 0.0});
        }
    }
}

void Sandbox::go_to_town(int step)
{
    field::Playground& world = *world_;
    if (world.map() == nullptr) {
        return;
    }
    std::vector<const field::Settlement*> towns;
    for (const field::Settlement& s : world.map()->settlements()) {
        if (s.tier >= 1 && s.fell < 0) {
            towns.push_back(&s);
        }
    }
    if (towns.empty()) {
        return;
    }
    town_ = ((town_ + step) % static_cast<int>(towns.size()) + static_cast<int>(towns.size())) % static_cast<int>(towns.size());
    const field::Settlement& s = *towns[static_cast<std::size_t>(town_)];
    const field::TerrainConfig& t = world.config().terrain;
    focus_ = field::Vec3{(s.x + 0.5) * t.cell, (s.y + 0.5) * t.cell, (s.ground + 1) * t.cell};
    focus_z_ = focus_.z;
    world.center_window_on(s.x / t.chunk, s.y / t.chunk, (s.ground + 4) / t.chunk);
    place_observer();
}

} // namespace fieldengine::sandbox

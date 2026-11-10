// The sandbox's panels: the tools, what the engine is doing, the law (every param live, every equation as it runs), and
// what is over the heads of the people.

#include "sandbox/Sandbox.hpp"

#include "world/Society.hpp"
#include "world/WorldMap.hpp"
#include "render/Units.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace fieldengine::sandbox {

namespace {

constexpr double kSim = render::units::sim_units_per_metre;

const char* tool_kind(int kind)
{
    static const char* const kNames[3] = {"throw", "pour", "spawn"};
    return kNames[std::clamp(kind, 0, 2)];
}

} // namespace

render::Vec2 Sandbox::to_overlay(field::Vec3 world_metres) const
{
    const ImGuiIO& io = ImGui::GetIO();
    const double sx = io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0;
    const double sy = io.DisplayFramebufferScale.y > 0.0f ? io.DisplayFramebufferScale.y : 1.0;
    const render::Vec2 p = camera_.world_to_screen(render::Vec3{world_metres.x * kSim, world_metres.y * kSim, world_metres.z * kSim});
    return render::Vec2{p.x / sx, p.y / sy};
}

void Sandbox::render_ui()
{
    draw_overlay(ImGui::GetBackgroundDrawList()); // under the panels, over the world
    if (!show_ui_) {
        return;
    }
    render_tools_panel();
    render_world_panel();
    if (show_law_) {
        render_law_panel();
    }
    if (show_profiler_) {
        render_profiler();
    }
    if (show_help_) {
        render_help();
    }
}

void Sandbox::render_tools_panel()
{
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 0.0f), ImGuiCond_Always);
    ImGui::Begin("Tools", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    for (std::size_t i = 0; i < tools_.size(); ++i) {
        const Tool& tool = tools_[i];
        char label[96];
        if (i < 10) {
            std::snprintf(label, sizeof(label), "%zu  %s", (i + 1) % 10, tool.name.c_str());
        } else {
            std::snprintf(label, sizeof(label), "   %s", tool.name.c_str());
        }
        if (ImGui::Selectable(label, tool_ == static_cast<int>(i))) {
            tool_ = static_cast<int>(i);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s (%s)", tool.about.c_str(), tool_kind(static_cast<int>(tool.kind)));
        }
    }
    ImGui::Separator();
    const Tool& tool = tools_[static_cast<std::size_t>(tool_)];
    ImGui::TextWrapped("%s", tool.about.c_str());
    if (tool.kind == Tool::Kind::Throw) {
        float speed = static_cast<float>(throw_speed_);
        if (ImGui::SliderFloat("speed, m/s", &speed, 5.0f, 60.0f, "%.0f")) {
            throw_speed_ = speed;
        }
        float radius = static_cast<float>(tool.radius);
        if (ImGui::SliderFloat("radius, m", &radius, 0.25f, 3.0f, "%.2f")) {
            tools_[static_cast<std::size_t>(tool_)].radius = radius;
        }
        ImGui::Checkbox("burst of force where it lands", &explode_on_impact_);
    } else if (tool.kind == Tool::Kind::Pour && tool.pour.empty() && !pour_names_.empty()) {
        pour_pick_ = std::clamp(pour_pick_, 0, static_cast<int>(pour_names_.size()) - 1);
        if (ImGui::BeginCombo("pour", pour_names_[static_cast<std::size_t>(pour_pick_)].c_str())) {
            for (std::size_t i = 0; i < pour_names_.size(); ++i) {
                if (ImGui::Selectable(pour_names_[i].c_str(), pour_pick_ == static_cast<int>(i))) {
                    pour_pick_ = static_cast<int>(i);
                }
            }
            ImGui::EndCombo();
        }
        const int slot = world_->pour_slot(pour_names_[static_cast<std::size_t>(pour_pick_)]);
        if (slot >= 0) {
            ImGui::TextWrapped("%s", world_->pours()[static_cast<std::size_t>(slot)].about.c_str());
        }
    } else if (tool.kind == Tool::Kind::Spawn && !creature_names_.empty()) {
        creature_pick_ = std::clamp(creature_pick_, 0, static_cast<int>(creature_names_.size()) - 1);
        if (ImGui::BeginCombo("kind", creature_names_[static_cast<std::size_t>(creature_pick_)].c_str())) {
            for (std::size_t i = 0; i < creature_names_.size(); ++i) {
                if (ImGui::Selectable(creature_names_[i].c_str(), creature_pick_ == static_cast<int>(i))) {
                    creature_pick_ = static_cast<int>(i);
                }
            }
            ImGui::EndCombo();
        }
    }
    ImGui::Separator();
    ImGui::Checkbox("pause (P)", &paused_);
    float scale = static_cast<float>(time_scale_);
    if (ImGui::SliderFloat("time", &scale, 0.1f, 2.0f, "x%.2f")) {
        time_scale_ = scale;
    }
    if (ImGui::Button("next town (T)")) {
        go_to_town(1);
    }
    ImGui::SameLine();
    if (ImGui::Button("law (F1)")) {
        show_law_ = !show_law_;
    }
    ImGui::SameLine();
    if (ImGui::Button("profiler (F2)")) {
        show_profiler_ = !show_profiler_;
    }
    ImGui::SameLine();
    if (ImGui::Button("help (H)")) {
        show_help_ = !show_help_;
    }
    ImGui::End();
}

void Sandbox::render_world_panel()
{
    const field::Playground& world = *world_;
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 10.0f, 10.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::Begin("Engine", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
    const field::Solver& solver = world.solver();
    ImGui::Text("law on %s", solver.on_gpu() ? solver.gpu_name().c_str() : "the CPU (OpenMP)");
    ImGui::Text("tick %.2f ms   field step %.1f ms   frame %.1f ms", step_ms_, world.field_ms(), 1000.0 * io.DeltaTime);
    const field::GridShape& g = world.fields().shape();
    ImGui::Text("window %d x %d x %d cells (%.0f x %.0f x %.0f m), %zu fields",
        g.nx, g.ny, g.nz, g.nx * g.cell, g.ny * g.cell, g.nz * g.cell, world.law().fields.size());
    const field::WorldMesher::Stats& s = world_view_.stats();
    ImGui::Text("detail level %d   regions %zu   faces %zu   building %zu", s.level, s.regions_drawn, s.faces, s.pending);
    ImGui::Text("creatures %zu   pieces in the air %zu   balls thrown %d", world.creatures().size(), world.pieces().size(), balls_thrown_);
    ImGui::TextDisabled("beyond the window: %zu chunks still going, %zu creatures walking", world.far_busy(), world.far_bodies().size());
    ImGui::TextDisabled("contracts broken so far: %zu", world.violations().size());
    if (world.map() != nullptr) {
        const field::WorldMap& map = *world.map();
        const field::TerrainConfig& t = world.config().terrain;
        const int wx = static_cast<int>(std::floor(focus_.x / t.cell));
        const int wy = static_cast<int>(std::floor(focus_.y / t.cell));
        ImGui::Separator();
        ImGui::Text("%s, %.0f C, %.0f m above the sea", field::biome_name(map.biome_at(wx, wy)), map.temperature_at(wx, wy),
            focus_z_ - t.sea_level * t.cell);
        if (const field::Settlement* near = map.nearest_settlement(wx, wy)) {
            const double away = std::hypot(static_cast<double>(wx - near->x), static_cast<double>(wy - near->y)) * t.cell;
            ImGui::Text("%s, %d people, %.0f m away", near->name.c_str(), near->people, away);
            ImGui::TextDisabled("they build: %s", field::style_words(map.styles()[static_cast<std::size_t>(near->style)]).c_str());
            int told = 0;
            for (auto it = map.chronicle().rbegin(); it != map.chronicle().rend() && told < 3; ++it) {
                if (it->settlement == near->id) {
                    ImGui::TextDisabled("age %d: %s", it->age, it->text.c_str());
                    ++told;
                }
            }
        }
    }
    ImGui::End();
}

// THE PROFILER: what the last field step cost, term by term, on the backend it ran on (CUDA events, the CPU's clock).
void Sandbox::render_profiler()
{
    field::Playground& world = *world_;
    const field::StepProfile p = world.field_profile();
    if (p.total_ms != profile_last_) {
        profile_last_ = p.total_ms;
        profile_history_.push_back(static_cast<float>(p.gpu_ms > 0.0 ? p.gpu_ms : p.total_ms));
        if (profile_history_.size() > 240) {
            profile_history_.erase(profile_history_.begin());
        }
    }
    ImGui::SetNextWindowSize(ImVec2(470.0f, 520.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Profiler (F2)", &show_profiler_);
    const field::Solver& solver = std::as_const(world).solver();
    ImGui::Text("%s on %s", p.backend.empty() ? solver.backend_api() : p.backend.c_str(),
        solver.on_gpu() ? solver.gpu_name().c_str() : "the CPU");
    ImGui::Text("step %.2f ms   card %.2f ms   %d launches", p.total_ms, p.gpu_ms, p.launches);
    ImGui::Text("up %.1f KiB (%.2f ms)   down %.1f KiB (%.2f ms)   derives %.2f ms", p.upload_bytes / 1024.0, p.upload_ms,
        p.download_bytes / 1024.0, p.download_ms, p.derive_ms);
    bool learned = world.learned();
    if (ImGui::Checkbox("learned schemes, under guard", &learned)) {
        world.set_learned(learned);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Terms with `learned d ...` lines run their learned scheme with its larger step; one that breaks a "
                          "contract of the term (range, finite, conserves) is undone and taken classically, with substeps.");
    }
    if (p.fallbacks > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%d undone", p.fallbacks);
    }
    if (!profile_history_.empty()) {
        float most = 0.0f;
        for (const float v : profile_history_) {
            most = std::max(most, v);
        }
        ImGui::PlotLines("##steps", profile_history_.data(), static_cast<int>(profile_history_.size()), 0, "ms a field step", 0.0f,
            most * 1.1f + 0.01f, ImVec2(-1.0f, 60.0f));
    }
    std::vector<const field::TermProfile*> rows;
    double most = 1e-6;
    for (const field::TermProfile& t : p.terms) {
        if (!t.idle) {
            rows.push_back(&t);
            most = std::max(most, t.ms);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const field::TermProfile* a, const field::TermProfile* b) { return a->ms > b->ms; });
    ImGui::TextDisabled("%zu of %zu terms ran (the rest had nothing to do: not dispatched)", rows.size(), p.terms.size());
    if (ImGui::BeginTable("terms", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("term");
        ImGui::TableSetupColumn("ms");
        ImGui::TableSetupColumn("substeps");
        ImGui::TableSetupColumn("kernels");
        ImGui::TableHeadersRow();
        for (const field::TermProfile* t : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s%s%s", t->name.c_str(), t->learned ? "  (learned)" : "", t->fell_back ? "  (undone)" : "");
            ImGui::TableNextColumn();
            char label[32];
            std::snprintf(label, sizeof(label), "%.3f", t->ms);
            ImGui::ProgressBar(static_cast<float>(t->ms / most), ImVec2(-1.0f, 0.0f), label);
            ImGui::TableNextColumn();
            ImGui::Text("%d", t->substeps);
            ImGui::TableNextColumn();
            ImGui::Text("%d", t->launches);
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

void Sandbox::render_law_panel()
{
    field::Playground& world = *world_;
    const field::Law& law = world.law();
    if (gif_dir_.empty()) {
        ImGui::SetNextWindowSize(ImVec2(620.0f, 560.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(330.0f, 60.0f), ImGuiCond_FirstUseEver);
    } else { // a capture: small, out of the way of the picture
        ImGui::SetNextWindowSize(ImVec2(470.0f, 190.0f), ImGuiCond_Always);
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 490.0f, ImGui::GetIO().DisplaySize.y - 300.0f), ImGuiCond_Always);
    }
    ImGui::Begin("The law (assets/law/*.law)", &show_law_);
    ImGui::InputText("filter", law_filter_, sizeof(law_filter_));
    const std::string filter = law_filter_;
    if (ImGui::BeginTabBar("##law")) {
        if (ImGui::BeginTabItem("Params")) {
            ImGui::TextDisabled("Every param of the running law. A change is taken up at the next field step.");
            ImGui::BeginChild("params");
            for (std::size_t i = 0; i < law.param_names.size(); ++i) {
                const std::string& name = law.param_names[i];
                if (!filter.empty() && name.find(filter) == std::string::npos) {
                    continue;
                }
                float value = static_cast<float>(law.param_values[i]);
                const float speed = std::max(std::abs(value) * 0.01f, 0.0005f);
                if (ImGui::DragFloat(name.c_str(), &value, speed, 0.0f, 0.0f, "%.4g")) {
                    world.set_param(name, value);
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Equations")) {
            ImGui::TextDisabled("The expression trees the solver integrates, printed back as mathematics.");
            ImGui::BeginChild("equations");
            for (const field::LawEquation& eq : field::law_equations(law)) {
                if (!filter.empty() && eq.lhs.find(filter) == std::string::npos) {
                    continue;
                }
                std::string rhs;
                for (std::size_t k = 0; k < eq.parts.size(); ++k) {
                    rhs += (k == 0 ? "" : " + ") + eq.parts[k];
                }
                ImGui::TextWrapped("%s = %s", eq.lhs.c_str(), rhs.empty() ? "0" : rhs.c_str());
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Fields")) {
            ImGui::BeginChild("fields");
            for (const field::FieldDecl& decl : law.fields) {
                if (!filter.empty() && decl.name.find(filter) == std::string::npos) {
                    continue;
                }
                ImGui::Text("%-14s total %.4g", decl.name.c_str(), world.total(decl.name));
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Contracts")) {
            if (world.violations().empty()) {
                ImGui::TextDisabled("No contract of the law has been broken.");
            }
            for (const field::Violation& v : world.violations()) {
                ImGui::TextWrapped("%s  [%s] %s: %s", field::to_string(v.kind), v.term.c_str(), v.field.c_str(), v.detail.c_str());
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void Sandbox::render_help()
{
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(10.0f, io.DisplaySize.y - 10.0f), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::Begin("##help", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
    ImGui::TextUnformatted("WASD / arrows  move      Shift  faster      right drag  pan      wheel  zoom (out to the whole continent)");
    ImGui::TextUnformatted("Q / E, middle drag  turn      left click  use the tool      1-0  tools      T  next town");
    ImGui::TextUnformatted("P  pause      F1  the law      Tab  hide panels      H  this help      hover a creature to read its mind");
    ImGui::End();
}

// Over the world: names and what is said over the heads of the people, and a creature's mind under the cursor.
void Sandbox::draw_overlay(ImDrawList* overlay)
{
    const field::Playground& world = *world_;
    const field::Vec3 o = world.origin();
    const double view_m = camera_.ortho_half_height() * 2.0 / kSim;
    const bool near_enough = view_m < 90.0;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const field::Creature* hovered = nullptr;
    float best = 24.0f;
    for (const field::Creature& c : world.creatures()) {
        const render::Vec2 s = to_overlay(c.body.position + o);
        const float d = std::hypot(static_cast<float>(s.x) - mouse.x, static_cast<float>(s.y) - mouse.y);
        if (d < best) {
            best = d;
            hovered = &c;
        }
        if (near_enough && c.town >= 0) {
            const render::Vec2 head = to_overlay(c.body.position + o + field::Vec3{0.0, 0.0, c.body.radius + 1.3});
            const ImVec2 size = ImGui::CalcTextSize(c.who.name.c_str());
            const ImVec2 at(static_cast<float>(head.x) - size.x * 0.5f, static_cast<float>(head.y) - size.y);
            overlay->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), IM_COL32(0, 0, 0, 160), c.who.name.c_str());
            overlay->AddText(at, c.hostile ? IM_COL32(255, 100, 90, 255) : IM_COL32(235, 235, 240, 230), c.who.name.c_str());
        }
    }
    if (near_enough) {
        for (const field::Bark& b : world.barks()) {
            const field::Creature* c = world.creature_with(b.id);
            if (c == nullptr) {
                continue;
            }
            const render::Vec2 s = to_overlay(c->body.position + o + field::Vec3{0.0, 0.0, c->body.radius + 1.3});
            const float wrap = 240.0f;
            const ImVec2 size = ImGui::CalcTextSize(b.text.c_str(), nullptr, false, wrap);
            const ImVec2 at(static_cast<float>(s.x) - size.x * 0.5f, static_cast<float>(s.y) - size.y - 22.0f);
            const int alpha = static_cast<int>(std::clamp(b.left * 2.0, 0.0, 1.0) * 215.0);
            overlay->AddRectFilled(ImVec2(at.x - 5.0f, at.y - 3.0f), ImVec2(at.x + size.x + 5.0f, at.y + size.y + 3.0f), IM_COL32(236, 228, 208, alpha), 3.0f);
            overlay->AddText(ImGui::GetFont(), ImGui::GetFontSize(), at, IM_COL32(24, 22, 28, std::min(255, alpha + 40)), b.text.c_str(), nullptr, wrap);
        }
    }
    // The live window: where the law runs. Its edge, along the ground, when it is not most of the screen.
    if (view_m > 60.0 && world.map() != nullptr) {
        const field::TerrainConfig& t = world.config().terrain;
        const double chunk_m = t.chunk * t.cell;
        const double x0 = world.window_cx() * chunk_m;
        const double y0 = world.window_cy() * chunk_m;
        const double x1 = (world.window_cx() + world.window_chunks()) * chunk_m;
        const double y1 = (world.window_cy() + world.window_chunks()) * chunk_m;
        const double corners[5][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}, {x0, y0}};
        constexpr int kSteps = 24;
        for (int side = 0; side < 4; ++side) {
            ImVec2 last{};
            for (int k = 0; k <= kSteps; ++k) {
                const double a = static_cast<double>(k) / kSteps;
                const double x = corners[side][0] + (corners[side + 1][0] - corners[side][0]) * a;
                const double y = corners[side][1] + (corners[side + 1][1] - corners[side][1]) * a;
                const render::Vec2 s = to_overlay(field::Vec3{x, y, world.terrain_at(x, y) + 0.3});
                const ImVec2 here(static_cast<float>(s.x), static_cast<float>(s.y));
                if (k > 0) {
                    overlay->AddLine(last, here, IM_COL32(255, 230, 140, 150), 1.5f);
                }
                last = here;
            }
        }
        const render::Vec2 label = to_overlay(field::Vec3{x0, y1, world.terrain_at(x0, y1) + 0.3});
        overlay->AddText(ImVec2(static_cast<float>(label.x) + 4.0f, static_cast<float>(label.y) - 18.0f), IM_COL32(255, 230, 140, 200), "live window: the law runs here");
    }
    hovered_ = hovered != nullptr ? hovered->id : 0;
    // Someone beyond the window under the cursor: who it is and what its day has it doing, with no body to read.
    if (hovered == nullptr && !ImGui::GetIO().WantCaptureMouse && world.society() != nullptr) {
        const field::Playground::PersonProxy* near = nullptr;
        float closest = 14.0f;
        for (const field::Playground::PersonProxy& p : proxies_) {
            const render::Vec2 s = to_overlay(p.at + field::Vec3{0.0, 0.0, 0.8});
            const float d = std::hypot(static_cast<float>(s.x) - mouse.x, static_cast<float>(s.y) - mouse.y);
            if (d < closest) {
                closest = d;
                near = &p;
            }
        }
        if (near != nullptr) {
            const field::Person who = world.society()->person(near->town, near->index);
            static const char* const kDoing[] = {"on the way to work, or at it", "at the market", "going home", "keeping watch", "", "", "", ""};
            ImGui::BeginTooltip();
            ImGui::Text("%s, %s of %s", who.name.c_str(), who.job.c_str(), world.society()->towns()[static_cast<std::size_t>(near->town)].name.c_str());
            ImGui::Text("%s", kDoing[static_cast<int>(near->doing)]);
            ImGui::TextDisabled("beyond the live window: no body, where the day has them; a body again if the window comes here");
            ImGui::EndTooltip();
        }
    }
    if (hovered == nullptr || ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    // A creature's mind is a few numbers and the lines of the law that move them.
    const field::CreatureDef& def = world.law().creatures[static_cast<std::size_t>(hovered->def)];
    const render::Vec2 s = to_overlay(hovered->body.position + o);
    overlay->AddCircle(ImVec2(static_cast<float>(s.x), static_cast<float>(s.y)), 18.0f, IM_COL32(255, 255, 255, 150), 24, 1.5f);
    ImGui::BeginTooltip();
    if (hovered->town >= 0 && world.society() != nullptr) {
        ImGui::Text("%s, %s of %s", hovered->who.name.c_str(), hovered->who.job.c_str(),
            world.society()->towns()[static_cast<std::size_t>(hovered->town)].name.c_str());
        ImGui::Text("%s", world.doing_words(*hovered));
    } else {
        ImGui::Text("%s #%u", def.name.c_str(), hovered->id);
    }
    ImGui::Text("hp %.0f / %.0f   (creature %s in the law)", hovered->body.hp, hovered->body.hp_max, def.name.c_str());
    for (std::size_t i = 1; i < def.locals.size() && i < hovered->locals.size(); ++i) {
        ImGui::TextDisabled("  %-10s %.3f", def.locals[i].c_str(), hovered->locals[i]);
    }
    ImGui::EndTooltip();
}

} // namespace fieldengine::sandbox

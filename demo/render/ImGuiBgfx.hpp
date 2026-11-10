#pragma once

#include <bgfx/bgfx.h>

struct ImDrawData;
struct ImFont;

namespace fieldengine::render {

// Minimal Dear ImGui rendering backend on bgfx. The platform side stays on ImGui_ImplSDL3 (input /
// display size); this replaces ImGui_ImplSDLRenderer3 now that the window is owned by bgfx. Draws the
// HUD on its own sequential view above the arena.
class ImGuiBgfx {
public:
    bool init();
    void shutdown();
    void render(ImDrawData* draw_data, bgfx::ViewId view);

    bool valid() const { return program_valid_; }
    // A monospace face for editing law text (null: the default font).
    ImFont* mono() const { return mono_; }

private:
    void create_font_texture();

    bool program_valid_ = false;
    bgfx::VertexLayout layout_{};
    bgfx::ProgramHandle program_ = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle font_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle sampler_ = BGFX_INVALID_HANDLE;
    ImFont* mono_ = nullptr;
};

} // namespace fieldengine::render

#include "render/ImGuiBgfx.hpp"

#include <bgfx/embedded_shader.h>
#include <bx/math.h>
#include <imgui.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "dx11/vs_imgui.sc.bin.h"
#include "dx11/fs_imgui.sc.bin.h"
#include "glsl/vs_imgui.sc.bin.h"
#include "glsl/fs_imgui.sc.bin.h"
#include "essl/vs_imgui.sc.bin.h"
#include "essl/fs_imgui.sc.bin.h"
#include "spirv/vs_imgui.sc.bin.h"
#include "spirv/fs_imgui.sc.bin.h"

namespace fieldengine::render {

namespace {

const bgfx::EmbeddedShader kImGuiShaders[] = {
    BGFX_EMBEDDED_SHADER(vs_imgui),
    BGFX_EMBEDDED_SHADER(fs_imgui),
    BGFX_EMBEDDED_SHADER_END(),
};

} // namespace

bool ImGuiBgfx::init()
{
    const bgfx::RendererType::Enum type = bgfx::getRendererType();
    if (type == bgfx::RendererType::Noop) {
        return false; // headless: no HUD rendering needed
    }

    layout_.begin()
        .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    const bgfx::ShaderHandle vsh = bgfx::createEmbeddedShader(kImGuiShaders, type, "vs_imgui");
    const bgfx::ShaderHandle fsh = bgfx::createEmbeddedShader(kImGuiShaders, type, "fs_imgui");
    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh)) {
        return false;
    }
    program_ = bgfx::createProgram(vsh, fsh, true);
    program_valid_ = bgfx::isValid(program_);
    if (!program_valid_) {
        return false;
    }

    sampler_ = bgfx::createUniform("s_tex", bgfx::UniformType::Sampler);
    create_font_texture();
    return true;
}

namespace {

bool readable(const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

} // namespace

void ImGuiBgfx::create_font_texture()
{
    ImGuiIO& io = ImGui::GetIO();
    // The law inspector prints equations (∂T/∂t = ∇·(k∇T) + ...), so the UI font needs mathematics:
    // a system sans for text with a symbol font merged in for operators, arrows, sub- and superscripts.
    // Without them (no such files) the built-in ASCII font stays, with the stand-ins below.
    static const ImWchar kText[] = {0x0020, 0x00FF, 0x0370, 0x03FF, 0x0400, 0x04FF, 0x2000, 0x206F, 0x2070, 0x209F,
        0x2100, 0x214F, 0x2190, 0x21FF, 0x2200, 0x22FF, 0x2300, 0x23FF, 0x25A0, 0x25FF, 0};
    static const ImWchar kSymbols[] = {0x0370, 0x03FF, 0x2070, 0x209F, 0x2100, 0x214F, 0x2190, 0x21FF, 0x2200,
        0x22FF, 0x2300, 0x23FF, 0x25A0, 0x25FF, 0};
    std::string fonts;
    if (const char* windir = std::getenv("WINDIR")) {
        fonts = std::string(windir) + "/Fonts/";
    }
    const std::string sans = fonts + "segoeui.ttf";
    const std::string symbols = fonts + "seguisym.ttf";
    const std::string mono = fonts + "consola.ttf";
    bool real_font = false;
    if (io.Fonts->Fonts.Size == 0 && !fonts.empty() && readable(sans)) {
        ImFontConfig base;
        base.OversampleH = 2;
        real_font = io.Fonts->AddFontFromFileTTF(sans.c_str(), 16.0f, &base, kText) != nullptr;
        if (real_font && readable(symbols)) {
            ImFontConfig merge;
            merge.MergeMode = true;
            io.Fonts->AddFontFromFileTTF(symbols.c_str(), 16.0f, &merge, kSymbols);
        }
        if (real_font && readable(mono)) {
            mono_ = io.Fonts->AddFontFromFileTTF(mono.c_str(), 15.0f, &base, kText);
            if (mono_ != nullptr && readable(symbols)) {
                ImFontConfig merge;
                merge.MergeMode = true;
                io.Fonts->AddFontFromFileTTF(symbols.c_str(), 15.0f, &merge, kSymbols);
            }
        }
    }
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    font_ = bgfx::createTexture2D(
        static_cast<std::uint16_t>(width),
        static_cast<std::uint16_t>(height),
        false,
        1,
        bgfx::TextureFormat::RGBA8,
        0,
        bgfx::copy(pixels, static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height) * 4));

    io.Fonts->SetTexID(static_cast<ImTextureID>(font_.idx));

    // The built-in font is ASCII-only, and the game's text is written with real punctuation (the
    // objective line read "Survive until nightfall ? the stars..." because "—" has no glyph). Point
    // each typographic character at its ASCII stand-in once, here, instead of policing every string.
    static constexpr std::pair<ImWchar, ImWchar> stand_ins[] = {
        {0x2014, '-'}, {0x2013, '-'}, {0x2012, '-'}, {0x2212, '-'},   // dashes, minus
        {0x2018, '\''}, {0x2019, '\''}, {0x201C, '"'}, {0x201D, '"'}, // quotes
        {0x2026, '.'}, {0x00B7, '.'}, {0x2022, '*'},                  // ellipsis, middle dot, bullet
        {0x2192, '>'}, {0x2190, '<'}, {0x00D7, 'x'}, {0x2265, '>'}, {0x2264, '<'}, {0x2248, '~'},
        {0x00A0, ' '},
    };
    if (!real_font) {
        for (ImFont* font : io.Fonts->Fonts) {
            for (const auto& [from, to] : stand_ins) {
                font->AddRemapChar(from, to);
            }
        }
    }
}

void ImGuiBgfx::shutdown()
{
    if (bgfx::isValid(font_)) {
        bgfx::destroy(font_);
        font_ = BGFX_INVALID_HANDLE;
    }
    if (bgfx::isValid(sampler_)) {
        bgfx::destroy(sampler_);
        sampler_ = BGFX_INVALID_HANDLE;
    }
    if (program_valid_) {
        bgfx::destroy(program_);
        program_ = BGFX_INVALID_HANDLE;
        program_valid_ = false;
    }
}

void ImGuiBgfx::render(ImDrawData* draw_data, bgfx::ViewId view)
{
    if (!program_valid_ || draw_data == nullptr || draw_data->CmdListsCount == 0) {
        return;
    }

    const int fb_w = static_cast<int>(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    const int fb_h = static_cast<int>(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (fb_w <= 0 || fb_h <= 0) {
        return;
    }

    bgfx::setViewName(view, "ImGui");
    bgfx::setViewMode(view, bgfx::ViewMode::Sequential);

    const bgfx::Caps* caps = bgfx::getCaps();
    const float l = draw_data->DisplayPos.x;
    const float r = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    const float t = draw_data->DisplayPos.y;
    const float b = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    float ortho[16];
    bx::mtxOrtho(ortho, l, r, b, t, 0.0f, 1000.0f, 0.0f, caps->homogeneousDepth);
    bgfx::setViewTransform(view, nullptr, ortho);
    bgfx::setViewRect(view, 0, 0, static_cast<std::uint16_t>(fb_w), static_cast<std::uint16_t>(fb_h));

    const ImVec2 clip_off = draw_data->DisplayPos;
    const ImVec2 clip_scale = draw_data->FramebufferScale;

    constexpr std::uint64_t state =
        BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_MSAA |
        BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA);

    for (int n = 0; n < draw_data->CmdListsCount; ++n) {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        const std::uint32_t num_vertices = static_cast<std::uint32_t>(cmd_list->VtxBuffer.Size);
        const std::uint32_t num_indices = static_cast<std::uint32_t>(cmd_list->IdxBuffer.Size);

        if (bgfx::getAvailTransientVertexBuffer(num_vertices, layout_) < num_vertices ||
            bgfx::getAvailTransientIndexBuffer(num_indices) < num_indices) {
            break;
        }

        bgfx::TransientVertexBuffer tvb;
        bgfx::TransientIndexBuffer tib;
        bgfx::allocTransientVertexBuffer(&tvb, num_vertices, layout_);
        bgfx::allocTransientIndexBuffer(&tib, num_indices);
        std::memcpy(tvb.data, cmd_list->VtxBuffer.Data, num_vertices * sizeof(ImDrawVert));
        std::memcpy(tib.data, cmd_list->IdxBuffer.Data, num_indices * sizeof(ImDrawIdx));

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; ++cmd_i) {
            const ImDrawCmd& cmd = cmd_list->CmdBuffer[cmd_i];
            if (cmd.UserCallback != nullptr) {
                cmd.UserCallback(cmd_list, &cmd);
                continue;
            }
            if (cmd.ElemCount == 0) {
                continue;
            }

            const float clip_min_x = (cmd.ClipRect.x - clip_off.x) * clip_scale.x;
            const float clip_min_y = (cmd.ClipRect.y - clip_off.y) * clip_scale.y;
            const float clip_max_x = (cmd.ClipRect.z - clip_off.x) * clip_scale.x;
            const float clip_max_y = (cmd.ClipRect.w - clip_off.y) * clip_scale.y;
            if (clip_max_x <= clip_min_x || clip_max_y <= clip_min_y) {
                continue;
            }

            bgfx::setScissor(
                static_cast<std::uint16_t>(clip_min_x < 0.0f ? 0.0f : clip_min_x),
                static_cast<std::uint16_t>(clip_min_y < 0.0f ? 0.0f : clip_min_y),
                static_cast<std::uint16_t>(clip_max_x - clip_min_x),
                static_cast<std::uint16_t>(clip_max_y - clip_min_y));
            bgfx::setState(state);

            bgfx::TextureHandle texture = font_;
            if (cmd.GetTexID() != 0) {
                texture.idx = static_cast<std::uint16_t>(cmd.GetTexID());
            }
            bgfx::setTexture(0, sampler_, texture);

            bgfx::setVertexBuffer(0, &tvb, cmd.VtxOffset, num_vertices);
            bgfx::setIndexBuffer(&tib, cmd.IdxOffset, cmd.ElemCount);
            bgfx::submit(view, program_);
        }
    }
}

} // namespace fieldengine::render

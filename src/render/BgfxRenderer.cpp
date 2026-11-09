#include "render/BgfxRenderer.hpp"


#include <bgfx/embedded_shader.h>
#include <bx/math.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Embedded shader blobs, compiled at build time by bgfx_compile_shaders(... AS_HEADERS) into the
// binary tree (see src/render/SHADERS.md). One header per renderer profile; BGFX_EMBEDDED_SHADER
// picks the right blob for the active backend at runtime.
#include "dx11/vs_basic.sc.bin.h"
#include "dx11/fs_basic.sc.bin.h"
#include "glsl/vs_basic.sc.bin.h"
#include "glsl/fs_basic.sc.bin.h"
#include "essl/vs_basic.sc.bin.h"
#include "essl/fs_basic.sc.bin.h"
#include "spirv/vs_basic.sc.bin.h"
#include "spirv/fs_basic.sc.bin.h"


#include "dx11/vs_post.sc.bin.h"
#include "dx11/fs_post.sc.bin.h"
#include "glsl/vs_post.sc.bin.h"
#include "glsl/fs_post.sc.bin.h"
#include "essl/vs_post.sc.bin.h"
#include "essl/fs_post.sc.bin.h"
#include "spirv/vs_post.sc.bin.h"
#include "spirv/fs_post.sc.bin.h"

namespace fieldengine::render {

namespace {

constexpr Color kBackground{18, 16, 24, 255};

const bgfx::EmbeddedShader kEmbeddedShaders[] = {
    BGFX_EMBEDDED_SHADER(vs_basic),
    BGFX_EMBEDDED_SHADER(fs_basic),
    BGFX_EMBEDDED_SHADER(vs_post),
    BGFX_EMBEDDED_SHADER(fs_post),
    BGFX_EMBEDDED_SHADER_END(),
};

// bgfx clear colour wants 0xRRGGBBAA (distinct from the 0xAABBGGRR vertex packing).
std::uint32_t to_rgba(const Color& c)
{
    return (static_cast<std::uint32_t>(c.r) << 24) | (static_cast<std::uint32_t>(c.g) << 16) |
           (static_cast<std::uint32_t>(c.b) << 8) | static_cast<std::uint32_t>(c.a);
}


// Frame-capture callback. bgfx only takes screenshots through a CallbackI, so the renderer installs
// this one and writes an uncompressed 32-bit TGA (no image-encoder dependency). It exists so a run
// can be verified visually offline — `CL_SMOKE_SHOT=<path>` on the client, or a visual regression
// check — and is inert until requestScreenShot() is called. Every other hook is the default no-op.
struct CaptureCallback final : public bgfx::CallbackI {
    ~CaptureCallback() override = default;

    void fatal(const char*, std::uint16_t, bgfx::Fatal::Enum, const char* str) override
    {
        std::fprintf(stderr, "bgfx fatal: %s\n", str != nullptr ? str : "(null)");
    }
    void traceVargs(const char*, std::uint16_t, const char*, va_list) override {}
    void profilerBegin(const char*, std::uint32_t, const char*, std::uint16_t) override {}
    void profilerBeginLiteral(const char*, std::uint32_t, const char*, std::uint16_t) override {}
    void profilerEnd() override {}
    std::uint32_t cacheReadSize(std::uint64_t) override { return 0; }
    bool cacheRead(std::uint64_t, void*, std::uint32_t) override { return false; }
    void cacheWrite(std::uint64_t, const void*, std::uint32_t) override {}

    void screenShot(const char* file_path, std::uint32_t width, std::uint32_t height,
                    std::uint32_t pitch, const void* data, std::uint32_t, bool yflip) override
    {
        if (file_path == nullptr || data == nullptr || width == 0 || height == 0) {
            return;
        }
        std::FILE* out = std::fopen(file_path, "wb");
        if (out == nullptr) {
            return;
        }
        // TGA header: uncompressed true-colour, 32 bpp, top-left origin unless bgfx flipped it.
        const std::uint8_t header[18] = {
            0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            static_cast<std::uint8_t>(width & 0xff), static_cast<std::uint8_t>(width >> 8),
            static_cast<std::uint8_t>(height & 0xff), static_cast<std::uint8_t>(height >> 8),
            32, static_cast<std::uint8_t>(yflip ? 0x08 : 0x28)};
        std::fwrite(header, 1, sizeof(header), out);
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::uint32_t row = 0; row < height; ++row) {
            std::fwrite(bytes + static_cast<std::size_t>(row) * pitch, 1, width * 4, out);
        }
        std::fclose(out);
    }

    void captureBegin(std::uint32_t, std::uint32_t, std::uint32_t, bgfx::TextureFormat::Enum, bool) override {}
    void captureEnd() override {}
    void captureFrame(const void*, std::uint32_t) override {}
};

CaptureCallback& capture_callback()
{
    static CaptureCallback callback;
    return callback;
}

} // namespace

bool BgfxRenderer::init(const RenderConfig& config)
{
    width_ = config.width;
    height_ = config.height;

    bgfx::Init init;
    init.type = config.use_noop_backend ? bgfx::RendererType::Noop : bgfx::RendererType::Count;
    init.resolution.width = static_cast<std::uint32_t>(width_);
    init.resolution.height = static_cast<std::uint32_t>(height_);
    init.resolution.reset = BGFX_RESET_VSYNC;
    reset_flags_ = BGFX_RESET_VSYNC;
    init.platformData.nwh = config.native_window_handle;
    init.platformData.ndt = config.native_display_handle;
    init.callback = &capture_callback(); // enables request_screenshot(); otherwise a no-op
    // Figures, the law's `show` cubes and the HUD all go through transient buffers each frame: a crowded view needs more than the
    // default, and what does not fit is dropped whole (the HUD last of all).
    init.limits.transientVbSize = 64u << 20;
    init.limits.transientIbSize = 16u << 20;

    if (!bgfx::init(init)) {
        return false;
    }
    initialized_ = true;
    frame_count_ = 0;

    layout_.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    // Noop has no renderer profile in the embedded table, so program creation is skipped; drawing
    // becomes a no-op while init/frame/shutdown still exercise the full path for the CI smoke.
    const bgfx::RendererType::Enum type = bgfx::getRendererType();
    if (type != bgfx::RendererType::Noop) {
        const bgfx::ShaderHandle vsh = bgfx::createEmbeddedShader(kEmbeddedShaders, type, "vs_basic");
        cut_uniform_ = bgfx::createUniform("u_cut", bgfx::UniformType::Vec4);
        cut_dir_uniform_ = bgfx::createUniform("u_cutDir", bgfx::UniformType::Vec4);
        cut_keep_uniform_ = bgfx::createUniform("u_cutKeep", bgfx::UniformType::Vec4);
        clip_uniform_ = bgfx::createUniform("u_clip", bgfx::UniformType::Vec4);
        const bgfx::ShaderHandle fsh = bgfx::createEmbeddedShader(kEmbeddedShaders, type, "fs_basic");
        if (bgfx::isValid(vsh) && bgfx::isValid(fsh)) {
            program_ = bgfx::createProgram(vsh, fsh, true);
            program_valid_ = bgfx::isValid(program_);
        }

        // Pixel-art post-process program + fullscreen-pass resources.
        const bgfx::ShaderHandle pvsh = bgfx::createEmbeddedShader(kEmbeddedShaders, type, "vs_post");
        const bgfx::ShaderHandle pfsh = bgfx::createEmbeddedShader(kEmbeddedShaders, type, "fs_post");
        if (bgfx::isValid(pvsh) && bgfx::isValid(pfsh)) {
            post_program_ = bgfx::createProgram(pvsh, pfsh, true);
            post_program_valid_ = bgfx::isValid(post_program_);
        }
        post_layout_.begin()
            .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .end();
        post_sampler_ = bgfx::createUniform("s_tex", bgfx::UniformType::Sampler);
        post_params_uniform_ = bgfx::createUniform("u_post", bgfx::UniformType::Vec4);
    }

    bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, to_rgba(kBackground), 1.0f, 0);
    bgfx::setViewRect(0, 0, 0, static_cast<std::uint16_t>(width_), static_cast<std::uint16_t>(height_));
    return true;
}

void BgfxRenderer::shutdown()
{
    if (!initialized_) {
        return;
    }
    destroy_offscreen();
    for (StaticMesh& mesh : static_meshes_) {
        release(mesh);
    }
    static_meshes_.clear();
    static_draws_.clear();
    if (bgfx::isValid(post_sampler_)) {
        bgfx::destroy(post_sampler_);
        post_sampler_ = BGFX_INVALID_HANDLE;
    }
    if (bgfx::isValid(post_params_uniform_)) {
        bgfx::destroy(post_params_uniform_);
        post_params_uniform_ = BGFX_INVALID_HANDLE;
    }
    if (post_program_valid_) {
        bgfx::destroy(post_program_);
        post_program_ = BGFX_INVALID_HANDLE;
        post_program_valid_ = false;
    }
    if (program_valid_) {
        bgfx::destroy(program_);
        program_ = BGFX_INVALID_HANDLE;
        program_valid_ = false;
    }
    bgfx::shutdown();
    initialized_ = false;
    items_.clear();
}

void BgfxRenderer::ensure_offscreen(int low_w, int low_h)
{
    low_w = std::max(low_w, 1);
    low_h = std::max(low_h, 1);
    if (bgfx::isValid(offscreen_fb_) && low_w == offscreen_w_ && low_h == offscreen_h_) {
        return;
    }
    destroy_offscreen();
    // Colour target (point-sampled for the pixelize upscale) + a depth target, so the depth-tested
    // 3D voxels render correctly into the offscreen buffer before the pixel-art pass.
    const std::uint64_t color_flags = BGFX_TEXTURE_RT | BGFX_SAMPLER_POINT |
                                      BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP;
    bgfx::TextureHandle attachments[2];
    attachments[0] = bgfx::createTexture2D(
        static_cast<std::uint16_t>(low_w), static_cast<std::uint16_t>(low_h), false, 1,
        bgfx::TextureFormat::RGBA8, color_flags);
    attachments[1] = bgfx::createTexture2D(
        static_cast<std::uint16_t>(low_w), static_cast<std::uint16_t>(low_h), false, 1,
        bgfx::TextureFormat::D24S8, BGFX_TEXTURE_RT_WRITE_ONLY);
    offscreen_fb_ = bgfx::createFrameBuffer(2, attachments, true); // owns + destroys the textures
    if (bgfx::isValid(offscreen_fb_)) {
        offscreen_tex_ = bgfx::getTexture(offscreen_fb_, 0);
        offscreen_w_ = low_w;
        offscreen_h_ = low_h;
    }
}

void BgfxRenderer::destroy_offscreen()
{
    if (bgfx::isValid(offscreen_fb_)) {
        bgfx::destroy(offscreen_fb_); // also releases the attached colour texture
        offscreen_fb_ = BGFX_INVALID_HANDLE;
        offscreen_tex_ = BGFX_INVALID_HANDLE;
    }
    offscreen_w_ = 0;
    offscreen_h_ = 0;
}

void BgfxRenderer::ensure_reset(int width, int height)
{
    if (width == width_ && height == height_) {
        return;
    }
    width_ = std::max(width, 1);
    height_ = std::max(height, 1);
    bgfx::reset(static_cast<std::uint32_t>(width_), static_cast<std::uint32_t>(height_), reset_flags_);
}

void BgfxRenderer::set_camera(const Camera& camera)
{
    camera_ = camera;
}

void BgfxRenderer::begin_frame()
{
    if (!initialized_) {
        return;
    }
    items_.clear();
    triangles_.clear();
    static_draws_.clear();

    const int w = static_cast<int>(camera_.viewport_width());
    const int h = static_cast<int>(camera_.viewport_height());
    if (w > 0 && h > 0) {
        ensure_reset(w, h);
    }

    // Decide whether this frame routes the world through the low-res offscreen target for the
    // pixel-art post pass. Read-only with respect to the simulation either way.
    post_active_ = params_.enabled && post_program_valid_;
    int view_w = width_;
    int view_h = height_;
    if (post_active_) {
        const int low_w = std::max(width_ / params_.pixel_size, 1);
        const int low_h = std::max(height_ / params_.pixel_size, 1);
        ensure_offscreen(low_w, low_h);
        if (bgfx::isValid(offscreen_fb_)) {
            bgfx::setViewFrameBuffer(kWorldView, offscreen_fb_);
            view_w = low_w;
            view_h = low_h;
        } else {
            post_active_ = false;
        }
    }
    if (!post_active_) {
        bgfx::setViewFrameBuffer(kWorldView, BGFX_INVALID_HANDLE);
    }

    bgfx::setViewRect(kWorldView, 0, 0, static_cast<std::uint16_t>(view_w), static_cast<std::uint16_t>(view_h));
    bgfx::setViewClear(kWorldView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, to_rgba(kBackground), 1.0f, 0);

    // Isometric 3D view-projection built from the camera with glm (same RH math the camera uses for
    // picking, so clicks line up with what's drawn). Depth convention follows the bgfx backend caps.
    const Vec3 e = camera_.eye();
    const Vec3 t = camera_.target();
    const Vec3 u = camera_.up();
    const glm::mat4 view = glm::lookAtRH(
        glm::vec3(e.x, e.y, e.z), glm::vec3(t.x, t.y, t.z), glm::vec3(u.x, u.y, u.z));
    const float hh = static_cast<float>(camera_.ortho_half_height());
    const float hw = static_cast<float>(camera_.ortho_half_width());
    const glm::mat4 proj = bgfx::getCaps()->homogeneousDepth
        ? glm::orthoRH_NO(-hw, hw, -hh, hh, 0.05f, static_cast<float>(camera_.far_plane()))
        : glm::orthoRH_ZO(-hw, hw, -hh, hh, 0.05f, static_cast<float>(camera_.far_plane()));
    bgfx::setViewTransform(kWorldView, glm::value_ptr(view), glm::value_ptr(proj));
    bgfx::touch(kWorldView);
}

namespace {

std::uint8_t shade8(std::uint8_t c, double f)
{
    return static_cast<std::uint8_t>(std::clamp(static_cast<double>(c) * f, 0.0, 255.0));
}

Color shade(const Color& c, double f)
{
    return Color{shade8(c.r, f), shade8(c.g, f), shade8(c.b, f), c.a};
}

} // namespace

void BgfxRenderer::tessellate(const DrawItem& item, std::vector<PosColorVertex>& out) const
{
    const float cx = static_cast<float>(item.position.x);
    const float cy = static_cast<float>(item.position.y);
    const float cz = static_cast<float>(item.position.z);

    // A quad is a ground decal in the xy plane at z=cz.
    const std::uint32_t abgr = item.color.to_abgr();
    auto flat = [&](float x, float y) { out.push_back(PosColorVertex{x, y, cz, abgr}); };

    switch (item.shape) {
    case Shape::Cube: {
        // A shaded voxel box: top brightest, sides dimmer, bottom darkest — reads as 3D without a
        // lighting pass. Spans [cx-hw,cx+hw] x [cy-hh,cy+hh] x [cz, cz+h].
        const float hw = static_cast<float>(item.half_w > 0.0 ? item.half_w : item.size);
        const float hh = static_cast<float>(item.half_h > 0.0 ? item.half_h : item.size);
        const float h = static_cast<float>(item.height > 0.0 ? item.height : item.size * 2.0);
        const float x0 = cx - hw, x1 = cx + hw, y0 = cy - hh, y1 = cy + hh, z0 = cz, z1 = cz + h;
        const std::uint32_t top = shade(item.color, 1.0).to_abgr();
        const std::uint32_t side_x = shade(item.color, 0.78).to_abgr();
        const std::uint32_t side_y = shade(item.color, 0.62).to_abgr();
        const std::uint32_t bottom = shade(item.color, 0.45).to_abgr();
        auto quad = [&](float ax, float ay, float az, float bx_, float by, float bz,
                        float ccx, float ccy, float ccz, float dx, float dy, float dz, std::uint32_t col) {
            out.push_back(PosColorVertex{ax, ay, az, col});
            out.push_back(PosColorVertex{bx_, by, bz, col});
            out.push_back(PosColorVertex{ccx, ccy, ccz, col});
            out.push_back(PosColorVertex{ax, ay, az, col});
            out.push_back(PosColorVertex{ccx, ccy, ccz, col});
            out.push_back(PosColorVertex{dx, dy, dz, col});
        };
        quad(x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1, top);    // top (+z)
        quad(x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, bottom); // bottom (-z)
        quad(x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1, side_x); // +x
        quad(x0, y0, z0, x0, y1, z0, x0, y1, z1, x0, y0, z1, side_x); // -x
        quad(x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1, side_y); // +y
        quad(x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1, side_y); // -y
        break;
    }
    case Shape::Quad: {
        const float hw = static_cast<float>(item.half_w > 0.0 ? item.half_w : item.size);
        const float hh = static_cast<float>(item.half_h > 0.0 ? item.half_h : item.size);
        flat(cx - hw, cy - hh);
        flat(cx + hw, cy - hh);
        flat(cx + hw, cy + hh);
        flat(cx - hw, cy - hh);
        flat(cx + hw, cy + hh);
        flat(cx - hw, cy + hh);
        break;
    }
    }
}

void BgfxRenderer::end_frame()
{
    if (!initialized_) {
        return;
    }

    if (program_valid_) {
        // The world's chunks first: opaque, depth-written, already on the GPU.
        static const float kNoClip[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (const auto& [id, clipped] : static_draws_) {
            const StaticMesh& mesh = static_meshes_[static_cast<std::size_t>(id)];
            if (!mesh.alive || !bgfx::isValid(mesh.vertices) || !bgfx::isValid(mesh.indices)) {
                continue;
            }
            bgfx::setUniform(clip_uniform_, clipped ? clip_ : kNoClip);
            bgfx::setUniform(cut_uniform_, cut_);
            bgfx::setUniform(cut_dir_uniform_, cut_dir_);
            bgfx::setUniform(cut_keep_uniform_, cut_keep_);
            bgfx::setVertexBuffer(0, mesh.vertices);
            bgfx::setIndexBuffer(mesh.indices);
            bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS |
                BGFX_STATE_MSAA);
            bgfx::submit(kWorldView, program_);
        }
    }

    // What is fading out in the way of you, over everything whole: the same meshes, blended, writing no depth.
    const bool fading = program_valid_ && (cut_[3] > 0.0f || cut_keep_[2] > 0.0f);
    const float cut_fading[4] = {1.0f, cut_keep_[1], cut_keep_[2], cut_keep_[3]};
    constexpr std::uint64_t kFadeState = BGFX_STATE_WRITE_RGB | BGFX_STATE_DEPTH_TEST_LEQUAL | BGFX_STATE_BLEND_ALPHA | BGFX_STATE_MSAA;
    if (fading) {
        static const float kNoClipFading[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (const auto& [id, clipped] : static_draws_) {
            const StaticMesh& mesh = static_meshes_[static_cast<std::size_t>(id)];
            if (!mesh.alive || !bgfx::isValid(mesh.vertices) || !bgfx::isValid(mesh.indices)) {
                continue;
            }
            bgfx::setUniform(clip_uniform_, clipped ? clip_ : kNoClipFading);
            bgfx::setUniform(cut_uniform_, cut_);
            bgfx::setUniform(cut_dir_uniform_, cut_dir_);
            bgfx::setUniform(cut_keep_uniform_, cut_fading);
            bgfx::setVertexBuffer(0, mesh.vertices);
            bgfx::setIndexBuffer(mesh.indices);
            bgfx::setState(kFadeState);
            bgfx::submit(kWorldView, program_);
        }
    }

    if (program_valid_ && (!items_.empty() || !triangles_.empty())) {
        std::stable_sort(items_.begin(), items_.end(), [](const DrawItem& a, const DrawItem& b) {
            return static_cast<int>(a.layer) < static_cast<int>(b.layer);
        });

        // Opaque voxels (alpha 255) write depth so they occlude correctly in 3D; translucent decals
        // (trails/telegraphs, alpha < 255) blend on top with depth test but no depth write.
        std::vector<PosColorVertex> opaque;
        std::vector<PosColorVertex> blended;
        for (const DrawItem& item : items_) {
            tessellate(item, item.color.a >= 255 ? opaque : blended);
        }

        opaque.insert(opaque.end(), triangles_.begin(), triangles_.end());
        // In chunks: one transient buffer has a size limit, and a batch that does not fit used to be
        // dropped whole — a big voxel world would simply vanish.
        auto submit_batch = [&](std::vector<PosColorVertex>& verts, std::uint64_t state) {
            constexpr std::uint32_t chunk = 3u * 20000u;
            const std::uint32_t total = static_cast<std::uint32_t>(verts.size());
            for (std::uint32_t first = 0; first < total; first += chunk) {
                const std::uint32_t num = std::min(chunk, total - first);
                if (bgfx::getAvailTransientVertexBuffer(num, layout_) < num) {
                    return;
                }
                bgfx::TransientVertexBuffer tvb;
                bgfx::allocTransientVertexBuffer(&tvb, num, layout_);
                std::memcpy(tvb.data, verts.data() + first, num * sizeof(PosColorVertex));
                static const float kNoClipHere[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                bgfx::setUniform(clip_uniform_, kNoClipHere);
                bgfx::setUniform(cut_uniform_, cut_);
                bgfx::setUniform(cut_dir_uniform_, cut_dir_);
                bgfx::setUniform(cut_keep_uniform_, cut_keep_);
                bgfx::setVertexBuffer(0, &tvb);
                bgfx::setState(state);
                bgfx::submit(kWorldView, program_);
            }
        };

        submit_batch(opaque,
            BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z |
            BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_MSAA);
        submit_batch(blended,
            BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A |
            BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_BLEND_ALPHA | BGFX_STATE_MSAA);
    }

    if (post_active_) {
        run_post_pass();
    }
}

void BgfxRenderer::run_post_pass()
{
    if (!post_program_valid_ || !bgfx::isValid(offscreen_tex_)) {
        return;
    }

    // Fullscreen oversized triangle in clip space; sample the low-res world target with point
    // filtering (the upscale produces the blocky pixels) and posterize to the limited palette.
    if (bgfx::getAvailTransientVertexBuffer(3, post_layout_) < 3) {
        return;
    }
    struct PostVtx {
        float x, y, u, v;
    };
    const bool flip = !bgfx::getCaps()->originBottomLeft;
    bgfx::TransientVertexBuffer tvb;
    bgfx::allocTransientVertexBuffer(&tvb, 3, post_layout_);
    PostVtx* v = reinterpret_cast<PostVtx*>(tvb.data);
    auto set = [&](int i, float x, float y) {
        v[i].x = x;
        v[i].y = y;
        v[i].u = (x + 1.0f) * 0.5f;
        const float vv = (y + 1.0f) * 0.5f;
        v[i].v = flip ? (1.0f - vv) : vv;
    };
    set(0, -1.0f, -1.0f);
    set(1, 3.0f, -1.0f);
    set(2, -1.0f, 3.0f);

    bgfx::setViewFrameBuffer(kPostView, BGFX_INVALID_HANDLE);
    bgfx::setViewRect(kPostView, 0, 0, static_cast<std::uint16_t>(width_), static_cast<std::uint16_t>(height_));
    bgfx::setViewClear(kPostView, BGFX_CLEAR_NONE);

    const float params[4] = {
        static_cast<float>(params_.pixel_size),
        static_cast<float>(params_.color_levels),
        0.0f,
        0.0f,
    };
    bgfx::setUniform(post_params_uniform_, params);
    bgfx::setTexture(0, post_sampler_, offscreen_tex_,
        BGFX_SAMPLER_POINT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
    bgfx::setVertexBuffer(0, &tvb);
    bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
    bgfx::submit(kPostView, post_program_);
}

void BgfxRenderer::present()
{
    if (!initialized_) {
        return;
    }
    bgfx::frame();
    ++frame_count_;
}

void BgfxRenderer::request_screenshot(const std::string& path)
{
    if (!initialized_ || path.empty()) {
        return;
    }
    // The capture lands in the callback while the NEXT frame is processed, so callers must keep
    // presenting for a frame or two afterwards.
    bgfx::requestScreenShot(BGFX_INVALID_HANDLE, path.c_str());
}

void BgfxRenderer::submit(const DrawItem& item)
{
    items_.push_back(item);
}

void BgfxRenderer::set_cutaway(Vec3 centre, double radius, Vec3 to_camera, double floor, double keep, double ceiling, double foliage, double foliage_keep)
{
    cut_keep_[2] = static_cast<float>(foliage);
    (void)foliage_keep;
    cut_keep_[3] = 0.0f;
    (void)keep;
    cut_keep_[0] = 0.0f; // the pass that is whole
    cut_keep_[1] = static_cast<float>(ceiling);
    cut_[0] = static_cast<float>(centre.x);
    cut_[1] = static_cast<float>(centre.y);
    cut_[2] = static_cast<float>(centre.z);
    cut_[3] = static_cast<float>(radius);
    cut_dir_[0] = static_cast<float>(to_camera.x);
    cut_dir_[1] = static_cast<float>(to_camera.y);
    cut_dir_[2] = static_cast<float>(to_camera.z);
    cut_dir_[3] = static_cast<float>(floor);
}

void BgfxRenderer::submit_triangles(const std::vector<ColorVertex>& vertices)
{
    triangles_.reserve(triangles_.size() + vertices.size());
    for (const ColorVertex& v : vertices) {
        triangles_.push_back(PosColorVertex{v.x, v.y, v.z, v.abgr});
    }
}

void BgfxRenderer::release(StaticMesh& mesh)
{
    if (bgfx::isValid(mesh.vertices)) {
        bgfx::destroy(mesh.vertices);
    }
    if (bgfx::isValid(mesh.indices)) {
        bgfx::destroy(mesh.indices);
    }
    mesh = StaticMesh{};
}

int BgfxRenderer::upload_static_mesh(int id, const std::vector<ColorVertex>& vertices,
    const std::vector<std::uint32_t>& indices)
{
    if (!initialized_) {
        return -1;
    }
    if (id < 0 || static_cast<std::size_t>(id) >= static_meshes_.size()) {
        id = -1;
        for (std::size_t i = 0; i < static_meshes_.size(); ++i) {
            if (!static_meshes_[i].alive) {
                id = static_cast<int>(i);
                break;
            }
        }
        if (id < 0) {
            static_meshes_.push_back(StaticMesh{});
            id = static_cast<int>(static_meshes_.size()) - 1;
        }
    }
    StaticMesh& mesh = static_meshes_[static_cast<std::size_t>(id)];
    release(mesh);
    mesh.alive = true;
    if (vertices.empty() || indices.empty()) {
        return id; // an empty chunk (all air) is a mesh that draws nothing
    }
    static_assert(sizeof(ColorVertex) == sizeof(PosColorVertex));
    const std::uint32_t vbytes = static_cast<std::uint32_t>(vertices.size() * sizeof(ColorVertex));
    const std::uint32_t ibytes = static_cast<std::uint32_t>(indices.size() * sizeof(std::uint32_t));
    mesh.vertices = bgfx::createVertexBuffer(bgfx::copy(vertices.data(), vbytes), layout_);
    mesh.indices = bgfx::createIndexBuffer(bgfx::copy(indices.data(), ibytes), BGFX_BUFFER_INDEX32);
    mesh.bytes = vbytes + ibytes;
    return id;
}

void BgfxRenderer::destroy_static_mesh(int id)
{
    if (id < 0 || static_cast<std::size_t>(id) >= static_meshes_.size()) {
        return;
    }
    release(static_meshes_[static_cast<std::size_t>(id)]);
}

void BgfxRenderer::draw_static_mesh(int id, bool clipped)
{
    if (id >= 0 && static_cast<std::size_t>(id) < static_meshes_.size() && static_meshes_[static_cast<std::size_t>(id)].alive) {
        static_draws_.emplace_back(id, clipped);
    }
}

void BgfxRenderer::set_clip_box(double x0, double y0, double x1, double y1)
{
    clip_[0] = static_cast<float>(x0);
    clip_[1] = static_cast<float>(y0);
    clip_[2] = static_cast<float>(x1);
    clip_[3] = static_cast<float>(y1);
}

std::size_t BgfxRenderer::static_mesh_count() const
{
    return static_cast<std::size_t>(std::count_if(
        static_meshes_.begin(), static_meshes_.end(), [](const StaticMesh& m) { return m.alive; }));
}

void BgfxRenderer::set_pixel_post(bool enabled)
{
    params_.enabled = enabled;
}

void BgfxRenderer::set_post_params(const PostProcessParams& params)
{
    params_ = sanitize(params);
}

const char* BgfxRenderer::backend_name() const
{
    return bgfx::getRendererName(bgfx::getRendererType());
}

} // namespace fieldengine::render

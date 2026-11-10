#pragma once

#include "render/Camera.hpp"
#include "render/PostProcess.hpp"
#include "render/RenderTypes.hpp"

#include <bgfx/bgfx.h>

#include <cstdint>
#include <string>
#include <vector>

namespace fieldengine::render {

// The GPU backend: orthographic camera, static meshes for the world, already-built DrawItems tessellated into
// transient buffers. It only ever reads the simulation. The smoke runs it on bgfx::RendererType::Noop
// (RenderConfig::use_noop_backend), which needs no shaders or window and never touches a real device.
class BgfxRenderer {
public:
    bool init(const RenderConfig& config);
    void shutdown();

    void begin_frame();
    void end_frame();
    void present(); // flushes to the device, so several layers can be submitted before one present

    void set_camera(const Camera& camera);
    void submit(const DrawItem& item);
    void submit_triangles(const std::vector<ColorVertex>& vertices); // opaque, three vertices per triangle
    // A tube from `centre` toward `to_camera` (world units), above height `floor`, is not drawn. radius 0: none.
    void set_cutaway(Vec3 centre, double radius, Vec3 to_camera, double floor, double keep = 0.0, double ceiling = 0.0, double foliage = 0.0, double foliage_keep = 0.0);

    // GEOMETRY THAT STAYS ON THE GPU: a chunk of the world is uploaded once, drawn every frame it is in
    // view, and replaced only when the law changes it. `id` < 0 creates a mesh; the id is returned.
    // Positions are world units, indices 32-bit, opaque.
    int upload_static_mesh(int id, const std::vector<ColorVertex>& vertices, const std::vector<std::uint32_t>& indices);
    void destroy_static_mesh(int id);
    void draw_static_mesh(int id, bool clipped = false); // this frame; clipped: not drawn inside the clip box
    // A box on the ground (world units) inside which clipped meshes are not drawn: where the live window is drawn at full
    // detail, the coarse regions under it step aside. x1 <= x0: none.
    void set_clip_box(double x0, double y0, double x1, double y1);
    std::size_t static_mesh_count() const;
    void set_pixel_post(bool enabled);
    void set_post_params(const PostProcessParams& params);

    const char* backend_name() const;

    // Capture the backbuffer to an uncompressed TGA at `path`. The write happens asynchronously in
    // bgfx's callback (a frame or two later), so keep presenting after the call. Used to verify a
    // real run's output offline (`CL_SMOKE_SHOT` on the client); no-op before init.
    void request_screenshot(const std::string& path);

    std::uint64_t frame_count() const { return frame_count_; }

private:
    struct PosColorVertex {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::uint32_t abgr = 0xffffffffu;
    };

    void ensure_reset(int width, int height);
    void ensure_offscreen(int low_w, int low_h);
    void destroy_offscreen();
    void run_post_pass();
    void tessellate(const DrawItem& item, std::vector<PosColorVertex>& out) const;

    static constexpr bgfx::ViewId kWorldView = 0;
    static constexpr bgfx::ViewId kPostView = 1;

    bool initialized_ = false;
    bool program_valid_ = false;
    std::uint64_t frame_count_ = 0;
    int width_ = 1280;
    int height_ = 720;
    std::uint32_t reset_flags_ = 0;
    PostProcessParams params_;
    bool post_active_ = false; // whether the current frame routed the world through the offscreen FB

    struct StaticMesh {
        bgfx::VertexBufferHandle vertices = BGFX_INVALID_HANDLE;
        bgfx::IndexBufferHandle indices = BGFX_INVALID_HANDLE;
        std::size_t bytes = 0;
        bool alive = false;
    };
    std::vector<StaticMesh> static_meshes_;
    std::vector<std::pair<int, bool>> static_draws_; // ids queued this frame, and whether each is clipped
    void release(StaticMesh& mesh);

    Camera camera_;
    std::vector<DrawItem> items_;
    std::vector<PosColorVertex> triangles_; // pre-built opaque geometry for this frame

    bgfx::VertexLayout layout_{};
    bgfx::UniformHandle cut_uniform_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle cut_dir_uniform_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle cut_keep_uniform_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle clip_uniform_ = BGFX_INVALID_HANDLE;
    float clip_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cut_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cut_dir_[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float cut_keep_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bgfx::ProgramHandle program_ = BGFX_INVALID_HANDLE;

    // Pixel-art post pass resources.
    bgfx::ProgramHandle post_program_ = BGFX_INVALID_HANDLE;
    bool post_program_valid_ = false;
    bgfx::VertexLayout post_layout_{};
    bgfx::UniformHandle post_sampler_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle post_params_uniform_ = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle offscreen_fb_ = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle offscreen_tex_ = BGFX_INVALID_HANDLE;
    int offscreen_w_ = 0;
    int offscreen_h_ = 0;
};

} // namespace fieldengine::render

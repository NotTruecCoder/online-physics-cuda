#include "world/Playground.hpp"
#include "world/VoxelMesh.hpp"
#include "render/BgfxRenderer.hpp"
#include "render/Camera.hpp"
#include "render/Units.hpp"

#include <iostream>

// GPU smoke: drive the real bgfx backend on the Noop renderer (no device, no window, no shaders).
// Proves init -> voxel mesh of the proving ground as triangles + a body as a draw item -> N frames ->
// shutdown runs without crashing. No pixel assertions; this gate only guards the bgfx shell.
int main()
{
    using namespace fieldengine;

    std::vector<std::string> errors;
    std::optional<field::Playground> world = field::Playground::load_default(errors);
    if (!world) {
        for (const std::string& e : errors) {
            std::cerr << "  " << e << "\n";
        }
        std::cerr << "render_gpu_smoke: the proving ground did not load\n";
        return 1;
    }

    render::BgfxRenderer renderer;
    render::RenderConfig config;
    config.use_noop_backend = true;
    config.width = 640;
    config.height = 480;
    if (!renderer.init(config)) {
        std::cerr << "render_gpu_smoke: bgfx init failed\n";
        return 1;
    }

    render::Camera camera(640.0, 480.0);
    camera.set_pan(render::Vec2{0.12, 0.12});
    renderer.set_pixel_post(true);

    std::vector<field::MeshVertex> mesh;
    std::vector<render::ColorVertex> triangles;
    constexpr int frames = 8;
    for (int frame = 0; frame < frames; ++frame) {
        world->step(field::Control{});
        field::build_voxel_mesh(world->fields(), mesh);
        triangles.clear();
        for (const field::MeshVertex& v : mesh) {
            triangles.push_back(render::ColorVertex{static_cast<float>(render::units::to_sim_units(v.x)),
                static_cast<float>(render::units::to_sim_units(v.y)), static_cast<float>(render::units::to_sim_units(v.z)),
                v.abgr});
        }
        renderer.set_camera(camera);
        renderer.begin_frame();
        renderer.submit_triangles(triangles);
        renderer.submit(render::DrawItem{.position = {0.05, 0.05, 0.0}, .size = 0.004});
        renderer.end_frame();
        renderer.present();
    }

    // A chunk's life on the GPU: uploaded, drawn, replaced, destroyed; the slot is reused.
    {
        const std::vector<std::uint32_t> indices{0, 1, 2};
        const int id = renderer.upload_static_mesh(-1, triangles, indices);
        const int empty = renderer.upload_static_mesh(-1, {}, {}); // an all-air chunk
        renderer.begin_frame();
        renderer.draw_static_mesh(id);
        renderer.draw_static_mesh(empty);
        renderer.end_frame();
        renderer.present();
        const bool replaced = renderer.upload_static_mesh(id, triangles, indices) == id;
        const std::size_t count = renderer.static_mesh_count();
        renderer.destroy_static_mesh(id);
        const bool freed = renderer.static_mesh_count() == count - 1;
        const int again = renderer.upload_static_mesh(-1, triangles, indices);
        renderer.draw_static_mesh(99999); // nothing there: refused, not a crash
        if (id < 0 || empty < 0 || !replaced || count != 2 || !freed || again != id) {
            std::cerr << "render_gpu_smoke: static meshes misbehave\n";
            renderer.shutdown();
            return 1;
        }
        renderer.destroy_static_mesh(again);
        renderer.destroy_static_mesh(empty);
    }

    if (mesh.empty() || renderer.frame_count() != static_cast<std::uint64_t>(frames + 1)) {
        std::cerr << "render_gpu_smoke: expected " << frames << " frames of a non-empty world, got "
                  << renderer.frame_count() << " frames, " << mesh.size() << " vertices\n";
        renderer.shutdown();
        return 1;
    }
    const char* backend = renderer.backend_name();
    std::cout << "render_gpu_smoke OK (backend=" << (backend ? backend : "?") << ", " << mesh.size() / 3
              << " triangles)\n";
    renderer.shutdown();
    return 0;
}

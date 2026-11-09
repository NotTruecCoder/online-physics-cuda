#include "sandbox/WorldView.hpp"

#include "render/Units.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::app {

void WorldView::reset(render::BgfxRenderer& renderer)
{
    mesher_.clear([&](int handle) { renderer.destroy_static_mesh(handle); });
}

void WorldView::update(const field::Playground& world, render::BgfxRenderer& renderer, const render::Camera& camera)
{
    constexpr double sim = render::units::sim_units_per_metre;
    const double half_h = camera.ortho_half_height() / sim; // metres of the view plane
    const double half_w = camera.ortho_half_width() / sim;
    field::LodView view;
    view.pixels_per_metre = camera.viewport_height() / (2.0 * half_h);
    view.centre_x = camera.pan().x / sim;
    view.centre_y = camera.pan().y / sim;
    // The ground is seen foreshortened by the sine of the pitch.
    // (The tree in plan() culls the rest; this only has to be enough for the highest ground.)
    const double depth = (half_h + 520.0) / std::sin(camera.pitch_degrees() * 3.14159265358979 / 180.0);
    view.radius_metres = std::hypot(half_w, depth);
    // A quarter of a screen around it, so walking and turning find the world already built.
    const double w = camera.viewport_width();
    const double h = camera.viewport_height();
    const double mx = 0.25 * w;
    const double my = 0.25 * h;
    view.visible = [&camera, w, h, mx, my](const double lo[3], const double hi[3]) {
        double x0 = 1e30;
        double x1 = -1e30;
        double y0 = 1e30;
        double y1 = -1e30;
        for (int corner = 0; corner < 8; ++corner) {
            const render::Vec2 p = camera.world_to_screen(render::Vec3{(corner & 1 ? hi[0] : lo[0]) * sim,
                (corner & 2 ? hi[1] : lo[1]) * sim, (corner & 4 ? hi[2] : lo[2]) * sim});
            x0 = std::min(x0, p.x);
            x1 = std::max(x1, p.x);
            y0 = std::min(y0, p.y);
            y1 = std::max(y1, p.y);
        }
        return x1 >= -mx && x0 <= w + mx && y1 >= -my && y0 <= h + my;
    };
    const double sx = 0.04 * w;
    const double sy = 0.04 * h;
    view.in_sight = [&camera, w, h, sx, sy](const double lo[3], const double hi[3]) {
        double x0 = 1e30;
        double x1 = -1e30;
        double y0 = 1e30;
        double y1 = -1e30;
        for (int corner = 0; corner < 8; ++corner) {
            const render::Vec2 p = camera.world_to_screen(render::Vec3{(corner & 1 ? hi[0] : lo[0]) * sim,
                (corner & 2 ? hi[1] : lo[1]) * sim, (corner & 4 ? hi[2] : lo[2]) * sim});
            x0 = std::min(x0, p.x);
            x1 = std::max(x1, p.x);
            y0 = std::min(y0, p.y);
            y1 = std::max(y1, p.y);
        }
        return x1 >= -sx && x0 <= w + sx && y1 >= -sy && y0 <= h + sy;
    };
    // Half a screen of margin moves slowly: the plan is redone every 2 m of panning (or a sixteenth of the view, if
    // that is more), or when the view turns.
    const auto mix = [](std::uint64_t h, double v) { return (h ^ static_cast<std::uint64_t>(std::llround(v * 16.0))) * 1099511628211ull; };
    const double stride = std::max(2.0, half_h / 8.0);
    view.stamp = mix(mix(mix(mix(mix(1469598103934665603ull, camera.yaw_degrees()), camera.zoom() * 1000.0), w), h), std::floor(view.centre_x / stride));
    view.stamp = mix(mix(view.stamp, std::floor(view.centre_y / stride)), std::floor(camera.target().z / sim / stride));
    mesher_.update(world, view,
        [&](int handle, const field::ChunkMesh& mesh) {
            vertices_.resize(mesh.vertices.size());
            for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
                const field::MeshVertex& v = mesh.vertices[i];
                vertices_[i] = render::ColorVertex{static_cast<float>(v.x * sim), static_cast<float>(v.y * sim),
                    static_cast<float>(v.z * sim), v.abgr};
            }
            return renderer.upload_static_mesh(handle, vertices_, mesh.indices);
        },
        [&](int handle) { renderer.destroy_static_mesh(handle); });
}

void WorldView::draw(render::BgfxRenderer& renderer, double chunk_metres) const
{
    // Zoomed out a little, the live window is drawn at full detail and the coarse regions step aside where it is.
    const bool overlay = mesher_.live_overlay();
    const std::array<int, 4> box = mesher_.window_box();
    constexpr double sim = render::units::sim_units_per_metre;
    if (overlay) {
        renderer.set_clip_box(box[0] * chunk_metres * sim, box[1] * chunk_metres * sim, box[2] * chunk_metres * sim, box[3] * chunk_metres * sim);
    }
    mesher_.for_each_drawn([&](int handle, int level, int, int) { renderer.draw_static_mesh(handle, overlay && level > 0); });
}

} // namespace fieldengine::app

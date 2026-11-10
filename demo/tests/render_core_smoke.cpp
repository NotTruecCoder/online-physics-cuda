// The render layer's arithmetic, without a GPU: the camera the picker and the culling rely on, the packing the
// backend relies on, the pixel-art parameters it clamps.
#include "render/Camera.hpp"
#include "render/PostProcess.hpp"
#include "render/RenderTypes.hpp"
#include "render/Units.hpp"
#include "tests/Gate.hpp"

#include <cmath>

using namespace fieldengine::render;
using fieldengine::gate::check;

namespace {

bool nearly(double a, double b, double eps = 1e-6)
{
    return std::abs(a - b) <= eps;
}

// world_to_screen and screen_to_world must be inverses on the ground plane under pan and zoom.
void test_camera_round_trip()
{
    Camera camera(1280.0, 720.0);
    camera.set_pan(Vec2{0.3, 0.7});
    camera.set_zoom(1.8);
    const Vec3 world_points[] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.15, 0.85, 0.0}, {1.0, 1.0, 0.0}, {0.42, 0.11, 0.0},
    };
    for (const Vec3& w : world_points) {
        const Vec3 back = camera.screen_to_world(camera.world_to_screen(w), 0.0);
        check(nearly(back.x, w.x, 1e-4) && nearly(back.y, w.y, 1e-4), "camera world->screen->world identity (z=0)");
    }
    const Vec2 screen_points[] = {{0.0, 0.0}, {640.0, 360.0}, {1279.0, 5.0}, {100.0, 700.0}};
    for (const Vec2& s : screen_points) {
        const Vec2 back = camera.world_to_screen(camera.screen_to_world(s, 0.0));
        check(nearly(back.x, s.x, 1e-3) && nearly(back.y, s.y, 1e-3), "camera screen->world->screen identity");
    }
}

// The pan target projects to the screen centre; elevation reads upward; zoom magnifies.
void test_camera_pan_zoom()
{
    Camera camera(800.0, 600.0);
    camera.set_pan(Vec2{0.5, 0.5});
    camera.set_zoom(1.0);
    const Vec2 centre = camera.world_to_screen(Vec3{0.5, 0.5, 0.0});
    check(nearly(centre.x, 400.0, 1e-3) && nearly(centre.y, 300.0, 1e-3), "pan target projects to screen centre");
    check(camera.world_to_screen(Vec3{0.5, 0.5, 0.3}).y < centre.y, "world +z lifts on screen");
    const Vec2 corner = camera.world_to_screen(Vec3{0.9, 0.5, 0.0});
    camera.set_zoom(2.0);
    const Vec2 corner_zoomed = camera.world_to_screen(Vec3{0.9, 0.5, 0.0});
    check(std::abs(corner_zoomed.x - 400.0) > std::abs(corner.x - 400.0), "higher zoom spreads points");
}

void test_post_process_params()
{
    const PostProcessParams low = sanitize(PostProcessParams{true, 0, 1});
    check(low.pixel_size == 1 && low.color_levels == 2, "post parameters clamp up");
    const PostProcessParams high = sanitize(PostProcessParams{true, 999, 9999});
    check(high.pixel_size == 32 && high.color_levels == 256, "post parameters clamp down");
    const PostProcessParams kept = sanitize(PostProcessParams{true, 5, 20});
    check(kept.pixel_size == 5 && kept.color_levels == 20, "in-range post parameters are kept");
}

} // namespace

int main()
{
    test_camera_round_trip();
    test_camera_pan_zoom();
    test_post_process_params();
    const Color c{0x12, 0x34, 0x56, 0x78};
    check(c.to_abgr() == 0x78563412u, "ABGR packing matches byte order");
    check(nearly(units::to_metres(1.0), 100.0) && nearly(units::to_sim_units(100.0), 1.0), "1 sim-unit is 100 m");
    return fieldengine::gate::report("render_core");
}

#pragma once

#include "render/RenderTypes.hpp"

namespace fieldengine::render {

// Isometric orthographic 3D camera. The world is full 3D (x,y on the ground, z = height/up); it only
// reads as 2D because of the fixed isometric view (GDD §14.3). Pure math (glm internally): it exposes
// eye/target/up + ortho extents for the GPU and an invertible world<->screen mapping (ground-plane
// ray) for mouse picking. No GPU state here.
//
// Canonical sim space stays normalized [0,1]^2 on the ground (see UNITS.md); z is height in the same
// unit. Picking resolves the cursor onto the z=0 ground plane.
class Camera {
public:
    Camera() = default;
    Camera(double viewport_w, double viewport_h);

    void set_viewport(double width, double height);
    void set_zoom(double zoom);                 // >1 zooms in
    void set_pan(Vec2 ground_center);           // world ground point the camera looks at
    void set_target_height(double z);           // ...and its height (a hill top stays mid-screen)
    void set_vertical_extent(double world_units); // world height visible at zoom 1
    void set_iso_angles(double yaw_degrees, double pitch_degrees);

    double viewport_width() const { return viewport_w_; }
    double viewport_height() const { return viewport_h_; }
    double zoom() const { return zoom_; }
    Vec2 pan() const { return pan_; }
    double yaw_degrees() const { return yaw_deg_; }
    double pitch_degrees() const { return pitch_deg_; }

    // Camera basis for the GPU view matrix.
    Vec3 eye() const;
    Vec3 target() const { return Vec3{pan_.x, pan_.y, target_z_}; }
    Vec3 up() const { return Vec3{0.0, 0.0, 1.0}; }
    double ortho_half_height() const;
    double ortho_half_width() const;
    double far_plane() const; // of the view volume, from the eye

    // World <-> screen for picking / overlays. screen_to_world intersects the ray with z=world_z.
    Vec2 world_to_screen(const Vec3& world) const;
    Vec2 world_to_screen(const Vec2& ground) const { return world_to_screen(Vec3{ground.x, ground.y, 0.0}); }
    Vec3 screen_to_world(const Vec2& screen, double world_z = 0.0) const;

private:
    double viewport_w_ = 1280.0;
    double viewport_h_ = 720.0;
    double zoom_ = 1.0;
    Vec2 pan_{0.5, 0.5};
    double target_z_ = 0.0;
    double vertical_extent_ = 1.4; // fits the [0,1] arena with margin at zoom 1
    double yaw_deg_ = 45.0;
    double pitch_deg_ = 35.264; // true isometric
    double eye_distance() const;
    double distance_ = 6.0;     // arbitrary for ortho; only direction matters
};

} // namespace fieldengine::render

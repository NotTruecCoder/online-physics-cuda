#include "render/Camera.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace fieldengine::render {

namespace {

constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

glm::dvec3 to_glm(const Vec3& v)
{
    return glm::dvec3{v.x, v.y, v.z};
}

} // namespace

Camera::Camera(double viewport_w, double viewport_h)
    : viewport_w_(viewport_w), viewport_h_(viewport_h)
{
}

void Camera::set_viewport(double width, double height)
{
    viewport_w_ = width > 1.0 ? width : 1.0;
    viewport_h_ = height > 1.0 ? height : 1.0;
}

void Camera::set_zoom(double zoom)
{
    zoom_ = std::clamp(zoom, 1e-5, 50.0);
}

void Camera::set_pan(Vec2 ground_center)
{
    pan_ = ground_center;
}

void Camera::set_target_height(double z)
{
    target_z_ = z;
}

void Camera::set_vertical_extent(double world_units)
{
    vertical_extent_ = world_units > 0.01 ? world_units : vertical_extent_;
}

void Camera::set_iso_angles(double yaw_degrees, double pitch_degrees)
{
    yaw_deg_ = yaw_degrees;
    pitch_deg_ = std::clamp(pitch_degrees, 5.0, 89.0);
}

Vec3 Camera::eye() const
{
    const double yaw = yaw_deg_ * kDeg2Rad;
    const double pitch = pitch_deg_ * kDeg2Rad;
    const Vec3 t = target();
    // Direction from target to eye: above the ground (sin pitch) and around it (yaw).
    const double dx = std::cos(pitch) * std::cos(yaw);
    const double dy = std::cos(pitch) * std::sin(yaw);
    const double dz = std::sin(pitch);
    const double distance = eye_distance();
    return Vec3{t.x + dx * distance, t.y + dy * distance, t.z + dz * distance};
}

double Camera::eye_distance() const
{
    // Far enough back that nothing on screen is behind the eye, however wide the view and however tall the world.
    return std::max(distance_, 3.0 * ortho_half_width() + 3.0 * ortho_half_height() + 6.0);
}

double Camera::far_plane() const
{
    return 2.0 * eye_distance();
}

double Camera::ortho_half_height() const
{
    return vertical_extent_ * 0.5 / zoom_;
}

double Camera::ortho_half_width() const
{
    return ortho_half_height() * (viewport_w_ / viewport_h_);
}

Vec2 Camera::world_to_screen(const Vec3& world) const
{
    const glm::dmat4 view = glm::lookAtRH(to_glm(eye()), to_glm(target()), to_glm(up()));
    const double hh = ortho_half_height();
    const double hw = ortho_half_width();
    const glm::dmat4 proj = glm::orthoRH_NO(-hw, hw, -hh, hh, 0.05, far_plane());
    const glm::dvec4 clip = proj * view * glm::dvec4(to_glm(world), 1.0);
    const glm::dvec3 ndc = glm::dvec3(clip) / clip.w;
    return Vec2{
        (ndc.x * 0.5 + 0.5) * viewport_w_,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_h_,
    };
}

Vec3 Camera::screen_to_world(const Vec2& screen, double world_z) const
{
    const glm::dmat4 view = glm::lookAtRH(to_glm(eye()), to_glm(target()), to_glm(up()));
    const double hh = ortho_half_height();
    const double hw = ortho_half_width();
    const glm::dmat4 proj = glm::orthoRH_NO(-hw, hw, -hh, hh, 0.05, far_plane());
    const glm::dmat4 inv = glm::inverse(proj * view);

    const double ndc_x = screen.x / viewport_w_ * 2.0 - 1.0;
    const double ndc_y = (1.0 - screen.y / viewport_h_) * 2.0 - 1.0;
    glm::dvec4 near_p = inv * glm::dvec4(ndc_x, ndc_y, -1.0, 1.0);
    glm::dvec4 far_p = inv * glm::dvec4(ndc_x, ndc_y, 1.0, 1.0);
    near_p /= near_p.w;
    far_p /= far_p.w;

    const glm::dvec3 ray = glm::dvec3(far_p) - glm::dvec3(near_p);
    if (std::abs(ray.z) < 1e-12) {
        return Vec3{near_p.x, near_p.y, world_z};
    }
    const double t = (world_z - near_p.z) / ray.z;
    const glm::dvec3 hit = glm::dvec3(near_p) + ray * t;
    return Vec3{hit.x, hit.y, world_z};
}

} // namespace fieldengine::render

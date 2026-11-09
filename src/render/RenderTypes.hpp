#pragma once

#include <cstdint>

namespace fieldengine::render {

// Plain math vectors for the render layer. The simulation owns its own coordinate types; the render
// layer never writes back, so these are intentionally trivial value types.
struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// One vertex of a pre-built, already-coloured triangle list in world units (a voxel mesh built from
// the fields, for one): the backend draws it as opaque geometry, three vertices per triangle.
struct ColorVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::uint32_t abgr = 0xffffffffu;
};

// 8-bit straight-alpha colour.
struct Color {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    // bgfx/most 2D APIs consume 0xAABBGGRR packed little-endian.
    constexpr std::uint32_t to_abgr() const
    {
        return (static_cast<std::uint32_t>(a) << 24) | (static_cast<std::uint32_t>(b) << 16) |
               (static_cast<std::uint32_t>(g) << 8) | static_cast<std::uint32_t>(r);
    }
};

// The window the backend draws into, opaque so this layer stays free of platform headers.
struct RenderConfig {
    int width = 1280;
    int height = 720;
    void* native_window_handle = nullptr;  // SDL/Win32 HWND etc.; null for headless/noop
    void* native_display_handle = nullptr; // X11/Wayland; null on Windows
    bool use_noop_backend = false;         // force bgfx RendererType::Noop for CI smoke
};

constexpr bool operator==(const Color& a, const Color& b)
{
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

enum class Shape {
    Quad,
    Cube,
};

// Draw ordering: lower layers render first.
enum class Layer {
    Field,
    Body,
};

// One renderable primitive in world space. The render layer is pull-only: a DrawItem is built from a
// read-only view of the simulation and never mutates it.
struct DrawItem {
    Vec3 position;        // world-space base point: x,y on the ground, z = bottom elevation
    Shape shape = Shape::Cube;
    double size = 0.01;   // half-extent where half_w, half_h or height are not given
    double half_w = 0.0;  // x half-extent for Quad/Cube
    double half_h = 0.0;  // y half-extent for Quad/Cube
    double height = 0.0;  // z extent for Cube voxels (0 => a cube of side 2*size); ignored otherwise
    Color color;
    Layer layer = Layer::Body;

};

} // namespace fieldengine::render

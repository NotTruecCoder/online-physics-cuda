#pragma once

#include <algorithm>

namespace fieldengine::render {

// Pixel-art post-process parameters (GDD §9/§14.3: a final shader pixelizes + palette-limits the
// frame for the Albion-like look). Data, not magic numbers in the shader: the renderer reads these
// to size the offscreen target and quantize colours. Pure and headless-testable.
struct PostProcessParams {
    bool enabled = false;
    int pixel_size = 3;     // downsample factor; the world is rendered at 1/pixel_size then upscaled
    int color_levels = 24;  // per-channel quantization levels (palette limiting)
};

// Clamp parameters into the valid, renderable range. Deterministic: same input -> same output.
constexpr PostProcessParams sanitize(PostProcessParams params)
{
    params.pixel_size = std::clamp(params.pixel_size, 1, 32);
    params.color_levels = std::clamp(params.color_levels, 2, 256);
    return params;
}

constexpr bool operator==(const PostProcessParams& a, const PostProcessParams& b)
{
    return a.enabled == b.enabled && a.pixel_size == b.pixel_size && a.color_levels == b.color_levels;
}

} // namespace fieldengine::render

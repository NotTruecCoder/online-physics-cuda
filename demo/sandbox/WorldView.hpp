#pragma once

#include "world/Playground.hpp"
#include "world/WorldMesher.hpp"
#include "render/BgfxRenderer.hpp"
#include "render/Camera.hpp"

#include <vector>

// The open world on the graphics card: what field::WorldMesher builds, as static meshes, for this camera.
namespace fieldengine::app {

class WorldView {
public:
    void reset(render::BgfxRenderer& renderer);
    void update(const field::Playground& world, render::BgfxRenderer& renderer, const render::Camera& camera);
    void draw(render::BgfxRenderer& renderer, double chunk_metres) const;

    field::WorldMesher& mesher() { return mesher_; }
    const field::WorldMesher::Stats& stats() const { return mesher_.stats(); }

private:
    field::WorldMesher mesher_;
    std::vector<render::ColorVertex> vertices_;
};

} // namespace fieldengine::app

#pragma once

#include "field/Playground.hpp"
#include "render/BgfxRenderer.hpp"

#include <cstdint>
#include <string>
#include <vector>

// WHAT A SPELL LOOKS AND FEELS LIKE, BESIDES WHAT THE LAW DOES WITH IT: cubes flying off, the view shaking, the screen
// going white as you gate. All of it runs on the frame and reads the world's events; none of it reaches the simulation.
namespace fieldengine::app {

struct Rgb {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

class Effects {
public:
    void update(double dt);
    void submit(render::BgfxRenderer& renderer, double sim_per_metre) const;
    void clear();

    // What the world's events look like.
    void burst(field::Vec3 at, Rgb colour, double radius);     // an impact: cubes out, a ring on the ground, the view shakes
    void ring(field::Vec3 at, Rgb colour, double radius);      // a ring of cubes going out along the ground
    void beam(field::Vec3 from, field::Vec3 to, Rgb colour, double radius);
    void column(field::Vec3 foot, double height, Rgb colour, double radius);
    void streak(field::Vec3 from, field::Vec3 to, Rgb colour);  // a dash, a blink: cubes hanging where you were
    void puff(field::Vec3 at, Rgb colour, int count, double speed); // at your feet: a jump, a step
    void spark(field::Vec3 at, Rgb colour, double size, double life); // one cube hanging there, shrinking
    void bit(field::Vec3 at, field::Vec3 velocity, std::uint32_t abgr, double size, double life); // one cube off your body
    void gather(field::Vec3 at, Rgb colour, double radius);    // cubes drawn in to a point
    void impact(field::Vec3 at, field::Vec3 way, Rgb colour, double power); // a blow landing: thrown on the way it was going
    void shockwave(field::Vec3 at, Rgb colour, double radius); // what a blow does to the ground: earth and dust, going out

    // How far the view is off where it should be, metres. Shake adds up, and falls away.
    void shake(double amount) { trauma_ = std::min(1.0, trauma_ + amount); }
    [[nodiscard]] field::Vec3 view_offset() const { return offset_; }

    // The screen going to a colour and coming back.
    void flash(Rgb colour, double alpha) { flash_ = colour; flash_alpha_ = std::max(flash_alpha_, alpha); }
    [[nodiscard]] Rgb flash_colour() const { return flash_; }
    [[nodiscard]] double flash_alpha() const { return flash_alpha_; }
    void hold_flash(double alpha) { flash_alpha_ = alpha; }

    // Numbers rising off what was hurt.
    struct Floater {
        field::Vec3 at{};
        double value = 0.0;
        double life = 0.0;
    };
    void floater(field::Vec3 at, double value);
    [[nodiscard]] const std::vector<Floater>& floaters() const { return floaters_; }


private:
    struct Particle {
        field::Vec3 at{};
        field::Vec3 velocity{};
        double size = 0.1;
        double life = 0.5;
        double life_max = 0.5;
        double gravity = 0.0;
        double drag = 0.0;
        std::uint32_t abgr = 0xffffffffu;
    };
    void add(const Particle& particle);
    double random();

    std::vector<Particle> particles_;
    std::size_t next_ = 0;
    std::uint32_t seed_ = 0x9e3779b9u;
    std::vector<Floater> floaters_;
    double trauma_ = 0.0;
    double time_ = 0.0;
    field::Vec3 offset_{};
    Rgb flash_{};
    double flash_alpha_ = 0.0;
};

} // namespace fieldengine::app

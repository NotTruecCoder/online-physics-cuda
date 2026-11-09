#include "sandbox/Effects.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::app {

namespace {

constexpr std::size_t kMaxParticles = 1600;
constexpr double kTau = 6.283185307179586;

std::uint32_t abgr(Rgb c, double scale = 1.0)
{
    const auto channel = [&](float v) { return static_cast<std::uint32_t>(std::clamp(static_cast<double>(v) * scale, 0.0, 1.0) * 255.0); };
    return 0xff000000u | (channel(c.b) << 16) | (channel(c.g) << 8) | channel(c.r);
}

} // namespace

double Effects::random()
{
    seed_ ^= seed_ << 13;
    seed_ ^= seed_ >> 17;
    seed_ ^= seed_ << 5;
    return static_cast<double>(seed_ & 0xffffffu) / 16777216.0;
}

void Effects::add(const Particle& particle)
{
    // Past the limit the oldest go first: an effect is never worth a frame.
    if (particles_.size() < kMaxParticles) {
        particles_.push_back(particle);
        return;
    }
    particles_[next_] = particle;
    next_ = (next_ + 1) % kMaxParticles;
}

void Effects::clear()
{
    particles_.clear();
    floaters_.clear();
    next_ = 0;
    trauma_ = 0.0;
}

void Effects::update(double dt)
{
    time_ += dt;
    for (std::size_t i = 0; i < particles_.size();) {
        Particle& p = particles_[i];
        p.life -= dt;
        if (p.life <= 0.0) {
            particles_[i] = particles_.back();
            particles_.pop_back();
            continue;
        }
        p.velocity.z -= p.gravity * dt;
        const double keep = std::max(0.0, 1.0 - p.drag * dt);
        p.velocity = p.velocity * keep;
        p.at = p.at + p.velocity * dt;
        ++i;
    }
    if (next_ >= particles_.size()) {
        next_ = 0;
    }
    for (std::size_t i = 0; i < floaters_.size();) {
        floaters_[i].life -= dt;
        floaters_[i].at.z += 1.4 * dt;
        if (floaters_[i].life <= 0.0) {
            floaters_[i] = floaters_.back();
            floaters_.pop_back();
        } else {
            ++i;
        }
    }
    trauma_ = std::max(0.0, trauma_ - 1.8 * dt);
    flash_alpha_ = std::max(0.0, flash_alpha_ - 2.2 * dt);
    // Shake goes with the square, so small knocks are small; smooth noise, so it is a shudder and not static.
    const double s = trauma_ * trauma_ * 0.9;
    offset_ = field::Vec3{s * (std::sin(time_ * 61.0) + 0.5 * std::sin(time_ * 37.0)), s * (std::sin(time_ * 53.0 + 1.3) + 0.5 * std::sin(time_ * 41.0)),
        s * 0.5 * std::sin(time_ * 47.0 + 2.1)};
}

void Effects::submit(render::BgfxRenderer& renderer, double sim) const
{
    for (const Particle& p : particles_) {
        // A cube stays a cube: it shrinks away, it does not fade.
        const double half = p.size * std::sqrt(p.life / p.life_max);
        render::DrawItem item;
        item.shape = render::Shape::Cube;
        item.layer = render::Layer::Body;
        item.position = render::Vec3{p.at.x * sim, p.at.y * sim, (p.at.z - half) * sim};
        item.half_w = item.half_h = half * sim;
        item.height = 2.0 * half * sim;
        item.color = render::Color{static_cast<std::uint8_t>(p.abgr & 255u), static_cast<std::uint8_t>((p.abgr >> 8) & 255u), static_cast<std::uint8_t>((p.abgr >> 16) & 255u), 255};
        renderer.submit(item);
    }
}

void Effects::burst(field::Vec3 at, Rgb colour, double wide)
{
    const double radius = std::min(wide, 8.0); // past this it is the law you are watching, not a few cubes
    const int count = static_cast<int>(std::clamp(5.0 + radius * 4.0, 5.0, 28.0));
    for (int k = 0; k < count; ++k) {
        const double a = random() * kTau;
        const double up = random() * 1.2 - 0.2;
        const double speed = (0.5 + random()) * (3.0 + radius * 3.5);
        const field::Vec3 way = field::normalize(field::Vec3{std::cos(a), std::sin(a), up});
        const double shade = 0.7 + 0.6 * random();
        add(Particle{.at = at + way * (random() * radius * 0.4), .velocity = way * speed, .size = (0.05 + 0.1 * random()) * (0.8 + radius * 0.25),
            .life = 0.35 + 0.5 * random(), .life_max = 0.85, .gravity = 12.0, .drag = 1.6, .abgr = abgr(colour, shade)});
    }
    // A heart of white, gone at once.
    for (int k = 0; k < 3; ++k) {
        const field::Vec3 way = field::normalize(field::Vec3{random() - 0.5, random() - 0.5, random() - 0.5});
        add(Particle{.at = at + way * (radius * 0.3 * random()), .velocity = way * (radius * 2.0), .size = 0.12 + radius * 0.12, .life = 0.14, .life_max = 0.14,
            .abgr = abgr(Rgb{1.0f, 1.0f, 0.92f})});
    }
    ring(at, colour, radius);
    shake(std::min(0.75, 0.16 + radius * 0.11));
}

void Effects::ring(field::Vec3 at, Rgb colour, double wide)
{
    const double radius = std::min(wide, 8.0);
    const int count = static_cast<int>(std::clamp(8.0 + radius * 3.0, 8.0, 28.0));
    const double speed = 5.0 + radius * 6.0;
    for (int k = 0; k < count; ++k) {
        const double a = kTau * k / count + random() * 0.1;
        const field::Vec3 way{std::cos(a), std::sin(a), 0.0};
        add(Particle{.at = at + way * 0.3 + field::Vec3{0.0, 0.0, 0.1}, .velocity = way * speed, .size = 0.09 + 0.05 * random(), .life = 0.36, .life_max = 0.36, .drag = 3.0, .abgr = abgr(colour, 1.1)});
    }
}

void Effects::beam(field::Vec3 from, field::Vec3 to, Rgb colour, double radius)
{
    const field::Vec3 line = to - from;
    const double len = field::length(line);
    const int count = static_cast<int>(std::clamp(len * 4.5, 6.0, 170.0));
    for (int k = 0; k < count; ++k) {
        const double t = (k + random()) / count;
        const double jag = radius * 0.5;
        const field::Vec3 off{(random() - 0.5) * jag, (random() - 0.5) * jag, (random() - 0.5) * jag};
        add(Particle{.at = from + line * t + off, .velocity = off * 2.0, .size = 0.09 + radius * 0.11 * random(), .life = 0.12 + 0.24 * random(), .life_max = 0.36, .abgr = abgr(colour, 0.85 + 0.4 * random())});
    }
    burst(to, colour, radius * 0.9);
    shake(0.18);
}

void Effects::column(field::Vec3 foot, double height, Rgb colour, double radius)
{
    const int count = static_cast<int>(std::clamp(height * radius * 18.0, 16.0, 150.0));
    for (int k = 0; k < count; ++k) {
        const double a = random() * kTau;
        const double r = radius * std::sqrt(random());
        const double up = height * random();
        add(Particle{.at = foot + field::Vec3{r * std::cos(a), r * std::sin(a), up}, .velocity = field::Vec3{0.0, 0.0, 1.0 + 3.0 * random()}, .size = 0.06 + 0.1 * random(), .life = 0.25 + 0.4 * random(), .life_max = 0.65, .abgr = abgr(colour, 0.8 + 0.4 * random())});
    }
    ring(foot, colour, radius + 0.5);
    shake(0.22);
}

void Effects::streak(field::Vec3 from, field::Vec3 to, Rgb colour)
{
    const field::Vec3 line = to - from;
    const int count = static_cast<int>(std::clamp(field::length(line) * 3.0, 8.0, 80.0));
    for (int k = 0; k < count; ++k) {
        const double t = (k + random()) / count;
        const field::Vec3 off{(random() - 0.5) * 0.6, (random() - 0.5) * 0.6, (random() - 0.5) * 1.0};
        add(Particle{.at = from + line * t + off, .velocity = off * 0.8, .size = 0.05 + 0.08 * random(), .life = 0.2 + 0.3 * random(), .life_max = 0.5, .abgr = abgr(colour, 0.8 + 0.4 * random())});
    }
}

void Effects::puff(field::Vec3 at, Rgb colour, int count, double speed)
{
    for (int k = 0; k < count; ++k) {
        const double a = random() * kTau;
        const field::Vec3 way{std::cos(a), std::sin(a), 0.15 + 0.4 * random()};
        add(Particle{.at = at + way * 0.15, .velocity = way * (speed * (0.4 + random())), .size = 0.05 + 0.07 * random(), .life = 0.2 + 0.25 * random(), .life_max = 0.45, .gravity = 2.0, .drag = 3.0, .abgr = abgr(colour, 0.8 + 0.3 * random())});
    }
}

void Effects::spark(field::Vec3 at, Rgb colour, double size, double life)
{
    const field::Vec3 off{(random() - 0.5) * size, (random() - 0.5) * size, (random() - 0.5) * size};
    add(Particle{.at = at + off, .velocity = off, .size = size * (0.5 + 0.5 * random()), .life = life * (0.6 + 0.4 * random()), .life_max = life, .abgr = abgr(colour, 0.8 + 0.4 * random())});
}

void Effects::bit(field::Vec3 at, field::Vec3 velocity, std::uint32_t colour, double size, double life)
{
    add(Particle{.at = at, .velocity = velocity, .size = size, .life = life, .life_max = life, .drag = 2.0, .abgr = colour});
}

void Effects::gather(field::Vec3 at, Rgb colour, double radius)
{
    for (int k = 0; k < 36; ++k) {
        const field::Vec3 way = field::normalize(field::Vec3{random() - 0.5, random() - 0.5, random() - 0.3});
        const double life = 0.22 + 0.1 * random();
        // Starts out there and reaches the middle as it ends.
        add(Particle{.at = at + way * radius, .velocity = way * (-radius / life), .size = 0.07 + 0.06 * random(), .life = life, .life_max = life, .abgr = abgr(colour, 0.9 + 0.3 * random())});
    }
}

void Effects::impact(field::Vec3 at, field::Vec3 way, Rgb colour, double power)
{
    const int count = static_cast<int>(std::clamp(4.0 + power * 0.3, 4.0, 16.0));
    const field::Vec3 ahead = field::normalize(way);
    for (int k = 0; k < count; ++k) {
        const field::Vec3 off{random() - 0.5, random() - 0.5, random() * 0.8 - 0.2};
        const field::Vec3 dir = field::normalize(ahead + off * 1.3);
        const double speed = (0.4 + random()) * (5.0 + power * 0.28);
        const double shade = 0.75 + 0.5 * random();
        add(Particle{.at = at, .velocity = dir * speed, .size = (0.05 + 0.09 * random()) * (1.0 + power * 0.012), .life = 0.2 + 0.4 * random(), .life_max = 0.6, .gravity = 14.0, .drag = 2.2,
            .abgr = abgr(Rgb{colour.r * static_cast<float>(shade), colour.g * static_cast<float>(shade), colour.b * static_cast<float>(shade)})});
    }
    // A bright cross, there and gone: you see the moment it landed.
    const field::Vec3 side = field::normalize(field::Vec3{-ahead.y, ahead.x, 0.0});
    for (int k = -7; k <= 7; ++k) {
        const double reach = 0.11 * k * (1.0 + power * 0.015);
        for (const field::Vec3 along : {side * reach, field::Vec3{0.0, 0.0, reach}}) {
            add(Particle{.at = at + along, .velocity = along * 2.0, .size = 0.11 * (1.0 - std::abs(k) / 9.0), .life = 0.1, .life_max = 0.1, .gravity = 0.0, .drag = 0.0, .abgr = abgr(Rgb{1.0f, 1.0f, 0.92f})});
        }
    }
}

void Effects::shockwave(field::Vec3 at, Rgb colour, double radius)
{
    // Clods thrown up, around.
    const int clods = static_cast<int>(std::clamp(radius * 4.0, 5.0, 24.0));
    for (int k = 0; k < clods; ++k) {
        const double a = random() * kTau;
        const double from = random() * radius;
        const field::Vec3 way{std::cos(a), std::sin(a), 0.0};
        add(Particle{.at = at + way * from, .velocity = way * (2.0 + 3.0 * random()) + field::Vec3{0.0, 0.0, 3.0 + 6.0 * random()}, .size = 0.07 + 0.14 * random(), .life = 0.5 + 0.5 * random(), .life_max = 1.0, .gravity = 20.0, .drag = 0.4,
            .abgr = abgr(Rgb{0.42f + 0.1f * static_cast<float>(random()), 0.34f, 0.24f})});
    }
    // Two rings: one quick and thin, one slow and low.
    ring(at, colour, radius * 1.3);
    ring(at + field::Vec3{0.0, 0.0, 0.1}, Rgb{0.78f, 0.72f, 0.6f}, radius);
    shake(std::min(0.8, 0.25 + radius * 0.12));
}

void Effects::floater(field::Vec3 at, double value)
{
    if (floaters_.size() < 64) {
        floaters_.push_back(Floater{.at = at, .value = value, .life = 0.9});
    }
}

} // namespace fieldengine::app

#include "field/LawBounds.hpp"

#include <algorithm>
#include <cmath>

namespace fieldengine::field {

double times(double a, double b)
{
    return a == 0.0 || b == 0.0 ? 0.0 : a * b; // 0 * infinity: nothing of it
}

Bounds bounds_of(const Node& n, const std::vector<Bounds>& fields, const std::vector<double>& params, double h)
{
    const auto arg = [&](std::size_t i) { return bounds_of(*n.args[i], fields, params, h); };
    const auto clean = [](double lo, double hi) {
        return Bounds{std::isnan(lo) ? -std::numeric_limits<double>::infinity() : lo, std::isnan(hi) ? std::numeric_limits<double>::infinity() : hi};
    };
    const auto spread_of = [](const Bounds& b) { return b.lo == b.hi ? 0.0 : b.hi - b.lo; };
    switch (n.op) {
    case Op::Const:
        return Bounds{n.number, n.number};
    case Op::Param: {
        const double v = static_cast<float>(params[static_cast<std::size_t>(n.index)]);
        return Bounds{v, v};
    }
    case Op::Cell:
        return Bounds{h, h};
    case Op::Rim:
        return Bounds{0.0, 1.0};
    case Op::Field:
        return fields[static_cast<std::size_t>(n.index)];
    case Op::Neg: {
        const Bounds a = arg(0);
        return Bounds{-a.hi, -a.lo};
    }
    case Op::Add: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        return clean(a.lo + b.lo, a.hi + b.hi);
    }
    case Op::Sub: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        return clean(a.lo - b.hi, a.hi - b.lo);
    }
    case Op::Mul: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        const double c[4] = {times(a.lo, b.lo), times(a.lo, b.hi), times(a.hi, b.lo), times(a.hi, b.hi)};
        return Bounds{*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
    }
    case Op::Div: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        if (a.none() || b.none()) {
            return Bounds{0.0, 0.0}; // dv(a, 0) is 0
        }
        if (b.lo <= 0.0 && b.hi >= 0.0) {
            return Bounds{};
        }
        const double c[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
        return clean(*std::min_element(c, c + 4), *std::max_element(c, c + 4));
    }
    case Op::Pow: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        if (a.none() && b.lo > 0.0) {
            return Bounds{0.0, 0.0};
        }
        if (a.lo >= 0.0 && b.lo == b.hi && b.lo > 0.0) {
            return clean(std::pow(a.lo, b.lo), std::pow(a.hi, b.lo));
        }
        return Bounds{};
    }
    case Op::Min: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        return Bounds{std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
    }
    case Op::Max: {
        const Bounds a = arg(0);
        const Bounds b = arg(1);
        return Bounds{std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
    }
    case Op::Abs: {
        const Bounds a = arg(0);
        const double most = std::max(std::abs(a.lo), std::abs(a.hi));
        return Bounds{a.lo <= 0.0 && a.hi >= 0.0 ? 0.0 : std::min(std::abs(a.lo), std::abs(a.hi)), most};
    }
    case Op::Sqrt: {
        const Bounds a = arg(0);
        return Bounds{std::sqrt(std::max(0.0, a.lo)), std::sqrt(std::max(0.0, a.hi))};
    }
    case Op::Exp: {
        const Bounds a = arg(0);
        return Bounds{std::exp(std::min(a.lo, 80.0)), std::exp(std::min(a.hi, 80.0))};
    }
    case Op::Log: {
        const Bounds a = arg(0);
        return Bounds{std::log(std::max(a.lo, 1e-30)), std::log(std::max(a.hi, 1e-30))};
    }
    case Op::Step: {
        const Bounds a = arg(0);
        return Bounds{a.lo > 0.0 ? 1.0 : 0.0, a.hi > 0.0 ? 1.0 : 0.0};
    }
    case Op::Pos: {
        const Bounds a = arg(0);
        return Bounds{std::max(a.lo, 0.0), std::max(a.hi, 0.0)};
    }
    case Op::Clamp: {
        const Bounds a = arg(0);
        const Bounds lo = arg(1);
        const Bounds hi = arg(2);
        return Bounds{std::min(std::max(a.lo, lo.lo), std::max(lo.hi, hi.hi)), std::min(std::max(a.hi, lo.hi), std::max(lo.hi, hi.hi))};
    }
    case Op::Shift:
    case Op::MaxOf:
    case Op::MinOf:
        return arg(0);
    case Op::Lap: {
        const double most = 6.0 * spread_of(arg(0)) / (h * h);
        return Bounds{-most, most};
    }
    case Op::Gx:
    case Op::Gy:
    case Op::Gz: {
        const double most = spread_of(arg(0)) / h;
        return Bounds{-most, most};
    }
    case Op::Diffuse: {
        const Bounds k = arg(0);
        const double most = times(12.0 * std::max(std::abs(k.lo), std::abs(k.hi)), spread_of(arg(1))) / (h * h);
        return Bounds{-most, most};
    }
    case Op::Flux: {
        // Nothing differs across any face: the two ways round cancel.
        for (std::size_t i = 0; i + 2 < n.args.size(); ++i) {
            const Bounds b = arg(i);
            if (!(b.lo == b.hi)) {
                return Bounds{};
            }
        }
        return Bounds{0.0, 0.0};
    }
    case Op::LDiffuse: {
        const Bounds f = arg(1);
        return f.lo == f.hi ? Bounds{0.0, 0.0} : Bounds{};
    }
    case Op::Adv:
    case Op::Fall:
    case Op::Spread:
    case Op::Topple:
    case Op::SumOf: {
        const Bounds a = arg(0);
        return a.none() ? a : Bounds{};
    }
    default:
        return Bounds{};
    }
}

bool always_zero(const Node& n, const std::vector<char>& zero, const std::vector<double>& params)
{
    auto arg = [&](std::size_t i) { return always_zero(*n.args[i], zero, params); };
    switch (n.op) {
    case Op::Const:
        return n.number == 0.0;
    case Op::Param:
        return params[static_cast<std::size_t>(n.index)] == 0.0;
    case Op::Field:
        return zero[static_cast<std::size_t>(n.index)] != 0;
    case Op::Neg:
    case Op::Abs:
    case Op::Sqrt:
    case Op::Step:
    case Op::Pos:
    case Op::Shift:
    case Op::Lap:
    case Op::Gx:
    case Op::Gy:
    case Op::Gz:
    case Op::MaxOf:
    case Op::MinOf:
    case Op::SumOf:
        return arg(0);
    case Op::Mul:
    case Op::Div:
        return arg(0) || arg(1);
    case Op::Add:
    case Op::Sub:
    case Op::Min:
    case Op::Max:
        return arg(0) && arg(1);
    case Op::Diffuse:
        return arg(0) || arg(1);
    case Op::Flux:
        return std::all_of(n.args.begin(), n.args.end() - 2, [&](const std::shared_ptr<Node>& a) { return always_zero(*a, zero, params); });
    case Op::LDiffuse:
        return arg(1);
    case Op::Adv:
    case Op::Fall:
    case Op::Spread:
    case Op::Topple:
        return arg(0);
    default:
        return false;
    }
}

} // namespace fieldengine::field

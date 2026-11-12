#pragma once

#include "field/Law.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// The compiled form of a law expression: what the CPU evaluator (Law.cpp) walks and the GPU compiler
// (GpuSolver.cpp) translates. Nothing outside those two looks inside a node.
namespace fieldengine::field {

// ---- expression tree -------------------------------------------------------------------------------

enum class Op {
    Const,
    Param,
    Cell, // h, the voxel edge in metres
    Rim,  // rim, 1 on the outermost cells of the lattice: what the law cannot see past
    Field,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Abs,
    Sqrt,
    Exp,
    Log,
    Step,  // 1 where x > 0
    Pos,   // max(x, 0)
    Clamp,
    Lap,   // Laplacian, zero-flux boundary
    Gx,    // central gradient components
    Gy,
    Gz,
    Diffuse, // div(k grad F), flux form: conservative to rounding
    Adv,     // -(u . grad F), upwind
    Distance, // signed distance (metres) to the 0.5 iso-surface of the argument
    Poisson,  // phi with lap(phi) = arg, phi = 0 outside the lattice; warm-started
    MaxOf,
    MinOf,
    SumOf,
    Fall,   // fall(F, B, speed): F drops into the free space below (1 - B - F), conservatively
    Spread, // spread(F, B, k): F levels out sideways into free neighbours, conservatively
    Topple, // topple(F, B, rate): F that cannot fall straight slides off a ledge, diagonally down
    Shift,  // shift(F, dx, dy, dz): F in the cell that far away (clamped at the edge)
    Local,  // a creature's own state variable (index into its locals); only read at a point
    // Networks (index = the net): what the law learned, in the same expression as what it was told.
    Mlp,      // mlp(NET, a, b, ...): the net's first output, cell by cell
    Flux,     // flux(NET, F, a, ...): F moved between neighbours by a learned, antisymmetric, limited face flux
    LDiffuse, // ldiffuse(NET, k, F): diffusion whose face flux is interpolated by learned weights over four cells
    Dt,       // dt, the step a term is taken for (0 in a derive): what an implicit scheme needs to build its matrix
    // tridiag_x|y|z(lo, di, up, rhs) (index = the axis): u with lo u[i-1] + di u[i] + up u[i+1] = rhs along every line of
    // the axis, the edge cell repeating past the ends (as shift reads it). Thomas' algorithm; the implicit half of ADI.
    Tridiag,
    // The transposes of the stencils, for adjoints (ir/Adjoint): shift_t(F, dx, dy, dz) adds F into the cell that shift
    // would have read it from (the edge cell gathers everything clamped onto it); gx_t, gy_t, gz_t are gx's, gy's, gz's
    // transposes, one-sided edges included. <shift(a, d), b> = <a, shift_t(b, d)> to rounding.
    ShiftT,
    GxT,
    GyT,
    GzT,
};

struct Node {
    Op op = Op::Const;
    double number = 0.0;
    int index = -1;
    int iterations = 0;
    std::vector<std::shared_ptr<Node>> args;
    std::vector<float> buffer;  // this node's result; persists (Poisson warm-starts from it)
    std::vector<float> scratch; // second buffer for iterative ops
    std::vector<double> wide;   // per-cell fluxes of the conservative ops (fall, spread, topple)
    // Global ops (distance, poisson) are expensive and their input rarely changes between steps:
    // the result is reused while the input's bits are the same.
    std::uint64_t cache_key = 0;
    bool cache_valid = false;
    // Subtree size: a product evaluates its cheaper factor first, and when that one is zero the
    // other is never computed.
    int cost = 1;
    // Used more than once in its term (common subexpression): its value is kept for one epoch.
    bool shared = false;
    std::uint64_t epoch = 0;
    Value cached{};
};

// Whether a node is a scalar no matter what the fields hold (reductions count as scalar).
inline bool always_scalar(const Node& node)
{
    switch (node.op) {
    case Op::Const:
    case Op::Param:
    case Op::Cell:
    case Op::MaxOf:
    case Op::MinOf:
    case Op::SumOf:
        return true;
    case Op::Flux:
    case Op::LDiffuse:
    case Op::Field:
    case Op::Distance:
    case Op::Poisson:
    case Op::Fall:
    case Op::Spread:
    case Op::Topple:
    case Op::Tridiag:
    case Op::ShiftT:
    case Op::GxT:
    case Op::GyT:
    case Op::GzT:
        return false;
    default:
        for (const std::shared_ptr<Node>& arg : node.args) {
            if (!always_scalar(*arg)) {
                return false;
            }
        }
        return true;
    }
}

} // namespace fieldengine::field

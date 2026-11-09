#pragma once

#include "field/Backend.hpp"

#include <memory>
#include <string>

// THE LAW IN CUDA, the main backend. When a law is compiled (or changed while the world runs) its IR (ir/LawIR.hpp) is
// turned into one CUDA translation unit, compiled by NVRTC for the card in the machine and cached on disk by its text:
//
//   - every term is one fused kernel (two when a field it writes is read off its own cell), with element-wise chains
//     and stencils inlined, and the stages the IR plans run before it: distance (exact EDT, three axis passes),
//     poisson (Jacobi, warm-started), reductions (two levels, fixed order), networks (fp32, or tensor cores);
//   - the lattice lives on the card as a ring: when the window moves, the origin moves and only the chunks that arrive
//     go up; Poisson's warm start and distance's seeds move with it for free;
//   - contracts (range, finite, conserves) are checked in kernels and read back once a step; a learned scheme
//     (`learned d ...`) runs under guard: snapshot, step, check, and undo plus classical substeps if it broke one;
//   - every term is timed with events (the profiler), and what goes up and comes down is counted.
//
// No result depends on thread order (no float atomics; counters only), so two runs on one card agree to the bit. The
// CPU Solver is the reference the gates hold this to.
namespace fieldengine::field {

class CudaSolver {
public:
    // Null with the reason when there is no CUDA device or NVRTC, or the law does not compile for it.
    static std::shared_ptr<LawBackend> create(Law& law, const GridShape& shape, std::string& why);
    // Why no law can run in CUDA on this machine; empty when one can.
    static std::string unavailable();
    // The generated source of a law (lawc --cuda prints it).
    static std::string source(const Law& law);
};

} // namespace fieldengine::field

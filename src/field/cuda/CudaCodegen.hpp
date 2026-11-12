#pragma once

#include "field/ir/LawIR.hpp"

#include <map>
#include <string>
#include <vector>

// From the law's IR to one CUDA translation unit (see CudaSolver.hpp for what runs it).
namespace fieldengine::field::cuda {

struct StageCode {
    enum class Kind { Stored, Distance, Poisson, Reduce, Net, Tridiag };
    Kind kind = Kind::Stored;
    int inst = -1;
    std::string kernel;      // computes it (Distance: the seed pass; Poisson: the source pass; Reduce: the plane pass)
    unsigned shared = 0;     // dynamic shared memory of `kernel` (nets: weights, fragments, rows)
    bool tensor = false;     // a net on tensor cores: one-dimensional launch of 128 threads
    int iterations = 0;      // Poisson
    int axis = 0;            // Tridiag: the axis its lines run along (a thread a line)
    Op reduce = Op::SumOf;
};

struct ProgramCode {
    std::string gather;      // empty: no equations
    bool in_place = false;
    std::vector<int> swaps;  // not in place: the fields whose two buffers swap after the kernel (ping-pong)
    std::vector<int> stages;
    std::vector<int> limit_stages;
    int max_dt = -1;         // an instruction; evaluated on the host
};

struct LawLayout {
    int aux_slots = 0;
    int uniforms = 0;
    int flags = 0;
    int net_floats = 1;
    int net_words = 2;
    std::vector<int> slot;    // per instruction: its first slot in A (stage values), or -1
    std::vector<int> uslot;   // per instruction: its value in U (reductions), or -1
    std::vector<int> flag;    // per instruction: its re-run flag (distance, poisson), or -1
    std::vector<int> net_image_at; // per net of the law: where its fp32 image starts in W
    std::vector<int> net_frag_at;  // and its fragments in NF (uint32 words)
    std::map<int, StageCode> stages;
    std::vector<ProgramCode> terms;
    std::vector<ProgramCode> learned;
    std::vector<ProgramCode> derives;
};

std::string generate(const ir::Module& module, const Law& law, LawLayout& layout);

// The generic kernels every law module has (ring addressing, distance passes, Jacobi, contracts, uploads).
const char* law_prelude();

} // namespace fieldengine::field::cuda

#pragma once

#include "field/Law.hpp"
#include "field/LawNode.hpp"

#include <cstdint>
#include <string>
#include <vector>

// THE LAW AS A PROGRAM. The parser's trees (LawNode.hpp) carry buffers and caches for the CPU interpreter; this is the
// same law with none of that: one immutable SSA graph for the whole law, every value computed once (structural
// hashing across all terms and derives), constants folded, each instruction knowing whether it is one number or one per
// cell and how far from its cell it reads. On top of the graph, one Program per term (and per learned scheme, and per
// derive) says what the backends must do and in what order:
//
//   stages   - values computed over the whole lattice before the term's kernel reads them: the global ops (distance,
//              poisson, reductions), the networks (they run warp-wide on tensor cores, so not inside another stencil),
//              and whatever the cost model says is cheaper to store once than to recompute at every point of a stencil;
//   gather   - every equation's rate at every cell, fused into one kernel: element-wise chains and stencils inline;
//   in_place - when no equation reads a field it writes anywhere but at its own cell, the store is fused into the same
//              kernel (one pass, no delta buffer); else a second kernel applies what the first gathered.
//
// The CUDA backend generates kernels from this, lawc prints it, and tools/lawjax builds the same steps in JAX from its
// JSON. The CPU interpreter is the reference all three are held to.
namespace fieldengine::field::ir {

enum class Shape : std::uint8_t { Scalar, Cell };

struct Inst {
    Op op = Op::Const;
    Shape shape = Shape::Scalar;
    double number = 0.0;      // Const
    int index = -1;           // Field, Param, Local, a net
    int iterations = 0;       // Poisson
    int offset[3] = {0, 0, 0}; // Shift
    std::vector<int> args;    // ids of earlier instructions
    int radius = 0;           // how far (cells, any axis) this op reads its arguments from its own cell
    bool global = false;      // needs the whole lattice of its argument first: distance, poisson, reductions
    bool net = false;         // runs a network
    int uses = 0;
    int cost = 1;             // to evaluate at one point, everything inlined
};

struct Equation {
    int target = -1;
    int value = -1;
};

struct Program {
    std::string name;
    std::vector<Equation> equations;
    std::vector<int> stages;  // instructions computed over the lattice first, nested first
    std::vector<int> limit_stages; // the reductions max_dt reads, computed once before the substeps
    bool in_place = false;
    int max_dt = -1;          // a scalar instruction, or none
    std::vector<int> reads;
    std::vector<int> writes;
    std::vector<std::vector<int>> conserves;
    bool accumulate = false;  // a derive that adds to what the ones before it set
};

struct Module {
    std::vector<Inst> insts;
    std::vector<Program> terms;
    std::vector<Program> learned; // per term; no equations = the term has none
    std::vector<Program> derives;
    // Why an instruction is a stage of some program: kept for printing.
    std::vector<char> materialized;
    int folded = 0; // constants folded and identities dropped while lowering
    int shared = 0; // values found twice and kept once

    [[nodiscard]] bool is_stage(int id) const;
};

// Stage instructions are stored, not inlined, when one point of their reader costs more than this.
inline constexpr int kInlineBudget = 48;

Module lower(const Law& law);

const char* op_name(Op op);

// For tools/lawjax and for reading: the graph, the programs, the fields, params and nets (with their weight files).
std::string to_json(const Module& module, const Law& law);
// A listing a person can read: one instruction a line, then each program's stages and equations.
std::string describe(const Module& module, const Law& law);

} // namespace fieldengine::field::ir

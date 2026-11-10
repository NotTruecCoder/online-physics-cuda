#pragma once

#include "field/Grid.hpp"
#include "field/Law.hpp"

#include <cstddef>
#include <string>
#include <vector>

// WHERE A LAW RUNS. The CPU Solver is the reference; a backend runs the same law somewhere faster and is held to it
// by the gates. Two exist: CUDA (kernels generated from the law's IR, compiled by NVRTC at run time; the main one) and
// Direct3D 11 (HLSL compute shaders; kept for cards without CUDA, and runs no nets).
namespace fieldengine::field {

// A contract the law declared and the world broke, named in words a designer can act on.
struct Violation {
    enum class Kind { Range, NotFinite, Conservation, SubstepCap };
    Kind kind = Kind::Range;
    std::string term; // empty for a derive
    std::string field;
    std::string detail;
};

const char* to_string(Violation::Kind kind);

// A box of cells, [x0, x1) x [y0, y1) x [z0, z1), in lattice coordinates.
struct CellBox {
    int x0 = 0, y0 = 0, z0 = 0, x1 = 0, y1 = 0, z1 = 0;
};

// What was written to the lattice from outside since the backend last saw it.
struct FieldEdits {
    bool all = true;
    std::vector<std::vector<char>> planes; // per field, per z-plane
    // The window moved: the cell now at c held what was at c + shift. Only `boxes` hold cells that were not in the
    // lattice before. A backend that keeps its lattice as a ring (CUDA) moves its origin and uploads only those.
    int shift[3] = {0, 0, 0};
    std::vector<CellBox> boxes;
    [[nodiscard]] bool shifted() const { return shift[0] != 0 || shift[1] != 0 || shift[2] != 0 || !boxes.empty(); }
};

// WHAT A STEP COST, term by term: the profiler panel draws this, the bench prints it.
struct TermProfile {
    std::string name;
    double ms = 0.0;       // GPU time (events) on a GPU backend, wall time on the CPU
    int launches = 0;      // kernels it sent
    int substeps = 0;
    bool idle = false;     // nothing to do: not dispatched
    bool learned = false;  // its learned scheme ran, and held
    bool fell_back = false; // its learned scheme broke a contract: undone, and the classical one ran
};

struct StepProfile {
    std::string backend;
    double total_ms = 0.0;    // the whole step, as the host waited for it
    double gpu_ms = 0.0;      // the sum of what the card spent on it
    double derive_ms = 0.0;
    double upload_ms = 0.0;
    double download_ms = 0.0;
    std::size_t upload_bytes = 0;
    std::size_t download_bytes = 0;
    int launches = 0;
    int fallbacks = 0;
    std::vector<TermProfile> terms;
};

class LawBackend {
public:
    virtual ~LawBackend() = default;
    // `changed` says, per field, whether any cell of it moved: only those came back into `fields`.
    virtual void step(double dt, bool respect_max_dt, bool learned, Law& law, FieldSet& fields, FieldEdits& edits,
        std::vector<char>& changed, std::vector<Violation>& violations, std::vector<int>& substeps) = 0;
    virtual void derive(Law& law, FieldSet& fields, FieldEdits& edits, std::vector<char>& changed) = 0;
    // The contracts cost as much as a good part of the law: look at them in one step of so many. (Gates look at every
    // one; a guarded learned term is always looked at.)
    virtual void check_contracts_every(int steps) = 0;
    // LAZY READBACK. On, a step leaves what changed on the card: the bounds the idle-term test needs are worked out
    // there, and `fields` is filled only by fetch(), for the fields asked for. A backend that cannot keeps copying.
    virtual void set_lazy(bool on) { (void)on; }
    virtual void fetch(FieldSet& fields, const std::vector<char>& which) { (void)fields; (void)which; }
    [[nodiscard]] virtual const std::string& adapter() const = 0;
    [[nodiscard]] virtual const char* api() const = 0;
    [[nodiscard]] virtual const StepProfile& profile() const = 0;
};

} // namespace fieldengine::field

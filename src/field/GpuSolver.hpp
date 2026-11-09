#pragma once

#include "field/Backend.hpp"
#include "field/Solver.hpp"

#include <memory>
#include <string>
#include <vector>

// THE LAW ON DIRECT3D 11 (the fallback backend; CUDA is the main one, field/cuda). Every equation of the law text is
// translated to a compute shader when the law is compiled, so a law on the GPU is the same text with the same
// contracts: `max_dt` sets the substeps, `conserves` and `range` are checked term by term and broken ones reported by
// name. It runs no nets (mlp, flux, ldiffuse) and no learned schemes: a law with nets needs CUDA, or the CPU.
//
// The CPU Solver stays the reference: a gate runs both from one start and compares them cell by cell. The GPU
// computes in single precision and the two do not agree to the bit, only to the rounding; `distance` is exact on both.
// Two runs on one card agree to the bit, since no shader's result depends on thread order.
namespace fieldengine::field {

class GpuSolver final : public LawBackend {
public:
    // Null (with the reason) when there is no hardware GPU or the law does not compile for one.
    static std::shared_ptr<GpuSolver> create(Law& law, const GridShape& shape, std::string& why);
    ~GpuSolver() override;
    // Why no law can run on this machine; empty when one can.
    static std::string unavailable();

    void step(double dt, bool respect_max_dt, bool learned, Law& law, FieldSet& fields, FieldEdits& edits,
        std::vector<char>& changed, std::vector<Violation>& violations, std::vector<int>& substeps) override;
    void derive(Law& law, FieldSet& fields, FieldEdits& edits, std::vector<char>& changed) override;
    void check_contracts_every(int steps) override;
    [[nodiscard]] const std::string& adapter() const override;
    [[nodiscard]] const char* api() const override { return "Direct3D 11"; }
    [[nodiscard]] const StepProfile& profile() const override;

private:
    struct Impl;
    explicit GpuSolver(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace fieldengine::field

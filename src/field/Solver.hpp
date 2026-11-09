#pragma once

#include "field/Backend.hpp"
#include "field/Grid.hpp"
#include "field/Law.hpp"
#include "field/LawBounds.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fieldengine::field {

// Integrates a compiled law on one lattice.
//
// One step = every term in declaration order (Lie splitting), each with as many substeps as its own
// `max_dt` asks for, each substep gather-then-apply (every equation of a term reads the same
// pre-state), then every `derive` in order. No clocks, no RNG: the same law from the same state gives the
// same bits. This is the reference; use_gpu() hands the same steps to the graphics card (GpuSolver.hpp).
class Solver {
public:
    Solver(Law law, GridShape shape);
    // A copy would share expression buffers with the original; build a second solver instead.
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = default;
    Solver& operator=(Solver&&) = default;

    // Mutable access assumes every field is about to change; edit_cell says exactly which plane will.
    [[nodiscard]] FieldSet& fields()
    {
        edits_.all = true;
        return fields_;
    }
    float& edit_cell(int field, std::size_t cell);
    // The window moved by (dx, dy, dz) cells: the caller slides the cells (what is now at c was at c + d) and fills
    // `arriving`. A backend that keeps a ring only uploads those boxes.
    [[nodiscard]] FieldSet& shift_window(int dx, int dy, int dz, const std::vector<CellBox>& arriving);
    [[nodiscard]] const FieldSet& fields() const { return fields_; }
    [[nodiscard]] const Law& law() const { return law_; }
    [[nodiscard]] int field(std::string_view name) const { return law_.field_id(name); }

    bool set_param(std::string_view name, double value);
    [[nodiscard]] std::optional<double> param(std::string_view name) const;

    // Recomputes every derived field. Call after editing a field from outside the law (digging,
    // placing a wall) so rays and bodies see the edit before the next step.
    void refresh_derived();
    void step(double dt);
    // Per field: did the last step (or refresh) change any cell of it? (Always yes on the CPU, which does not look.)
    [[nodiscard]] const std::vector<char>& last_changed() const { return changed_; }

    // From here on the law runs on the GPU: CUDA where there is one, else Direct3D 11 (FIELD_BACKEND=cuda|hlsl|cpu
    // chooses); false (with the reason) leaves it on the CPU. fields() stays the truth between steps: edits go up and
    // results come back on their own.
    bool use_gpu(std::string& why);
    [[nodiscard]] bool on_gpu() const { return gpu_ != nullptr; }
    [[nodiscard]] std::string gpu_name() const;
    [[nodiscard]] const char* backend_api() const { return gpu_ ? gpu_->api() : "CPU"; }
    // What the last step cost, term by term.
    [[nodiscard]] const StepProfile& profile() const { return gpu_ ? gpu_->profile() : profile_; }

    // Learned schemes (`learned d ...` in a term) run instead of the classical ones, under guard: a step that breaks a
    // contract of its term is undone and taken classically. Off: the law is exactly what it was told.
    void set_learned(bool on) { learned_ = on; }
    [[nodiscard]] bool learned() const { return learned_; }
    // Networks whose weights changed (a trainer, online learning): the backend takes them on the next step.
    void nets_changed() { ++nets_version_; }
    [[nodiscard]] std::uint64_t nets_version() const { return nets_version_; }
    [[nodiscard]] Law& law_for_training() { return law_; }
    void check_contracts_every(int steps);

    [[nodiscard]] const std::vector<Violation>& violations() const { return violations_; }
    // Substeps each term took in the last step (declaration order).
    [[nodiscard]] const std::vector<int>& last_substeps() const { return substeps_; }

    // FALSIFICATION HOOK: ignore every declared max_dt. Exists so a gate can show the stability
    // contract is what keeps the world bounded — never set in the game.
    void set_respect_max_dt(bool respect) { respect_max_dt_ = respect; }

    static constexpr int kMaxSubsteps = 4096;

private:
    void apply_term(Term& term, double dt);
    void integrate(Term& term, std::vector<Equation>& equations, Node* max_dt, double dt, std::size_t index);
    void run_derives();
    void check_field(int id, const std::string& term);
    void update_zero(int id);
    void update_all_zero();
    void update_bounds(int id);

    Law law_;
    FieldSet fields_;
    std::vector<Violation> violations_;
    std::vector<int> substeps_;
    std::vector<std::vector<float>> deltas_;
    std::vector<char> zero_; // per field: every cell exactly 0 (see EvalContext::zero)
    std::vector<Bounds> bounds_; // per field: what its cells go between: a term with nothing to do is not worked out
    std::uint64_t epoch_ = 0; // one per gather (EvalContext::epoch)
    bool respect_max_dt_ = true;
    std::shared_ptr<LawBackend> gpu_;
    StepProfile profile_;
    bool learned_ = false;
    std::uint64_t nets_version_ = 0;
    FieldEdits edits_;
    std::vector<char> changed_;
};

} // namespace fieldengine::field

#include "field/Solver.hpp"

#include "field/GpuSolver.hpp"
#include "field/cuda/CudaSolver.hpp"
#include "field/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <sstream>

namespace fieldengine::field {

const char* to_string(Violation::Kind kind)
{
    switch (kind) {
    case Violation::Kind::Range:
        return "range";
    case Violation::Kind::NotFinite:
        return "not-finite";
    case Violation::Kind::Conservation:
        return "conservation";
    case Violation::Kind::SubstepCap:
        return "substep-cap";
    }
    return "?";
}

namespace {

double total(const std::vector<float>& values)
{
    return parallel_sum(values.size(), [&](std::size_t i) { return static_cast<double>(values[i]); });
}

double magnitude(const std::vector<float>& values)
{
    return parallel_sum(values.size(), [&](std::size_t i) { return std::abs(static_cast<double>(values[i])); });
}

std::string cell_name(const GridShape& shape, std::size_t i)
{
    const std::size_t nx = static_cast<std::size_t>(shape.nx);
    const std::size_t ny = static_cast<std::size_t>(shape.ny);
    std::ostringstream out;
    out << "(" << i % nx << "," << (i / nx) % ny << "," << i / (nx * ny) << ")";
    return out.str();
}

} // namespace

Solver::Solver(Law law, GridShape shape) : law_(clone_law(law)), fields_(shape)
{
    limit_threads();
    for (const FieldDecl& decl : law_.fields) {
        fields_.add(decl.name, static_cast<float>(decl.init));
    }
    substeps_.assign(law_.terms.size(), 0);
    refresh_derived();
}

bool Solver::set_param(std::string_view name, double value)
{
    const int id = law_.param_id(name);
    if (id < 0) {
        return false;
    }
    law_.param_values[static_cast<std::size_t>(id)] = value;
    return true;
}

std::optional<double> Solver::param(std::string_view name) const
{
    const int id = law_.param_id(name);
    if (id < 0) {
        return std::nullopt;
    }
    return law_.param_values[static_cast<std::size_t>(id)];
}

void Solver::update_zero(int id)
{
    const std::vector<float>& values = fields_.values(id);
    zero_[static_cast<std::size_t>(id)] =
        std::all_of(values.begin(), values.end(), [](float v) { return v == 0.0f; }) ? 1 : 0;
}

void Solver::update_bounds(int id)
{
    const std::vector<float>& values = fields_.values(id);
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    bounds_[static_cast<std::size_t>(id)] = Bounds{*lo, *hi};
    zero_[static_cast<std::size_t>(id)] = *lo == 0.0f && *hi == 0.0f ? 1 : 0;
}

void Solver::update_all_zero()
{
    zero_.assign(fields_.count(), 0);
    bounds_.resize(fields_.count());
    for (std::size_t id = 0; id < fields_.count(); ++id) {
        update_bounds(static_cast<int>(id));
    }
}

void Solver::fetch(const std::vector<char>& which) const
{
    if (!gpu_) {
        return;
    }
    const std::lock_guard<std::mutex> lock(*fetch_mutex_);
    std::vector<char> still(which.size(), 0); // another thread may have fetched some while this one waited
    bool any = false;
    for (std::size_t f = 0; f < which.size() && f < stale_.size(); ++f) {
        still[f] = which[f] && stale_[f] ? 1 : 0;
        any = any || still[f];
    }
    if (!any) {
        return;
    }
    gpu_->fetch(fields_, still);
    for (std::size_t f = 0; f < still.size(); ++f) {
        stale_[f] = still[f] ? 0 : stale_[f];
    }
}

void Solver::fetch_all() const
{
    std::vector<char> all;
    {
        const std::lock_guard<std::mutex> lock(*fetch_mutex_);
        if (std::none_of(stale_.begin(), stale_.end(), [](char c) { return c != 0; })) {
            return;
        }
        all.assign(stale_.size(), 1);
    }
    fetch(all);
}

const std::vector<float>& Solver::values(int field) const
{
    const auto f = static_cast<std::size_t>(field);
    std::vector<char> one(stale_.size(), 0);
    if (f < one.size()) {
        one[f] = 1;
        fetch(one);
    }
    return fields_.values(field);
}

void Solver::set_lazy_readback(bool on)
{
    fetch_all();
    lazy_ = on;
    if (gpu_) {
        gpu_->set_lazy(on);
    }
}

float& Solver::edit_cell(int field, std::size_t cell)
{
    (void)values(field);
    if (gpu_ && !edits_.all) {
        edits_.planes.resize(fields_.count());
        std::vector<char>& planes = edits_.planes[static_cast<std::size_t>(field)];
        const GridShape& g = fields_.shape();
        planes.resize(static_cast<std::size_t>(g.nz), 0);
        planes[cell / (static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny))] = 1;
    }
    return fields_.values(field)[cell];
}

bool Solver::use_gpu(std::string& why)
{
    fetch_all();
    edits_.all = true;
    const char* forced = std::getenv("FIELD_BACKEND");
    const std::string want = forced != nullptr ? forced : "";
    gpu_.reset();
    if (want == "cpu") {
        why = "FIELD_BACKEND=cpu";
        return false;
    }
    std::string cuda_why;
    if (want != "hlsl") {
        gpu_ = CudaSolver::create(law_, fields_.shape(), cuda_why);
        if (gpu_) {
            return true;
        }
        if (want == "cuda") {
            why = cuda_why;
            return false;
        }
    }
    gpu_ = GpuSolver::create(law_, fields_.shape(), why);
    if (!gpu_ && !cuda_why.empty()) {
        why = "CUDA: " + cuda_why + "; Direct3D 11: " + why;
    }
    return gpu_ != nullptr;
}

FieldSet& Solver::shift_window(int dx, int dy, int dz, const std::vector<CellBox>& arriving)
{
    fetch_all();
    if (gpu_ && !edits_.all) {
        const bool planes = std::any_of(edits_.planes.begin(), edits_.planes.end(), [](const std::vector<char>& p) {
            return std::any_of(p.begin(), p.end(), [](char c) { return c != 0; });
        });
        if (planes) {
            edits_.all = true; // edited in the old lattice's cells: simpler to send everything once
        } else {
            const GridShape& g = fields_.shape();
            // Boxes from an earlier move not yet sent are now that much further along.
            for (CellBox& b : edits_.boxes) {
                b = CellBox{std::max(0, b.x0 - dx), std::max(0, b.y0 - dy), std::max(0, b.z0 - dz), std::min(g.nx, b.x1 - dx),
                    std::min(g.ny, b.y1 - dy), std::min(g.nz, b.z1 - dz)};
            }
            edits_.shift[0] += dx;
            edits_.shift[1] += dy;
            edits_.shift[2] += dz;
            edits_.boxes.insert(edits_.boxes.end(), arriving.begin(), arriving.end());
        }
    }
    return fields_;
}

std::string Solver::gpu_name() const
{
    return gpu_ ? gpu_->adapter() : std::string{};
}

void Solver::check_contracts_every(int steps)
{
    if (gpu_) {
        gpu_->check_contracts_every(steps);
    }
}

void Solver::refresh_derived()
{
    if (gpu_) {
        fetch_all();
        gpu_->derive(law_, fields_, edits_, changed_);
        if (lazy_) {
            stale_ = changed_;
        }
        return;
    }
    changed_.assign(fields_.count(), 1);
    update_all_zero(); // fields may have been edited from outside since the last look
    run_derives();
}

void Solver::run_derives()
{
    std::vector<char> started(fields_.count(), 0);
    for (Derive& derive : law_.derives) {
        const EvalContext context{.fields = &fields_, .params = &law_.param_values, .zero = &zero_, .epoch = ++epoch_, .nets = &law_.nets};
        const Value value = evaluate(*derive.rhs, context);
        std::vector<float>& out = fields_.values(derive.target);
        char& begun = started[static_cast<std::size_t>(derive.target)];
        const bool add = derive.accumulate && begun;
        parallel_blocks(out.size(), [&](std::size_t first, std::size_t end) {
            for (std::size_t i = first; i < end; ++i) {
                out[i] = static_cast<float>((add ? static_cast<double>(out[i]) : 0.0) + value.at(i));
            }
        });
        begun = 1;
        update_bounds(derive.target);
    }
}

void Solver::step(double dt)
{
    violations_.clear();
    if (gpu_) {
        gpu_->step(dt, respect_max_dt_, learned_, law_, fields_, edits_, changed_, violations_, substeps_);
        if (lazy_) {
            stale_.resize(changed_.size(), 0);
            for (std::size_t f = 0; f < changed_.size(); ++f) {
                stale_[f] = stale_[f] || changed_[f];
            }
        }
        return;
    }
    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    profile_.backend = "CPU";
    profile_.terms.resize(law_.terms.size());
    profile_.fallbacks = 0;
    profile_.launches = 0;
    changed_.assign(fields_.count(), 1);
    update_all_zero(); // a spell may have poured into a field since the last step
    for (Term& term : law_.terms) {
        apply_term(term, dt);
    }
    const auto derives = clock::now();
    run_derives(); // the zero flags are current: every term kept them so
    for (const Derive& derive : law_.derives) {
        check_field(derive.target, "");
    }
    const auto done = clock::now();
    profile_.derive_ms = std::chrono::duration<double, std::milli>(done - derives).count();
    profile_.total_ms = std::chrono::duration<double, std::milli>(done - started).count();
    profile_.gpu_ms = 0.0;
}

void Solver::apply_term(Term& term, double dt)
{
    const std::size_t index = static_cast<std::size_t>(&term - law_.terms.data());
    const auto started = std::chrono::steady_clock::now();
    TermProfile& p = profile_.terms[index];
    p = TermProfile{.name = term.name};
    if (learned_ && !term.learned.empty()) {
        // GUARDED: the learned scheme first, on a copy of what it writes; a broken contract undoes it.
        std::vector<std::vector<float>> saved;
        for (const int field : term.writes) {
            saved.push_back(fields_.values(field));
        }
        const std::size_t mark = violations_.size();
        integrate(term, term.learned, term.learned_max_dt.get(), dt, index);
        if (violations_.size() == mark) {
            p.learned = true;
            p.substeps = substeps_[index];
            p.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            return;
        }
        violations_.resize(mark); // the guard took them: they never reach the world
        for (std::size_t w = 0; w < term.writes.size(); ++w) {
            fields_.values(term.writes[w]) = std::move(saved[w]);
            update_bounds(term.writes[w]);
        }
        p.fell_back = true;
        ++profile_.fallbacks;
    }
    integrate(term, term.equations, term.max_dt.get(), dt, index);
    p.substeps = substeps_[index];
    p.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void Solver::integrate(Term& term, std::vector<Equation>& equations, Node* max_dt, double dt, std::size_t index)
{
    const EvalContext context{.fields = &fields_, .params = &law_.param_values, .zero = &zero_, .nets = &law_.nets};
    substeps_[index] = 1;
    if (std::all_of(equations.begin(), equations.end(), [&](const Equation& eq) { return bounds_of(*eq.rhs, bounds_, law_.param_values, fields_.shape().cell).none(); })) {
        profile_.terms[index].idle = true;
        return; // stone does not melt where nothing is near the melting point
    }
    int substeps = 1;
    if (respect_max_dt_ && max_dt != nullptr) {
        const std::optional<double> limit = evaluate_scalar(*max_dt, context);
        if (limit && *limit > 0.0 && std::isfinite(*limit) && dt > *limit) {
            const double wanted = std::ceil(dt / *limit);
            if (wanted > kMaxSubsteps) {
                violations_.push_back(Violation{.kind = Violation::Kind::SubstepCap, .term = term.name,
                    .detail = "max_dt " + std::to_string(*limit) + " asks for " + std::to_string(wanted) +
                        " substeps; capped at " + std::to_string(kMaxSubsteps)});
                substeps = kMaxSubsteps;
            } else {
                substeps = static_cast<int>(wanted);
            }
        }
    }
    substeps_[index] = substeps;
    const double sub_dt = dt / substeps;

    const std::size_t n = fields_.shape().cells();
    deltas_.resize(std::max(deltas_.size(), equations.size()));
    std::vector<char> idle(equations.size(), 0);
    // The conservation totals are taken just before the first change, never for a term that turns
    // out idle: an idle term changes nothing, and its contract holds without counting.
    std::vector<double> before;
    std::vector<double> scale;
    bool touched = false;
    for (int s = 0; s < substeps; ++s) {
        // Gather: every equation reads the same pre-state. Copying out is what makes that true even
        // when an equation's right-hand side IS another equation's target.
        const EvalContext gather{.fields = &fields_, .params = &law_.param_values, .zero = &zero_, .epoch = ++epoch_,
            .nets = &law_.nets, .dt = sub_dt};
        bool any = false;
        for (std::size_t e = 0; e < equations.size(); ++e) {
            const Value value = evaluate(*equations[e].rhs, gather);
            idle[e] = value.is_scalar() && value.scalar == 0.0 ? 1 : 0;
            if (idle[e]) {
                continue; // nothing to add: a term about mana nobody has cast stops here
            }
            any = true;
            std::vector<float>& delta = deltas_[e];
            delta.resize(n);
            if (value.is_scalar()) {
                std::fill(delta.begin(), delta.end(), static_cast<float>(value.scalar * sub_dt));
            } else {
                parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t i = begin; i < end; ++i) {
                        delta[i] = static_cast<float>(static_cast<double>(value.array[i]) * sub_dt);
                    }
                });
            }
        }
        if (!any) {
            break; // every equation is idle, and stays so: the pre-state did not change
        }
        if (!touched) {
            for (const std::vector<int>& group : term.conserves) {
                double sum = 0.0;
                double mag = 0.0;
                for (const int field : group) {
                    sum += total(fields_.values(field));
                    mag += magnitude(fields_.values(field));
                }
                before.push_back(sum);
                scale.push_back(mag);
            }
        }
        touched = true;
        // Apply.
        for (std::size_t e = 0; e < equations.size(); ++e) {
            if (idle[e]) {
                continue;
            }
            std::vector<float>& target = fields_.values(equations[e].target);
            const std::vector<float>& delta = deltas_[e];
            parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    target[i] += delta[i];
                }
            });
        }
        for (const Equation& equation : equations) {
            update_zero(equation.target);
        }
    }
    if (touched) {
        for (const Equation& equation : equations) {
            update_bounds(equation.target);
        }
    }

    for (std::size_t c = 0; c < before.size(); ++c) {
        double after = 0.0;
        std::string names;
        for (const int field : term.conserves[c]) {
            after += total(fields_.values(field));
            names += (names.empty() ? "" : " + ") + law_.fields[static_cast<std::size_t>(field)].name;
        }
        // Float storage rounds every cell on every substep, so the budget scales with the amount of
        // stuff and the work done; a real leak is orders of magnitude above it.
        const double tolerance = 1e-5 * (scale[c] + 1.0) * std::sqrt(static_cast<double>(substeps));
        if (std::abs(after - before[c]) > tolerance) {
            std::ostringstream detail;
            detail << "total went " << before[c] << " -> " << after << " (tolerance " << tolerance << ")";
            violations_.push_back(Violation{.kind = Violation::Kind::Conservation, .term = term.name,
                .field = names, .detail = detail.str()});
        }
    }
    if (touched) {
        for (const int field : term.writes) {
            check_field(field, term.name);
        }
    }
}

void Solver::check_field(int id, const std::string& term)
{
    const FieldDecl& decl = law_.fields[static_cast<std::size_t>(id)];
    std::vector<float>& values = fields_.values(id);
    // Counted in parallel; only a field that broke its contract is walked again for the report.
    const double broken = parallel_sum(values.size(), [&](std::size_t i) {
        const float v = values[i];
        if (!std::isfinite(v)) {
            return 1.0;
        }
        if (v < decl.lo || v > decl.hi) {
            if (decl.clamp) {
                values[i] = static_cast<float>(std::clamp(static_cast<double>(v), decl.lo, decl.hi));
                return 0.0;
            }
            return 1.0;
        }
        return 0.0;
    });
    if (broken == 0.0) {
        return;
    }
    std::size_t bad = 0;
    std::size_t first = 0;
    bool non_finite = false;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float v = values[i];
        if (!std::isfinite(v)) {
            if (bad++ == 0) {
                first = i;
            }
            non_finite = true;
            continue;
        }
        if (v < decl.lo || v > decl.hi) {
            if (bad++ == 0) {
                first = i;
            }
        }
    }
    if (bad == 0) {
        return;
    }
    std::ostringstream detail;
    detail << bad << " cell(s); first " << decl.name << " = " << values[first] << " at "
           << cell_name(fields_.shape(), first) << ", declared range [" << decl.lo << ", " << decl.hi << "]";
    violations_.push_back(Violation{.kind = non_finite ? Violation::Kind::NotFinite : Violation::Kind::Range,
        .term = term, .field = decl.name, .detail = detail.str()});
}

} // namespace fieldengine::field

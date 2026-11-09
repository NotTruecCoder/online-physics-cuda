// lawc: the law compiler on the command line.
//
//   lawc --ir LAW...                    the IR as a listing (instructions, stages, fusion)
//   lawc --ir-json out.json LAW...      the IR for tools/lawjax
//   lawc --cuda out.cu LAW...           the generated CUDA translation unit
//   lawc --scene S --steps N [--dt X] [--backend cpu|cuda|hlsl] [--learned] [--profile]
//   lawc --scene S --steps N --compare  the CPU reference against a GPU backend, field by field
//   lawc --scene S --steps N --dump DIR [--every K]   states as raw float32 (DIR/step_K/FIELD.bin)
//   --init DIR                          start from a state in that format (after the scene's ops; fields it lacks keep theirs)
//   --param NAME VALUE                  set a param (after the scene's)
//   --json FILE [--warmup N]            timings as JSON: mean step, card time and per-term time over the steps after N
//
// Laws come from the command line, or from the scene's `law` lines.
#include "field/Law.hpp"
#include "field/Scene.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaSolver.hpp"
#include "field/ir/LawIR.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>
#include <string>
#include <vector>

using namespace fieldengine::field;

namespace {

void set_backend(const std::string& backend)
{
#ifdef _WIN32
    _putenv_s("FIELD_BACKEND", backend.c_str());
#else
    setenv("FIELD_BACKEND", backend.c_str(), 1);
#endif
}

void print_profile(const StepProfile& p)
{
    std::cout << "  " << p.backend << ": step " << std::fixed << std::setprecision(3) << p.total_ms << " ms";
    if (p.gpu_ms > 0.0) {
        std::cout << " (card " << p.gpu_ms << " ms, " << p.launches << " launches)";
    }
    std::cout << ", derives " << p.derive_ms << " ms, up " << p.upload_bytes / 1024 << " KiB, down " << p.download_bytes / 1024 << " KiB";
    if (p.fallbacks > 0) {
        std::cout << ", " << p.fallbacks << " learned step(s) undone";
    }
    std::cout << "\n";
    for (const TermProfile& t : p.terms) {
        if (t.idle) {
            continue;
        }
        std::cout << "    " << std::left << std::setw(22) << t.name << std::right << std::setw(9) << t.ms << " ms  x" << t.substeps;
        if (t.launches > 0) {
            std::cout << "  " << t.launches << " launches";
        }
        std::cout << (t.learned ? "  learned" : "") << (t.fell_back ? "  FELL BACK" : "") << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> law_paths;
    std::optional<std::string> scene_path;
    std::optional<std::string> ir_json;
    std::optional<std::string> cuda_out;
    std::optional<std::string> dump;
    std::string backend = "cpu";
    bool listing = false;
    bool compare = false;
    bool learned = false;
    bool profile = false;
    int steps = 0;
    int every = 0;
    double dt = 0.05;
    std::optional<std::string> init_dir;
    std::optional<std::string> json_out;
    int warmup = 0;
    std::vector<std::pair<std::string, double>> param_overrides;
    std::optional<GridShape> grid;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << a << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--ir") listing = true;
        else if (a == "--ir-json") ir_json = next();
        else if (a == "--cuda") cuda_out = next();
        else if (a == "--scene") scene_path = next();
        else if (a == "--steps") steps = std::atoi(next().c_str());
        else if (a == "--dt") dt = std::atof(next().c_str());
        else if (a == "--backend") backend = next();
        else if (a == "--compare") compare = true;
        else if (a == "--learned") learned = true;
        else if (a == "--profile") profile = true;
        else if (a == "--dump") dump = next();
        else if (a == "--every") every = std::atoi(next().c_str());
        else if (a == "--init") init_dir = next();
        else if (a == "--json") json_out = next();
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--param") {
            const std::string name = next();
            param_overrides.emplace_back(name, std::atof(next().c_str()));
        }
        else if (a == "--grid") {
            GridShape g;
            g.nx = std::atoi(next().c_str());
            g.ny = std::atoi(next().c_str());
            g.nz = std::atoi(next().c_str());
            g.cell = std::atof(next().c_str());
            grid = g;
        } else if (a == "--help" || a == "-h") {
            std::cout << "lawc [--ir] [--ir-json F] [--cuda F] [--scene S] [--grid NX NY NZ H] [--steps N] [--dt X]\n"
                         "     [--backend cpu|cuda|hlsl] [--learned] [--profile] [--compare] [--dump DIR [--every K]]\n"
                         "     [--init DIR] [--param NAME VALUE]... [--json FILE [--warmup N]] LAW...\n";
            return 0;
        } else {
            law_paths.push_back(a);
        }
    }

    Scene scene;
    std::string error;
    if (scene_path && !load_scene(*scene_path, scene, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (law_paths.empty()) {
        law_paths = scene.laws;
    }
    if (grid) {
        scene.grid = *grid;
    }
    std::vector<LawSource> sources;
    for (const std::string& path : law_paths) {
        const std::optional<LawSource> source = read_law_file(path);
        if (!source) {
            std::cerr << "no law at " << path << "\n";
            return 1;
        }
        sources.push_back(*source);
    }
    if (sources.empty()) {
        std::cerr << "no law: give LAW files or a scene with 'law' lines (--help)\n";
        return 2;
    }
    CompileResult compiled = compile_law(sources);
    for (const std::string& e : compiled.errors) {
        std::cerr << e << "\n";
    }
    if (!compiled.law) {
        return 1;
    }
    const Law& law = *compiled.law;
    for (const NetDecl& net : law.nets) {
        if (!net.loaded) {
            std::cerr << "note: net '" << net.name << "' has no weights at " << net.file << " (it starts untrained: its output is 0)\n";
        }
    }

    if (listing || ir_json) {
        const ir::Module m = ir::lower(law);
        if (listing) {
            std::cout << ir::describe(m, law);
        }
        if (ir_json) {
            std::ofstream(*ir_json) << ir::to_json(m, law);
            std::cout << "IR written to " << *ir_json << "\n";
        }
    }
    if (cuda_out) {
        std::ofstream(*cuda_out) << CudaSolver::source(law);
        std::cout << "CUDA written to " << *cuda_out << "\n";
    }
    if (steps <= 0) {
        return 0;
    }

    auto make = [&](const std::string& where) -> std::optional<Solver> {
        Solver solver(law, scene.grid);
        if (!apply_scene(scene, solver, error)) {
            std::cerr << error << "\n";
            return std::nullopt;
        }
        for (const auto& [name, value] : param_overrides) {
            if (!solver.set_param(name, value)) {
                std::cerr << "the law has no param '" << name << "'\n";
                return std::nullopt;
            }
        }
        if (init_dir) {
            if (!load_fields(solver.fields(), *init_dir, error)) {
                std::cerr << error << "\n";
                return std::nullopt;
            }
            solver.refresh_derived();
        }
        solver.set_learned(learned);
        if (where != "cpu") {
            set_backend(where);
            std::string why;
            if (!solver.use_gpu(why)) {
                std::cerr << "no " << where << " backend: " << why << "\n";
                return std::nullopt;
            }
            solver.check_contracts_every(1);
        }
        return solver;
    };

    std::optional<Solver> main_solver = make(compare ? "cpu" : backend);
    if (!main_solver) {
        return 1;
    }
    std::optional<Solver> other;
    if (compare) {
        other = make(backend == "cpu" ? "cuda" : backend);
        if (!other) {
            return 1;
        }
    }
    std::cout << law.fields.size() << " fields, " << law.terms.size() << " terms, " << law.nets.size() << " nets; grid " << scene.grid.nx << "x"
              << scene.grid.ny << "x" << scene.grid.nz << " at " << scene.grid.cell << " m; " << steps << " steps of " << dt << " s on "
              << main_solver->backend_api() << (other ? std::string(" against ") + other->backend_api() : std::string()) << "\n";
    if (dump) {
        dump_fields(main_solver->fields(), *dump + "/step_0", error);
    }
    double total_ms = 0.0;
    std::size_t violations = 0;
    // What --json reports: the steps after the warm-up (the first steps compile kernels and fill caches).
    double timed_ms = 0.0;
    double timed_gpu_ms = 0.0;
    int timed = 0;
    int fallbacks = 0;
    std::vector<double> term_ms(law.terms.size(), 0.0);
    std::vector<double> step_ms;
    for (int s = 1; s <= steps; ++s) {
        const auto start = std::chrono::steady_clock::now();
        main_solver->step(dt);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        total_ms += ms;
        fallbacks += main_solver->profile().fallbacks;
        if (s > warmup) {
            const StepProfile& p = main_solver->profile();
            timed_ms += ms;
            timed_gpu_ms += p.gpu_ms;
            step_ms.push_back(ms);
            ++timed;
            for (std::size_t t = 0; t < p.terms.size() && t < term_ms.size(); ++t) {
                term_ms[t] += p.terms[t].ms;
            }
        }
        violations += main_solver->violations().size();
        for (const Violation& v : main_solver->violations()) {
            if (violations < 12) {
                std::cerr << "  step " << s << ": [" << v.term << "] " << v.field << " " << to_string(v.kind) << ": " << v.detail << "\n";
            }
        }
        if (other) {
            other->step(dt);
        }
        if (dump && every > 0 && s % every == 0) {
            dump_fields(std::as_const(*main_solver).fields(), *dump + "/step_" + std::to_string(s), error);
        }
        if (profile && (s == 1 || s == steps)) {
            print_profile(main_solver->profile());
        }
    }
    if (dump && (every <= 0 || steps % every != 0)) {
        dump_fields(std::as_const(*main_solver).fields(), *dump + "/step_" + std::to_string(steps), error);
    }
    std::cout << "mean step " << total_ms / steps << " ms, " << violations << " contract violation(s)\n";
    if (json_out) {
        std::sort(step_ms.begin(), step_ms.end());
        std::ofstream out(*json_out);
        out << std::setprecision(9) << "{\"backend\":\"" << main_solver->backend_api() << "\",\"cells\":"
            << static_cast<long long>(scene.grid.nx) * scene.grid.ny * scene.grid.nz << ",\"steps\":" << timed
            << ",\"mean_ms\":" << (timed ? timed_ms / timed : 0.0) << ",\"median_ms\":" << (step_ms.empty() ? 0.0 : step_ms[step_ms.size() / 2])
            << ",\"gpu_ms\":" << (timed ? timed_gpu_ms / timed : 0.0) << ",\"violations\":" << violations << ",\"fallbacks\":" << fallbacks
            << ",\"substeps\":[";
        const std::vector<int>& sub = main_solver->last_substeps();
        for (std::size_t t = 0; t < sub.size(); ++t) {
            out << (t ? "," : "") << sub[t];
        }
        out << "],\"terms\":{";
        for (std::size_t t = 0; t < term_ms.size(); ++t) {
            out << (t ? "," : "") << '"' << law.terms[t].name << "\":" << (timed ? term_ms[t] / timed : 0.0);
        }
        out << "}}\n";
    }
    if (other) {
        const FieldSet& a = std::as_const(*main_solver).fields();
        const FieldSet& b = std::as_const(*other).fields();
        double worst = 0.0;
        for (std::size_t f = 0; f < a.count(); ++f) {
            const std::vector<float>& va = a.values(static_cast<int>(f));
            const std::vector<float>& vb = b.values(static_cast<int>(f));
            double most = 0.0;
            double scale = 1e-6;
            for (std::size_t i = 0; i < va.size(); ++i) {
                most = std::max(most, std::abs(static_cast<double>(va[i]) - vb[i]));
                scale = std::max(scale, std::abs(static_cast<double>(va[i])));
            }
            worst = std::max(worst, most / scale);
            if (most > 0.0) {
                std::cout << "  " << std::left << std::setw(14) << a.at(static_cast<int>(f)).name << std::right << " max |cpu - gpu| "
                          << std::scientific << std::setprecision(3) << most << "  (" << most / scale << " of its largest)\n"
                          << std::defaultfloat;
            }
        }
        std::cout << "worst relative difference " << worst << "\n";
    }
    return 0;
}

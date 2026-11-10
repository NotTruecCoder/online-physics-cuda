// Gate: THE LAW IN CUDA IS THE LAW. The kernels generated from the IR hold to the CPU reference on the shipped laws and on
// every op alone; two runs on one card agree to the bit; a window that moves (the ring's origin, only the arriving
// boxes uploaded) gives what uploading the whole moved lattice gives; the IR shares and fuses.
//
// Falsified by (each run once): the edge clamp in ex()/ey()/ez() dropped (stencils read past the lattice: parity
// breaks on lap/diffuse); `in_place` forced on for every term (a term reading a field it writes off its own cell races:
// parity and determinism break on spread/fall); the ring's origin not moved on a shift (the shift check fails); CSE
// off in the IR (the "found twice" count is 0).
#include "tests/Gate.hpp"

#include "field/Law.hpp"
#include "field/Scene.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaSolver.hpp"
#include "field/ir/LawIR.hpp"

#include <utility>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>

using namespace fieldengine::field;
using fieldengine::gate::check;

namespace {

void force(const char* backend)
{
#ifdef _WIN32
    _putenv_s("FIELD_BACKEND", backend);
#else
    setenv("FIELD_BACKEND", backend, 1);
#endif
}

std::optional<Law> compile(const std::vector<std::string>& paths, const std::string& extra = "")
{
    std::vector<LawSource> sources;
    for (const std::string& p : paths) {
        if (std::optional<LawSource> s = read_law_file(p)) {
            sources.push_back(*s);
        }
    }
    if (!extra.empty()) {
        sources.push_back(LawSource{"extra", extra});
    }
    CompileResult r = compile_law(sources);
    for (const std::string& e : r.errors) {
        std::cerr << e << "\n";
    }
    return r.law;
}

double worst_difference(const FieldSet& a, const FieldSet& b)
{
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
    }
    return worst;
}

} // namespace

int main()
{
    if (!CudaSolver::unavailable().empty()) {
        std::cout << "no CUDA here (" << CudaSolver::unavailable() << "): nothing to hold to the reference\n";
        return fieldengine::gate::report("cuda_parity");
    }
    force("cuda");

    // ---- 1. the shipped laws, a scene with something for every op ----------------------------------------------------
    Scene scene;
    std::string error;
    check(load_scene("assets/scenes/fire_test.scene", scene, error), "the test scene loads");
    const std::optional<Law> world = compile({"assets/law/world.law", "assets/law/magic.law"});
    check(world.has_value(), "the shipped laws compile");
    if (world) {
        const ir::Module m = ir::lower(*world);
        std::size_t fused = 0;
        for (const ir::Program& p : m.terms) {
            fused += p.in_place ? 1 : 0;
        }
        std::cout << "  IR: " << m.insts.size() << " instructions, " << m.shared << " shared, " << fused << " of " << m.terms.size()
                  << " terms fused into one in-place kernel\n";
        check(m.shared > 0, "the IR keeps a value found twice once");
        check(fused > 0 && fused < m.terms.size(), "some terms store in place, and those that read their own fields off their cell do not");

        Solver cpu(*world, scene.grid);
        Solver gpu(*world, scene.grid);
        Solver again(*world, scene.grid);
        apply_scene(scene, cpu, error);
        apply_scene(scene, gpu, error);
        apply_scene(scene, again, error);
        std::string why;
        check(gpu.use_gpu(why) && std::string(gpu.backend_api()) == "CUDA", "the law runs in CUDA: " + why);
        check(again.use_gpu(why), "a second CUDA solver opens");
        for (int s = 0; s < 12; ++s) {
            cpu.step(0.25);
            gpu.step(0.25);
            again.step(0.25);
        }
        const double worst = worst_difference(std::as_const(cpu).fields(), std::as_const(gpu).fields());
        std::cout << "  12 steps of the world law: worst relative difference to the CPU " << worst << "\n";
        check(worst < 1e-4, "CUDA holds to the CPU reference on the shipped laws (to float rounding)");
        check(std::as_const(gpu).fields().digest() == std::as_const(again).fields().digest(), "two CUDA runs agree to the bit");
        check(gpu.violations().empty() == cpu.violations().empty(), "the same contracts break (or hold) on both");
        const StepProfile& p = gpu.profile();
        check(p.backend == "CUDA" && p.gpu_ms > 0.0 && p.launches > 0, "the step is profiled: card time and launches");
    }

    // ---- 2. every op alone --------------------------------------------------------------------------------------------
    const char* ops[] = {
        "d A = lap(A)", "d A = gx(A) + gy(A) + gz(A)", "d A = diffuse(0.5 + B, A)", "d A = adv(A, B - 0.5, 0.3, -0.2)",
        "d A = fall(A, B, 2)", "d A = spread(A, B, 0.5)", "d A = topple(A, B, 1)", "d A = shift(A, 1, -1, 0) - A",
        "d A = step(B - 0.5) * sqrt(abs(A)) - clamp(A, 0.2, 0.4) + min(A, B) * max(A, 0.1)", "d A = abs(B) ^ 1.5 - exp(-A) + log(1 + B)",
        "d A = 0.01 * (maxof(B) - minof(A)) + 0.0001 * sumof(B)", "d A = 0.1 * distance(B)", "d A = 0.1 * poisson(B - 0.5, 16)",
    };
    for (const char* op : ops) {
        const std::string text = std::string("field A init 0 range -1e9 1e9\nfield B init 0 range -1e9 1e9\nterm t\n  ") + op + "\n";
        const std::optional<Law> law = compile({}, text);
        if (!law) {
            check(false, std::string("compiles: ") + op);
            continue;
        }
        const GridShape g{20, 14, 9, 0.5};
        Solver cpu(*law, g);
        Solver gpu(*law, g);
        for (Solver* s : {&cpu, &gpu}) {
            FieldSet& f = s->fields();
            for (int z = 0; z < g.nz; ++z) {
                for (int y = 0; y < g.ny; ++y) {
                    for (int x = 0; x < g.nx; ++x) {
                        const std::size_t i = g.index(x, y, z);
                        f.values(0)[i] = static_cast<float>(0.5 + 0.4 * std::sin(0.7 * x + 0.3 * y) * std::cos(0.5 * z));
                        f.values(1)[i] = (x + 2 * y + 3 * z) % 7 < 3 ? 1.0f : 0.0f;
                    }
                }
            }
            s->refresh_derived();
        }
        std::string why;
        gpu.use_gpu(why);
        for (int s = 0; s < 4; ++s) {
            cpu.step(0.02);
            gpu.step(0.02);
        }
        const double worst = worst_difference(std::as_const(cpu).fields(), std::as_const(gpu).fields());
        check(worst < 2e-5, std::string("CUDA holds to the CPU on: ") + op + " (" + std::to_string(worst) + ")");
    }

    // ---- 3. the ring: a moved window, only the arriving boxes sent, is the moved lattice ----------------------------
    {
        const std::optional<Law> law = compile({}, "field A init 0 range -1e9 1e9\nterm t\n  d A = diffuse(0.3, A)\n  conserves A\nderive D = distance(A)\n");
        const GridShape g{32, 24, 8, 1.0};
        Solver ring(*law, g);
        Solver whole(*law, g);
        std::string why;
        auto paint = [&](FieldSet& f, int x0, int x1) {
            for (int z = 0; z < g.nz; ++z) {
                for (int y = 0; y < g.ny; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        f.values(0)[g.index(x, y, z)] = ((x * 7 + y * 3 + z) % 11) / 10.0f;
                    }
                }
            }
        };
        paint(ring.fields(), 0, g.nx);
        paint(whole.fields(), 0, g.nx);
        ring.use_gpu(why);
        whole.use_gpu(why);
        ring.step(0.1);
        whole.step(0.1);
        for (int move = 0; move < 3; ++move) {
            // The lattice slides 8 cells in x (what was at x + 8 is now at x); the last 8 columns are new.
            const int d = 8;
            for (Solver* s : {&ring, &whole}) {
                FieldSet& f = s == &ring ? s->shift_window(d, 0, 0, {CellBox{g.nx - d, 0, 0, g.nx, g.ny, g.nz}}) : s->fields();
                std::vector<float>& v = f.values(0);
                for (int z = 0; z < g.nz; ++z) {
                    for (int y = 0; y < g.ny; ++y) {
                        for (int x = 0; x + d < g.nx; ++x) {
                            v[g.index(x, y, z)] = v[g.index(x + d, y, z)];
                        }
                    }
                }
                paint(f, g.nx - d, g.nx);
                s->refresh_derived();
                s->step(0.1);
            }
        }
        check(std::as_const(ring).fields().digest() == std::as_const(whole).fields().digest(),
            "a window moved as a ring (only what arrived uploaded) is to the bit the window moved whole");
    }
    return fieldengine::gate::report("cuda_parity");
}

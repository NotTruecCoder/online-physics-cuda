// Gate: NETWORKS IN THE LAW. A net's backward pass is its forward pass's derivative (against finite differences); a
// learned flux moves T without making or losing any, whatever its weights are, on the CPU and in CUDA; its limiter
// keeps T in its range at any step; tensor cores give what fp32 gives, to fp16; an untrained ldiffuse is diffuse.
//
// Falsified by (each run once): flux as g(i,j) instead of half the difference of both ways round (conservation is
// reported); the limiter off (T goes below 0 at a big step); the A fragment's rows swapped in CudaNet (the tensor-core
// parity breaks); ldiffuse's base weights not [0 -1 1 0] (the untrained check fails).
#include "tests/Gate.hpp"

#include "field/Law.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaSolver.hpp"
#include "field/nn/Net.hpp"

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

std::optional<Law> compile(const std::string& text)
{
    CompileResult r = compile_law({LawSource{"gate", text}});
    for (const std::string& e : r.errors) {
        std::cerr << e << "\n";
    }
    return r.law;
}

// Random weights everywhere (an untrained net is all zero at the end: nothing to test).
void scramble(Law& law, float amount)
{
    for (NetDecl& net : law.nets) {
        for (std::size_t i = 0; i < net.weights.size(); ++i) {
            net.weights[i] = amount * std::sin(1.7f * static_cast<float>(i) + 0.3f);
        }
    }
}

void paint(Solver& s)
{
    FieldSet& f = s.fields();
    const GridShape& g = f.shape();
    for (int z = 0; z < g.nz; ++z) {
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x) {
                f.values(0)[g.index(x, y, z)] = static_cast<float>(400.0 + 300.0 * std::sin(0.5 * x) * std::cos(0.4 * y + z));
                if (f.count() > 1) {
                    f.values(1)[g.index(x, y, z)] = (x + y) % 3 == 0 ? 0.8f : 0.1f;
                }
            }
        }
    }
    s.refresh_derived();
}

double total(const Solver& s, int field)
{
    double t = 0.0;
    for (const float v : s.fields().values(field)) {
        t += v;
    }
    return t;
}

} // namespace

int main()
{
    // ---- 1. backward is the derivative of forward ----------------------------------------------------------------
    {
        NetDecl net{.name = "probe"};
        net.sizes = {3, 8, 8, 2};
        init_net(net, 7);
        for (std::size_t i = 0; i < net.weights.size(); ++i) {
            net.weights[i] += 0.2f * std::cos(static_cast<float>(i));
        }
        const float x[3] = {0.3f, -0.7f, 1.1f};
        std::vector<float> tape(18);
        float y[2];
        net_forward_taped(net, x, y, tape.data());
        const float dy[2] = {1.0f, -0.5f};
        std::vector<float> grad(net.parameter_count(), 0.0f);
        std::vector<float> scratch(16);
        float dx[3];
        net_backward(net, x, tape.data(), dy, grad.data(), dx, scratch.data());
        double worst = 0.0;
        for (std::size_t p = 0; p < net.weights.size(); p += 7) {
            const float keep = net.weights[p];
            float hi[2];
            float lo[2];
            net.weights[p] = keep + 1e-2f;
            net_forward(net, x, hi, scratch.data());
            net.weights[p] = keep - 1e-2f;
            net_forward(net, x, lo, scratch.data());
            net.weights[p] = keep;
            const double fd = ((hi[0] - lo[0]) * dy[0] + (hi[1] - lo[1]) * dy[1]) / 2e-2;
            worst = std::max(worst, std::abs(fd - grad[p]) / (std::abs(fd) + 1e-2));
        }
        std::cout << "  backward against finite differences: worst relative " << worst << "\n";
        check(worst < 2e-2, "the backward pass is the derivative of the forward pass");
    }

    // ---- 2. a learned flux conserves, by construction, and stays in range -------------------------------------------
    const std::string flux_law = "field T init 290 range 0 6000\nfield fuel init 0 range 0 1.01\n"
                                 "net fx in 4 hidden 16 16 out 1 act tanh file none\nterm f\n  d T = flux(fx, T, fuel)\n  conserves T\n";
    if (std::optional<Law> law = compile(flux_law)) {
        scramble(*law, 3.0f);
        law->nets[0].out_scale = 400.0f;
        const GridShape g{24, 16, 4, 1.0};
        Solver cpu(*law, g);
        paint(cpu);
        const double before = total(cpu, 0);
        for (int s = 0; s < 10; ++s) {
            cpu.step(0.5); // a big step: without the limiter T would go below 0
        }
        float least = 1e30f;
        for (const float v : std::as_const(cpu).fields().values(0)) {
            least = std::min(least, v);
        }
        const double drift = std::abs(total(cpu, 0) - before) / before;
        std::cout << "  learned flux, random weights, 10 big steps: total moved by " << drift << " of itself, least T " << least << "\n";
        check(drift < 1e-6, "a learned flux makes and loses no T (antisymmetric by construction)");
        check(cpu.violations().empty(), "no contract breaks: conserves and range hold whatever the weights");
        check(least >= 0.0f, "the limiter keeps T in its range at a big step");
        // The same law where the net's raw flux is used one way only (no half difference): conservation breaks.
        std::optional<Law> mutant = compile("field T init 290 range 0 6000\nfield fuel init 0 range 0 1.01\n"
                                            "net fx in 2 hidden 16 16 out 1 act tanh file none\nterm f\n  d T = mlp(fx, T, fuel)\n  conserves T\n");
        scramble(*mutant, 3.0f);
        mutant->nets[0].out_scale = 400.0f;
        Solver bad(*mutant, g);
        paint(bad);
        bad.step(0.5);
        bool caught = false;
        for (const Violation& v : bad.violations()) {
            caught = caught || v.kind == Violation::Kind::Conservation;
        }
        check(caught, "control: a net that writes T cell by cell is caught making T");

        if (CudaSolver::unavailable().empty()) {
            force("cuda");
            // One step (the random net is chaotic: rounding grows over steps), against what the step changed.
            scramble(*law, 0.7f);
            // Inputs scaled to about one, as every trainer here leaves them (fp16 holds 400 K to a quarter of a kelvin).
            law->nets[0].in_shift = {400.0f, 0.4f, 400.0f, 0.4f};
            law->nets[0].in_scale = {1.0f / 300.0f, 2.0f, 1.0f / 300.0f, 2.0f};
            for (const bool half : {false, true}) {
                Law l = *law;
                l.nets[0].half = half;
                Solver ref(l, g);
                Solver gpu(l, g);
                paint(ref);
                paint(gpu);
                const std::vector<float> before = std::as_const(ref).fields().values(0);
                std::string why;
                check(gpu.use_gpu(why), "the law with a net runs in CUDA: " + why);
                ref.step(0.2);
                gpu.step(0.2);
                double worst = 0.0;
                double moved = 0.0;
                for (std::size_t i = 0; i < before.size(); ++i) {
                    const double c = std::as_const(ref).fields().values(0)[i];
                    worst = std::max(worst, std::abs(c - std::as_const(gpu).fields().values(0)[i]));
                    moved = std::max(moved, std::abs(c - before[i]));
                }
                std::cout << "  CUDA " << (half ? "tensor cores (fp16)" : "fp32") << ": worst |T - T_cpu| " << worst << " K of a step that moved T by up to "
                          << moved << " K\n";
                check(moved > 1.0, "the learned flux moves heat");
                check(worst <= (half ? 0.03 : 1e-4) * moved + 1e-3, half ? "tensor cores hold to the CPU to fp16" : "the fp32 net kernel holds to the CPU");
                for (int s = 0; s < 4; ++s) {
                    gpu.step(0.2);
                }
                const double drift = std::abs(total(gpu, 0) - total(ref, 0) - 0.0) / total(ref, 0);
                (void)drift;
                double t0 = 0.0;
                for (const float v : before) {
                    t0 += v;
                }
                check(std::abs(total(gpu, 0) - t0) / t0 < 1e-6, "in CUDA too the learned flux keeps the total");
            }
        }
    } else {
        check(false, "the flux law compiles");
    }

    // ---- 3. an untrained ldiffuse is diffuse ------------------------------------------------------------------------
    {
        std::optional<Law> a = compile("field T init 290 range 0 6000\nterm t\n  d T = diffuse(0.8, T)\n");
        std::optional<Law> b = compile("field T init 290 range 0 6000\nnet ld in 5 hidden 8 out 4 act tanh file none\nterm t\n  d T = ldiffuse(ld, 0.8, T)\n");
        const GridShape g{16, 12, 3, 1.0};
        Solver sa(*a, g);
        Solver sb(*b, g);
        paint(sa);
        paint(sb);
        for (int s = 0; s < 5; ++s) {
            sa.step(0.1);
            sb.step(0.1);
        }
        double worst = 0.0;
        for (std::size_t i = 0; i < sa.fields().values(0).size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::abs(std::as_const(sa).fields().values(0)[i] - std::as_const(sb).fields().values(0)[i])));
        }
        std::cout << "  untrained ldiffuse against diffuse: worst " << worst << " K\n";
        check(worst < 1e-2, "an untrained learned discretization is the classical one");
    }
    return fieldengine::gate::report("nets");
}

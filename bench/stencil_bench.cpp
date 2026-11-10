// bench_stencil: what the generated code costs against what the card can do.
//
// One law term, explicit heat conduction (assets/law/bench/heat.law: d T = diffuse(k, T), conserves T, a range), on an
// N^3 lattice, three ways on the same card:
//
//   copy        a device-to-device copy kernel: the bandwidth the card actually reaches (the roofline's roof);
//   handwritten one fused kernel, the 7-point flux-form update, ping-pong buffers: the least traffic the step can have
//               (one read of T, one write; 8 bytes a cell);
//   engine      the law compiled by the engine (IR -> CUDA -> NVRTC), with its contract checks every step or never:
//               the term's own time on the card (its CUDA events: the kernels and the contract checks), and the whole
//               step as the host sees it, with the readback of T that the engine does after every step so the CPU side
//               of the world (mesher, bodies) sees the state.
//
// The handwritten and the engine's results are compared cell by cell after the same steps (they are the same arithmetic,
// up to the order of operations). Also the CPU reference (OpenMP) for the same step. Writes JSON to stdout or --out FILE.
//
//   bench_stencil [--sizes 64,128,256] [--steps 50] [--out FILE]
#include "field/Law.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaApi.hpp"

#include <utility>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace fieldengine::field;
namespace cu = fieldengine::field::cuda;

namespace {

const char* kKernels = R"(
extern "C" __global__ void copy(const float* __restrict__ a, float* __restrict__ b, unsigned long long n)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        b[i] = a[i];
}
// The flux-form explicit step with zero-flux edges (the edge cell repeats), as the law's diffuse() with a uniform k.
extern "C" __global__ void heat(const float* __restrict__ T, float* __restrict__ out, int nx, int ny, int nz, float c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long long plane = (long long)nx * ny;
    for (int z = 0; z < nz; ++z) {
        const long long i = z * plane + (long long)y * nx + x;
        const float t = T[i];
        float s = 0.f;
        s += (x > 0 ? T[i - 1] : t) - t;
        s += (x + 1 < nx ? T[i + 1] : t) - t;
        s += (y > 0 ? T[i - nx] : t) - t;
        s += (y + 1 < ny ? T[i + nx] : t) - t;
        s += (z > 0 ? T[i - plane] : t) - t;
        s += (z + 1 < nz ? T[i + plane] : t) - t;
        out[i] = t + c * s;
    }
}
)";

float start_value(std::size_t i)
{
    std::uint64_t h = i * 0x9e3779b97f4a7c15ull;
    h ^= h >> 31;
    h *= 0x94d049bb133111ebull;
    h ^= h >> 29;
    return 290.0f + 1000.0f * static_cast<float>(h >> 40) / 16777216.0f;
}

struct Events {
    cu::CUevent a = nullptr;
    cu::CUevent b = nullptr;
    Events()
    {
        const cu::Device& d = cu::device();
        d.api.cuEventCreate(&a, 0);
        d.api.cuEventCreate(&b, 0);
    }
    void begin() const { cu::device().api.cuEventRecord(a, cu::device().stream); }
    float end() const
    {
        const cu::Device& d = cu::device();
        d.api.cuEventRecord(b, d.stream);
        d.api.cuEventSynchronize(b);
        float ms = 0.0f;
        d.api.cuEventElapsedTime(&ms, a, b);
        return ms;
    }
};

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<int> sizes = {64, 128, 256};
    int steps = 50;
    std::optional<std::string> out_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--sizes" && i + 1 < argc) {
            sizes.clear();
            std::stringstream list(argv[++i]);
            std::string item;
            while (std::getline(list, item, ',')) {
                sizes.push_back(std::atoi(item.c_str()));
            }
        }
    }
    const cu::Device& dev = cu::device();
    if (!dev.ok) {
        std::cerr << "no CUDA: " << dev.why << "\n";
        return 1;
    }
    dev.bind();
    std::string log;
    bool cached = false;
    const std::vector<char> cubin = cu::compile(kKernels, "bench_stencil.cu", {"--std=c++17", "--fmad=false"}, log, cached);
    cu::Module module;
    std::string why;
    if (cubin.empty() || !module.load(cubin, why)) {
        std::cerr << "the bench kernels do not compile: " << log << why << "\n";
        return 1;
    }
    cu::CUfunction copy = module.function("copy");
    cu::CUfunction heat = module.function("heat");

    const std::optional<LawSource> source = read_law_file("assets/law/bench/heat.law");
    if (!source) {
        std::cerr << "run from the build directory (assets/law/bench/heat.law)\n";
        return 1;
    }
    CompileResult compiled = compile_law({*source});
    if (!compiled.law) {
        std::cerr << "heat.law does not compile\n";
        return 1;
    }
    const double k = 1.0;
    const double h = 1.0;
    const double dt = 0.1; // under max_dt = 0.15 h^2 / k: one substep
    const float c = static_cast<float>(k * dt / (h * h));

    std::ostringstream json;
    json << std::setprecision(6) << "{\"card\":\"" << dev.name << "\",\"sm\":\"" << dev.major << dev.minor << "\",\"steps\":" << steps << ",\"sizes\":[";
    for (std::size_t si = 0; si < sizes.size(); ++si) {
        const int n = sizes[si];
        const std::size_t cells = static_cast<std::size_t>(n) * n * n;
        std::vector<float> start(cells);
        for (std::size_t i = 0; i < cells; ++i) {
            start[i] = start_value(i);
        }

        // ---- the roof: a copy -----------------------------------------------------------------------------------
        cu::Buffer a;
        cu::Buffer b;
        a.allocate(cells * 4);
        b.allocate(cells * 4);
        dev.api.cuMemcpyHtoDAsync(a.ptr(), start.data(), cells * 4, dev.stream);
        Events ev;
        std::vector<double> copy_ms;
        for (int r = 0; r < steps + 3; ++r) {
            cu::CUdeviceptr pa = a.ptr();
            cu::CUdeviceptr pb = b.ptr();
            unsigned long long count = cells;
            ev.begin();
            cu::launch(copy, static_cast<unsigned>(dev.sms * 16), 1, 1, 256, 1, 1, 0, {&pa, &pb, &count});
            const float ms = ev.end();
            if (r >= 3) {
                copy_ms.push_back(ms);
            }
        }
        const double copy_gbs = 8.0 * cells / (median(copy_ms) * 1e6);

        // ---- handwritten ----------------------------------------------------------------------------------------
        dev.api.cuMemcpyHtoDAsync(a.ptr(), start.data(), cells * 4, dev.stream);
        std::vector<double> hand_ms;
        cu::CUdeviceptr in = a.ptr();
        cu::CUdeviceptr out = b.ptr();
        for (int r = 0; r < steps; ++r) {
            int nx = n;
            int ny = n;
            int nz = n;
            float cv = c;
            ev.begin();
            cu::launch(heat, static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>((n + 7) / 8), 1, 32, 8, 1, 0, {&in, &out, &nx, &ny, &nz, &cv});
            hand_ms.push_back(ev.end());
            std::swap(in, out);
        }
        std::vector<float> hand(cells);
        dev.api.cuMemcpyDtoHAsync(hand.data(), in, cells * 4, dev.stream);
        dev.api.cuStreamSynchronize(dev.stream);
        const double hand_med = median(hand_ms);

        // ---- the engine -----------------------------------------------------------------------------------------
        auto run_engine = [&](int check_every, bool lazy, int count, double& card_ms, double& step_ms, double& down_ms,
                              std::vector<double>* walls) {
#ifdef _WIN32
            _putenv_s("FIELD_BACKEND", "cuda");
#else
            setenv("FIELD_BACKEND", "cuda", 1);
#endif
            Solver solver(*compiled.law, GridShape{n, n, n, h});
            solver.set_param("k_heat", k);
            solver.fields().values(solver.field("T")) = start;
            std::string reason;
            if (!solver.use_gpu(reason)) {
                std::cerr << "engine on CUDA: " << reason << "\n";
                return false;
            }
            solver.check_contracts_every(check_every);
            solver.set_lazy_readback(lazy);
            std::vector<double> card;
            std::vector<double> wall;
            std::vector<double> down;
            for (int r = 0; r < count + 3; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                solver.step(dt);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (r >= 3) {
                    card.push_back(solver.profile().terms.empty() ? 0.0 : solver.profile().terms[0].ms);
                    down.push_back(solver.profile().download_ms);
                    wall.push_back(ms);
                }
            }
            card_ms = median(card);
            step_ms = median(wall);
            down_ms = median(down);
            if (walls) {
                *walls = wall;
            }
            return true;
        };
        double eng_card = 0.0;
        double eng_wall = 0.0;
        double eng_card_nc = 0.0;
        double eng_wall_nc = 0.0;
        double down_ms = 0.0;
        double down_nc = 0.0;
        double lazy_card = 0.0;
        double lazy_wall = 0.0;
        double lazy_down = 0.0;
        std::vector<double> tails;
        std::vector<float> engine_result;
        // The tails: many steps with lazy readback and no contracts, so a step is what the card and the launches cost.
        const int tail_steps = std::max(steps, n >= 256 ? 1000 : 3000);
        if (!run_engine(1, false, steps, eng_card, eng_wall, down_ms, nullptr) || !run_engine(1000000, false, steps, eng_card_nc, eng_wall_nc, down_nc, nullptr) ||
            !run_engine(1000000, true, tail_steps, lazy_card, lazy_wall, lazy_down, &tails)) {
            return 1;
        }
        std::sort(tails.begin(), tails.end());
        const auto pct = [&](double q) { return tails[std::min(tails.size() - 1, static_cast<std::size_t>(q * static_cast<double>(tails.size())))]; };
        // The same number of steps as the handwritten run, for the comparison of values.
        {
#ifdef _WIN32
            _putenv_s("FIELD_BACKEND", "cuda");
#else
            setenv("FIELD_BACKEND", "cuda", 1);
#endif
            Solver solver(*compiled.law, GridShape{n, n, n, h});
            solver.set_param("k_heat", k);
            solver.fields().values(solver.field("T")) = start;
            std::string reason;
            solver.use_gpu(reason);
            solver.set_lazy_readback(true); // the cells come back once, when read: this checks the fetch too
            for (int r = 0; r < steps; ++r) {
                solver.step(dt);
            }
            engine_result = std::as_const(solver).values(solver.field("T"));
        }
        double worst = 0.0;
        for (std::size_t i = 0; i < cells; ++i) {
            worst = std::max(worst, static_cast<double>(std::abs(engine_result[i] - hand[i])));
        }

        // ---- the CPU reference (a few steps: it is slow) --------------------------------------------------------
        double cpu_ms = 0.0;
        {
            Solver solver(*compiled.law, GridShape{n, n, n, h});
            solver.set_param("k_heat", k);
            solver.fields().values(solver.field("T")) = start;
            std::vector<double> wall;
            const int cpu_steps = n >= 256 ? 3 : 8;
            for (int r = 0; r < cpu_steps + 1; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                solver.step(dt);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (r >= 1) {
                    wall.push_back(ms);
                }
            }
            cpu_ms = median(wall);
        }

        const double mcells = static_cast<double>(cells) / 1e6;
        std::cerr << "n=" << n << ": copy " << copy_gbs << " GB/s; handwritten " << hand_med << " ms (" << 8.0 * cells / (hand_med * 1e6)
                  << " GB/s of 8 B/cell); engine term " << eng_card << " ms with contracts, " << eng_card_nc << " ms without; step "
                  << eng_wall << " ms, of it readback " << down_ms << " ms; lazy readback step " << lazy_wall << " ms (p50 " << pct(0.5)
                  << ", p99 " << pct(0.99) << ", p99.9 " << pct(0.999) << " of " << tails.size() << "); CPU " << cpu_ms
                  << " ms; |engine - handwritten| " << worst << " K\n";
        json << (si ? "," : "") << "{\"n\":" << n << ",\"cells\":" << cells << ",\"copy_gbs\":" << copy_gbs << ",\"hand_ms\":" << hand_med
             << ",\"hand_gbs_8b\":" << 8.0 * cells / (hand_med * 1e6) << ",\"engine_card_ms\":" << eng_card
             << ",\"engine_card_ms_no_contracts\":" << eng_card_nc << ",\"engine_step_ms\":" << eng_wall
             << ",\"engine_step_ms_no_contracts\":" << eng_wall_nc << ",\"engine_readback_ms\":" << down_ms
             << ",\"lazy_step_ms\":" << lazy_wall << ",\"lazy_card_ms\":" << lazy_card << ",\"lazy_p50_ms\":" << pct(0.5)
             << ",\"lazy_p99_ms\":" << pct(0.99) << ",\"lazy_p999_ms\":" << pct(0.999) << ",\"lazy_max_ms\":" << tails.back()
             << ",\"tail_steps\":" << tails.size() << ",\"cpu_ms\":" << cpu_ms
             << ",\"mcells_per_s_hand\":" << mcells / (hand_med * 1e-3) << ",\"mcells_per_s_engine_card\":" << mcells / (eng_card_nc * 1e-3)
             << ",\"max_abs_diff_K\":" << worst << "}";
    }
    json << "]}\n";
    if (out_path) {
        std::ofstream(*out_path) << json.str();
    }
    std::cout << json.str();
    return 0;
}

#include "field/mind/WorldModel.hpp"

#include "field/LawNode.hpp"
#include "field/cuda/CudaApi.hpp"
#include "field/cuda/CudaNet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <sstream>

namespace fieldengine::field {
using cuda::CUdeviceptr;
using cuda::CUfunction;

namespace {

constexpr int kBatch = 1024;
constexpr int kHidden = 64;
constexpr float kSigma = 0.6f;
constexpr int kCpuSamples = 48;

std::uint64_t mix(std::uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

// The same Gaussian on both sides: a hash of where it is used, Box-Muller.
float gauss(std::uint64_t key)
{
    const std::uint64_t a = mix(key);
    const std::uint64_t b = mix(a ^ 0x9e3779b97f4a7c15ull);
    const float u1 = (static_cast<float>(a >> 40) + 0.5f) / 16777216.0f;
    const float u2 = (static_cast<float>(b >> 40) + 0.5f) / 16777216.0f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

std::uint64_t noise_key(std::uint32_t id, std::uint64_t tick, int k, int t, int d)
{
    return (static_cast<std::uint64_t>(id) * 0x100000001b3ull) ^ (tick * 0x9e3779b97f4a7c15ull) ^
        (static_cast<std::uint64_t>(k) << 20) ^ (static_cast<std::uint64_t>(t) << 8) ^ static_cast<std::uint64_t>(d);
}

std::string literal(double v)
{
    char text[48];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(static_cast<float>(std::isfinite(v) ? v : 0.0)));
    std::string s = text;
    if (s.find_first_of(".eEn") == std::string::npos) {
        s += ".0";
    }
    return "(" + s + "f)";
}

} // namespace

// ---- the device side ----------------------------------------------------------------------------------------------

struct WorldModel::Gpu {
    cuda::Module module;
    cuda::Buffer W, scale, obs, mean, eps, cost, ids, X, T, pre, del, xn, loss, grad, m1, m2;
    int agents = 0;
    int image = 0;
    int params = 0;
    int taped = 0;
    std::uint64_t adam_t = 0;
    CUfunction rollout = nullptr;
    CUfunction update = nullptr;
    CUfunction rows = nullptr;
    CUfunction grads = nullptr;
    CUfunction adam = nullptr;
};

std::string WorldModel::cost_code() const
{
    const CreatureDef& def = law_->creatures[static_cast<std::size_t>(def_)];
    std::function<std::string(const Node&)> e = [&](const Node& n) -> std::string {
        auto a = [&](std::size_t k) { return e(*n.args[k]); };
        switch (n.op) {
        case Op::Const: return literal(n.number);
        case Op::Param: return literal(law_->param_values[static_cast<std::size_t>(n.index)]);
        case Op::Cell: return "1.f";
        case Op::Local: return n.index < locals_ ? "s[" + std::to_string(n.index) + "]" : "0.f";
        case Op::Field: {
            const auto found = std::find(observed_.begin(), observed_.end(), n.index);
            return found == observed_.end() ? "0.f" : "s[" + std::to_string(locals_ + (found - observed_.begin()) * (1 + kRing)) + "]";
        }
        case Op::Neg: return "(-" + a(0) + ")";
        case Op::Abs: return "fabsf(" + a(0) + ")";
        case Op::Sqrt: return "sqrtf(fmaxf(0.f, " + a(0) + "))";
        case Op::Exp: return "expf(fminf(" + a(0) + ", 80.f))";
        case Op::Log: return "logf(fmaxf(" + a(0) + ", 1e-30f))";
        case Op::Step: return "(" + a(0) + " > 0.f ? 1.f : 0.f)";
        case Op::Pos: return "fmaxf(" + a(0) + ", 0.f)";
        case Op::Add: return "(" + a(0) + " + " + a(1) + ")";
        case Op::Sub: return "(" + a(0) + " - " + a(1) + ")";
        case Op::Mul: return "(" + a(0) + " * " + a(1) + ")";
        case Op::Div: return "(" + a(1) + " == 0.f ? 0.f : " + a(0) + " / " + a(1) + ")";
        case Op::Pow: return "powf(" + a(0) + ", " + a(1) + ")";
        case Op::Min: return "fminf(" + a(0) + ", " + a(1) + ")";
        case Op::Max: return "fmaxf(" + a(0) + ", " + a(1) + ")";
        case Op::Clamp: return "fminf(fmaxf(" + a(0) + ", " + a(1) + "), fmaxf(" + a(1) + ", " + a(2) + "))";
        default: return "0.f";
        }
    };
    return "__device__ __forceinline__ float wm_cost(const float* s) { return " + (def.plan ? e(*def.plan) : std::string("0.f")) + "; }\n";
}

float WorldModel::cost(const float* s) const
{
    const CreatureDef& def = law_->creatures[static_cast<std::size_t>(def_)];
    std::function<float(const Node&)> e = [&](const Node& n) -> float {
        auto a = [&](std::size_t k) { return e(*n.args[k]); };
        switch (n.op) {
        case Op::Const: return static_cast<float>(n.number);
        case Op::Param: return static_cast<float>(law_->param_values[static_cast<std::size_t>(n.index)]);
        case Op::Cell: return 1.0f;
        case Op::Local: return n.index < locals_ ? s[n.index] : 0.0f;
        case Op::Field: {
            const auto found = std::find(observed_.begin(), observed_.end(), n.index);
            return found == observed_.end() ? 0.0f : s[locals_ + (found - observed_.begin()) * (1 + kRing)];
        }
        case Op::Neg: return -a(0);
        case Op::Abs: return std::abs(a(0));
        case Op::Sqrt: return std::sqrt(std::max(0.0f, a(0)));
        case Op::Exp: return std::exp(std::min(a(0), 80.0f));
        case Op::Log: return std::log(std::max(a(0), 1e-30f));
        case Op::Step: return a(0) > 0.0f ? 1.0f : 0.0f;
        case Op::Pos: return std::max(a(0), 0.0f);
        case Op::Add: return a(0) + a(1);
        case Op::Sub: return a(0) - a(1);
        case Op::Mul: return a(0) * a(1);
        case Op::Div: { const float b = a(1); return b == 0.0f ? 0.0f : a(0) / b; }
        case Op::Pow: return std::pow(a(0), a(1));
        case Op::Min: return std::min(a(0), a(1));
        case Op::Max: return std::max(a(0), a(1));
        case Op::Clamp: { const float lo = a(1); return std::min(std::max(a(0), lo), std::max(lo, a(2))); }
        default: return 0.0f;
        }
    };
    return def.plan ? e(*def.plan) : 0.0f;
}

WorldModel::WorldModel(const Law& law, int creature) : law_(&law), def_(creature)
{
    const CreatureDef& def = law.creatures[static_cast<std::size_t>(creature)];
    locals_ = static_cast<int>(def.locals.size());
    observed_ = def.observe;
    features_ = locals_ + static_cast<int>(observed_.size()) * (1 + kRing);
    horizon_ = def.horizon;
    samples_ = std::clamp(def.samples, 8, 1024);
    dt_ = def.plan_dt;
    radius_ = 3.0 * def.sense;
    net_.name = "wm_" + def.name;
    net_.sizes = {features_ + 2, kHidden, kHidden, features_};
    net_.act = Activation::Tanh;
    init_net(net_, 0x5eed);
    scale_.assign(static_cast<std::size_t>(features_), 1.0f);
    x_.assign(kCapacity * static_cast<std::size_t>(features_ + 2), 0.0f);
    t_.assign(kCapacity * static_cast<std::size_t>(features_), 0.0f);

    const cuda::Device& d = cuda::device();
    if (!d.ok) {
        why_cpu_ = d.why;
        return;
    }
    d.bind();
    const int F = features_;
    const int H = horizon_;
    const cuda::NetLayout layout = cuda::net_layout(net_);
    std::ostringstream src;
    src << "typedef unsigned long long u64;\n" << cuda::net_prelude() << cuda::net_device_code(net_, "dyn") << cuda::net_train_code(net_, "dyn")
        << cost_code() << R"(
__device__ __forceinline__ u64 mix(u64 h) { h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33; return h; }
__device__ __forceinline__ float gauss(u64 key)
{
    const u64 a = mix(key); const u64 b = mix(a ^ 0x9e3779b97f4a7c15ull);
    const float u1 = ((float)(a >> 40) + 0.5f) / 16777216.f; const float u2 = ((float)(b >> 40) + 0.5f) / 16777216.f;
    return sqrtf(-2.f * logf(u1)) * cosf(6.28318530718f * u2);
}
)";
    src << "constexpr int F = " << F << ", H = " << H << ", IMAGE = " << layout.image_floats << ";\n" << R"(
// A block a creature, a thread a future: H steps of the learned ODE from where it is, the law's cost of each.
extern "C" __global__ void mppi_rollout(const float* __restrict__ W, const float* __restrict__ SCALE, const float* __restrict__ OBS,
    const float* __restrict__ MEAN, const unsigned* __restrict__ IDS, float* EPS, float* COST, int K, u64 tick, float sigma, float dt)
{
    extern __shared__ float sm[];
    for (int q = threadIdx.x; q < IMAGE; q += blockDim.x) sm[q] = W[q];
    float* scale = sm + IMAGE;
    for (int q = threadIdx.x; q < F; q += blockDim.x) scale[q] = SCALE[q];
    __syncthreads();
    const int agent = blockIdx.x; const int k = threadIdx.x; if (k >= K) return;
    float s[F]; float x[F + 2]; float ds[F];
    for (int f = 0; f < F; ++f) s[f] = OBS[agent * F + f];
    float J = 0.f;
    const u64 id = IDS[agent];
    for (int t = 0; t < H; ++t) {
        float a[2];
        for (int d = 0; d < 2; ++d) {
            const float e = k == 0 ? 0.f : gauss((id * 0x100000001b3ull) ^ (tick * 0x9e3779b97f4a7c15ull) ^ ((u64)k << 20) ^ ((u64)t << 8) ^ (u64)d);
            a[d] = fminf(fmaxf(MEAN[(agent * H + t) * 2 + d] + sigma * e, -1.f), 1.f);
            EPS[(((u64)agent * K + k) * H + t) * 2 + d] = a[d];
        }
        for (int f = 0; f < F; ++f) x[f] = s[f];
        x[F] = a[0]; x[F + 1] = a[1];
        dyn_fwd(sm, x, ds);
        for (int f = 0; f < F; ++f) s[f] += dt * ds[f] * scale[f];
        J += wm_cost(s) + 0.01f * (a[0] * a[0] + a[1] * a[1]);
    }
    COST[agent * K + k] = J;
}

// The futures weighed by their cost (softmin, the temperature a quarter of the spread): the new plan. A thread a step
// and an axis, the futures summed in order.
extern "C" __global__ void mppi_update(const float* __restrict__ COST, const float* __restrict__ EPS, float* MEAN, int K)
{
    const int agent = blockIdx.x; const int q = threadIdx.x; if (q >= H * 2) return;
    float lo = 1e30f; float sum = 0.f;
    for (int k = 0; k < K; ++k) { const float c = COST[agent * K + k]; lo = fminf(lo, c); sum += c; }
    const float lam = fmaxf(1e-4f, 0.25f * (sum / K - lo));
    float w = 0.f; float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const float e = expf(-(COST[agent * K + k] - lo) / lam);
        w += e; acc += e * EPS[((u64)agent * K + k) * H * 2 + q];
    }
    MEAN[agent * H * 2 + q] = acc / w;
}
)";
    std::string log;
    bool cached = false;
    const std::vector<char> cubin = cuda::compile(src.str(), "mind.cu", {"--std=c++17"}, log, cached);
    if (cubin.empty()) {
        why_cpu_ = "the world model did not compile: " + log.substr(0, 400);
        return;
    }
    auto g = std::make_unique<Gpu>();
    if (!g->module.load(cubin, why_cpu_)) {
        return;
    }
    g->rollout = g->module.function("mppi_rollout");
    g->update = g->module.function("mppi_update");
    g->rows = g->module.function("dyn_train_rows");
    g->grads = g->module.function("dyn_train_grad");
    g->adam = g->module.function("dyn_adam");
    g->image = layout.image_floats;
    g->params = static_cast<int>(net_.parameter_count());
    g->taped = 2 * kHidden + F;
    const bool ok = g->W.allocate(4ull * g->image) && g->scale.allocate(4ull * F) && g->X.allocate(4ull * kBatch * (F + 2)) &&
        g->T.allocate(4ull * kBatch * F) && g->pre.allocate(4ull * kBatch * g->taped) && g->del.allocate(4ull * kBatch * g->taped) &&
        g->xn.allocate(4ull * kBatch * (F + 2)) && g->loss.allocate(4ull * kBatch) && g->grad.allocate(4ull * g->params) &&
        g->m1.allocate(4ull * g->params) && g->m2.allocate(4ull * g->params);
    if (!ok || !g->rollout || !g->update || !g->rows || !g->grads || !g->adam) {
        why_cpu_ = "the world model's kernels or memory are missing";
        return;
    }
    d.api.cuMemsetD32Async(g->m1.ptr(), 0, static_cast<std::size_t>(g->params), d.stream);
    d.api.cuMemsetD32Async(g->m2.ptr(), 0, static_cast<std::size_t>(g->params), d.stream);
    const std::vector<float> image = cuda::net_image(net_);
    d.api.cuMemcpyHtoDAsync(g->W.ptr(), image.data(), image.size() * 4, d.stream);
    d.api.cuMemcpyHtoDAsync(g->scale.ptr(), scale_.data(), scale_.size() * 4, d.stream);
    d.api.cuStreamSynchronize(d.stream);
    gpu_ = std::move(g);
}

WorldModel::~WorldModel() = default;

const NetDecl& WorldModel::net() const
{
    return net_;
}

void WorldModel::observe(const FieldSet& fields, double x, double y, double z, const std::vector<double>& locals, float* out) const
{
    for (int i = 0; i < locals_; ++i) {
        out[i] = i < static_cast<int>(locals.size()) ? static_cast<float>(locals[static_cast<std::size_t>(i)]) : 0.0f;
    }
    int at = locals_;
    for (const int f : observed_) {
        out[at++] = fields.sample(f, x, y, z);
        for (int k = 0; k < kRing; ++k) {
            const double a = 6.283185307179586 * k / kRing;
            out[at++] = fields.sample(f, x + radius_ * std::cos(a), y + radius_ * std::sin(a), z);
        }
    }
}

void WorldModel::remember(std::uint32_t id, const float* obs, double dt)
{
    Last& last = last_[id];
    const int F = features_;
    if (last.acted && static_cast<int>(last.obs.size()) == F && dt > 1e-6) {
        float* x = x_.data() + head_ * static_cast<std::size_t>(F + 2);
        float* t = t_.data() + head_ * static_cast<std::size_t>(F);
        for (int f = 0; f < F; ++f) {
            x[f] = last.obs[static_cast<std::size_t>(f)];
            t[f] = static_cast<float>((obs[f] - last.obs[static_cast<std::size_t>(f)]) / dt);
        }
        x[F] = last.action[0];
        x[F + 1] = last.action[1];
        head_ = (head_ + 1) % kCapacity;
        count_ = std::min(count_ + 1, kCapacity);
    }
    last.obs.assign(obs, obs + F);
}

void WorldModel::forget(std::uint32_t id)
{
    last_.erase(id);
    mean_.erase(id);
}

// Once it has seen enough: what its inputs and its changes are in units of, fixed from then on.
void WorldModel::normalise()
{
    const int F = features_;
    const std::size_t n = count_;
    net_.in_shift.assign(static_cast<std::size_t>(F + 2), 0.0f);
    net_.in_scale.assign(static_cast<std::size_t>(F + 2), 1.0f);
    for (int f = 0; f < F; ++f) {
        double sum = 0.0;
        double sq = 0.0;
        double tsum = 0.0;
        double tsq = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const double v = x_[r * static_cast<std::size_t>(F + 2) + static_cast<std::size_t>(f)];
            const double t = t_[r * static_cast<std::size_t>(F) + static_cast<std::size_t>(f)];
            sum += v;
            sq += v * v;
            tsum += t;
            tsq += t * t;
        }
        const double mean = sum / n;
        const double sd = std::sqrt(std::max(0.0, sq / n - mean * mean));
        const double tmean = tsum / n;
        const double tsd = std::sqrt(std::max(0.0, tsq / n - tmean * tmean));
        net_.in_shift[static_cast<std::size_t>(f)] = static_cast<float>(mean);
        net_.in_scale[static_cast<std::size_t>(f)] = static_cast<float>(1.0 / std::max(sd, 1e-3));
        scale_[static_cast<std::size_t>(f)] = static_cast<float>(std::max(std::sqrt(tsd * tsd + tmean * tmean), 1e-4));
    }
    scaled_ = true;
    if (gpu_) {
        const cuda::Device& d = cuda::device();
        d.bind();
        const std::vector<float> image = cuda::net_image(net_);
        d.api.cuMemcpyHtoDAsync(gpu_->W.ptr(), image.data(), image.size() * 4, d.stream);
        d.api.cuMemcpyHtoDAsync(gpu_->scale.ptr(), scale_.data(), scale_.size() * 4, d.stream);
        d.api.cuStreamSynchronize(d.stream);
    }
}

void WorldModel::predict(const float* s, const float* a, float* ds_dt) const
{
    std::vector<float> x(s, s + features_);
    x.push_back(a[0]);
    x.push_back(a[1]);
    std::vector<float> y(static_cast<std::size_t>(features_));
    std::vector<float> scratch(2 * static_cast<std::size_t>(net_.widest()));
    net_forward(net_, x.data(), y.data(), scratch.data());
    for (int f = 0; f < features_; ++f) {
        ds_dt[f] = y[static_cast<std::size_t>(f)] * scale_[static_cast<std::size_t>(f)];
    }
}

void WorldModel::learn(int iterations)
{
    if (!scaled_) {
        if (count_ < 512) {
            return;
        }
        normalise();
    }
    const auto started = std::chrono::steady_clock::now();
    const int F = features_;
    const int rows = static_cast<int>(std::min<std::size_t>(kBatch, count_));
    std::vector<float> X(static_cast<std::size_t>(rows) * (F + 2));
    std::vector<float> T(static_cast<std::size_t>(rows) * F);
    double loss_sum = 0.0;
    for (int it = 0; it < iterations; ++it) {
        ++learn_round_;
        for (int r = 0; r < rows; ++r) {
            const std::size_t pick = static_cast<std::size_t>(mix(learn_round_ * 0x9e3779b97f4a7c15ull + static_cast<std::uint64_t>(r)) % count_);
            std::memcpy(&X[static_cast<std::size_t>(r) * (F + 2)], &x_[pick * static_cast<std::size_t>(F + 2)], 4ull * (F + 2));
            for (int f = 0; f < F; ++f) {
                T[static_cast<std::size_t>(r) * F + f] = t_[pick * static_cast<std::size_t>(F) + f] / scale_[static_cast<std::size_t>(f)];
            }
        }
        const float lr = 2e-3f;
        const float b1 = 0.9f;
        const float b2 = 0.999f;
        if (gpu_) {
            Gpu& g = *gpu_;
            const cuda::Device& d = cuda::device();
            d.bind();
            d.api.cuMemcpyHtoDAsync(g.X.ptr(), X.data(), X.size() * 4, d.stream);
            d.api.cuMemcpyHtoDAsync(g.T.ptr(), T.data(), T.size() * 4, d.stream);
            CUdeviceptr w = g.W.ptr(), xp = g.X.ptr(), tp = g.T.ptr(), pre = g.pre.ptr(), del = g.del.ptr(), xn = g.xn.ptr(), lp = g.loss.ptr(),
                        gp = g.grad.ptr(), m1 = g.m1.ptr(), m2 = g.m2.ptr();
            CUdeviceptr none = 0;
            int n = rows;
            cuda::launch(g.rows, static_cast<unsigned>((rows + 127) / 128), 1, 1, 128, 1, 1, 0, {&w, &xp, &tp, &none, &n, &pre, &del, &xn, &lp});
            cuda::launch(g.grads, static_cast<unsigned>((g.params + 127) / 128), 1, 1, 128, 1, 1, 0, {&pre, &del, &xn, &n, &gp});
            ++g.adam_t;
            float lrv = lr;
            float b1v = b1;
            float b2v = b2;
            float c1 = 1.0f - std::pow(b1, static_cast<float>(g.adam_t));
            float c2 = 1.0f - std::pow(b2, static_cast<float>(g.adam_t));
            float clip = 5.0f;
            cuda::launch(g.adam, static_cast<unsigned>((g.params + 127) / 128), 1, 1, 128, 1, 1, 0, {&w, &gp, &m1, &m2, &lrv, &b1v, &b2v, &c1, &c2, &clip});
            if (it + 1 == iterations) {
                std::vector<float> losses(static_cast<std::size_t>(rows));
                d.api.cuMemcpyDtoHAsync(losses.data(), g.loss.ptr(), losses.size() * 4, d.stream);
                std::vector<float> image(static_cast<std::size_t>(g.image));
                d.api.cuMemcpyDtoHAsync(image.data(), g.W.ptr(), image.size() * 4, d.stream);
                d.api.cuStreamSynchronize(d.stream);
                for (const float v : losses) {
                    loss_sum += v;
                }
                cuda::net_from_image(net_, image);
            }
        } else {
            // The same on the host: forward with the tape, back, the rows summed in order, Adam.
            static thread_local std::vector<float> m1;
            static thread_local std::vector<float> m2;
            static thread_local std::uint64_t t = 0;
            const std::size_t P = net_.parameter_count();
            if (m1.size() != P) {
                m1.assign(P, 0.0f);
                m2.assign(P, 0.0f);
                t = 0;
            }
            std::vector<float> grad(P, 0.0f);
            std::vector<float> tape(2 * kHidden + F);
            std::vector<float> y(static_cast<std::size_t>(F));
            std::vector<float> dy(static_cast<std::size_t>(F));
            std::vector<float> scratch(2 * static_cast<std::size_t>(net_.widest()));
            double total = 0.0;
            for (int r = 0; r < rows; ++r) {
                const float* xr = &X[static_cast<std::size_t>(r) * (F + 2)];
                net_forward_taped(net_, xr, y.data(), tape.data());
                for (int f = 0; f < F; ++f) {
                    const float e = y[static_cast<std::size_t>(f)] - T[static_cast<std::size_t>(r) * F + f];
                    dy[static_cast<std::size_t>(f)] = e / rows;
                    total += 0.5 * e * e;
                }
                net_backward(net_, xr, tape.data(), dy.data(), grad.data(), nullptr, scratch.data());
            }
            ++t;
            const float c1 = 1.0f - std::pow(b1, static_cast<float>(t));
            const float c2 = 1.0f - std::pow(b2, static_cast<float>(t));
            for (std::size_t p = 0; p < P; ++p) {
                const float gv = std::clamp(grad[p], -5.0f, 5.0f);
                m1[p] = b1 * m1[p] + (1.0f - b1) * gv;
                m2[p] = b2 * m2[p] + (1.0f - b2) * gv * gv;
                net_.weights[p] -= lr * (m1[p] / c1) / (std::sqrt(m2[p] / c2) + 1e-8f);
            }
            if (it + 1 == iterations) {
                loss_sum = total;
            }
        }
        ++updates_;
    }
    loss_ = loss_sum / std::max(rows, 1);
    learn_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void WorldModel::rollout_cpu(const float* obs, std::uint32_t id, std::uint64_t tick, std::vector<float>& mean)
{
    const int F = features_;
    const int H = horizon_;
    const int K = std::min(samples_, kCpuSamples);
    std::vector<float> costs(static_cast<std::size_t>(K));
    std::vector<float> taken(static_cast<std::size_t>(K) * H * 2);
    std::vector<float> s(static_cast<std::size_t>(F));
    std::vector<float> ds(static_cast<std::size_t>(F));
    for (int k = 0; k < K; ++k) {
        s.assign(obs, obs + F);
        float J = 0.0f;
        for (int t = 0; t < H; ++t) {
            float a[2];
            for (int d = 0; d < 2; ++d) {
                const float e = k == 0 ? 0.0f : gauss(noise_key(id, tick, k, t, d));
                a[d] = std::clamp(mean[static_cast<std::size_t>(t * 2 + d)] + kSigma * e, -1.0f, 1.0f);
                taken[(static_cast<std::size_t>(k) * H + t) * 2 + d] = a[d];
            }
            predict(s.data(), a, ds.data());
            for (int f = 0; f < F; ++f) {
                s[static_cast<std::size_t>(f)] += static_cast<float>(dt_) * ds[static_cast<std::size_t>(f)];
            }
            J += cost(s.data()) + 0.01f * (a[0] * a[0] + a[1] * a[1]);
        }
        costs[static_cast<std::size_t>(k)] = J;
    }
    float lo = 1e30f;
    float sum = 0.0f;
    for (const float c : costs) {
        lo = std::min(lo, c);
        sum += c;
    }
    const float lam = std::max(1e-4f, 0.25f * (sum / K - lo));
    for (int q = 0; q < H * 2; ++q) {
        float w = 0.0f;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float e = std::exp(-(costs[static_cast<std::size_t>(k)] - lo) / lam);
            w += e;
            acc += e * taken[static_cast<std::size_t>(k) * H * 2 + static_cast<std::size_t>(q)];
        }
        mean[static_cast<std::size_t>(q)] = acc / w;
    }
}

void WorldModel::plan(const std::vector<float>& obs, const std::vector<std::uint32_t>& ids, std::uint64_t tick, std::vector<float>& actions)
{
    const auto started = std::chrono::steady_clock::now();
    const int n = static_cast<int>(ids.size());
    const int F = features_;
    const int H = horizon_;
    actions.assign(2 * static_cast<std::size_t>(n), 0.0f);
    if (n == 0) {
        return;
    }
    std::vector<float> means(static_cast<std::size_t>(n) * H * 2, 0.0f);
    for (int i = 0; i < n; ++i) {
        const auto found = mean_.find(ids[static_cast<std::size_t>(i)]);
        if (found != mean_.end()) {
            std::copy(found->second.begin(), found->second.end(), means.begin() + static_cast<std::ptrdiff_t>(i) * H * 2);
        }
    }
    if (gpu_) {
        Gpu& g = *gpu_;
        const cuda::Device& d = cuda::device();
        d.bind();
        if (n > g.agents) {
            g.agents = std::max(n, 2 * g.agents);
            g.obs.allocate(4ull * g.agents * F);
            g.mean.allocate(4ull * g.agents * H * 2);
            g.eps.allocate(4ull * g.agents * samples_ * H * 2);
            g.cost.allocate(4ull * g.agents * samples_);
            g.ids.allocate(4ull * g.agents);
        }
        d.api.cuMemcpyHtoDAsync(g.obs.ptr(), obs.data(), 4ull * n * F, d.stream);
        d.api.cuMemcpyHtoDAsync(g.mean.ptr(), means.data(), means.size() * 4, d.stream);
        d.api.cuMemcpyHtoDAsync(g.ids.ptr(), ids.data(), 4ull * n, d.stream);
        CUdeviceptr w = g.W.ptr(), sc = g.scale.ptr(), op = g.obs.ptr(), mp = g.mean.ptr(), ip = g.ids.ptr(), ep = g.eps.ptr(), cp = g.cost.ptr();
        int K = samples_;
        unsigned long long t = tick;
        float sigma = kSigma;
        float dt = static_cast<float>(dt_);
        const unsigned shared = static_cast<unsigned>(4 * (g.image + F));
        cuda::launch(g.rollout, static_cast<unsigned>(n), 1, 1, static_cast<unsigned>(K), 1, 1, shared, {&w, &sc, &op, &mp, &ip, &ep, &cp, &K, &t, &sigma, &dt});
        cuda::launch(g.update, static_cast<unsigned>(n), 1, 1, static_cast<unsigned>((H * 2 + 31) / 32 * 32), 1, 1, 0, {&cp, &ep, &mp, &K});
        d.api.cuMemcpyDtoHAsync(means.data(), g.mean.ptr(), means.size() * 4, d.stream);
        d.api.cuStreamSynchronize(d.stream);
    } else {
        for (int i = 0; i < n; ++i) {
            std::vector<float> m(means.begin() + static_cast<std::ptrdiff_t>(i) * H * 2, means.begin() + static_cast<std::ptrdiff_t>(i + 1) * H * 2);
            rollout_cpu(&obs[static_cast<std::size_t>(i) * F], ids[static_cast<std::size_t>(i)], tick, m);
            std::copy(m.begin(), m.end(), means.begin() + static_cast<std::ptrdiff_t>(i) * H * 2);
        }
    }
    for (int i = 0; i < n; ++i) {
        float* m = &means[static_cast<std::size_t>(i) * H * 2];
        // What it does is its plan, and a little more while it knows little: without trying, nothing is learned
        // about what trying does. The noise fades as it learns (a hash, like the planner's).
        const float explore = 0.5f / (1.0f + static_cast<float>(updates_) / 200.0f);
        float ax = m[0] + explore * gauss(noise_key(ids[static_cast<std::size_t>(i)], tick, 4095, 63, 0));
        float ay = m[1] + explore * gauss(noise_key(ids[static_cast<std::size_t>(i)], tick, 4095, 63, 1));
        const float norm = std::sqrt(ax * ax + ay * ay);
        if (norm > 1.0f) {
            ax /= norm;
            ay /= norm;
        }
        actions[2 * static_cast<std::size_t>(i)] = ax;
        actions[2 * static_cast<std::size_t>(i) + 1] = ay;
        Last& last = last_[ids[static_cast<std::size_t>(i)]];
        last.action[0] = ax;
        last.action[1] = ay;
        last.acted = true;
        // Next time it starts from the rest of this plan.
        std::vector<float> next(m + 2, m + H * 2);
        next.push_back(0.0f);
        next.push_back(0.0f);
        mean_[ids[static_cast<std::size_t>(i)]] = std::move(next);
    }
    plan_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

} // namespace fieldengine::field

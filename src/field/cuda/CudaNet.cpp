#include "field/cuda/CudaNet.hpp"

#include "field/cuda/CudaApi.hpp"

#include <algorithm>
#include <sstream>

namespace fieldengine::field::cuda {

namespace {

int pad(int v, int to)
{
    return (v + to - 1) / to * to;
}

const char* act_fn(Activation a)
{
    switch (a) {
    case Activation::Tanh:
        return "act_tanh";
    case Activation::Relu:
        return "act_relu";
    case Activation::Silu:
        return "act_silu";
    case Activation::Linear:
        return "act_linear";
    }
    return "act_linear";
}

const char* act_grad_fn(Activation a)
{
    switch (a) {
    case Activation::Tanh:
        return "grad_tanh";
    case Activation::Relu:
        return "grad_relu";
    case Activation::Silu:
        return "grad_silu";
    case Activation::Linear:
        return "grad_linear";
    }
    return "grad_linear";
}

// Small layers are unrolled whole (their activations stay in registers); bigger ones only by 8 (the activations then
// live in local memory, which L1 holds). Measured on the 5-32-32-4 net of ldiffuse over 1024^2 cells (sm_86): with the
// 32 x 32 layer unrolled whole (the old limit was 2048 weights) the stage took 101 ms, unrolled by 8 it takes 8 ms; a
// 64-64 net (already by 8) ran faster than the 32-32 one did. Bigger layers also take NVRTC minutes to compile whole.
constexpr int kWholeUnroll = 512;

const char* outer_unroll(int ni, int no)
{
    return ni * no > kWholeUnroll ? "#pragma unroll 1\n" : "#pragma unroll\n";
}

const char* inner_unroll(int ni, int no)
{
    return ni * no > kWholeUnroll ? "#pragma unroll 8\n" : "#pragma unroll\n";
}

} // namespace

const char* net_prelude()
{
    return R"(
#ifndef FIELD_FRAG2
#define FIELD_FRAG2
struct alignas(8) frag2 { unsigned x, y; };
#endif
__device__ __forceinline__ float act_tanh(float x) { return tanhf(x); }
__device__ __forceinline__ float act_relu(float x) { return x > 0.f ? x : 0.f; }
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_linear(float x) { return x; }
__device__ __forceinline__ float grad_tanh(float x) { float t = tanhf(x); return 1.f - t * t; }
__device__ __forceinline__ float grad_relu(float x) { return x > 0.f ? 1.f : 0.f; }
__device__ __forceinline__ float grad_silu(float x) { float s = 1.f / (1.f + expf(-x)); return s * (1.f + x * (1.f - s)); }
__device__ __forceinline__ float grad_linear(float) { return 1.f; }
// Two floats as one f16x2 register, the first in the low half.
__device__ __forceinline__ unsigned pack_h2(float lo, float hi)
{
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// D = A(16x16, f16, row) * B(16x8, f16, col) + D (f32): one warp, in place.
__device__ __forceinline__ void mma16816(float* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
)";
}

NetLayout net_layout(const NetDecl& net)
{
    NetLayout l;
    l.in = net.inputs();
    l.out = net.outputs();
    l.widest = net.widest();
    l.weights_at = pad(2 * l.in + 1, 4);
    const int layers = net.layers();
    int at = l.weights_at;
    for (int k = 0; k < layers; ++k) {
        l.layer_at.push_back(at);
        at += (net.sizes[static_cast<std::size_t>(k)] + 1) * net.sizes[static_cast<std::size_t>(k) + 1];
    }
    l.image_floats = at;
    l.tensor = net.half;
    int frag = 0;
    for (int k = 0; k < layers; ++k) {
        const int kp = pad(net.sizes[static_cast<std::size_t>(k)], 16);
        const int np = k + 1 == layers ? pad(net.sizes[static_cast<std::size_t>(k) + 1], 8) : pad(net.sizes[static_cast<std::size_t>(k) + 1], 16);
        l.k_pad.push_back(kp);
        l.n_pad.push_back(np);
        l.frag_at.push_back(frag);
        frag += (kp / 16) * (np / 8) * 32;
    }
    l.frag_words = 2 * frag;
    return l;
}

std::vector<float> net_image(const NetDecl& net)
{
    const NetLayout l = net_layout(net);
    std::vector<float> image(static_cast<std::size_t>(l.image_floats), 0.0f);
    for (int i = 0; i < l.in; ++i) {
        image[static_cast<std::size_t>(i)] = i < static_cast<int>(net.in_shift.size()) ? net.in_shift[static_cast<std::size_t>(i)] : 0.0f;
        image[static_cast<std::size_t>(l.in + i)] = i < static_cast<int>(net.in_scale.size()) ? net.in_scale[static_cast<std::size_t>(i)] : 1.0f;
    }
    image[static_cast<std::size_t>(2 * l.in)] = net.out_scale;
    std::copy(net.weights.begin(), net.weights.end(), image.begin() + l.weights_at);
    return image;
}

void net_from_image(NetDecl& net, const std::vector<float>& image)
{
    const NetLayout l = net_layout(net);
    net.weights.assign(image.begin() + l.weights_at, image.begin() + l.image_floats);
}

std::vector<std::uint32_t> net_fragments(const NetDecl& net)
{
    const NetLayout l = net_layout(net);
    std::vector<std::uint32_t> out(static_cast<std::size_t>(l.frag_words), 0u);
    const int layers = net.layers();
    for (int k = 0; k < layers; ++k) {
        const int in = net.sizes[static_cast<std::size_t>(k)];
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const std::size_t off = net.weight_offset(k);
        auto w = [&](int n, int kk) { return n < no && kk < in ? net.weights[off + static_cast<std::size_t>(n) * static_cast<std::size_t>(in) + static_cast<std::size_t>(kk)] : 0.0f; };
        auto pair = [&](float lo, float hi) { return static_cast<std::uint32_t>(to_half(lo)) | (static_cast<std::uint32_t>(to_half(hi)) << 16); };
        const int kt_n = l.k_pad[static_cast<std::size_t>(k)] / 16;
        const int nt_n = l.n_pad[static_cast<std::size_t>(k)] / 8;
        for (int kt = 0; kt < kt_n; ++kt) {
            for (int nt = 0; nt < nt_n; ++nt) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int g = lane >> 2;
                    const int t = lane & 3;
                    const int n = nt * 8 + g;
                    const int k0 = kt * 16 + 2 * t;
                    const std::size_t at = 2 * (static_cast<std::size_t>(l.frag_at[static_cast<std::size_t>(k)]) +
                        (static_cast<std::size_t>(kt) * static_cast<std::size_t>(nt_n) + static_cast<std::size_t>(nt)) * 32 + static_cast<std::size_t>(lane));
                    out[at] = pair(w(n, k0), w(n, k0 + 1));
                    out[at + 1] = pair(w(n, k0 + 8), w(n, k0 + 9));
                }
            }
        }
    }
    return out;
}

std::string net_device_code(const NetDecl& net, const std::string& id)
{
    const NetLayout l = net_layout(net);
    const int layers = net.layers();
    std::ostringstream s;
    s << "// ---- net " << net.name << ": ";
    for (std::size_t k = 0; k < net.sizes.size(); ++k) {
        s << (k ? "-" : "") << net.sizes[k];
    }
    s << ", " << to_string(net.act) << (net.half ? ", tensor cores" : "") << "\n";
    s << "constexpr int " << id << "_in = " << l.in << ", " << id << "_out = " << l.out << ", " << id << "_image = " << l.image_floats
      << ", " << id << "_frags = " << l.frag_words << ";\n";

    // ---- fp32, one row: the reference's arithmetic in its order.
    s << "__device__ __forceinline__ void " << id << "_fwd(const float* __restrict__ w, const float* x, float* y)\n{\n";
    s << "    float a[" << l.widest << "], b[" << l.widest << "];\n";
    s << "#pragma unroll\n    for (int i = 0; i < " << l.in << "; ++i) a[i] = (x[i] - w[i]) * w[" << l.in << " + i];\n";
    for (int k = 0; k < layers; ++k) {
        const int ni = net.sizes[static_cast<std::size_t>(k)];
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const int at = l.layer_at[static_cast<std::size_t>(k)];
        const bool last = k + 1 == layers;
        s << outer_unroll(ni, no) << "    for (int o = 0; o < " << no << "; ++o) {\n"
          << "        float acc = w[" << at + ni * no << " + o];\n"
          << inner_unroll(ni, no) << "        for (int i = 0; i < " << ni << "; ++i) acc += w[" << at << " + o * " << ni << " + i] * a[i];\n";
        if (last) {
            s << "        y[o] = acc * w[" << 2 * l.in << "];\n    }\n";
        } else {
            s << "        b[o] = " << act_fn(net.act) << "(acc);\n    }\n";
            s << "#pragma unroll\n    for (int o = 0; o < " << no << "; ++o) a[o] = b[o];\n";
        }
    }
    s << "}\n";

    // ---- the same with every pre-activation kept (rows of the trainer).
    int taped = 0;
    for (int k = 0; k < layers; ++k) {
        taped += net.sizes[static_cast<std::size_t>(k) + 1];
    }
    s << "constexpr int " << id << "_taped = " << taped << ";\n";

    if (!l.tensor) {
        return s.str();
    }
    // ---- tensor cores: a warp, 32 rows of xs (raw inputs, stride k_pad[0]) -> ys (stride n_pad[last]).
    const int kp0 = l.k_pad[0];
    const int npl = l.n_pad.back();
    s << "constexpr int " << id << "_xs = " << kp0 << ", " << id << "_ys = " << npl << ";\n";
    s << "__device__ __forceinline__ void " << id << "_tc(const float* __restrict__ w, const frag2* __restrict__ frag, const float* xs, float* ys)\n{\n"
      << "    const int lane = threadIdx.x & 31; const int g = lane >> 2; const int t = lane & 3;\n"
      << "#pragma unroll\n    for (int mt = 0; mt < 2; ++mt) {\n"
      << "        const int r0 = mt * 16 + g; const int r1 = r0 + 8;\n";
    // Layer 0's operand from the normalised inputs.
    const int kt0 = kp0 / 16;
    s << "        unsigned a0[" << kt0 << "][4];\n"
      << "#pragma unroll\n        for (int kt = 0; kt < " << kt0 << "; ++kt) {\n"
      << "            const int k0 = kt * 16 + 2 * t;\n"
      << "            float v[8];\n"
      << "#pragma unroll\n            for (int q = 0; q < 8; ++q) {\n"
      << "                const int k = k0 + (q & 1) + ((q >> 2) << 3); const int r = (q & 2) ? r1 : r0;\n"
      << "                v[q] = k < " << l.in << " ? (xs[r * " << kp0 << " + k] - w[k]) * w[" << l.in << " + k] : 0.f;\n"
      << "            }\n"
      << "            a0[kt][0] = pack_h2(v[0], v[1]); a0[kt][1] = pack_h2(v[2], v[3]); a0[kt][2] = pack_h2(v[4], v[5]); a0[kt][3] = pack_h2(v[6], v[7]);\n"
      << "        }\n";
    for (int k = 0; k < layers; ++k) {
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const int ni = net.sizes[static_cast<std::size_t>(k)];
        const int kt_n = l.k_pad[static_cast<std::size_t>(k)] / 16;
        const int nt_n = l.n_pad[static_cast<std::size_t>(k)] / 8;
        const int bias = l.layer_at[static_cast<std::size_t>(k)] + ni * no;
        const bool last = k + 1 == layers;
        s << "        float c" << k << "[" << nt_n << "][4];\n"
          << "#pragma unroll\n        for (int nt = 0; nt < " << nt_n << "; ++nt) {\n"
          << "            const int col = nt * 8 + 2 * t;\n"
          << "            const float b0 = col < " << no << " ? w[" << bias << " + col] : 0.f; const float b1 = col + 1 < " << no << " ? w[" << bias << " + col + 1] : 0.f;\n"
          << "            c" << k << "[nt][0] = b0; c" << k << "[nt][1] = b1; c" << k << "[nt][2] = b0; c" << k << "[nt][3] = b1;\n"
          << "#pragma unroll\n            for (int kt = 0; kt < " << kt_n << "; ++kt) {\n"
          << "                const frag2 f = frag[" << l.frag_at[static_cast<std::size_t>(k)] << " + (kt * " << nt_n << " + nt) * 32 + lane];\n"
          << "                mma16816(c" << k << "[nt], a" << k << "[kt], f.x, f.y);\n"
          << "            }\n"
          << "        }\n";
        if (!last) {
            const int kt_next = nt_n / 2;
            s << "        unsigned a" << k + 1 << "[" << kt_next << "][4];\n"
              << "#pragma unroll\n        for (int kt = 0; kt < " << kt_next << "; ++kt) {\n"
              << "            float* p = c" << k << "[2 * kt]; float* q = c" << k << "[2 * kt + 1];\n"
              << "            a" << k + 1 << "[kt][0] = pack_h2(" << act_fn(net.act) << "(p[0]), " << act_fn(net.act) << "(p[1]));\n"
              << "            a" << k + 1 << "[kt][1] = pack_h2(" << act_fn(net.act) << "(p[2]), " << act_fn(net.act) << "(p[3]));\n"
              << "            a" << k + 1 << "[kt][2] = pack_h2(" << act_fn(net.act) << "(q[0]), " << act_fn(net.act) << "(q[1]));\n"
              << "            a" << k + 1 << "[kt][3] = pack_h2(" << act_fn(net.act) << "(q[2]), " << act_fn(net.act) << "(q[3]));\n"
              << "        }\n";
        } else {
            s << "        const float scale = w[" << 2 * l.in << "];\n"
              << "#pragma unroll\n        for (int nt = 0; nt < " << nt_n << "; ++nt) {\n"
              << "            const int col = nt * 8 + 2 * t;\n"
              << "            ys[r0 * " << npl << " + col] = c" << k << "[nt][0] * scale; ys[r0 * " << npl << " + col + 1] = c" << k << "[nt][1] * scale;\n"
              << "            ys[r1 * " << npl << " + col] = c" << k << "[nt][2] * scale; ys[r1 * " << npl << " + col + 1] = c" << k << "[nt][3] * scale;\n"
              << "        }\n";
        }
    }
    s << "    }\n    __syncwarp();\n}\n";
    return s.str();
}

std::string net_train_code(const NetDecl& net, const std::string& id)
{
    const NetLayout l = net_layout(net);
    const int layers = net.layers();
    std::ostringstream s;
    std::vector<int> taped_at;
    int taped = 0;
    for (int k = 0; k < layers; ++k) {
        taped_at.push_back(taped);
        taped += net.sizes[static_cast<std::size_t>(k) + 1];
    }
    // Rows: forward with every pre-activation kept, the loss, and the deltas back to the first layer. A row a thread.
    s << "extern \"C\" __global__ void " << id << "_train_rows(const float* __restrict__ w, const float* __restrict__ X,\n"
      << "    const float* __restrict__ T, const float* __restrict__ M, int rows, float* PRE, float* DEL, float* XN, float* LOSS)\n{\n"
      << "    const int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;\n"
      << "    float a[" << l.widest << "]; float pre[" << taped << "]; float del[" << taped << "];\n"
      << "#pragma unroll\n    for (int i = 0; i < " << l.in << "; ++i) { a[i] = (X[r * " << l.in << " + i] - w[i]) * w[" << l.in << " + i]; XN[r * " << l.in << " + i] = a[i]; }\n";
    for (int k = 0; k < layers; ++k) {
        const int ni = net.sizes[static_cast<std::size_t>(k)];
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const int at = l.layer_at[static_cast<std::size_t>(k)];
        const int tp = taped_at[static_cast<std::size_t>(k)];
        s << outer_unroll(ni, no) << "    for (int o = 0; o < " << no << "; ++o) { float acc = w[" << at + ni * no << " + o];\n"
          << inner_unroll(ni, no) << "        for (int i = 0; i < " << ni << "; ++i) acc += w[" << at << " + o * " << ni << " + i] * a[i];\n"
          << "        pre[" << tp << " + o] = acc; }\n";
        if (k + 1 < layers) {
            s << "#pragma unroll\n    for (int o = 0; o < " << no << "; ++o) a[o] = " << act_fn(net.act) << "(pre[" << tp << " + o]);\n";
        }
    }
    const int last_tp = taped_at.back();
    s << "    const float scale = w[" << 2 * l.in << "]; const float m = M ? M[r] : 1.f; float loss = 0.f;\n"
      << "#pragma unroll\n    for (int o = 0; o < " << l.out << "; ++o) {\n"
      << "        const float e = pre[" << last_tp << " + o] * scale - T[r * " << l.out << " + o];\n"
      << "        loss += 0.5f * m * e * e; del[" << last_tp << " + o] = m * e * scale; }\n";
    for (int k = layers - 1; k > 0; --k) {
        const int ni = net.sizes[static_cast<std::size_t>(k)];
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const int at = l.layer_at[static_cast<std::size_t>(k)];
        const int tp = taped_at[static_cast<std::size_t>(k)];
        const int below = taped_at[static_cast<std::size_t>(k) - 1];
        s << outer_unroll(ni, no) << "    for (int i = 0; i < " << ni << "; ++i) { float acc = 0.f;\n"
          << inner_unroll(ni, no) << "        for (int o = 0; o < " << no << "; ++o) acc += w[" << at << " + o * " << ni << " + i] * del[" << tp << " + o];\n"
          << "        del[" << below << " + i] = acc * " << act_grad_fn(net.act) << "(pre[" << below << " + i]); }\n";
    }
    s << "#pragma unroll\n    for (int k = 0; k < " << taped << "; ++k) { PRE[(unsigned long long)r * " << taped << " + k] = pre[k]; DEL[(unsigned long long)r * " << taped << " + k] = del[k]; }\n"
      << "    LOSS[r] = loss;\n}\n";

    // Weight gradients: a thread a parameter, the rows summed in order (no atomics: the same batch, the same bits).
    int params = 0;
    for (int k = 0; k < layers; ++k) {
        params += (net.sizes[static_cast<std::size_t>(k)] + 1) * net.sizes[static_cast<std::size_t>(k) + 1];
    }
    s << "constexpr int " << id << "_params = " << params << ";\n"
      << "extern \"C\" __global__ void " << id << "_train_grad(const float* __restrict__ PRE, const float* __restrict__ DEL,\n"
      << "    const float* __restrict__ XN, int rows, float* G)\n{\n"
      << "    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= " << params << ") return;\n"
      << "    const int at = p; float acc = 0.f;\n";
    int start = 0;
    for (int k = 0; k < layers; ++k) {
        const int ni = net.sizes[static_cast<std::size_t>(k)];
        const int no = net.sizes[static_cast<std::size_t>(k) + 1];
        const int tp = taped_at[static_cast<std::size_t>(k)];
        const int count = (ni + 1) * no;
        s << "    " << (k ? "else " : "") << "if (p < " << start + count << ") {\n"
          << "        const int q = p - " << start << ";\n"
          << "        if (q < " << ni * no << ") { const int o = q / " << ni << "; const int i = q % " << ni << ";\n"
          << "            for (int r = 0; r < rows; ++r) {\n"
          << "                const float input = ";
        if (k == 0) {
            s << "XN[(unsigned long long)r * " << l.in << " + i];\n";
        } else {
            s << act_fn(net.act) << "(PRE[(unsigned long long)r * " << taped << " + " << taped_at[static_cast<std::size_t>(k) - 1] << " + i]);\n";
        }
        s << "                acc += DEL[(unsigned long long)r * " << taped << " + " << tp << " + o] * input; }\n"
          << "        } else { const int o = q - " << ni * no << ";\n"
          << "            for (int r = 0; r < rows; ++r) acc += DEL[(unsigned long long)r * " << taped << " + " << tp << " + o]; }\n"
          << "    }\n";
        start += count;
    }
    s << "    G[at] = acc / (float)(rows > 0 ? rows : 1);\n}\n";

    // Adam on the weights (the normalisation in front of them is the trainer's choice, not learned here).
    s << "extern \"C\" __global__ void " << id << "_adam(float* w, const float* __restrict__ G, float* M1, float* M2, float lr,\n"
      << "    float b1, float b2, float c1, float c2, float clip)\n{\n"
      << "    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= " << params << ") return;\n"
      << "    float g = G[p]; g = fminf(fmaxf(g, -clip), clip);\n"
      << "    const float m = b1 * M1[p] + (1.f - b1) * g; const float v = b2 * M2[p] + (1.f - b2) * g * g;\n"
      << "    M1[p] = m; M2[p] = v;\n"
      << "    w[" << l.weights_at << " + p] -= lr * (m / c1) / (sqrtf(v / c2) + 1e-8f);\n}\n";
    return s.str();
}

} // namespace fieldengine::field::cuda

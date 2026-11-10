#include "field/cuda/CudaCodegen.hpp"

#include "field/cuda/CudaNet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

namespace fieldengine::field::cuda {

const char* law_prelude()
{
    return R"(
typedef unsigned long long u64;
#ifndef FIELD_FRAG2
#define FIELD_FRAG2
struct alignas(8) frag2 { unsigned x, y; };
#endif
#define FAR 1e30f
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
struct Grid {
    int nx, ny, nz, nxy, cells, ox, oy, oz;
    float h;
    int pad;
    float* S;        // the fields, a ring: logical cell (x, y, z) lives at I(x, y, z)
    float* A;        // stages: stored values, distance and poisson buffers, net face fluxes
    float* D;        // deltas, for terms that cannot store in place
    float* U;        // reductions: one number each
    unsigned* C;     // per field: did any cell change
    float* L;        // linear staging: uploads, downloads, partial sums
    unsigned* R;     // contract records
    float* W;        // net weights (fp32 images)
    unsigned* F;     // re-run flags of distance and poisson
    const frag2* NF; // net weights as tensor-core fragments
};
__constant__ Grid G;
__constant__ float P[1024];
// Where each field lives now, and the buffer a term that reads its own neighbours writes its next values to: after it
// runs, the host swaps the two (ping-pong), so the step moves one read and one write a cell, not a delta through D.
struct FieldPair { float* now; float* next; };
__constant__ FieldPair FT[256];

__device__ __forceinline__ int I(int x, int y, int z)
{
    int px = x + G.ox; if (px >= G.nx) px -= G.nx;
    int py = y + G.oy; if (py >= G.ny) py -= G.ny;
    int pz = z + G.oz; if (pz >= G.nz) pz -= G.nz;
    return px + py * G.nx + pz * G.nxy;
}
__device__ __forceinline__ int ex(int v) { return v < 0 ? 0 : (v >= G.nx ? G.nx - 1 : v); }
__device__ __forceinline__ int ey(int v) { return v < 0 ? 0 : (v >= G.ny ? G.ny - 1 : v); }
__device__ __forceinline__ int ez(int v) { return v < 0 ? 0 : (v >= G.nz ? G.nz - 1 : v); }
__device__ __forceinline__ bool inside(int x, int y, int z) { return x >= 0 && y >= 0 && z >= 0 && x < G.nx && y < G.ny && z < G.nz; }
__device__ __forceinline__ float dv(float a, float b) { return b == 0.f ? 0.f : a / b; }
__device__ __forceinline__ float sat(float v) { return fminf(fmaxf(v, 0.f), 1.f); }
__device__ __forceinline__ float face(float ka, float kb, float va, float vb)
{
    const float s = ka + kb;
    return (s > 0.f ? 2.f * ka * kb / s : 0.f) * (vb - va);
}
#define AT(slot, x, y, z) G.A[(u64)(slot) * G.cells + I(x, y, z)]
#define FIELD(f, x, y, z) FT[f].now[I(x, y, z)]
#define CELL_HEAD const int x = blockIdx.x * blockDim.x + threadIdx.x; const int y = blockIdx.y * blockDim.y + threadIdx.y; \
    const int z = blockIdx.z; if (x >= G.nx || y >= G.ny) return;

// Small sums that do not lose the small to the large (Neumaier).
struct Sum { float s; float c; };
__device__ __forceinline__ void add(Sum& a, float v)
{
    const float t = a.s + v;
    const float low = fabsf(a.s) >= fabsf(v) ? (a.s - t) + v : (v - t) + a.s;
    a.c += low;
    a.s = t;
}

// The flux limiter of flux(): the same on both sides of the face, so it stays antisymmetric.
__device__ __forceinline__ float limit_face(float r, float fi, float fj, float lo, float hi, float per)
{
    if (per <= 0.f) return r;
    const bool has_lo = lo > -1e29f; const bool has_hi = hi < 1e29f;
    if (r > 0.f) {
        if (has_lo) r = fminf(r, fmaxf(0.f, (fi - lo) * per));
        if (has_hi) r = fminf(r, fmaxf(0.f, (hi - fj) * per));
    } else if (r < 0.f) {
        if (has_lo) r = fmaxf(r, -fmaxf(0.f, (fj - lo) * per));
        if (has_hi) r = fmaxf(r, -fmaxf(0.f, (hi - fi) * per));
    }
    return r;
}

// "Field t changed this step": one flag a field, set by one lane a warp at most (a ballot), not by every cell that
// changed, which would all write the same word.
__device__ __forceinline__ void mark(int t, bool changed)
{
    const unsigned m = __activemask();
    const unsigned any = __ballot_sync(m, changed);
    const unsigned lane = (threadIdx.x + threadIdx.y * blockDim.x) & 31u;
    if (any != 0u && lane == (unsigned)(__ffs(m) - 1) && G.C[t] == 0u) G.C[t] = 1u;
}

extern "C" __global__ void k_fill(int slot, unsigned bits)
{
    CELL_HEAD
    ((unsigned*)G.A)[(u64)slot * G.cells + I(x, y, z)] = bits;
}

// One axis of the squared distance transform: outward from each cell until no farther cell can beat what is found.
// Exact, cheap near a surface, and only run when a seed changed.
extern "C" __global__ void k_scan(int from, int to, int axis, int flag)
{
    if (G.F[flag] == 0u) return;
    CELL_HEAD
    const int n = axis == 0 ? G.nx : (axis == 1 ? G.ny : G.nz);
    const int q = axis == 0 ? x : (axis == 1 ? y : z);
    float best = FAR;
    for (int r = 0; r < n; ++r) {
        const float rr = (float)(r * r);
        if (rr >= best) break;
        const int lo = q - r; const int hi = q + r;
        if (lo < 0 && hi >= n) break;
        if (lo >= 0) best = fminf(best, AT(from, axis == 0 ? lo : x, axis == 1 ? lo : y, axis == 2 ? lo : z) + rr);
        if (hi < n) best = fminf(best, AT(from, axis == 0 ? hi : x, axis == 1 ? hi : y, axis == 2 ? hi : z) + rr);
    }
    AT(to, x, y, z) = best;
}

// to-inside, to-outside, the seed -> signed metres. The surface lies half a cell from the centre of the last solid cell,
// and beyond the lattice is air.
extern "C" __global__ void k_signed(int to_in, int to_out, int seed, int result, int flag)
{
    if (G.F[flag] == 0u) return;
    CELL_HEAD
    const float far_m = (G.nx + G.ny + G.nz) * G.h;
    const bool in = AT(seed, x, y, z) == 0.f;
    float d;
    if (in) {
        float border = 1e30f;
        if (G.nx > 1) border = fminf(border, (float)imin(x + 1, G.nx - x));
        if (G.ny > 1) border = fminf(border, (float)imin(y + 1, G.ny - y));
        if (G.nz > 1) border = fminf(border, (float)imin(z + 1, G.nz - z));
        const float o = AT(to_out, x, y, z);
        d = -(fminf(o < 1e29f ? sqrtf(o) : far_m / G.h, border) - 0.5f) * G.h;
    } else {
        const float o = AT(to_in, x, y, z);
        d = o < 1e29f ? (sqrtf(o) - 0.5f) * G.h : far_m;
    }
    AT(result, x, y, z) = d;
}

// One Jacobi sweep: phi (from) -> phi (to), the source in `source`; phi = 0 outside the lattice.
extern "C" __global__ void k_jacobi(int from, int to, int source, int flag)
{
    if (G.F[flag] == 0u) return;
    CELL_HEAD
    float sum = 0.f; int axes = 0;
    if (G.nx > 1) { ++axes; if (x > 0) sum += AT(from, x - 1, y, z); if (x + 1 < G.nx) sum += AT(from, x + 1, y, z); }
    if (G.ny > 1) { ++axes; if (y > 0) sum += AT(from, x, y - 1, z); if (y + 1 < G.ny) sum += AT(from, x, y + 1, z); }
    if (G.nz > 1) { ++axes; if (z > 0) sum += AT(from, x, y, z - 1); if (z + 1 < G.nz) sum += AT(from, x, y, z + 1); }
    AT(to, x, y, z) = (sum - G.h * G.h * AT(source, x, y, z)) / (2.f * (float)imax(axes, 1));
}

extern "C" __global__ void k_copy(int from, int to, int flag)
{
    if (G.F[flag] == 0u) return;
    CELL_HEAD
    AT(to, x, y, z) = AT(from, x, y, z);
}

// A field's range contract: only a cell that breaks it touches the counters (integer: the order does not matter).
// Three words: out of range, not finite, ~(the first bad cell) so the least index wins.
extern "C" __global__ void k_range(int field, int slot, float lo, float hi)
{
    CELL_HEAD
    const float v = FIELD(field, x, y, z);
    const bool nan = (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u;
    if (nan || v < lo || v > hi) {
        atomicAdd(&G.R[slot + (nan ? 1 : 0)], 1u);
        atomicMax(&G.R[slot + 2], ~(unsigned)(x + y * G.nx + z * G.nxy));
    }
}

// A field's total in two levels, in a fixed order: every row, then the rows of each plane. A warp a row: lane l sums
// x = l, l + 32, ... (coalesced reads), then the lanes are folded in a fixed tree (16, 8, 4, 2, 1), so the bits of a
// total do not depend on timing. The launch covers whole warps, so a warp returns or works as one.
extern "C" __global__ void k_rows(int field)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = t >> 5; const int lane = t & 31;
    if (r >= G.ny * G.nz) return;
    const int y = r % G.ny; const int z = r / G.ny;
    Sum sum = {0.f, 0.f}; Sum mag = {0.f, 0.f};
    for (int x = lane; x < G.nx; x += 32) { const float v = FIELD(field, x, y, z); add(sum, v); add(mag, fabsf(v)); }
    for (int o = 16; o > 0; o >>= 1) {
        const float s2 = __shfl_down_sync(0xffffffffu, sum.s, o); const float c2 = __shfl_down_sync(0xffffffffu, sum.c, o);
        const float m2 = __shfl_down_sync(0xffffffffu, mag.s, o); const float d2 = __shfl_down_sync(0xffffffffu, mag.c, o);
        add(sum, s2); sum.c += c2; add(mag, m2); mag.c += d2;
    }
    if (lane == 0) { G.L[2 * r] = sum.s + sum.c; G.L[2 * r + 1] = mag.s + mag.c; }
}

extern "C" __global__ void k_planes(int slot)
{
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= G.nz) return;
    Sum sum = {0.f, 0.f}; Sum mag = {0.f, 0.f};
    for (int y = 0; y < G.ny; ++y) { add(sum, G.L[2 * (z * G.ny + y)]); add(mag, G.L[2 * (z * G.ny + y) + 1]); }
    G.R[slot + 2 * z] = __float_as_uint(sum.s + sum.c); G.R[slot + 2 * z + 1] = __float_as_uint(mag.s + mag.c);
}

extern "C" __global__ void k_clamp(int field, float lo, float hi)
{
    CELL_HEAD
    float& v = FIELD(field, x, y, z);
    if (v < lo) v = lo; else if (v > hi) v = hi;
}

// The ring to and from the host's linear order.
extern "C" __global__ void k_linearize(int field)
{
    CELL_HEAD
    G.L[x + y * G.nx + z * G.nxy] = FIELD(field, x, y, z);
}

extern "C" __global__ void k_scatter(int field, int x0, int y0, int z0, int bx, int by, int bz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; const int j = blockIdx.y * blockDim.y + threadIdx.y; const int k = blockIdx.z;
    if (i >= bx || j >= by || k >= bz) return;
    FIELD(field, x0 + i, y0 + j, z0 + k) = G.L[i + j * bx + k * bx * by];
}

// The planes of a reduction (sum, least, most each) into one number.
extern "C" __global__ void k_reduce_final(int kind, int uslot)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    Sum sum = {0.f, 0.f}; float lo = G.L[1]; float hi = G.L[2];
    for (int z = 0; z < G.nz; ++z) { add(sum, G.L[3 * z]); lo = fminf(lo, G.L[3 * z + 1]); hi = fmaxf(hi, G.L[3 * z + 2]); }
    G.U[uslot] = kind == 0 ? sum.s + sum.c : (kind == 1 ? hi : lo);
}
)";
}

namespace {

std::string literal(double v)
{
    if (!std::isfinite(v)) {
        return v > 0 ? "1e30f" : (v < 0 ? "(-1e30f)" : "0.f");
    }
    const float f = static_cast<float>(v);
    if (!std::isfinite(f)) {
        // A number past float, as the CPU's double would round it when it lands in a cell: infinity.
        return f > 0 ? "__int_as_float(0x7f800000)" : "__int_as_float(0xff800000)";
    }
    char text[48];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(f));
    std::string s = text;
    if (s.find_first_of(".eEn") == std::string::npos) {
        s += ".0";
    }
    return "(" + s + "f)";
}

class Generator {
public:
    Generator(const ir::Module& m, const Law& law, LawLayout& layout) : m_(m), law_(law), lay_(layout) {}

    std::string run()
    {
        allocate();
        out_ << law_prelude() << net_prelude();
        nets();
        for (std::size_t id = 0; id < m_.insts.size(); ++id) {
            emit(static_cast<int>(id));
        }
        for (const auto& [id, stage] : lay_.stages) {
            stage_kernel(id);
        }
        for (std::size_t t = 0; t < m_.terms.size(); ++t) {
            lay_.terms.push_back(program(m_.terms[t], "t" + std::to_string(t), false));
            lay_.learned.push_back(program(m_.learned[t], "l" + std::to_string(t), false));
        }
        for (std::size_t d = 0; d < m_.derives.size(); ++d) {
            lay_.derives.push_back(program(m_.derives[d], "d" + std::to_string(d), true));
        }
        return out_.str();
    }

private:
    [[nodiscard]] const ir::Inst& at(int id) const { return m_.insts[static_cast<std::size_t>(id)]; }
    static std::string v(int id, const std::string& x = "x", const std::string& y = "y", const std::string& z = "z")
    {
        return "v" + std::to_string(id) + "(" + x + ", " + y + ", " + z + ")";
    }

    void allocate()
    {
        lay_.slot.assign(m_.insts.size(), -1);
        lay_.uslot.assign(m_.insts.size(), -1);
        lay_.flag.assign(m_.insts.size(), -1);
        // Every stage any program uses (classical or learned) gets its storage.
        std::set<int> used;
        auto take = [&](const ir::Program& p) {
            used.insert(p.stages.begin(), p.stages.end());
            used.insert(p.limit_stages.begin(), p.limit_stages.end());
            lay_.max_equations = std::max(lay_.max_equations, static_cast<int>(p.equations.size()));
        };
        for (const ir::Program& p : m_.terms) take(p);
        for (const ir::Program& p : m_.learned) take(p);
        for (const ir::Program& p : m_.derives) take(p);
        for (const int id : used) {
            const ir::Inst& in = at(id);
            StageCode s;
            s.inst = id;
            switch (in.op) {
            case Op::Distance:
                s.kind = StageCode::Kind::Distance;
                lay_.slot[static_cast<std::size_t>(id)] = lay_.aux_slots;
                lay_.aux_slots += 7; // seed in, seed out, two scratch, to inside, to outside, the answer
                lay_.flag[static_cast<std::size_t>(id)] = lay_.flags++;
                break;
            case Op::Poisson:
                s.kind = StageCode::Kind::Poisson;
                s.iterations = in.iterations;
                lay_.slot[static_cast<std::size_t>(id)] = lay_.aux_slots;
                lay_.aux_slots += 3; // the source as last solved, phi (the answer), phi'
                lay_.flag[static_cast<std::size_t>(id)] = lay_.flags++;
                break;
            case Op::MaxOf: case Op::MinOf: case Op::SumOf:
                s.kind = StageCode::Kind::Reduce;
                s.reduce = in.op;
                lay_.uslot[static_cast<std::size_t>(id)] = lay_.uniforms++;
                break;
            case Op::Mlp:
                s.kind = StageCode::Kind::Net;
                lay_.slot[static_cast<std::size_t>(id)] = lay_.aux_slots;
                lay_.aux_slots += 1;
                break;
            case Op::Flux: case Op::LDiffuse:
                s.kind = StageCode::Kind::Net;
                lay_.slot[static_cast<std::size_t>(id)] = lay_.aux_slots;
                lay_.aux_slots += 3; // the flux through each cell's +x, +y, +z face
                break;
            default:
                s.kind = StageCode::Kind::Stored;
                lay_.slot[static_cast<std::size_t>(id)] = lay_.aux_slots;
                lay_.aux_slots += 1;
                break;
            }
            if (s.kind == StageCode::Kind::Net) {
                s.tensor = law_.nets[static_cast<std::size_t>(in.index)].half;
            }
            s.kernel = "st" + std::to_string(id);
            lay_.stages.emplace(id, s);
        }
    }

    void nets()
    {
        int floats = 0;
        int words = 0;
        for (std::size_t n = 0; n < law_.nets.size(); ++n) {
            const NetDecl& net = law_.nets[n];
            const NetLayout l = net_layout(net);
            lay_.net_image_at.push_back(floats);
            lay_.net_frag_at.push_back(words);
            floats += (l.image_floats + 3) / 4 * 4;
            words += l.tensor ? l.frag_words : 0;
            out_ << net_device_code(net, "net" + std::to_string(n));
        }
        lay_.net_floats = std::max(floats, 1);
        lay_.net_words = std::max(words, 2);
    }

    // v<id>(x, y, z): the instruction's value at a cell, as its readers see it.
    void emit(int id)
    {
        const ir::Inst& in = at(id);
        const auto stage = lay_.stages.find(id);
        const bool stored = stage != lay_.stages.end();
        const std::string head = "__device__ __forceinline__ float v" + std::to_string(id) + "(int x, int y, int z)";
        const std::string body_head = "__device__ __forceinline__ float b" + std::to_string(id) + "(int x, int y, int z)";
        auto a = [&](std::size_t k, const std::string& x = "x", const std::string& y = "y", const std::string& z = "z") {
            return v(in.args[k], x, y, z);
        };
        std::ostringstream f;
        auto simple = [&](const std::string& expr) { f << "{ return " << expr << "; }\n"; };
        f << (stored && stage->second.kind == StageCode::Kind::Stored ? body_head : head);
        switch (in.op) {
        case Op::Const: simple(literal(in.number)); break;
        case Op::Param: simple("P[" + std::to_string(in.index) + "]"); break;
        case Op::Cell: simple("G.h"); break;
        case Op::Rim: simple("(x == 0 || y == 0 || z == 0 || x + 1 == G.nx || y + 1 == G.ny || z + 1 == G.nz) ? 1.f : 0.f"); break;
        case Op::Local: simple("0.f"); break;
        case Op::Field: simple("FIELD(" + std::to_string(in.index) + ", x, y, z)"); break;
        case Op::Neg: simple("-" + a(0)); break;
        case Op::Abs: simple("fabsf(" + a(0) + ")"); break;
        case Op::Sqrt: simple("sqrtf(fmaxf(0.f, " + a(0) + "))"); break;
        case Op::Exp: simple("expf(fminf(" + a(0) + ", 80.f))"); break;
        case Op::Log: simple("logf(fmaxf(" + a(0) + ", 1e-30f))"); break;
        case Op::Step: simple(a(0) + " > 0.f ? 1.f : 0.f"); break;
        case Op::Pos: simple("fmaxf(" + a(0) + ", 0.f)"); break;
        case Op::Add: simple(a(0) + " + " + a(1)); break;
        case Op::Sub: simple(a(0) + " - " + a(1)); break;
        case Op::Mul: simple(a(0) + " * " + a(1)); break;
        case Op::Div: simple("dv(" + a(0) + ", " + a(1) + ")"); break;
        case Op::Pow: simple("powf(" + a(0) + ", " + a(1) + ")"); break;
        case Op::Min: simple("fminf(" + a(0) + ", " + a(1) + ")"); break;
        case Op::Max: simple("fmaxf(" + a(0) + ", " + a(1) + ")"); break;
        case Op::Clamp:
            f << "{ const float lo = " << a(1) << "; return fminf(fmaxf(" << a(0) << ", lo), fmaxf(lo, " << a(2) << ")); }\n";
            break;
        case Op::Shift:
            simple(a(0, "ex(x + " + std::to_string(in.offset[0]) + ")", "ey(y + " + std::to_string(in.offset[1]) + ")",
                "ez(z + " + std::to_string(in.offset[2]) + ")"));
            break;
        case Op::Lap:
            // Differences first: a flat field is exactly flat.
            f << "{ const float c = " << a(0) << "; return ((" << a(0, "ex(x - 1)") << " - c) + (" << a(0, "ex(x + 1)") << " - c) + ("
              << a(0, "x", "ey(y - 1)") << " - c) + (" << a(0, "x", "ey(y + 1)") << " - c) + (" << a(0, "x", "y", "ez(z - 1)")
              << " - c) + (" << a(0, "x", "y", "ez(z + 1)") << " - c)) / (G.h * G.h); }\n";
            break;
        case Op::Gx:
            f << "{ const int hi = ex(x + 1); const int lo = ex(x - 1); const int span = hi - lo; return span > 0 ? (" << a(0, "hi")
              << " - " << a(0, "lo") << ") / (G.h * span) : 0.f; }\n";
            break;
        case Op::Gy:
            f << "{ const int hi = ey(y + 1); const int lo = ey(y - 1); const int span = hi - lo; return span > 0 ? (" << a(0, "x", "hi")
              << " - " << a(0, "x", "lo") << ") / (G.h * span) : 0.f; }\n";
            break;
        case Op::Gz:
            f << "{ const int hi = ez(z + 1); const int lo = ez(z - 1); const int span = hi - lo; return span > 0 ? ("
              << a(0, "x", "y", "hi") << " - " << a(0, "x", "y", "lo") << ") / (G.h * span) : 0.f; }\n";
            break;
        case Op::Diffuse: {
            f << "{ const float ka = " << a(0) << "; const float c = " << a(1) << "; float acc = 0.f;\n";
            const char* sides[6][4] = {{"x > 0", "x - 1", "y", "z"}, {"x + 1 < G.nx", "x + 1", "y", "z"},
                {"y > 0", "x", "y - 1", "z"}, {"y + 1 < G.ny", "x", "y + 1", "z"}, {"z > 0", "x", "y", "z - 1"},
                {"z + 1 < G.nz", "x", "y", "z + 1"}};
            for (const auto& s : sides) {
                f << "    if (" << s[0] << ") acc += face(ka, " << a(0, s[1], s[2], s[3]) << ", c, " << a(1, s[1], s[2], s[3]) << ");\n";
            }
            f << "    return acc / (G.h * G.h); }\n";
            break;
        }
        case Op::Adv: {
            auto up = [&](const std::string& speed, const char* lo, const char* hi, const char* pos, const char* count,
                          const std::string& at_lo, const std::string& at_hi) {
                return "(" + speed + " > 0.f ? (" + std::string(pos) + " > 0 ? " + speed + " * (c - " + at_lo + ") / G.h : 0.f) : (" +
                    pos + " + 1 < " + count + " ? " + speed + " * (" + at_hi + " - c) / G.h : 0.f))";
            };
            f << "{ const float c = " << a(0) << "; const float u = " << a(1) << "; const float w = " << a(2) << "; const float s = " << a(3) << ";\n"
              << "    return -(" << up("u", "", "", "x", "G.nx", a(0, "x - 1"), a(0, "x + 1")) << " + "
              << up("w", "", "", "y", "G.ny", a(0, "x", "y - 1"), a(0, "x", "y + 1")) << " + "
              << up("s", "", "", "z", "G.nz", a(0, "x", "y", "z - 1"), a(0, "x", "y", "z + 1")) << "); }\n";
            break;
        }
        case Op::Fall: {
            // The flux from a cell into the one below: its content times the room there.
            const std::string n = std::to_string(id);
            std::ostringstream pre;
            pre << "__device__ __forceinline__ float fall" << n << "(int x, int y, int z) { if (z == 0) return 0.f;\n"
                << "    return fmaxf(0.f, " << a(2) << ") / G.h * fmaxf(0.f, " << a(0) << ") * sat(1.f - " << a(1, "x", "y", "z - 1")
                << " - " << a(0, "x", "y", "z - 1") << "); }\n";
            out_ << pre.str();
            f << "{ float o = -fall" << n << "(x, y, z); if (z + 1 < G.nz) o += fall" << n << "(x, y, z + 1); return o; }\n";
            break;
        }
        case Op::Spread: {
            const std::string n = std::to_string(id);
            std::ostringstream pre;
            pre << "__device__ __forceinline__ float spread" << n << "(int ax, int ay, int az, int bx, int by, int bz) {\n"
                << "    const float fa = " << a(0, "ax", "ay", "az") << "; const float fb = " << a(0, "bx", "by", "bz") << ";\n"
                << "    const bool to_b = fa > fb; const int rx = to_b ? bx : ax; const int ry = to_b ? by : ay; const int rz = to_b ? bz : az;\n"
                << "    return 0.5f * (fmaxf(0.f, " << a(2, "ax", "ay", "az") << ") + fmaxf(0.f, " << a(2, "bx", "by", "bz")
                << ")) / (G.h * G.h) * (fa - fb) * sat(1.f - " << a(1, "rx", "ry", "rz") << " - " << a(0, "rx", "ry", "rz") << "); }\n";
            out_ << pre.str();
            f << "{ float o = 0.f;\n"
              << "    if (y > 0) o += spread" << n << "(x, y - 1, z, x, y, z);\n"
              << "    if (x > 0) o += spread" << n << "(x - 1, y, z, x, y, z);\n"
              << "    if (x + 1 < G.nx) o -= spread" << n << "(x, y, z, x + 1, y, z);\n"
              << "    if (y + 1 < G.ny) o -= spread" << n << "(x, y, z, x, y + 1, z);\n"
              << "    return o; }\n";
            break;
        }
        case Op::Topple: {
            const std::string n = std::to_string(id);
            std::ostringstream pre;
            pre << "__device__ __forceinline__ float room" << n << "(int x, int y, int z) { return sat(1.f - " << a(1) << " - " << a(0) << "); }\n"
                << "__device__ __forceinline__ float send" << n << "(int x, int y, int z, int sx, int sy) {\n"
                << "    const float here = " << a(0) << "; if (here <= 0.f || z == 0) return 0.f;\n"
                << "    if (room" << n << "(x, y, z - 1) > 0.01f) return 0.f;\n"
                << "    const int jx = x + sx; const int jy = y + sy; if (!inside(jx, jy, z)) return 0.f;\n"
                << "    return 0.25f * fmaxf(0.f, " << a(2) << ") / G.h * here * fminf(room" << n << "(jx, jy, z), room" << n << "(jx, jy, z - 1)); }\n";
            out_ << pre.str();
            f << "{ float o = 0.f;\n"
              << "    o -= send" << n << "(x, y, z, -1, 0); o -= send" << n << "(x, y, z, 1, 0); o -= send" << n << "(x, y, z, 0, -1); o -= send" << n << "(x, y, z, 0, 1);\n"
              << "    if (z + 1 < G.nz) {\n"
              << "        if (x + 1 < G.nx) o += send" << n << "(x + 1, y, z + 1, -1, 0);\n"
              << "        if (x > 0) o += send" << n << "(x - 1, y, z + 1, 1, 0);\n"
              << "        if (y + 1 < G.ny) o += send" << n << "(x, y + 1, z + 1, 0, -1);\n"
              << "        if (y > 0) o += send" << n << "(x, y - 1, z + 1, 0, 1);\n"
              << "    }\n    return o; }\n";
            break;
        }
        case Op::Distance: {
            const int s = lay_.slot[static_cast<std::size_t>(id)];
            simple(s >= 0 ? "AT(" + std::to_string(s + 6) + ", x, y, z)" : "0.f");
            break;
        }
        case Op::Poisson: {
            const int s = lay_.slot[static_cast<std::size_t>(id)];
            simple(s >= 0 ? "AT(" + std::to_string(s + 1) + ", x, y, z)" : "0.f");
            break;
        }
        case Op::MaxOf: case Op::MinOf: case Op::SumOf: {
            const int u = lay_.uslot[static_cast<std::size_t>(id)];
            simple(u >= 0 ? "G.U[" + std::to_string(u) + "]" : "0.f");
            break;
        }
        case Op::Mlp: {
            const int s = lay_.slot[static_cast<std::size_t>(id)];
            simple(s >= 0 ? "AT(" + std::to_string(s) + ", x, y, z)" : "0.f");
            break;
        }
        case Op::Flux: case Op::LDiffuse: {
            const int s = lay_.slot[static_cast<std::size_t>(id)];
            if (s < 0) {
                simple("0.f");
                break;
            }
            // What comes in through the -x face, minus what goes out through the +x one, ..., over h.
            f << "{ float o = 0.f;\n"
              << "    if (x > 0) o += AT(" << s << ", x - 1, y, z); if (x + 1 < G.nx) o -= AT(" << s << ", x, y, z);\n"
              << "    if (y > 0) o += AT(" << s + 1 << ", x, y - 1, z); if (y + 1 < G.ny) o -= AT(" << s + 1 << ", x, y, z);\n"
              << "    if (z > 0) o += AT(" << s + 2 << ", x, y, z - 1); if (z + 1 < G.nz) o -= AT(" << s + 2 << ", x, y, z);\n"
              << "    return o / G.h; }\n";
            break;
        }
        }
        out_ << f.str();
        if (stored && stage->second.kind == StageCode::Kind::Stored) {
            out_ << head << " { return AT(" << lay_.slot[static_cast<std::size_t>(id)] << ", x, y, z); }\n";
        }
    }

    // The kernel that computes a stage over the lattice.
    void stage_kernel(int id)
    {
        StageCode& s = lay_.stages.at(id);
        const ir::Inst& in = at(id);
        const int slot = lay_.slot[static_cast<std::size_t>(id)];
        const std::string name = s.kernel;
        switch (s.kind) {
        case StageCode::Kind::Stored:
            out_ << "extern \"C\" __global__ void " << name << "(float DT) { CELL_HEAD AT(" << slot << ", x, y, z) = b" << id << "(x, y, z); }\n";
            break;
        case StageCode::Kind::Distance:
            // The seeds: what is solid now; a change arms the axis passes.
            out_ << "extern \"C\" __global__ void " << name << "(float DT) { CELL_HEAD\n"
                 << "    const float seed = " << v(in.args[0]) << " >= 0.5f ? 0.f : FAR; const u64 k = (u64)" << slot << " * G.cells + I(x, y, z);\n"
                 << "    if (__float_as_uint(G.A[k]) != __float_as_uint(seed)) { G.A[k] = seed; G.A[k + G.cells] = seed == 0.f ? FAR : 0.f; G.F["
                 << lay_.flag[static_cast<std::size_t>(id)] << "] = 1u; } }\n";
            break;
        case StageCode::Kind::Poisson:
            out_ << "extern \"C\" __global__ void " << name << "(float DT) { CELL_HEAD\n"
                 << "    const float v = " << v(in.args[0]) << "; const u64 k = (u64)" << slot << " * G.cells + I(x, y, z);\n"
                 << "    if (__float_as_uint(G.A[k]) != __float_as_uint(v)) { G.A[k] = v; G.F[" << lay_.flag[static_cast<std::size_t>(id)] << "] = 1u; } }\n";
            break;
        case StageCode::Kind::Reduce:
            // A block a plane, a fixed tree: the same sum every time.
            out_ << "extern \"C\" __global__ void " << name << "(float DT) {\n"
                 << "    __shared__ float ss[256]; __shared__ float sc[256]; __shared__ float slo[256]; __shared__ float shi[256];\n"
                 << "    const int z = blockIdx.x; const int t = threadIdx.x; Sum sum = {0.f, 0.f}; float lo = 1e30f; float hi = -1e30f;\n"
                 << "    for (int p = t; p < G.nx * G.ny; p += 256) { const int x = p % G.nx; const int y = p / G.nx; const float v = " << v(in.args[0])
                 << "; add(sum, v); lo = fminf(lo, v); hi = fmaxf(hi, v); }\n"
                 << "    ss[t] = sum.s; sc[t] = sum.c; slo[t] = lo; shi[t] = hi; __syncthreads();\n"
                 << "    for (int w = 128; w > 0; w >>= 1) { if (t < w) { Sum a = {ss[t], sc[t]}; add(a, ss[t + w]); a.c += sc[t + w]; ss[t] = a.s; sc[t] = a.c;\n"
                 << "        slo[t] = fminf(slo[t], slo[t + w]); shi[t] = fmaxf(shi[t], shi[t + w]); } __syncthreads(); }\n"
                 << "    if (t == 0) { G.L[3 * z] = ss[0] + sc[0]; G.L[3 * z + 1] = slo[0]; G.L[3 * z + 2] = shi[0]; } }\n";
            break;
        case StageCode::Kind::Net:
            net_kernel(id, s);
            break;
        }
    }

    // The values a net op reads, at a cell: for mlp its arguments; for flux the moved field and its companions.
    void net_kernel(int id, StageCode& s)
    {
        const ir::Inst& in = at(id);
        const NetDecl& net = law_.nets[static_cast<std::size_t>(in.index)];
        const NetLayout l = net_layout(net);
        const std::string nn = "net" + std::to_string(in.index);
        const int slot = lay_.slot[static_cast<std::size_t>(id)];
        const int image_at = lay_.net_image_at[static_cast<std::size_t>(in.index)];
        const int image = (l.image_floats + 3) / 4 * 4;
        const bool tc = net.half;
        const int xs = tc ? l.k_pad[0] : 0;
        const int ys = tc ? l.n_pad.back() : 0;
        const int frag_pairs = tc ? l.frag_words / 2 : 0;
        s.tensor = tc;
        s.shared = static_cast<unsigned>(4 * image + (tc ? 8 * frag_pairs + 4 * 4 * 32 * (xs + ys) : 0));
        std::ostringstream k;
        k << "extern \"C\" __global__ void " << s.kernel << "(float DT) {\n"
          << "    extern __shared__ float sm[];\n"
          << "    const int threads = blockDim.x * blockDim.y; const int me = threadIdx.x + threadIdx.y * blockDim.x;\n"
          << "    for (int q = me; q < " << l.image_floats << "; q += threads) sm[q] = G.W[" << image_at << " + q];\n";
        if (tc) {
            k << "    frag2* sf = (frag2*)(sm + " << image << ");\n"
              << "    for (int q = me; q < " << frag_pairs << "; q += threads) sf[q] = G.NF[" << lay_.net_frag_at[static_cast<std::size_t>(in.index)] / 2 << " + q];\n"
              << "    float* xs = (float*)(sf + " << frag_pairs << ") + (threadIdx.x >> 5) * " << 32 * (xs + ys) << "; float* ys = xs + " << 32 * xs << ";\n"
              << "    __syncthreads();\n"
              << "    const int lane = threadIdx.x & 31;\n"
              << "    const int li = blockIdx.x * blockDim.x + threadIdx.x; const bool live = li < G.cells; const int lc = live ? li : G.cells - 1;\n"
              << "    const int x = lc % G.nx; const int y = (lc / G.nx) % G.ny; const int z = lc / G.nxy;\n";
        } else {
            k << "    __syncthreads();\n    CELL_HEAD\n";
        }
        const std::string per = "(DT > 0.f ? 0.999f * G.h / (6.f * DT) : 0.f)";
        if (in.op == Op::Mlp) {
            if (tc) {
                for (std::size_t a = 0; a < in.args.size(); ++a) {
                    k << "    xs[lane * " << xs << " + " << a << "] = " << v(in.args[a]) << ";\n";
                }
                k << "    __syncwarp(); " << nn << "_tc(sm, sf, xs, ys);\n"
                  << "    if (live) AT(" << slot << ", x, y, z) = ys[lane * " << ys << "];\n}\n";
            } else {
                k << "    float in[" << in.args.size() << "]; float out[" << l.out << "];\n";
                for (std::size_t a = 0; a < in.args.size(); ++a) {
                    k << "    in[" << a << "] = " << v(in.args[a]) << ";\n";
                }
                k << "    " << nn << "_fwd(sm, in, out); AT(" << slot << ", x, y, z) = out[0];\n}\n";
            }
        } else if (in.op == Op::Flux) {
            const std::size_t m = in.args.size() - 2;
            const std::string lo = literal(at(in.args[m]).number);
            const std::string hi = literal(at(in.args[m + 1]).number);
            k << "    float vi[" << m << "]; float vj[" << m << "]; float g[6];\n";
            for (std::size_t a = 0; a < m; ++a) {
                k << "    vi[" << a << "] = " << v(in.args[a]) << ";\n";
            }
            k << "    const int dx[3] = {1, 0, 0}; const int dy[3] = {0, 1, 0}; const int dz[3] = {0, 0, 1};\n"
              << "    bool open[3]; open[0] = x + 1 < G.nx; open[1] = y + 1 < G.ny; open[2] = z + 1 < G.nz;\n"
              << "    float fj[3];\n"
              // A face holds two whole nets: one copy of that code in the loop, not three (the instruction cache).
              << "#pragma unroll 1\n    for (int f = 0; f < 3; ++f) {\n"
              << "        const int jx = open[f] ? x + dx[f] : x; const int jy = open[f] ? y + dy[f] : y; const int jz = open[f] ? z + dz[f] : z;\n";
            for (std::size_t a = 0; a < m; ++a) {
                k << "        vj[" << a << "] = " << v(in.args[a], "jx", "jy", "jz") << ";\n";
            }
            k << "        fj[f] = vj[0];\n";
            if (tc) {
                k << "        for (int q = 0; q < " << m << "; ++q) { xs[lane * " << xs << " + q] = vi[q]; xs[lane * " << xs << " + " << m << " + q] = vj[q]; }\n"
                  << "        __syncwarp(); " << nn << "_tc(sm, sf, xs, ys); g[2 * f] = ys[lane * " << ys << "]; __syncwarp();\n"
                  << "        for (int q = 0; q < " << m << "; ++q) { xs[lane * " << xs << " + q] = vj[q]; xs[lane * " << xs << " + " << m << " + q] = vi[q]; }\n"
                  << "        __syncwarp(); " << nn << "_tc(sm, sf, xs, ys); g[2 * f + 1] = ys[lane * " << ys << "]; __syncwarp();\n";
            } else {
                k << "        float in[" << 2 * m << "]; float out[" << l.out << "];\n"
                  << "        for (int q = 0; q < " << m << "; ++q) { in[q] = vi[q]; in[" << m << " + q] = vj[q]; }\n"
                  << "        " << nn << "_fwd(sm, in, out); g[2 * f] = out[0];\n"
                  << "        for (int q = 0; q < " << m << "; ++q) { in[q] = vj[q]; in[" << m << " + q] = vi[q]; }\n"
                  << "        " << nn << "_fwd(sm, in, out); g[2 * f + 1] = out[0];\n";
            }
            k << "    }\n"
              << (tc ? "    if (!live) return;\n" : "")
              << "    const float per = " << per << ";\n"
              << "#pragma unroll\n    for (int f = 0; f < 3; ++f) {\n"
              << "        const float r = open[f] ? limit_face(0.5f * (g[2 * f] - g[2 * f + 1]), vi[0], fj[f], " << lo << ", " << hi << ", per) : 0.f;\n"
              << "        AT(" << slot << " + f, x, y, z) = r;\n    }\n}\n";
        } else { // LDiffuse
            // Window F[i-1] F[i] | F[j] F[j+1] along each axis; both ways round; half the difference.
            k << "    float g[12]; float win[3][4]; float kf[3]; bool open[3];\n"
              << "    const int cx[3] = {x, y, z}; const int count[3] = {G.nx, G.ny, G.nz};\n"
              // Six nets a cell: one copy of the code in the loops, not six (the instruction cache); a closed face (the
              // lattice's edge, or a flat axis) costs no net on the fp32 path.
              << "#pragma unroll 1\n    for (int f = 0; f < 3; ++f) {\n"
              << "        open[f] = cx[f] + 1 < count[f];\n"
              << "        const int ux = f == 0; const int uy = f == 1; const int uz = f == 2;\n"
              << "        const int jx = open[f] ? x + ux : x; const int jy = open[f] ? y + uy : y; const int jz = open[f] ? z + uz : z;\n"
              << "        const int mx = cx[f] > 0 ? x - ux : x; const int my = cx[f] > 0 ? y - uy : y; const int mz = cx[f] > 0 ? z - uz : z;\n"
              << "        const bool far_open = cx[f] + 2 < count[f];\n"
              << "        const int px = far_open ? jx + ux : jx; const int py = far_open ? jy + uy : jy; const int pz = far_open ? jz + uz : jz;\n"
              << "        win[f][0] = " << v(in.args[1], "mx", "my", "mz") << "; win[f][1] = " << v(in.args[1]) << "; win[f][2] = "
              << v(in.args[1], "jx", "jy", "jz") << "; win[f][3] = " << v(in.args[1], "px", "py", "pz") << ";\n"
              << "        const float ka = " << v(in.args[0]) << "; const float kb = " << v(in.args[0], "jx", "jy", "jz") << ";\n"
              << "        kf[f] = (ka + kb) > 0.f ? 2.f * ka * kb / (ka + kb) : 0.f;\n"
              << (tc ? "" : "        if (!open[f]) { g[4 * f] = 0.f; g[4 * f + 1] = 0.f; continue; }\n")
              << "#pragma unroll 1\n        for (int o = 0; o < 2; ++o) {\n"
              << "            float w[4]; for (int q = 0; q < 4; ++q) w[q] = o == 0 ? win[f][q] : win[f][3 - q];\n"
              << "            const float centre = 0.5f * (w[1] + w[2]); float spread = 1e-6f;\n"
              << "            for (int q = 0; q < 4; ++q) spread = fmaxf(spread, fabsf(w[q] - centre));\n";
            if (tc) {
                k << "            for (int q = 0; q < 4; ++q) xs[lane * " << xs << " + q] = (w[q] - centre) / spread;\n"
                  << "            xs[lane * " << xs << " + 4] = kf[f] * DT / (G.h * G.h);\n"
                  << "            __syncwarp(); " << nn << "_tc(sm, sf, xs, ys);\n"
                  << "            float out[4]; for (int q = 0; q < 4; ++q) out[q] = ys[lane * " << ys << " + q]; __syncwarp();\n";
            } else {
                k << "            float in[5]; for (int q = 0; q < 4; ++q) in[q] = (w[q] - centre) / spread; in[4] = kf[f] * DT / (G.h * G.h);\n"
                  << "            float out[4]; " << nn << "_fwd(sm, in, out);\n";
            }
            k << "            const float mean = 0.25f * (out[0] + out[1] + out[2] + out[3]);\n"
              << "            const float base[4] = {0.f, -1.f, 1.f, 0.f}; float acc = 0.f;\n"
              << "            for (int q = 0; q < 4; ++q) acc += (base[q] + (out[q] - mean)) * w[q];\n"
              << "            g[4 * f + o] = -kf[f] / G.h * acc;\n"
              << "        }\n    }\n"
              << (tc ? "    if (!live) return;\n" : "")
              << "#pragma unroll\n    for (int f = 0; f < 3; ++f) AT(" << slot << " + f, x, y, z) = open[f] ? 0.5f * (g[4 * f] - g[4 * f + 1]) : 0.f;\n}\n";
        }
        out_ << k.str();
    }

    ProgramCode program(const ir::Program& p, const std::string& tag, bool derive)
    {
        ProgramCode code;
        code.stages = p.stages;
        code.limit_stages = p.limit_stages;
        code.max_dt = p.max_dt;
        if (p.equations.empty()) {
            return code;
        }
        if (derive) {
            const ir::Equation& eq = p.equations[0];
            const bool reads_itself = std::find(p.reads.begin(), p.reads.end(), eq.target) != p.reads.end();
            code.in_place = !reads_itself;
            const std::string t = std::to_string(eq.target);
            code.gather = "g_" + tag;
            out_ << "extern \"C\" __global__ void " << code.gather << "(float DT) { CELL_HEAD\n"
                 << "    const float v = " << v(eq.value) << "; const int i = I(x, y, z); const float old = FT[" << t << "].now[i];\n"
                 << "    const float next = " << (p.accumulate ? "old + " : "0.f + ") << "v;\n"
                 << "    const bool ch = __float_as_uint(old) != __float_as_uint(next); ";
            if (code.in_place) {
                out_ << "if (ch) FT[" << t << "].now[i] = next;";
            } else {
                out_ << "FT[" << t << "].next[i] = next;"; // every cell: the buffers are swapped after
                code.swaps.push_back(eq.target);
            }
            out_ << " mark(" << t << ", ch); }\n";
            return code;
        }
        code.in_place = p.in_place;
        code.gather = "g_" + tag;
        std::ostringstream g;
        g << "extern \"C\" __global__ void " << code.gather << "(float DT) { CELL_HEAD\n    const int i = I(x, y, z);\n";
        for (std::size_t e = 0; e < p.equations.size(); ++e) {
            g << "    const float r" << e << " = " << v(p.equations[e].value) << ";\n";
        }
        g << "    float d;\n";
        if (code.in_place) {
            for (std::size_t e = 0; e < p.equations.size(); ++e) {
                const std::string t = std::to_string(p.equations[e].target);
                g << "    d = r" << e << " * DT; if (d != 0.f) FT[" << t << "].now[i] += d; mark(" << t << ", d != 0.f);\n";
            }
        } else {
            // Ping-pong: every right side was read from the fields as they were; each target's next value goes to its
            // other buffer, the equations on one target added in their order (as the stores were).
            std::vector<int> targets;
            for (const ir::Equation& eq : p.equations) {
                if (std::find(targets.begin(), targets.end(), eq.target) == targets.end()) {
                    targets.push_back(eq.target);
                }
            }
            for (const int target : targets) {
                const std::string t = std::to_string(target);
                g << "    { float n = FT[" << t << "].now[i]; bool ch = false;\n";
                for (std::size_t e = 0; e < p.equations.size(); ++e) {
                    if (p.equations[e].target == target) {
                        g << "      d = r" << e << " * DT; if (d != 0.f) { n += d; ch = true; }\n";
                    }
                }
                g << "      FT[" << t << "].next[i] = n; mark(" << t << ", ch); }\n";
            }
            code.swaps = targets;
        }
        g << "}\n";
        out_ << g.str();
        return code;
    }

    const ir::Module& m_;
    const Law& law_;
    LawLayout& lay_;
    std::ostringstream out_;
};

} // namespace

std::string generate(const ir::Module& module, const Law& law, LawLayout& layout)
{
    layout = LawLayout{};
    return Generator(module, law, layout).run();
}

} // namespace fieldengine::field::cuda

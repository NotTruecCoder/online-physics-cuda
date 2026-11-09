#include "field/GpuSolver.hpp"
#include "field/LawBounds.hpp"

#ifndef _WIN32

namespace fieldengine::field {

struct GpuSolver::Impl {
    std::string name;
};
GpuSolver::GpuSolver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuSolver::~GpuSolver() = default;
std::shared_ptr<GpuSolver> GpuSolver::create(Law&, const GridShape&, std::string& why)
{
    why = GpuSolver::unavailable();
    return nullptr;
}
std::string GpuSolver::unavailable()
{
    return "the GPU solver runs on Direct3D 11 (Windows)";
}
void GpuSolver::step(double, bool, bool, Law&, FieldSet&, FieldEdits&, std::vector<char>&, std::vector<Violation>&, std::vector<int>&) {}
void GpuSolver::derive(Law&, FieldSet&, FieldEdits&, std::vector<char>&) {}
void GpuSolver::check_contracts_every(int) {}
const std::string& GpuSolver::adapter() const { return impl_->name; }
const StepProfile& GpuSolver::profile() const
{
    static const StepProfile none;
    return none;
}

} // namespace fieldengine::field

#else

#include "field/LawNode.hpp"
#include "field/Parallel.hpp"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

using Microsoft::WRL::ComPtr;

namespace fieldengine::field {

namespace {

constexpr int kMaxUniforms = 1024; // params + reductions, in cbuffer Grid
constexpr float kFar = 1e30f;      // "no feature in reach" in the distance passes
constexpr int kGroup = 8;          // [numthreads(8,8,1)]

// ---- one device, one shader cache, for the whole process ------------------------------------------------

struct Device {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::string name;
    std::string why;
    std::mutex mutex; // one solver at a time on the immediate context
    std::map<std::string, ComPtr<ID3D11ComputeShader>> shaders; // by source text: a changed law recompiles what changed
};

Device& device()
{
    static Device out;
    static const bool opened = [] {
        ComPtr<IDXGIFactory6> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            out.why = "no DXGI 1.6 factory";
            return false;
        }
        // The laptop's discrete card, not whichever adapter happens to drive the screen.
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)))) {
            out.why = "no graphics adapter";
            return false;
        }
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            out.why = "only a software adapter";
            return false;
        }
        char name[128]{};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
        const D3D_FEATURE_LEVEL wanted = D3D_FEATURE_LEVEL_11_0;
        if (FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &wanted, 1, D3D11_SDK_VERSION,
                &out.device, nullptr, &out.context))) {
            out.why = std::string("no Direct3D 11 device on ") + name;
            return false;
        }
        out.name = name;
        return true;
    }();
    (void)opened;
    return out;
}

// ---- the shared part of every shader ---------------------------------------------------------------------

const char* const kPrelude = R"(
cbuffer Grid : register(b0) { int NX, NY, NZ, NXY, CELLS; float H; int2 PAD; float4 P[256]; };
cbuffer Run : register(b1) { int F0, F1, F2, F3; float DT, A0, A1, A2; };
int I(int3 c) { return c.x + c.y * NX + c.z * NXY; }
int3 edge(int3 c) { return clamp(c, int3(0, 0, 0), int3(NX - 1, NY - 1, NZ - 1)); }
float dv(float a, float b) { return b == 0 ? 0 : a / b; }
float pw(float a, float b)
{
    if (b == 0) return 1;
    if (a == 0) return b > 0 ? 0 : 1e30;
    if (a < 0 && floor(b) == b) { float r = pow(-a, b); return fmod(abs(b), 2) == 0 ? r : -r; }
    return pow(a, b);
}
float face(float ka, float kb, float va, float vb)
{
    float s = ka + kb;
    return (s > 0 ? 2 * ka * kb / s : 0) * (vb - va);
}
)";

const char* const kBuffers = R"(
StructuredBuffer<float> S : register(t0);
RWStructuredBuffer<float> D : register(u0);
RWStructuredBuffer<float> A : register(u1);
RWByteAddressBuffer ARGS : register(u2);
)";

const char* const kStoreBuffers = R"(
StructuredBuffer<float> D : register(t0);
RWStructuredBuffer<float> S : register(u0);
RWStructuredBuffer<uint> CHANGED : register(u1);
)";

// A pass over every cell.
const char* const kEachCell = "[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID)\n"
                              "{\n    if ((int)id.x >= NX || (int)id.y >= NY) return;\n    int3 c = int3(id); int i = I(c);\n";

// Scalars and arrays of one plane, summed without losing the small to the large.
const char* const kPlaneSum = R"(
struct Sum { float s; float c; };
void add(inout Sum a, float v)
{
    precise float t = a.s + v;
    precise float low = abs(a.s) >= abs(v) ? (a.s - t) + v : (v - t) + a.s;
    a.c += low;
    a.s = t;
}
)";

// Compiled shaders outlive the process, so only a law's first run ever waits for the compiler.
std::string shader_cache_path(const std::string& text)
{
    std::uint64_t h = 1469598103934665603ull;
    for (const char c : text) {
        h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
    char temp[MAX_PATH + 1]{};
    GetTempPathA(MAX_PATH, temp);
    std::string dir = std::string(temp) + "field_engine_law_shaders";
    CreateDirectoryA(dir.c_str(), nullptr);
    std::ostringstream name;
    name << dir << "\\" << std::hex << h << ".cso";
    return name.str();
}

ComPtr<ID3DBlob> compile_shader(const std::string& text, std::string& error)
{
    const std::string path = shader_cache_path(text);
    ComPtr<ID3DBlob> blob;
    if (std::ifstream in(path, std::ios::binary | std::ios::ate); in) {
        const auto size = static_cast<SIZE_T>(in.tellg());
        if (size > 0 && SUCCEEDED(D3DCreateBlob(size, &blob))) {
            in.seekg(0);
            in.read(static_cast<char*>(blob->GetBufferPointer()), static_cast<std::streamsize>(size));
            if (in) {
                return blob;
            }
        }
    }
    ComPtr<ID3DBlob> log;
    if (FAILED(D3DCompile(text.data(), text.size(), "law", nullptr, nullptr, "main", "cs_5_0",
            D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL1, 0, &blob, &log))) {
        error = log ? static_cast<const char*>(log->GetBufferPointer()) : "shader compiler failed";
        return nullptr;
    }
    std::ofstream(path, std::ios::binary).write(static_cast<const char*>(blob->GetBufferPointer()), static_cast<std::streamsize>(blob->GetBufferSize()));
    return blob;
}

std::string number(double v)
{
    if (!std::isfinite(v)) {
        return v > 0 ? "1e30" : (v < 0 ? "-1e30" : "0");
    }
    std::ostringstream out;
    out.precision(9);
    out << std::scientific << static_cast<float>(v);
    return "(" + out.str() + ")";
}

std::string uniform_slot(int slot)
{
    return "P[" + std::to_string(slot / 4) + "]." + "xyzw"[slot % 4];
}

} // namespace

// ---- the solver ---------------------------------------------------------------------------------------------

struct GpuSolver::Impl {
    struct Mat {
        const Node* node = nullptr;
        bool poisson = false;
        int compare = -1;  // writes the seeds (or the source), arms the indirect dispatch if they changed
        int result = -1;   // slot of the answer in A
        int seed = -1;     // first slot of what it keeps
        int iterations = 0;
    };
    struct Reduction {
        Node* node = nullptr;
        int uniform = -1;
        int shader = -1;
    };
    struct Program {
        int gather = -1;
        int store = -1;
        std::vector<int> mats;       // into Impl::mats, nested first
        std::vector<int> reductions; // into Impl::reductions
    };
    struct Check {
        int slot = -1;
        int field = -1;
    };
    struct ConserveCheck {
        std::size_t term = 0;
        std::size_t group = 0;
        int first_before = -1; // slots, one per field of the group
        int first_after = -1;
        int substeps = 1;
    };
    struct RangeCheck {
        std::size_t term = 0; // law.terms.size() = a derive
        int field = -1;
        int slot = -1;
    };

    Device* dev = nullptr;
    GridShape shape;
    int fields_n = 0;
    int cells = 0;
    int gx = 0;
    int gy = 0;

    ComPtr<ID3D11Buffer> state, delta, aux, args, records, changed;
    ComPtr<ID3D11Buffer> state_read, records_read, small_read, changed_read;
    ComPtr<ID3D11ShaderResourceView> state_srv, delta_srv;
    ComPtr<ID3D11UnorderedAccessView> state_uav, delta_uav, aux_uav, args_uav, records_uav, changed_uav;
    ComPtr<ID3D11Buffer> grid_cb, pass_cb;
    int aux_slots = 0;
    int record_words = 0;
    int uniforms = 0;

    std::vector<std::string> sources;
    std::vector<ComPtr<ID3D11ComputeShader>> shaders;
    int fill_cs = -1, scan_cs = -1, signed_cs = -1, jacobi_cs = -1, copy_cs = -1, range_cs = -1, rows_cs = -1, planes_cs = -1, clamp_cs = -1;

    std::deque<Mat> mats;
    std::vector<Reduction> reductions;
    std::map<const Node*, int> mat_of;
    std::map<const Node*, int> reduction_of;
    std::vector<Program> term_programs;
    std::vector<Program> derive_programs;
    std::vector<char> derive_begun; // this derive adds to what an earlier one set

    bool uploaded_once = false;
    std::vector<char> known_zero; // per field: every cell exactly 0, as of the end of the last step
    std::vector<char> zero;       // the same, as far as anything in this step can tell
    std::vector<char> touched;
    std::vector<Bounds> known;    // per field: what its cells went between, as of the end of the last step
    std::vector<Bounds> flowing;  // the same, as far as anything in this step can tell
    std::vector<char> dirty;      // per field: can it differ from what the derives last saw?
    int check_every = 1;          // the contracts are looked at in one step of so many
    int steps_taken = 0;
    bool checking = true;
    std::vector<float> grid_data;
    std::vector<ConserveCheck> conserve_checks;
    std::vector<RangeCheck> range_checks;
    int next_record = 0;
    StepProfile profile;

    // ---- text -------------------------------------------------------------------------------------------

    int add_shader(std::string text)
    {
        const auto found = std::find(sources.begin(), sources.end(), text);
        if (found != sources.end()) {
            return static_cast<int>(found - sources.begin());
        }
        sources.push_back(std::move(text));
        return static_cast<int>(sources.size()) - 1;
    }

    int reserve_aux(int count)
    {
        const int first = aux_slots;
        aux_slots += count;
        return first;
    }

    // One expression tree as HLSL functions of the cell coordinate. A neighbour's value is the same
    // function at another coordinate, so stencils nest in anything.
    class Emitter {
    public:
        Emitter(Impl& impl, Program& program) : impl_(impl), program_(program) {}

        std::string emit(Node& n)
        {
            if (const auto found = names_.find(&n); found != names_.end()) {
                return found->second;
            }
            std::vector<std::string> a;
            const bool leaves_args = n.op == Op::Distance || n.op == Op::Poisson || n.op == Op::MaxOf ||
                n.op == Op::MinOf || n.op == Op::SumOf;
            if (n.op == Op::Shift) {
                a.push_back(emit(*n.args[0])); // the rest are numbers
            } else if (!leaves_args) {
                for (const std::shared_ptr<Node>& arg : n.args) {
                    a.push_back(emit(*arg));
                }
            }
            const std::string name = "n" + std::to_string(names_.size());
            std::ostringstream f;
            auto simple = [&](const std::string& expr) {
                f << "float " << name << "(int3 c) { return " << expr << "; }\n";
            };
            auto at = [](const std::string& fn, const char* where) { return fn + "(" + where + ")"; };
            switch (n.op) {
            case Op::Const:
                simple(number(n.number));
                break;
            case Op::Param:
                simple(uniform_slot(n.index));
                break;
            case Op::Cell:
                simple("H");
                break;
            case Op::Rim:
                simple("(c.x == 0 || c.y == 0 || c.z == 0 || c.x + 1 == NX || c.y + 1 == NY || c.z + 1 == NZ) ? 1.0 : 0.0");
                break;
            case Op::Local:
                simple("0");
                break;
            case Op::Field:
                simple("S[" + std::to_string(n.index) + " * CELLS + I(c)]");
                break;
            case Op::Neg:
                simple("-" + a[0] + "(c)");
                break;
            case Op::Abs:
                simple("abs(" + a[0] + "(c))");
                break;
            case Op::Sqrt:
                simple("sqrt(max(0, " + a[0] + "(c)))");
                break;
            case Op::Exp:
                simple("exp(min(" + a[0] + "(c), 80))");
                break;
            case Op::Log:
                simple("log(max(" + a[0] + "(c), 1e-30))");
                break;
            case Op::Step:
                simple(a[0] + "(c) > 0 ? 1.0 : 0.0");
                break;
            case Op::Pos:
                simple("max(" + a[0] + "(c), 0)");
                break;
            case Op::Add:
                simple(a[0] + "(c) + " + a[1] + "(c)");
                break;
            case Op::Sub:
                simple(a[0] + "(c) - " + a[1] + "(c)");
                break;
            case Op::Mul:
                simple(a[0] + "(c) * " + a[1] + "(c)");
                break;
            case Op::Div:
                simple("dv(" + a[0] + "(c), " + a[1] + "(c))");
                break;
            case Op::Pow:
                simple("pw(" + a[0] + "(c), " + a[1] + "(c))");
                break;
            case Op::Min:
                simple("min(" + a[0] + "(c), " + a[1] + "(c))");
                break;
            case Op::Max:
                simple("max(" + a[0] + "(c), " + a[1] + "(c))");
                break;
            case Op::Clamp:
                f << "float " << name << "(int3 c) { float lo = " << a[1] << "(c); return clamp(" << a[0]
                  << "(c), lo, max(lo, " << a[2] << "(c))); }\n";
                break;
            case Op::Shift: {
                std::ostringstream offset;
                offset << "int3(" << static_cast<int>(n.args[1]->number) << "," << static_cast<int>(n.args[2]->number) << ","
                       << static_cast<int>(n.args[3]->number) << ")";
                simple(a[0] + "(edge(c + " + offset.str() + "))");
                break;
            }
            case Op::Lap:
                f << "float " << name << "(int3 c) { float v = " << a[0] << "(c); return (" << at(a[0], "edge(c - int3(1,0,0))")
                  << " + " << at(a[0], "edge(c + int3(1,0,0))") << " + " << at(a[0], "edge(c - int3(0,1,0))") << " + "
                  << at(a[0], "edge(c + int3(0,1,0))") << " + " << at(a[0], "edge(c - int3(0,0,1))") << " + "
                  << at(a[0], "edge(c + int3(0,0,1))") << " - 6 * v) / (H * H); }\n";
                break;
            case Op::Gx:
            case Op::Gy:
            case Op::Gz: {
                const char* unit = n.op == Op::Gx ? "int3(1,0,0)" : (n.op == Op::Gy ? "int3(0,1,0)" : "int3(0,0,1)");
                f << "float " << name << "(int3 c) { int3 hi = edge(c + " << unit << "); int3 lo = edge(c - " << unit
                  << "); int span = hi.x - lo.x + hi.y - lo.y + hi.z - lo.z; return span > 0 ? (" << a[0] << "(hi) - " << a[0]
                  << "(lo)) / (H * span) : 0; }\n";
                break;
            }
            case Op::Diffuse: {
                const std::string& k = a[0];
                const std::string& v = a[1];
                f << "float " << name << "(int3 c) { float ka = " << k << "(c); float v = " << v
                  << "(c); float acc = 0; int3 o;\n";
                const char* sides[6][2] = {{"c.x > 0", "c - int3(1,0,0)"}, {"c.x + 1 < NX", "c + int3(1,0,0)"},
                    {"c.y > 0", "c - int3(0,1,0)"}, {"c.y + 1 < NY", "c + int3(0,1,0)"}, {"c.z > 0", "c - int3(0,0,1)"},
                    {"c.z + 1 < NZ", "c + int3(0,0,1)"}};
                for (const auto& side : sides) {
                    f << "  if (" << side[0] << ") { o = " << side[1] << "; acc += face(ka, " << k << "(o), v, " << v << "(o)); }\n";
                }
                f << "  return acc / (H * H); }\n";
                break;
            }
            case Op::Adv:
                f << "float up_" << name << "(int3 c, float speed, int3 unit, int pos, int count) {\n"
                  << "  if (speed > 0) return pos > 0 ? speed * (" << a[0] << "(c) - " << a[0] << "(c - unit)) / H : 0;\n"
                  << "  return pos + 1 < count ? speed * (" << a[0] << "(c + unit) - " << a[0] << "(c)) / H : 0; }\n"
                  << "float " << name << "(int3 c) { return -(up_" << name << "(c, " << a[1]
                  << "(c), int3(1,0,0), c.x, NX) + up_" << name << "(c, " << a[2] << "(c), int3(0,1,0), c.y, NY) + up_" << name
                  << "(c, " << a[3] << "(c), int3(0,0,1), c.z, NZ)); }\n";
                break;
            case Op::Fall:
                if (always_scalar(*n.args[0])) {
                    simple("0");
                    break;
                }
                f << "float flux_" << name << "(int3 p) { if (p.z == 0) return 0; int3 b = p - int3(0,0,1);\n"
                  << "  return max(0, " << a[2] << "(p)) / H * max(0, " << a[0] << "(p)) * saturate(1 - " << a[1] << "(b) - " << a[0]
                  << "(b)); }\n"
                  << "float " << name << "(int3 c) { float o = -flux_" << name << "(c); if (c.z + 1 < NZ) o += flux_" << name
                  << "(c + int3(0,0,1)); return o; }\n";
                break;
            case Op::Spread:
                if (always_scalar(*n.args[0])) {
                    simple("0");
                    break;
                }
                f << "float flux_" << name << "(int3 a, int3 b) { float fa = " << a[0] << "(a); float fb = " << a[0] << "(b);\n"
                  << "  int3 r = fa > fb ? b : a;\n"
                  << "  return 0.5 * (max(0, " << a[2] << "(a)) + max(0, " << a[2] << "(b))) / (H * H) * (fa - fb) * saturate(1 - "
                  << a[1] << "(r) - " << a[0] << "(r)); }\n"
                  << "float " << name << "(int3 c) { float o = 0;\n"
                  << "  if (c.y > 0) o += flux_" << name << "(c - int3(0,1,0), c);\n"
                  << "  if (c.x > 0) o += flux_" << name << "(c - int3(1,0,0), c);\n"
                  << "  if (c.x + 1 < NX) o -= flux_" << name << "(c, c + int3(1,0,0));\n"
                  << "  if (c.y + 1 < NY) o -= flux_" << name << "(c, c + int3(0,1,0));\n"
                  << "  return o; }\n";
                break;
            case Op::Topple:
                if (always_scalar(*n.args[0])) {
                    simple("0");
                    break;
                }
                f << "float room_" << name << "(int3 j) { return saturate(1 - " << a[1] << "(j) - " << a[0] << "(j)); }\n"
                  << "float send_" << name << "(int3 p, int3 side) {\n"
                  << "  float here = " << a[0] << "(p); if (here <= 0 || p.z == 0) return 0;\n"
                  << "  int3 down = int3(0,0,1); if (room_" << name << "(p - down) > 0.01) return 0;\n"
                  << "  int3 j = p + side; if (any(j != edge(j))) return 0;\n"
                  << "  return 0.25 * max(0, " << a[2] << "(p)) / H * here * min(room_" << name << "(j), room_" << name << "(j - down)); }\n"
                  << "float " << name << "(int3 c) { float o = 0; int3 X = int3(1,0,0); int3 Y = int3(0,1,0); int3 Z = int3(0,0,1);\n"
                  << "  o -= send_" << name << "(c, -X); o -= send_" << name << "(c, X); o -= send_" << name << "(c, -Y); o -= send_" << name << "(c, Y);\n"
                  << "  if (c.z + 1 < NZ) {\n"
                  << "    int3 up = c + Z;\n"
                  << "    if (c.x + 1 < NX) o += send_" << name << "(up + X, -X);\n"
                  << "    if (c.x > 0) o += send_" << name << "(up - X, X);\n"
                  << "    if (c.y + 1 < NY) o += send_" << name << "(up + Y, -Y);\n"
                  << "    if (c.y > 0) o += send_" << name << "(up - Y, Y);\n"
                  << "  }\n  return o; }\n";
                break;
            case Op::Distance:
            case Op::Poisson: {
                const int mat = impl_.mat_for(n, program_);
                simple("A[" + std::to_string(impl_.mats[static_cast<std::size_t>(mat)].result) + " * CELLS + I(c)]");
                break;
            }
            case Op::MaxOf:
            case Op::MinOf:
            case Op::SumOf: {
                const int r = impl_.reduction_for(n, program_);
                simple(uniform_slot(impl_.reductions[static_cast<std::size_t>(r)].uniform));
                break;
            }
            }
            text_ += f.str();
            names_.emplace(&n, name);
            return name;
        }

        [[nodiscard]] const std::string& text() const { return text_; }

    private:
        Impl& impl_;
        Program& program_;
        std::map<const Node*, std::string> names_;
        std::string text_;
    };

    int mat_for(Node& node, Program& program)
    {
        if (const auto found = mat_of.find(&node); found != mat_of.end()) {
            if (std::find(program.mats.begin(), program.mats.end(), found->second) == program.mats.end()) {
                program.mats.push_back(found->second);
            }
            return found->second;
        }
        Mat mat;
        mat.node = &node;
        mat.poisson = node.op == Op::Poisson;
        // Distance: seed-in, seed-out, two scratch rows, to-inside, to-outside, the answer.
        // Poisson: the source as last solved, two buffers of phi (the first holds the answer).
        mat.seed = reserve_aux(mat.poisson ? 3 : 7);
        mat.result = mat.poisson ? mat.seed + 1 : mat.seed + 6;
        mat.iterations = node.iterations;
        Emitter emitter(*this, program);
        const std::string arg = emitter.emit(*node.args[0]);
        std::ostringstream main;
        main << kPrelude << kBuffers << emitter.text() << kEachCell;
        const std::string arm = "ARGS.InterlockedOr(0, " + std::to_string(gx) + "); ARGS.InterlockedOr(4, " +
            std::to_string(gy) + "); ARGS.InterlockedOr(8, " + std::to_string(shape.nz) + ");";
        const std::string seed = std::to_string(mat.seed);
        if (mat.poisson) {
            main << "    float v = " << arg << "(c);\n    int s = " << seed << " * CELLS + i;\n"
                 << "    if (asuint(A[s]) != asuint(v)) { A[s] = v; " << arm << " }\n}\n";
        } else {
            main << "    float seed = " << arg << "(c) >= 0.5 ? 0 : " << number(kFar) << ";\n    int s = " << seed << " * CELLS + i;\n"
                 << "    if (asuint(A[s]) != asuint(seed)) { A[s] = seed; A[s + CELLS] = seed == 0 ? " << number(kFar) << " : 0; " << arm
                 << " }\n}\n";
        }
        mat.compare = add_shader(main.str());
        mats.push_back(mat);
        const int id = static_cast<int>(mats.size()) - 1;
        mat_of.emplace(&node, id);
        program.mats.push_back(id);
        return id;
    }

    int reduction_for(Node& node, Program& program)
    {
        int id = -1;
        if (const auto found = reduction_of.find(&node); found != reduction_of.end()) {
            id = found->second;
        } else {
            Reduction r;
            r.node = &node;
            r.uniform = uniforms++;
            Emitter emitter(*this, program);
            const std::string arg = emitter.emit(*node.args[0]);
            std::ostringstream main;
            // Plane by plane into D, as sum, least, most.
            main << kPrelude << kBuffers << kPlaneSum << emitter.text() << "[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID)\n{\n"
                 << "    int z = id.x; Sum sum; sum.s = 0; sum.c = 0; float lo = 1e30; float hi = -1e30;\n"
                 << "    for (int y = 0; y < NY; ++y) for (int x = 0; x < NX; ++x) { float v = " << arg
                 << "(int3(x, y, z)); add(sum, v); lo = min(lo, v); hi = max(hi, v); }\n"
                 << "    D[3 * z] = sum.s + sum.c; D[3 * z + 1] = lo; D[3 * z + 2] = hi;\n}\n";
            r.shader = add_shader(main.str());
            reductions.push_back(r);
            id = static_cast<int>(reductions.size()) - 1;
            reduction_of.emplace(&node, id);
        }
        if (std::find(program.reductions.begin(), program.reductions.end(), id) == program.reductions.end()) {
            program.reductions.push_back(id);
        }
        return id;
    }

    Program compile_term(Term& term)
    {
        Program program;
        Emitter emitter(*this, program);
        std::vector<std::string> names;
        for (Equation& eq : term.equations) {
            names.push_back(emitter.emit(*eq.rhs));
        }
        std::ostringstream gather;
        gather << kPrelude << kBuffers << emitter.text() << kEachCell;
        for (std::size_t e = 0; e < names.size(); ++e) {
            gather << "    D[" << e << " * CELLS + i] = " << names[e] << "(c) * DT;\n";
        }
        gather << "}\n";
        program.gather = add_shader(gather.str());
        std::ostringstream store;
        store << kPrelude << kStoreBuffers << kEachCell << "    float d;\n";
        for (std::size_t e = 0; e < names.size(); ++e) {
            const int t = term.equations[e].target;
            store << "    d = D[" << e << " * CELLS + i]; if (d != 0) { S[" << t << " * CELLS + i] += d; if (CHANGED[" << t << "] == 0) CHANGED[" << t << "] = 1; }\n";
        }
        store << "}\n";
        program.store = add_shader(store.str());
        return program;
    }

    Program compile_derive(Derive& derive, bool add)
    {
        Program program;
        Emitter emitter(*this, program);
        const std::string name = emitter.emit(*derive.rhs);
        std::ostringstream gather;
        gather << kPrelude << kBuffers << emitter.text() << kEachCell << "    D[i] = " << name << "(c);\n}\n";
        program.gather = add_shader(gather.str());
        std::ostringstream store;
        const int t = derive.target;
        store << kPrelude << kStoreBuffers << kEachCell << "    int k = " << t << " * CELLS + i; float next = " << (add ? "S[k] + " : "") << "D[i];\n"
              << "    if (asuint(S[k]) != asuint(next)) { S[k] = next; if (CHANGED[" << t << "] == 0) CHANGED[" << t << "] = 1; }\n}\n";
        program.store = add_shader(store.str());
        return program;
    }

    void add_generic_shaders()
    {
        const std::string head = std::string(kPrelude) + kBuffers;
        std::ostringstream fill;
        fill << head << kEachCell << "    A[F0 * CELLS + i] = asfloat((uint)F1);\n}\n";
        fill_cs = add_shader(fill.str());
        // One axis of the squared distance transform: outward from each cell until no farther cell can
        // beat what is already found. Exact, cheap near a surface, and only ever run when a seed changed.
        std::ostringstream scan;
        scan << head << kEachCell
             << "    int n = F2 == 0 ? NX : (F2 == 1 ? NY : NZ); int q = F2 == 0 ? c.x : (F2 == 1 ? c.y : c.z);\n"
             << "    int stride = F2 == 0 ? 1 : (F2 == 1 ? NX : NXY); int base = F0 * CELLS + i - q * stride; float best = " << number(kFar) << ";\n"
             << "    [loop] for (int r = 0; r < n; ++r) {\n"
             << "        float rr = (float)(r * r); if (rr >= best) break;\n"
             << "        int lo = q - r; int hi = q + r; if (lo < 0 && hi >= n) break;\n"
             << "        if (lo >= 0) best = min(best, A[base + lo * stride] + rr);\n"
             << "        if (hi < n) best = min(best, A[base + hi * stride] + rr);\n"
             << "    }\n    A[F1 * CELLS + i] = best;\n}\n";
        scan_cs = add_shader(scan.str());
        // to-inside (F0), to-outside (F1), the seed (F2) -> signed metres (F3). The surface lies half a cell
        // from the centre of the last solid cell, and beyond the lattice is air.
        std::ostringstream sign;
        sign << head << kEachCell
             << "    float far_m = (NX + NY + NZ) * H; bool inside = A[F2 * CELLS + i] == 0; float d;\n"
             << "    if (inside) {\n"
             << "        float border = 1e30;\n"
             << "        if (NX > 1) border = min(border, min(c.x + 1, NX - c.x));\n"
             << "        if (NY > 1) border = min(border, min(c.y + 1, NY - c.y));\n"
             << "        if (NZ > 1) border = min(border, min(c.z + 1, NZ - c.z));\n"
             << "        float o = A[F1 * CELLS + i]; d = -(min(o < 1e29 ? sqrt(o) : far_m / H, border) - 0.5) * H;\n"
             << "    } else {\n"
             << "        float o = A[F0 * CELLS + i]; d = o < 1e29 ? (sqrt(o) - 0.5) * H : far_m;\n"
             << "    }\n    A[F3 * CELLS + i] = d;\n}\n";
        signed_cs = add_shader(sign.str());
        // One Jacobi sweep: phi (F0) -> phi (F1), source F2; phi = 0 outside the lattice.
        std::ostringstream jacobi;
        jacobi << head << kEachCell
               << "    int p = F0 * CELLS + i; float sum = 0; int axes = 0;\n"
               << "    if (NX > 1) { ++axes; if (c.x > 0) sum += A[p - 1]; if (c.x + 1 < NX) sum += A[p + 1]; }\n"
               << "    if (NY > 1) { ++axes; if (c.y > 0) sum += A[p - NX]; if (c.y + 1 < NY) sum += A[p + NX]; }\n"
               << "    if (NZ > 1) { ++axes; if (c.z > 0) sum += A[p - NXY]; if (c.z + 1 < NZ) sum += A[p + NXY]; }\n"
               << "    A[F1 * CELLS + i] = (sum - H * H * A[F2 * CELLS + i]) / (2.0 * max(axes, 1));\n}\n";
        jacobi_cs = add_shader(jacobi.str());
        std::ostringstream copy;
        copy << head << kEachCell << "    A[F1 * CELLS + i] = A[F0 * CELLS + i];\n}\n";
        copy_cs = add_shader(copy.str());
        // A field's contract, over every cell: only a cell that breaks it touches the counters. Three words a
        // check: cells out of range, cells not finite, and ~(the first bad cell), so the least index wins.
        std::ostringstream range;
        range << kPrelude << "StructuredBuffer<float> S : register(t0);\nRWStructuredBuffer<uint> R : register(u0);\n" << kEachCell
              << "    float v = S[F0 * CELLS + i]; bool nan = (asuint(v) & 0x7f800000) == 0x7f800000;\n"
              << "    if (nan || v < A0 || v > A1) { uint o; InterlockedAdd(R[F1 + (nan ? 1 : 0)], 1, o); InterlockedMax(R[F1 + 2], ~(uint)i, o); }\n}\n";
        range_cs = add_shader(range.str());
        // A field's total, in two levels so all of the card works on it: every row into D, then the rows of
        // each plane into two words of the record (sum, sum of magnitudes).
        std::ostringstream rows;
        rows << kPrelude << kPlaneSum << "StructuredBuffer<float> S : register(t0);\nRWStructuredBuffer<float> D : register(u0);\n"
             << "[numthreads(32,1,1)] void main(uint3 id : SV_DispatchThreadID)\n{\n    int r = id.x; if (r >= NY * NZ) return;\n"
             << "    Sum sum; sum.s = 0; sum.c = 0; Sum mag; mag.s = 0; mag.c = 0; int base = F0 * CELLS + r * NX;\n"
             << "    [loop] for (int x = 0; x < NX; ++x) { float v = S[base + x]; add(sum, v); add(mag, abs(v)); }\n"
             << "    D[2 * r] = sum.s + sum.c; D[2 * r + 1] = mag.s + mag.c;\n}\n";
        rows_cs = add_shader(rows.str());
        std::ostringstream planes;
        planes << kPrelude << kPlaneSum << "StructuredBuffer<float> D : register(t0);\nRWStructuredBuffer<float> R : register(u0);\n"
               << "[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID)\n{\n    int z = id.x; Sum sum; sum.s = 0; sum.c = 0; Sum mag; mag.s = 0; mag.c = 0;\n"
               << "    [loop] for (int y = 0; y < NY; ++y) { add(sum, D[2 * (z * NY + y)]); add(mag, D[2 * (z * NY + y) + 1]); }\n"
               << "    R[F1 + 2 * z] = sum.s + sum.c; R[F1 + 2 * z + 1] = mag.s + mag.c;\n}\n";
        planes_cs = add_shader(planes.str());
        std::ostringstream clamp;
        clamp << kPrelude << "RWStructuredBuffer<float> S : register(u0);\n" << kEachCell
              << "    int k = F0 * CELLS + i; float v = S[k]; if (v < A0) S[k] = A0; else if (v > A1) S[k] = A1;\n}\n";
        clamp_cs = add_shader(clamp.str());
    }

    bool compile_all(std::string& why)
    {
        shaders.resize(sources.size());
        // Text not seen before goes to the compiler on all cores; a law edit recompiles only what it changed.
        std::vector<int> missing;
        for (std::size_t s = 0; s < sources.size(); ++s) {
            const auto found = dev->shaders.find(sources[s]);
            if (found != dev->shaders.end()) {
                shaders[s] = found->second;
            } else {
                missing.push_back(static_cast<int>(s));
            }
        }
        std::vector<std::string> errors(missing.size());
#pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < static_cast<int>(missing.size()); ++m) {
            const std::string& text = sources[static_cast<std::size_t>(missing[static_cast<std::size_t>(m)])];
            const ComPtr<ID3DBlob> blob = compile_shader(text, errors[static_cast<std::size_t>(m)]);
            if (!blob) {
                continue;
            }
            dev->device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr,
                &shaders[static_cast<std::size_t>(missing[static_cast<std::size_t>(m)])]);
        }
        for (std::size_t m = 0; m < missing.size(); ++m) {
            const auto s = static_cast<std::size_t>(missing[m]);
            if (!errors[m].empty() || !shaders[s]) {
                why = "a shader did not compile: " + errors[m].substr(0, 400);
                return false;
            }
            dev->shaders.emplace(sources[s], shaders[s]);
        }
        return true;
    }

    // ---- resources ----------------------------------------------------------------------------------------

    bool make_buffer(std::size_t words, ComPtr<ID3D11Buffer>& buffer, ComPtr<ID3D11ShaderResourceView>* srv,
        ComPtr<ID3D11UnorderedAccessView>& uav)
    {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<UINT>(words * 4);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | (srv != nullptr ? D3D11_BIND_SHADER_RESOURCE : 0);
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = 4;
        if (FAILED(dev->device->CreateBuffer(&desc, nullptr, &buffer))) {
            return false;
        }
        if (srv != nullptr && FAILED(dev->device->CreateShaderResourceView(buffer.Get(), nullptr, srv->GetAddressOf()))) {
            return false;
        }
        return SUCCEEDED(dev->device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
    }

    bool make_staging(std::size_t bytes, ComPtr<ID3D11Buffer>& buffer)
    {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<UINT>(bytes);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        return SUCCEEDED(dev->device->CreateBuffer(&desc, nullptr, &buffer));
    }

    bool make_constants(std::size_t bytes, ComPtr<ID3D11Buffer>& buffer)
    {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<UINT>(bytes);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        return SUCCEEDED(dev->device->CreateBuffer(&desc, nullptr, &buffer));
    }

    bool make_resources(const Law& law, std::string& why)
    {
        std::size_t equations = 1;
        std::size_t words = 16;
        for (const Term& term : law.terms) {
            equations = std::max(equations, term.equations.size());
            words += 3 * term.writes.size();
            for (const std::vector<int>& group : term.conserves) {
                words += 2 * group.size() * 2 * static_cast<std::size_t>(shape.nz);
            }
        }
        words += 3 * law.derives.size();
        record_words = static_cast<int>(words);
        const std::size_t n = static_cast<std::size_t>(cells);
        bool ok = make_buffer(n * static_cast<std::size_t>(fields_n), state, &state_srv, state_uav) &&
            make_buffer(std::max(n * equations, 2 * static_cast<std::size_t>(shape.ny) * static_cast<std::size_t>(shape.nz)), delta, &delta_srv, delta_uav) &&
            make_buffer(n * static_cast<std::size_t>(std::max(aux_slots, 1)), aux, nullptr, aux_uav) &&
            make_buffer(static_cast<std::size_t>(record_words), records, nullptr, records_uav) &&
            make_buffer(static_cast<std::size_t>(fields_n), changed, nullptr, changed_uav) &&
            make_staging(static_cast<std::size_t>(fields_n) * 4, changed_read) &&
            make_staging(n * static_cast<std::size_t>(fields_n) * 4, state_read) &&
            make_staging(static_cast<std::size_t>(record_words) * 4, records_read) &&
            make_staging(static_cast<std::size_t>(shape.nz) * 12, small_read) &&
            make_constants(32 + 16 * 256, grid_cb) && make_constants(32, pass_cb);
        if (ok) {
            D3D11_BUFFER_DESC desc{};
            desc.ByteWidth = 12;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
            desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            ok = SUCCEEDED(dev->device->CreateBuffer(&desc, nullptr, &args));
            if (ok) {
                D3D11_UNORDERED_ACCESS_VIEW_DESC view{};
                view.Format = DXGI_FORMAT_R32_TYPELESS;
                view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
                view.Buffer.NumElements = 3;
                view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
                ok = SUCCEEDED(dev->device->CreateUnorderedAccessView(args.Get(), &view, &args_uav));
            }
        }
        if (!ok) {
            why = "could not allocate the lattice on the GPU";
        }
        return ok;
    }

    // ---- dispatching ----------------------------------------------------------------------------------------

    void set_pass(int f0, int f1, int f2, int f3, float dt, float a0 = 0.0f, float a1 = 0.0f)
    {
        struct {
            int f[4];
            float dt, a0, a1, a2;
        } data{{f0, f1, f2, f3}, dt, a0, a1, 0.0f};
        D3D11_MAPPED_SUBRESOURCE mapped{};
        dev->context->Map(pass_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        std::memcpy(mapped.pData, &data, sizeof(data));
        dev->context->Unmap(pass_cb.Get(), 0);
    }

    void write_grid()
    {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        dev->context->Map(grid_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        std::memcpy(mapped.pData, grid_data.data(), grid_data.size() * 4);
        dev->context->Unmap(grid_cb.Get(), 0);
    }

    void bind(ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* u0, ID3D11UnorderedAccessView* u1 = nullptr,
        ID3D11UnorderedAccessView* u2 = nullptr)
    {
        ID3D11ShaderResourceView* no_srv = nullptr;
        ID3D11UnorderedAccessView* no_uavs[3] = {nullptr, nullptr, nullptr};
        ID3D11DeviceContext* c = dev->context.Get();
        c->CSSetShaderResources(0, 1, &no_srv);
        c->CSSetUnorderedAccessViews(0, 3, no_uavs, nullptr);
        ID3D11UnorderedAccessView* uavs[3] = {u0, u1, u2};
        c->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);
        c->CSSetShaderResources(0, 1, &srv);
    }

    void run(int shader)
    {
        dev->context->CSSetShader(shaders[static_cast<std::size_t>(shader)].Get(), nullptr, 0);
        dev->context->Dispatch(static_cast<UINT>(gx), static_cast<UINT>(gy), static_cast<UINT>(shape.nz));
    }

    // Runs only if the compare pass before it armed the arguments.
    void run_armed(int shader)
    {
        dev->context->CSSetShader(shaders[static_cast<std::size_t>(shader)].Get(), nullptr, 0);
        dev->context->DispatchIndirect(args.Get(), 0);
    }

    void fill_slot(int slot, std::uint32_t bits)
    {
        bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get());
        set_pass(slot, static_cast<int>(bits), 0, 0, 0.0f);
        run(fill_cs);
    }

    void run_mat(const Mat& mat)
    {
        const UINT nothing[4] = {0, 0, 0, 0};
        dev->context->ClearUnorderedAccessViewUint(args_uav.Get(), nothing);
        bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get(), args_uav.Get());
        run(mat.compare);
        bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get());
        if (mat.poisson) {
            int from = mat.seed + 1;
            int to = mat.seed + 2;
            for (int it = 0; it < mat.iterations; ++it) {
                set_pass(from, to, mat.seed, 0, 0.0f);
                run_armed(jacobi_cs);
                std::swap(from, to);
            }
            if (from != mat.result) {
                set_pass(from, mat.result, 0, 0, 0.0f);
                run_armed(copy_cs);
            }
            return;
        }
        const int seed_in = mat.seed;
        const int seed_out = mat.seed + 1;
        const int scratch_a = mat.seed + 2;
        const int scratch_b = mat.seed + 3;
        const int to_in = mat.seed + 4;
        const int to_out = mat.seed + 5;
        for (const auto& [seed, result] : {std::pair{seed_in, to_in}, std::pair{seed_out, to_out}}) {
            set_pass(seed, scratch_a, 0, 0, 0.0f);
            run_armed(scan_cs);
            set_pass(scratch_a, scratch_b, 1, 0, 0.0f);
            run_armed(scan_cs);
            set_pass(scratch_b, result, 2, 0, 0.0f);
            run_armed(scan_cs);
        }
        set_pass(to_in, to_out, seed_in, mat.result, 0.0f);
        run_armed(signed_cs);
    }

    // Stalls for the card: taken only for a term that is about to run and whose law asks for a reduction.
    double run_reduction(const Reduction& r, const Law& law)
    {
        if (always_zero(*r.node, zero, law.param_values)) {
            return 0.0;
        }
        bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get());
        dev->context->CSSetShader(shaders[static_cast<std::size_t>(r.shader)].Get(), nullptr, 0);
        dev->context->Dispatch(static_cast<UINT>(shape.nz), 1, 1);
        const D3D11_BOX box{0, 0, 0, static_cast<UINT>(shape.nz) * 12, 1, 1};
        dev->context->CopySubresourceRegion(small_read.Get(), 0, 0, 0, 0, delta.Get(), 0, &box);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        dev->context->Map(small_read.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        const auto* v = static_cast<const float*>(mapped.pData);
        double sum = 0.0;
        double lo = v[1];
        double hi = v[2];
        for (int z = 0; z < shape.nz; ++z) {
            sum += v[3 * z];
            lo = std::min(lo, static_cast<double>(v[3 * z + 1]));
            hi = std::max(hi, static_cast<double>(v[3 * z + 2]));
        }
        dev->context->Unmap(small_read.Get(), 0);
        return r.node->op == Op::SumOf ? sum : (r.node->op == Op::MaxOf ? hi : lo);
    }

    // A max_dt, on the CPU: it is built of numbers and reductions.
    double scalar(const Node& n, const Law& law)
    {
        auto a = [&](std::size_t i) { return scalar(*n.args[i], law); };
        switch (n.op) {
        case Op::Const:
            return n.number;
        case Op::Param:
            return law.param_values[static_cast<std::size_t>(n.index)];
        case Op::Cell:
            return shape.cell;
        case Op::Neg:
            return -a(0);
        case Op::Abs:
            return std::abs(a(0));
        case Op::Sqrt:
            return std::sqrt(std::max(0.0, a(0)));
        case Op::Exp:
            return std::exp(std::min(a(0), 80.0));
        case Op::Log:
            return std::log(std::max(a(0), 1e-30));
        case Op::Step:
            return a(0) > 0.0 ? 1.0 : 0.0;
        case Op::Pos:
            return std::max(a(0), 0.0);
        case Op::Add:
            return a(0) + a(1);
        case Op::Sub:
            return a(0) - a(1);
        case Op::Mul:
            return a(0) * a(1);
        case Op::Div: {
            const double b = a(1);
            return b == 0.0 ? 0.0 : a(0) / b;
        }
        case Op::Pow:
            return std::pow(a(0), a(1));
        case Op::Min:
            return std::min(a(0), a(1));
        case Op::Max:
            return std::max(a(0), a(1));
        case Op::Clamp: {
            const double lo = a(1);
            return std::clamp(a(0), lo, std::max(lo, a(2)));
        }
        case Op::MaxOf:
        case Op::MinOf:
        case Op::SumOf: {
            const auto found = reduction_of.find(&n);
            return found == reduction_of.end() ? 0.0 : run_reduction(reductions[static_cast<std::size_t>(found->second)], law);
        }
        default:
            return 0.0;
        }
    }

    int add_range(int field, const FieldDecl& decl)
    {
        const int slot = next_record;
        next_record += 3;
        bind(state_srv.Get(), records_uav.Get());
        set_pass(field, slot, 0, 0, 0.0f, static_cast<float>(decl.lo), static_cast<float>(decl.hi));
        run(range_cs);
        return slot;
    }

    int add_total(int field)
    {
        const int slot = next_record;
        next_record += 2 * shape.nz;
        bind(state_srv.Get(), delta_uav.Get());
        set_pass(field, slot, 0, 0, 0.0f);
        dev->context->CSSetShader(shaders[static_cast<std::size_t>(rows_cs)].Get(), nullptr, 0);
        dev->context->Dispatch(static_cast<UINT>((shape.ny * shape.nz + 31) / 32), 1, 1);
        bind(delta_srv.Get(), records_uav.Get());
        dev->context->CSSetShader(shaders[static_cast<std::size_t>(planes_cs)].Get(), nullptr, 0);
        dev->context->Dispatch(static_cast<UINT>(shape.nz), 1, 1);
        return slot;
    }

    void run_program_inputs(const Program& program, const Law& law)
    {
        bool changed = false;
        for (const int r : program.reductions) {
            const Reduction& reduction = reductions[static_cast<std::size_t>(r)];
            grid_data[8 + static_cast<std::size_t>(reduction.uniform)] = static_cast<float>(run_reduction(reduction, law));
            changed = true;
        }
        if (changed) {
            write_grid();
        }
        for (const int m : program.mats) {
            run_mat(mats[static_cast<std::size_t>(m)]);
        }
    }

    void apply_term(std::size_t index, Term& term, double dt, bool respect, Law& law, std::vector<Violation>& violations,
        std::vector<int>& substeps)
    {
        const bool idle = std::all_of(term.equations.begin(), term.equations.end(), [&](const Equation& eq) {
            return always_zero(*eq.rhs, zero, law.param_values) || bounds_of(*eq.rhs, flowing, law.param_values, shape.cell).none();
        });
        substeps[index] = 1;
        if (idle) {
            return;
        }
        int count = 1;
        if (respect && term.max_dt) {
            const double limit = scalar(*term.max_dt, law);
            if (limit > 0.0 && std::isfinite(limit) && dt > limit) {
                const double wanted = std::ceil(dt / limit);
                if (wanted > Solver::kMaxSubsteps) {
                    violations.push_back(Violation{.kind = Violation::Kind::SubstepCap, .term = term.name,
                        .detail = "max_dt " + std::to_string(limit) + " asks for " + std::to_string(wanted) +
                            " substeps; capped at " + std::to_string(Solver::kMaxSubsteps)});
                    count = Solver::kMaxSubsteps;
                } else {
                    count = static_cast<int>(wanted);
                }
            }
        }
        substeps[index] = count;
        const Program& program = term_programs[index];
        std::vector<ConserveCheck> checks;
        for (std::size_t g = 0; checking && g < term.conserves.size(); ++g) {
            ConserveCheck check{.term = index, .group = g, .substeps = count};
            for (std::size_t i = 0; i < term.conserves[g].size(); ++i) {
                const int slot = add_total(term.conserves[g][i]);
                if (i == 0) {
                    check.first_before = slot;
                }
            }
            checks.push_back(check);
        }
        const float sub_dt = static_cast<float>(dt / count);
        for (int s = 0; s < count; ++s) {
            run_program_inputs(program, law);
            bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get());
            set_pass(0, 0, 0, 0, sub_dt);
            run(program.gather);
            bind(delta_srv.Get(), state_uav.Get(), changed_uav.Get());
            run(program.store);
        }
        for (const int field : term.writes) {
            const FieldDecl& decl = law.fields[static_cast<std::size_t>(field)];
            if (decl.clamp) {
                bind(nullptr, state_uav.Get());
                set_pass(field, 0, 0, 0, 0.0f, static_cast<float>(decl.lo), static_cast<float>(decl.hi));
                run(clamp_cs);
            }
            if (checking) {
                range_checks.push_back(RangeCheck{.term = index, .field = field, .slot = add_range(field, decl)});
            }
        }
        flow_on(term, dt, law);
        for (ConserveCheck& check : checks) {
            const std::vector<int>& group = term.conserves[check.group];
            for (std::size_t i = 0; i < group.size(); ++i) {
                const int slot = add_total(group[i]);
                if (i == 0) {
                    check.first_after = slot;
                }
            }
            conserve_checks.push_back(check);
        }
    }

    // What this term may have done to the bounds of what it writes, for the terms after it. If this is ever short, the
    // next term only starts a step later: the next step's bounds are read from the cells themselves.
    void flow_on(const Term& term, double dt, const Law& law)
    {
        for (int round = 0; round < 2; ++round) {
            std::vector<Bounds> next = flowing;
            for (const Equation& eq : term.equations) {
                // Heat going down its own gradient stays between the hottest and the coldest, at any stable dt.
                const bool within = eq.rhs->op == Op::Diffuse && eq.rhs->args[1]->op == Op::Field && eq.rhs->args[1]->index == eq.target;
                const auto t = static_cast<std::size_t>(eq.target);
                if (!within) {
                    const Bounds rate = bounds_of(*eq.rhs, flowing, law.param_values, shape.cell);
                    next[t].lo = std::min(next[t].lo, flowing[t].lo + times(dt, std::min(rate.lo, 0.0)));
                    next[t].hi = std::max(next[t].hi, flowing[t].hi + times(dt, std::max(rate.hi, 0.0)));
                    const FieldDecl& decl = law.fields[t];
                    next[t].lo = std::max(next[t].lo, decl.lo);
                    next[t].hi = std::min(next[t].hi, decl.hi);
                }
                dirty[t] = 1;
            }
            flowing = std::move(next);
        }
    }

    void run_derives(Law& law, bool all, bool check)
    {
        // Parts of one field run together, so a field that one source adds to and another sets stays whole.
        // A derive reads the ones before it, so what is zero is decided as it comes up.
        std::vector<char> decided(law.fields.size(), 0);
        std::vector<char> needed(law.fields.size(), 0);
        for (std::size_t d = 0; d < law.derives.size(); ++d) {
            Derive& derive = law.derives[d];
            const auto t = static_cast<std::size_t>(derive.target);
            if (!decided[t]) {
                decided[t] = 1;
                bool nothing = true;
                for (const Derive& part : law.derives) {
                    nothing = nothing && (part.target != derive.target || always_zero(*part.rhs, zero, law.param_values));
                }
                // The same inputs give the same answer: nothing it reads moved, nothing to work out.
                bool moved = false;
                for (const Derive& part : law.derives) {
                    if (part.target == derive.target) {
                        moved = moved || std::any_of(part.reads.begin(), part.reads.end(), [&](int f) { return dirty[static_cast<std::size_t>(f)] != 0; });
                    }
                }
                needed[t] = all || (moved && (!zero[t] || !nothing));
                zero[t] = nothing;
                dirty[t] = needed[t] ? 1 : dirty[t];
            }
            if (!needed[t]) {
                continue;
            }
            const Program& program = derive_programs[d];
            run_program_inputs(program, law);
            bind(state_srv.Get(), delta_uav.Get(), aux_uav.Get());
            set_pass(0, 0, 0, 0, 0.0f);
            run(program.gather);
            bind(delta_srv.Get(), state_uav.Get(), changed_uav.Get());
            run(program.store);
        }
        if (check && checking) {
            std::set<int> seen;
            for (const Derive& derive : law.derives) {
                const FieldDecl& decl = law.fields[static_cast<std::size_t>(derive.target)];
                if (needed[static_cast<std::size_t>(derive.target)] && seen.insert(derive.target).second) {
                    range_checks.push_back(RangeCheck{.term = law.terms.size(), .field = derive.target, .slot = add_range(derive.target, decl)});
                }
            }
        }
    }

    // ---- the lattice, between CPU and GPU ---------------------------------------------------------------------

    void upload(const FieldSet& fields, FieldEdits& edits, const Law& law, bool derives_follow = false)
    {
        const std::size_t plane = static_cast<std::size_t>(shape.nx) * static_cast<std::size_t>(shape.ny);
        auto scan = [&](int id) {
            look_over(id, fields.values(id).data());
        };
        known_zero.resize(fields.count(), 0);
        known.resize(fields.count());
        dirty.resize(fields.count(), 1);
        // Most fields of most lattices are all zero, here and on the card: those need not go up at all.
        std::vector<char> was_zero = known_zero;
        std::vector<int> going;
        for (std::size_t id = 0; id < fields.count(); ++id) {
            const std::vector<char>* marks = !edits.all && id < edits.planes.size() ? &edits.planes[id] : nullptr;
            if (!edits.all && (marks == nullptr || marks->empty())) {
                continue;
            }
            // What is about to be derived again need not go up; the derive compares with what the card holds,
            // which is what the CPU held before.
            if (derives_follow && edits.all && law.fields[id].derived && uploaded_once) {
                continue;
            }
            going.push_back(static_cast<int>(id));
        }
#pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < static_cast<int>(going.size()); ++k) {
            scan(going[static_cast<std::size_t>(k)]);
        }
        for (const int field : going) {
            const std::size_t id = static_cast<std::size_t>(field);
            if (uploaded_once && was_zero[id] && known_zero[id]) {
                continue;
            }
            dirty[id] = 1;
            const std::vector<char>* marks = !edits.all && id < edits.planes.size() ? &edits.planes[id] : nullptr;
            for (int z = 0; z < shape.nz; ++z) {
                if (marks != nullptr && !(*marks)[static_cast<std::size_t>(z)]) {
                    continue;
                }
                int end = z + 1;
                while (end < shape.nz && (marks == nullptr || (*marks)[static_cast<std::size_t>(end)])) {
                    ++end;
                }
                const std::size_t base = id * static_cast<std::size_t>(cells) + static_cast<std::size_t>(z) * plane;
                const std::size_t bytes = static_cast<std::size_t>(end - z) * plane * 4;
                const D3D11_BOX box{static_cast<UINT>(base * 4), 0, 0, static_cast<UINT>(base * 4 + bytes), 1, 1};
                dev->context->UpdateSubresource(state.Get(), 0, &box, fields.values(field).data() + static_cast<std::size_t>(z) * plane, 0, 0);
                z = end;
            }
        }
        edits.all = false;
        edits.planes.clear();
        edits.shift[0] = edits.shift[1] = edits.shift[2] = 0;
        edits.boxes.clear();
        uploaded_once = true;
        zero = known_zero;
        for (std::size_t p = 0; p < law.param_values.size(); ++p) {
            const float value = static_cast<float>(law.param_values[p]);
            if (grid_data[8 + p] != value) {
                std::fill(dirty.begin(), dirty.end(), 1); // what the derives worked out, they worked out with the old one
            }
            grid_data[8 + p] = value;
        }
        write_grid();
        flowing = known;
    }

    // How far a field goes, and whether it is all zero.
    void look_over(int id, const float* v)
    {
        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < cells; ++i) {
            lo = std::min(lo, v[i]);
            hi = std::max(hi, v[i]);
        }
        const auto f = static_cast<std::size_t>(id);
        known[f] = Bounds{lo, hi};
        known_zero[f] = lo == 0.0f && hi == 0.0f ? 1 : 0;
    }

    // What a step can change: every term still able to write gets to mark what it writes, until nothing
    // new turns up. A field left zero by all of them is exactly zero all the way through the step.
    void settle_zero_flags(const Law& law)
    {
        bool grew = true;
        while (grew) {
            grew = false;
            for (const Term& term : law.terms) {
                for (const Equation& eq : term.equations) {
                    if (zero[static_cast<std::size_t>(eq.target)] && !always_zero(*eq.rhs, zero, law.param_values)) {
                        zero[static_cast<std::size_t>(eq.target)] = 0;
                        grew = true;
                    }
                }
            }
        }
    }

    // What the shaders said changed comes back; a field no cell of which moved is not copied anywhere.
    void download(FieldSet& fields, std::vector<char>& changed_out)
    {
        dev->context->CopyResource(changed_read.Get(), changed.Get());
        {
            D3D11_MAPPED_SUBRESOURCE flags{};
            dev->context->Map(changed_read.Get(), 0, D3D11_MAP_READ, 0, &flags);
            const auto* words = static_cast<const std::uint32_t*>(flags.pData);
            touched.resize(fields.count());
            for (std::size_t id = 0; id < fields.count(); ++id) {
                touched[id] = words[id] != 0 ? 1 : 0;
            }
            dev->context->Unmap(changed_read.Get(), 0);
        }
        changed_out = touched;
        if (std::none_of(touched.begin(), touched.end(), [](char c) { return c != 0; })) {
            return;
        }
        for (std::size_t id = 0; id < fields.count(); ++id) {
            if (touched[id]) {
                const UINT base = static_cast<UINT>(id * static_cast<std::size_t>(cells) * 4);
                const D3D11_BOX box{base, 0, 0, base + static_cast<UINT>(cells) * 4, 1, 1};
                dev->context->CopySubresourceRegion(state_read.Get(), 0, base, 0, 0, state.Get(), 0, &box);
            }
        }
        D3D11_MAPPED_SUBRESOURCE mapped{};
        dev->context->Map(state_read.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        const auto* data = static_cast<const float*>(mapped.pData);
#pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < static_cast<int>(fields.count()); ++id) {
            if (touched[static_cast<std::size_t>(id)]) {
                const float* from = data + static_cast<std::size_t>(id) * static_cast<std::size_t>(cells);
                std::memcpy(fields.values(id).data(), from, static_cast<std::size_t>(cells) * 4);
                look_over(id, from);
            }
        }
        dev->context->Unmap(state_read.Get(), 0);
    }

    void read_contracts(const Law& law, std::vector<Violation>& violations)
    {
        if (next_record == 0) {
            return;
        }
        const D3D11_BOX box{0, 0, 0, static_cast<UINT>(next_record) * 4, 1, 1};
        dev->context->CopySubresourceRegion(records_read.Get(), 0, 0, 0, 0, records.Get(), 0, &box);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        dev->context->Map(records_read.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        const auto* words = static_cast<const std::uint32_t*>(mapped.pData);
        struct Totals {
            double sum = 0.0;
            double mag = 0.0;
        };
        auto read = [&](int slot) {
            Totals t;
            for (int z = 0; z < 2 * shape.nz; z += 2) {
                float sum = 0.0f;
                float mag = 0.0f;
                std::memcpy(&sum, &words[slot + z], 4);
                std::memcpy(&mag, &words[slot + z + 1], 4);
                t.sum += sum;
                t.mag += mag;
            }
            return t;
        };
        for (const ConserveCheck& check : conserve_checks) {
            const Term& term = law.terms[check.term];
            const std::vector<int>& group = term.conserves[check.group];
            double before = 0.0;
            double after = 0.0;
            double scale = 0.0;
            std::string names;
            for (std::size_t i = 0; i < group.size(); ++i) {
                const Totals b = read(check.first_before + static_cast<int>(i) * 2 * shape.nz);
                const Totals a = read(check.first_after + static_cast<int>(i) * 2 * shape.nz);
                before += b.sum;
                after += a.sum;
                scale += b.mag;
                names += (names.empty() ? "" : " + ") + law.fields[static_cast<std::size_t>(group[i])].name;
            }
            const double tolerance = 1e-5 * (scale + 1.0) * std::sqrt(static_cast<double>(check.substeps));
            if (std::abs(after - before) > tolerance) {
                std::ostringstream detail;
                detail << "total went " << before << " -> " << after << " (tolerance " << tolerance << ")";
                violations.push_back(Violation{.kind = Violation::Kind::Conservation, .term = term.name, .field = names, .detail = detail.str()});
            }
        }
        for (const RangeCheck& check : range_checks) {
            const std::uint32_t bad = words[check.slot];
            const std::uint32_t nan = words[check.slot + 1];
            if (bad == 0 && nan == 0) {
                continue;
            }
            const FieldDecl& decl = law.fields[static_cast<std::size_t>(check.field)];
            const std::size_t first = static_cast<std::size_t>(~words[check.slot + 2]);
            const std::size_t nx = static_cast<std::size_t>(shape.nx);
            const std::size_t ny = static_cast<std::size_t>(shape.ny);
            std::ostringstream detail;
            detail << bad + nan << " cell(s); first " << decl.name << " at (" << first % nx << "," << (first / nx) % ny << ","
                   << first / (nx * ny) << "), declared range [" << decl.lo << ", " << decl.hi << "]";
            violations.push_back(Violation{.kind = nan != 0 ? Violation::Kind::NotFinite : Violation::Kind::Range,
                .term = check.term < law.terms.size() ? law.terms[check.term].name : std::string{}, .field = decl.name, .detail = detail.str()});
        }
        dev->context->Unmap(records_read.Get(), 0);
    }

    void bind_constants()
    {
        ID3D11Buffer* cb[2] = {grid_cb.Get(), pass_cb.Get()};
        dev->context->CSSetConstantBuffers(0, 2, cb);
    }
};

GpuSolver::GpuSolver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuSolver::~GpuSolver() = default;

void GpuSolver::check_contracts_every(int steps)
{
    impl_->check_every = std::max(steps, 1);
}

const std::string& GpuSolver::adapter() const
{
    return impl_->dev->name;
}

const StepProfile& GpuSolver::profile() const
{
    return impl_->profile;
}

std::string GpuSolver::unavailable()
{
    return device().why;
}

std::shared_ptr<GpuSolver> GpuSolver::create(Law& law, const GridShape& shape, std::string& why)
{
    Device& dev = device();
    if (!dev.device) {
        why = dev.why;
        return nullptr;
    }
    // Nets are CUDA's (and the CPU's): this backend has no kernels for them.
    std::function<bool(const Node&)> uses_net = [&](const Node& n) {
        if (n.op == Op::Mlp || n.op == Op::Flux || n.op == Op::LDiffuse) {
            return true;
        }
        return std::any_of(n.args.begin(), n.args.end(), [&](const std::shared_ptr<Node>& a) { return uses_net(*a); });
    };
    for (const Term& term : law.terms) {
        for (const Equation& eq : term.equations) {
            if (uses_net(*eq.rhs)) {
                why = "term '" + term.name + "' uses a net, which Direct3D 11 does not run";
                return nullptr;
            }
        }
    }
    for (const Derive& derive : law.derives) {
        if (uses_net(*derive.rhs)) {
            why = "a derive uses a net, which Direct3D 11 does not run";
            return nullptr;
        }
    }
    const std::lock_guard<std::mutex> lock(dev.mutex);
    auto impl = std::make_unique<Impl>();
    Impl& g = *impl;
    g.dev = &dev;
    g.shape = shape;
    g.fields_n = static_cast<int>(law.fields.size());
    g.cells = static_cast<int>(shape.cells());
    g.gx = (shape.nx + kGroup - 1) / kGroup;
    g.gy = (shape.ny + kGroup - 1) / kGroup;
    g.uniforms = static_cast<int>(law.param_values.size());
    g.add_generic_shaders();
    for (Term& term : law.terms) {
        g.term_programs.push_back(g.compile_term(term));
        // Reductions a max_dt asks for get their shader here; the term's own program never runs them.
        if (term.max_dt) {
            std::function<void(Node&)> walk = [&](Node& n) {
                if (n.op == Op::MaxOf || n.op == Op::MinOf || n.op == Op::SumOf) {
                    Impl::Program scratch;
                    g.reduction_for(n, scratch);
                    return;
                }
                for (const std::shared_ptr<Node>& arg : n.args) {
                    walk(*arg);
                }
            };
            walk(*term.max_dt);
        }
    }
    std::set<int> started;
    for (Derive& derive : law.derives) {
        const bool add = derive.accumulate && started.contains(derive.target);
        started.insert(derive.target);
        g.derive_programs.push_back(g.compile_derive(derive, add));
    }
    if (g.uniforms > kMaxUniforms) {
        why = "too many params for one constant buffer";
        return nullptr;
    }
    if (!g.compile_all(why) || !g.make_resources(law, why)) {
        return nullptr;
    }
    // cbuffer Grid: NX NY NZ NXY CELLS H PAD PAD, then the uniforms.
    g.grid_data.assign(8 + 4 * 256, 0.0f);
    const int head[5] = {shape.nx, shape.ny, shape.nz, shape.nx * shape.ny, g.cells};
    std::memcpy(g.grid_data.data(), head, sizeof(head));
    g.grid_data[5] = static_cast<float>(shape.cell);
    g.write_grid();
    g.bind_constants();
    const UINT clear[4] = {0, 0, 0, 0};
    dev.context->ClearUnorderedAccessViewUint(g.state_uav.Get(), clear);
    dev.context->ClearUnorderedAccessViewUint(g.aux_uav.Get(), clear);
    dev.context->ClearUnorderedAccessViewUint(g.changed_uav.Get(), clear);
    // What the distance and Poisson passes keep starts unlike anything a field can hold.
    for (const Impl::Mat& mat : g.mats) {
        g.fill_slot(mat.seed, 0xffffffffu);
    }
    return std::shared_ptr<GpuSolver>(new GpuSolver(std::move(impl)));
}

void GpuSolver::step(double dt, bool respect_max_dt, bool learned, Law& law, FieldSet& fields, FieldEdits& edits,
    std::vector<char>& changed, std::vector<Violation>& violations, std::vector<int>& substeps)
{
    (void)learned; // no learned schemes here: the classical law runs
    Impl& g = *impl_;
    const std::lock_guard<std::mutex> lock(g.dev->mutex);
    const auto started = std::chrono::steady_clock::now();
    if (edits.shifted()) {
        edits.all = true; // no ring here: a moved window goes up whole
    }
    g.bind_constants();
    g.next_record = 0;
    const UINT nothing[4] = {0, 0, 0, 0};
    g.dev->context->ClearUnorderedAccessViewUint(g.records_uav.Get(), nothing);
    g.conserve_checks.clear();
    g.range_checks.clear();
    g.checking = g.steps_taken++ % g.check_every == 0;
    g.upload(fields, edits, law);
    g.settle_zero_flags(law);
    substeps.assign(law.terms.size(), 0);
    for (std::size_t t = 0; t < law.terms.size(); ++t) {
        g.apply_term(t, law.terms[t], dt, respect_max_dt, law, violations, substeps);
    }
    g.run_derives(law, false, true);
    g.bind(nullptr, nullptr);
    g.download(fields, changed);
    g.dev->context->ClearUnorderedAccessViewUint(g.changed_uav.Get(), nothing);
    g.read_contracts(law, violations);
    std::fill(g.dirty.begin(), g.dirty.end(), 0);
    g.profile.backend = "Direct3D 11";
    g.profile.terms.resize(law.terms.size());
    for (std::size_t t = 0; t < law.terms.size(); ++t) {
        g.profile.terms[t] = TermProfile{.name = law.terms[t].name, .substeps = substeps[t]}; // no timestamps here: the step as a whole
    }
    g.profile.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void GpuSolver::derive(Law& law, FieldSet& fields, FieldEdits& edits, std::vector<char>& changed)
{
    Impl& g = *impl_;
    const std::lock_guard<std::mutex> lock(g.dev->mutex);
    if (edits.shifted()) {
        edits.all = true;
    }
    g.bind_constants();
    g.next_record = 0;
    g.upload(fields, edits, law, true);
    g.run_derives(law, true, false);
    g.bind(nullptr, nullptr);
    g.download(fields, changed);
    std::fill(g.dirty.begin(), g.dirty.end(), 0);
    const UINT nothing[4] = {0, 0, 0, 0};
    g.dev->context->ClearUnorderedAccessViewUint(g.changed_uav.Get(), nothing);
}

} // namespace fieldengine::field

#endif

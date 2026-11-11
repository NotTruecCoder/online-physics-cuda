#include "field/cuda/CudaSolver.hpp"

#include "field/LawBounds.hpp"
#include "field/LawNode.hpp"
#include "field/Solver.hpp"
#include "field/cuda/CudaApi.hpp"
#include "field/cuda/CudaCodegen.hpp"
#include "field/cuda/CudaNet.hpp"
#include "field/ir/LawIR.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace fieldengine::field {

namespace {

using namespace cuda;

// The device's Grid, byte for byte (see law_prelude()).
struct GridHost {
    int nx, ny, nz, nxy, cells, ox, oy, oz;
    float h;
    int pad;
    unsigned long long S, A, U, C, L, R, W, F, NF;
};
static_assert(sizeof(GridHost) == 112, "GridHost must match the device's Grid");

constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 4;
constexpr int kGroup = 16; // fields a download moves at once through the pinned buffer

std::mutex& device_mutex()
{
    static std::mutex m; // one law at a time on the stream
    return m;
}

class Impl final : public LawBackend {
public:
    struct ConserveCheck {
        std::size_t term = 0;
        std::size_t group = 0;
        int first_before = -1;
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
    int cells = 0;
    int fields_n = 0;
    ir::Module module;
    LawLayout lay;
    cuda::Module mod;
    std::map<std::string, CUfunction> fns;
    CUdeviceptr grid_symbol = 0;
    CUdeviceptr param_symbol = 0;
    GridHost grid{};
    Buffer S, A, U, C, L, R, W, NF, F, snapshot;
    // Each field's buffer now and its other buffer (ping-pong; fields no program writes out of place have none), as the
    // device's FT table.
    struct FieldPair { CUdeviceptr now; CUdeviceptr next; };
    Buffer spare;
    Buffer bounds_keys; // lazy readback: 2 words a field
    bool lazy = false;
    std::vector<char> on_card; // lazy: the card's cells are newer than the host's
    int lazy_steps = 0;
    static constexpr int kTightenEvery = 8; // lazy: steps between measured bounds of integrated fields
    std::vector<FieldPair> pairs;
    CUdeviceptr table_symbol = 0;
    static constexpr int kTableSlots = 4096;
    Pinned table_ring; // swap_fields' staging, kTableSlots entries
    int next_slot = 0;
    Pinned pinned;
    std::vector<unsigned> changed_words;
    std::vector<float> params;
    std::uint64_t nets_key = 0;
    int record_words = 0;
    int next_record = 0;
    int check_every = 1;
    int steps_taken = 0;
    bool checking = true;
    bool uploaded_once = false;
    std::vector<char> known_zero, zero, touched, dirty;
    std::vector<Bounds> known, flowing;
    std::vector<ConserveCheck> conserve_checks;
    std::vector<RangeCheck> range_checks;
    std::string name;
    StepProfile profile_;
    std::vector<CUevent> events;
    int next_event = 0;
    std::vector<std::pair<int, int>> term_events; // per term: start, end (or -1)

    // ---- set up ----------------------------------------------------------------------------------------------------

    bool build(Law& law, std::string& why)
    {
        dev = &device();
        if (!dev->ok) {
            why = dev->why;
            return false;
        }
        dev->bind();
        name = dev->name;
        cells = static_cast<int>(shape.cells());
        fields_n = static_cast<int>(law.fields.size());
        module = ir::lower(law);
        const std::string source = generate(module, law, lay);
        std::string log;
        bool cached = false;
        const std::vector<char> cubin = compile(source, "law.cu", {"--std=c++17", "--fmad=false"}, log, cached);
        if (cubin.empty()) {
            why = "the law did not compile for CUDA: " + log.substr(0, 600);
            return false;
        }
        if (!mod.load(cubin, why)) {
            return false;
        }
        grid_symbol = mod.global("G");
        param_symbol = mod.global("P");
        table_symbol = mod.global("FT");
        if (fields_n > 256) {
            why = "more than 256 fields";
            return false;
        }
        if (grid_symbol == 0 || param_symbol == 0 || table_symbol == 0) {
            why = "the law module has no grid";
            return false;
        }
        if (law.param_values.size() > 1024) {
            why = "more than 1024 params";
            return false;
        }
        // Kernels with more shared memory than the default ask for it.
        for (const auto& [id, stage] : lay.stages) {
            if (stage.shared > 48 * 1024) {
                if (CUfunction f = fn(stage.kernel)) {
                    dev->api.cuFuncSetAttribute(f, kFuncMaxDynamicShared, static_cast<int>(stage.shared));
                }
            }
        }
        if (!allocate(law, why)) {
            return false;
        }
        upload_nets(law, true);
        // What distance and poisson keep starts unlike anything a field can hold, so the first look always runs them.
        for (const auto& [id, stage] : lay.stages) {
            if (stage.kind == StageCode::Kind::Distance || stage.kind == StageCode::Kind::Poisson) {
                int slot = lay.slot[static_cast<std::size_t>(id)];
                unsigned bits = 0xffffffffu;
                cells_launch("k_fill", {&slot, &bits});
            }
        }
        events.resize(2 * law.terms.size() + 16);
        for (CUevent& e : events) {
            dev->api.cuEventCreate(&e, 0);
        }
        dev->api.cuStreamSynchronize(dev->stream);
        return true;
    }

    bool allocate(const Law& law, std::string& why)
    {
        std::size_t words = 16;
        for (const Term& term : law.terms) {
            words += 2 * 3 * term.writes.size();
            for (const std::vector<int>& group : term.conserves) {
                words += 2 * 2 * group.size() * 2 * static_cast<std::size_t>(shape.nz);
            }
        }
        words += 3 * law.derives.size();
        record_words = static_cast<int>(words);
        const std::size_t n = static_cast<std::size_t>(cells);
        std::size_t writes_most = 1;
        for (const Term& term : law.terms) {
            writes_most = std::max(writes_most, term.writes.size());
        }
        const bool ok = S.allocate(4 * n * static_cast<std::size_t>(fields_n)) &&
            A.allocate(4 * n * static_cast<std::size_t>(std::max(lay.aux_slots, 1))) &&
            U.allocate(4 * static_cast<std::size_t>(std::max(lay.uniforms, 1))) &&
            C.allocate(4 * static_cast<std::size_t>(fields_n)) &&
            L.allocate(4 * std::max(n, 3 * static_cast<std::size_t>(shape.ny) * static_cast<std::size_t>(shape.nz) + 16)) &&
            R.allocate(4 * static_cast<std::size_t>(record_words)) &&
            W.allocate(4 * static_cast<std::size_t>(lay.net_floats)) &&
            NF.allocate(4 * static_cast<std::size_t>(lay.net_words)) &&
            F.allocate(4 * static_cast<std::size_t>(std::max(lay.flags, 1))) &&
            snapshot.allocate(4 * n * writes_most) &&
            bounds_keys.allocate(8 * static_cast<std::size_t>(fields_n)) &&
            table_ring.allocate(sizeof(FieldPair) * kTableSlots) &&
            pinned.allocate(4 * n * static_cast<std::size_t>(std::min(fields_n, kGroup)));
        if (!ok) {
            why = "could not allocate the lattice on " + name;
            return false;
        }
        const Api& a = dev->api;
        a.cuMemsetD32Async(S.ptr(), 0, n * static_cast<std::size_t>(fields_n), dev->stream);
        a.cuMemsetD32Async(A.ptr(), 0, n * static_cast<std::size_t>(std::max(lay.aux_slots, 1)), dev->stream);
        a.cuMemsetD32Async(U.ptr(), 0, static_cast<std::size_t>(std::max(lay.uniforms, 1)), dev->stream);
        a.cuMemsetD32Async(C.ptr(), 0, static_cast<std::size_t>(fields_n), dev->stream);
        a.cuMemsetD32Async(F.ptr(), 0, static_cast<std::size_t>(std::max(lay.flags, 1)), dev->stream);
        grid = GridHost{shape.nx, shape.ny, shape.nz, shape.nx * shape.ny, cells, 0, 0, 0, static_cast<float>(shape.cell), 0,
            S.ptr(), A.ptr(), U.ptr(), C.ptr(), L.ptr(), R.ptr(), W.ptr(), F.ptr(), NF.ptr()};
        write_grid();
        // The fields some program writes out of place get a second buffer.
        std::vector<int> swapped;
        for (const auto* list : {&lay.terms, &lay.learned, &lay.derives}) {
            for (const ProgramCode& code : *list) {
                for (const int f : code.swaps) {
                    if (std::find(swapped.begin(), swapped.end(), f) == swapped.end()) {
                        swapped.push_back(f);
                    }
                }
            }
        }
        if (!swapped.empty() && !spare.allocate(4 * n * swapped.size())) {
            why = "could not allocate the lattice on " + name;
            return false;
        }
        pairs.assign(static_cast<std::size_t>(fields_n), FieldPair{});
        for (int f = 0; f < fields_n; ++f) {
            pairs[static_cast<std::size_t>(f)].now = S.ptr() + 4ull * n * static_cast<std::size_t>(f);
            pairs[static_cast<std::size_t>(f)].next = pairs[static_cast<std::size_t>(f)].now;
        }
        for (std::size_t k = 0; k < swapped.size(); ++k) {
            pairs[static_cast<std::size_t>(swapped[k])].next = spare.ptr() + 4ull * n * k;
        }
        if (!pairs.empty()) {
            a.cuMemcpyHtoDAsync(table_symbol, pairs.data(), pairs.size() * sizeof(FieldPair), dev->stream);
            a.cuStreamSynchronize(dev->stream);
        }
        params.assign(law.param_values.size(), std::numeric_limits<float>::quiet_NaN());
        changed_words.resize(static_cast<std::size_t>(fields_n));
        return true;
    }

    // Where field f lives now.
    CUdeviceptr base(int f) const { return pairs[static_cast<std::size_t>(f)].now; }

    // After a kernel wrote these fields' next values: they are the fields now. The new entries go up from a ring of
    // pinned slots: a copy from pageable memory "may synchronize with the stream" (the host would wait for the kernel
    // before every swap), from pinned memory it is queued and the host goes on. A slot is reused only after a sync that
    // the copy reading it is behind (begin_step's, or the one here when the ring is full).
    void swap_fields(const ProgramCode& code)
    {
        for (const int f : code.swaps) {
            FieldPair& pair = pairs[static_cast<std::size_t>(f)];
            std::swap(pair.now, pair.next);
            if (next_slot == kTableSlots) {
                dev->api.cuStreamSynchronize(dev->stream);
                next_slot = 0;
            }
            FieldPair* slot = static_cast<FieldPair*>(table_ring.data()) + next_slot++;
            *slot = pair;
            dev->api.cuMemcpyHtoDAsync(table_symbol + sizeof(FieldPair) * static_cast<std::size_t>(f), slot, sizeof(FieldPair), dev->stream);
        }
    }

    void write_grid()
    {
        dev->api.cuMemcpyHtoDAsync(grid_symbol, &grid, sizeof(grid), dev->stream);
        dev->api.cuStreamSynchronize(dev->stream); // `grid` is a host value the copy reads
    }

    // Weights changed (a trainer wrote them, or online learning): the images and fragments go up again.
    void upload_nets(const Law& law, bool force)
    {
        std::uint64_t key = 1469598103934665603ull;
        for (const NetDecl& net : law.nets) {
            for (const float w : net.weights) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &w, 4);
                key = (key ^ bits) * 1099511628211ull;
            }
            key = (key ^ static_cast<std::uint64_t>(net.out_scale * 1e6f)) * 1099511628211ull;
        }
        if (!force && key == nets_key) {
            return;
        }
        nets_key = key;
        std::vector<float> image(static_cast<std::size_t>(lay.net_floats), 0.0f);
        std::vector<std::uint32_t> frags(static_cast<std::size_t>(lay.net_words), 0u);
        for (std::size_t n = 0; n < law.nets.size(); ++n) {
            const std::vector<float> im = net_image(law.nets[n]);
            std::copy(im.begin(), im.end(), image.begin() + lay.net_image_at[n]);
            if (law.nets[n].half) {
                const std::vector<std::uint32_t> f = net_fragments(law.nets[n]);
                std::copy(f.begin(), f.end(), frags.begin() + lay.net_frag_at[n]);
            }
        }
        dev->api.cuMemcpyHtoDAsync(W.ptr(), image.data(), image.size() * 4, dev->stream);
        dev->api.cuMemcpyHtoDAsync(NF.ptr(), frags.data(), frags.size() * 4, dev->stream);
        dev->api.cuStreamSynchronize(dev->stream);
        std::fill(dirty.begin(), dirty.end(), 1);
    }

    CUfunction fn(const std::string& kernel)
    {
        if (const auto found = fns.find(kernel); found != fns.end()) {
            return found->second;
        }
        CUfunction f = mod.function(kernel.c_str());
        fns.emplace(kernel, f);
        return f;
    }

    // ---- launching -------------------------------------------------------------------------------------------------

    void cells_launch(const std::string& kernel, std::initializer_list<void*> args, unsigned shared = 0)
    {
        const unsigned gx = (static_cast<unsigned>(shape.nx) + kBlockX - 1) / kBlockX;
        const unsigned gy = (static_cast<unsigned>(shape.ny) + kBlockY - 1) / kBlockY;
        launch(fn(kernel), gx, gy, static_cast<unsigned>(shape.nz), kBlockX, kBlockY, 1, shared, args);
        ++profile_.launches;
    }

    void line_launch(const std::string& kernel, unsigned count, unsigned block, std::initializer_list<void*> args, unsigned shared = 0)
    {
        launch(fn(kernel), (count + block - 1) / block, 1, 1, block, 1, 1, shared, args);
        ++profile_.launches;
    }

    int event()
    {
        if (next_event >= static_cast<int>(events.size())) {
            return -1;
        }
        dev->api.cuEventRecord(events[static_cast<std::size_t>(next_event)], dev->stream);
        return next_event++;
    }

    double elapsed(int a, int b) const
    {
        if (a < 0 || b < 0) {
            return 0.0;
        }
        float ms = 0.0f;
        dev->api.cuEventElapsedTime(&ms, events[static_cast<std::size_t>(a)], events[static_cast<std::size_t>(b)]);
        return ms;
    }

    // ---- stages ----------------------------------------------------------------------------------------------------

    void run_stage(int id, float dt)
    {
        const StageCode& s = lay.stages.at(id);
        const Api& a = dev->api;
        int slot = lay.slot[static_cast<std::size_t>(id)];
        switch (s.kind) {
        case StageCode::Kind::Stored:
            cells_launch(s.kernel, {&dt});
            return;
        case StageCode::Kind::Net:
            if (s.tensor) {
                line_launch(s.kernel, static_cast<unsigned>(cells), 128, {&dt}, s.shared);
            } else {
                cells_launch(s.kernel, {&dt}, s.shared);
            }
            return;
        case StageCode::Kind::Reduce: {
            launch(fn(s.kernel), static_cast<unsigned>(shape.nz), 1, 1, 256, 1, 1, 0, {&dt});
            int kind = s.reduce == Op::SumOf ? 0 : (s.reduce == Op::MaxOf ? 1 : 2);
            int u = lay.uslot[static_cast<std::size_t>(id)];
            line_launch("k_reduce_final", 1, 32, {&kind, &u});
            profile_.launches += 1;
            return;
        }
        case StageCode::Kind::Distance: {
            int flag = lay.flag[static_cast<std::size_t>(id)];
            a.cuMemsetD32Async(F.ptr() + 4ull * static_cast<unsigned long long>(flag), 0, 1, dev->stream);
            cells_launch(s.kernel, {&dt});
            int seed_in = slot;
            int seed_out = slot + 1;
            int scratch_a = slot + 2;
            int scratch_b = slot + 3;
            int to_in = slot + 4;
            int to_out = slot + 5;
            int result = slot + 6;
            for (const auto& [seed, to] : {std::pair{seed_in, to_in}, std::pair{seed_out, to_out}}) {
                int from = seed;
                int to_a = scratch_a;
                int to_b = scratch_b;
                int final_to = to;
                int ax0 = 0;
                int ax1 = 1;
                int ax2 = 2;
                cells_launch("k_scan", {&from, &to_a, &ax0, &flag});
                cells_launch("k_scan", {&to_a, &to_b, &ax1, &flag});
                cells_launch("k_scan", {&to_b, &final_to, &ax2, &flag});
            }
            cells_launch("k_signed", {&to_in, &to_out, &seed_in, &result, &flag});
            return;
        }
        case StageCode::Kind::Poisson: {
            int flag = lay.flag[static_cast<std::size_t>(id)];
            a.cuMemsetD32Async(F.ptr() + 4ull * static_cast<unsigned long long>(flag), 0, 1, dev->stream);
            cells_launch(s.kernel, {&dt});
            int source = slot;
            int from = slot + 1;
            int to = slot + 2;
            for (int it = 0; it < s.iterations; ++it) {
                cells_launch("k_jacobi", {&from, &to, &source, &flag});
                std::swap(from, to);
            }
            if (from != slot + 1) {
                int answer = slot + 1;
                cells_launch("k_copy", {&from, &answer, &flag});
            }
            return;
        }
        }
    }

    // A max_dt on the host: numbers, params, h and the reductions it reads (computed on the card, read back).
    double scalar_ir(int id, const Law& law)
    {
        const ir::Inst& in = module.insts[static_cast<std::size_t>(id)];
        auto a = [&](std::size_t k) { return scalar_ir(in.args[k], law); };
        switch (in.op) {
        case Op::Const: return in.number;
        case Op::Param: return law.param_values[static_cast<std::size_t>(in.index)];
        case Op::Cell: return shape.cell;
        case Op::Neg: return -a(0);
        case Op::Abs: return std::abs(a(0));
        case Op::Sqrt: return std::sqrt(std::max(0.0, a(0)));
        case Op::Exp: return std::exp(std::min(a(0), 80.0));
        case Op::Log: return std::log(std::max(a(0), 1e-30));
        case Op::Step: return a(0) > 0.0 ? 1.0 : 0.0;
        case Op::Pos: return std::max(a(0), 0.0);
        case Op::Add: return a(0) + a(1);
        case Op::Sub: return a(0) - a(1);
        case Op::Mul: return a(0) * a(1);
        case Op::Div: { const double b = a(1); return b == 0.0 ? 0.0 : a(0) / b; }
        case Op::Pow: return std::pow(a(0), a(1));
        case Op::Min: return std::min(a(0), a(1));
        case Op::Max: return std::max(a(0), a(1));
        case Op::Clamp: { const double lo = a(1); return std::clamp(a(0), lo, std::max(lo, a(2))); }
        case Op::MaxOf: case Op::MinOf: case Op::SumOf: {
            const int u = lay.uslot[static_cast<std::size_t>(id)];
            if (u < 0) {
                return 0.0;
            }
            float value = 0.0f;
            dev->api.cuMemcpyDtoHAsync(&value, U.ptr() + 4ull * static_cast<unsigned long long>(u), 4, dev->stream);
            dev->api.cuStreamSynchronize(dev->stream);
            return value;
        }
        default:
            return 0.0;
        }
    }

    // ---- contracts -------------------------------------------------------------------------------------------------

    int add_range(int field, const FieldDecl& decl)
    {
        int slot = next_record;
        next_record += 3;
        float lo = static_cast<float>(decl.lo);
        float hi = static_cast<float>(decl.hi);
        cells_launch("k_range", {&field, &slot, &lo, &hi});
        return slot;
    }

    int add_total(int field)
    {
        int slot = next_record;
        next_record += 2 * shape.nz;
        line_launch("k_rows", static_cast<unsigned>(shape.ny * shape.nz) * 32u, 128, {&field}); // a warp a row
        line_launch("k_planes", static_cast<unsigned>(shape.nz), 32, {&slot});
        return slot;
    }

    struct Totals {
        double sum = 0.0;
        double mag = 0.0;
    };

    static Totals totals_at(const std::vector<std::uint32_t>& words, int slot, int nz)
    {
        Totals t;
        for (int z = 0; z < 2 * nz; z += 2) {
            float sum = 0.0f;
            float mag = 0.0f;
            std::memcpy(&sum, &words[static_cast<std::size_t>(slot + z)], 4);
            std::memcpy(&mag, &words[static_cast<std::size_t>(slot + z + 1)], 4);
            t.sum += sum;
            t.mag += mag;
        }
        return t;
    }

    // Checks into violations; with `only_from` >= 0, only the checks from that record on (a guarded term's own).
    void judge(const Law& law, const std::vector<std::uint32_t>& words, const std::vector<ConserveCheck>& conserves,
        const std::vector<RangeCheck>& ranges, std::vector<Violation>& violations) const
    {
        for (const ConserveCheck& check : conserves) {
            const Term& term = law.terms[check.term];
            const std::vector<int>& group = term.conserves[check.group];
            double before = 0.0;
            double after = 0.0;
            double scale = 0.0;
            std::string names;
            for (std::size_t i = 0; i < group.size(); ++i) {
                const Totals b = totals_at(words, check.first_before + static_cast<int>(i) * 2 * shape.nz, shape.nz);
                const Totals a = totals_at(words, check.first_after + static_cast<int>(i) * 2 * shape.nz, shape.nz);
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
        for (const RangeCheck& check : ranges) {
            const std::uint32_t bad = words[static_cast<std::size_t>(check.slot)];
            const std::uint32_t nan = words[static_cast<std::size_t>(check.slot) + 1];
            if (bad == 0 && nan == 0) {
                continue;
            }
            const FieldDecl& decl = law.fields[static_cast<std::size_t>(check.field)];
            const std::size_t first = static_cast<std::size_t>(~words[static_cast<std::size_t>(check.slot) + 2]);
            const std::size_t nx = static_cast<std::size_t>(shape.nx);
            const std::size_t ny = static_cast<std::size_t>(shape.ny);
            std::ostringstream detail;
            detail << bad + nan << " cell(s); first " << decl.name << " at (" << first % nx << "," << (first / nx) % ny << ","
                   << first / (nx * ny) << "), declared range [" << decl.lo << ", " << decl.hi << "]";
            violations.push_back(Violation{.kind = nan != 0 ? Violation::Kind::NotFinite : Violation::Kind::Range,
                .term = check.term < law.terms.size() ? law.terms[check.term].name : std::string{}, .field = decl.name, .detail = detail.str()});
        }
    }

    std::vector<std::uint32_t> read_records(int upto)
    {
        std::vector<std::uint32_t> words(static_cast<std::size_t>(std::max(upto, 1)), 0u);
        if (upto > 0) { // nothing recorded (contracts not looked at this step): no wait for the card
            dev->api.cuMemcpyDtoHAsync(words.data(), R.ptr(), static_cast<std::size_t>(upto) * 4, dev->stream);
            dev->api.cuStreamSynchronize(dev->stream);
        }
        return words;
    }

    // ---- terms -----------------------------------------------------------------------------------------------------

    // One program (a term's classical or learned equations) with its substeps; false when it was idle.
    bool run_program(std::size_t index, const Term& term, const std::vector<Equation>& equations, const ProgramCode& code,
        double dt, bool respect, Law& law, bool check, std::vector<ConserveCheck>& conserves, std::vector<RangeCheck>& ranges,
        std::vector<Violation>& violations, int& count)
    {
        const bool idle = code.gather.empty() || std::all_of(equations.begin(), equations.end(), [&](const Equation& eq) {
            return always_zero(*eq.rhs, zero, law.param_values) || bounds_of(*eq.rhs, flowing, law.param_values, shape.cell).none();
        });
        count = 1;
        if (idle) {
            return false;
        }
        if (respect && code.max_dt >= 0) {
            for (const int s : code.limit_stages) {
                run_stage(s, 0.0f);
            }
            const double limit = scalar_ir(code.max_dt, law);
            if (limit > 0.0 && std::isfinite(limit) && dt > limit) {
                const double wanted = std::ceil(dt / limit);
                if (wanted > Solver::kMaxSubsteps) {
                    violations.push_back(Violation{.kind = Violation::Kind::SubstepCap, .term = term.name,
                        .detail = "max_dt " + std::to_string(limit) + " asks for " + std::to_string(wanted) + " substeps; capped at " +
                            std::to_string(Solver::kMaxSubsteps)});
                    count = Solver::kMaxSubsteps;
                } else {
                    count = static_cast<int>(wanted);
                }
            }
        }
        std::vector<ConserveCheck> mine;
        for (std::size_t g = 0; check && g < term.conserves.size(); ++g) {
            ConserveCheck c{.term = index, .group = g, .substeps = count};
            for (std::size_t i = 0; i < term.conserves[g].size(); ++i) {
                const int slot = add_total(term.conserves[g][i]);
                if (i == 0) {
                    c.first_before = slot;
                }
            }
            mine.push_back(c);
        }
        float sub_dt = static_cast<float>(dt / count);
        for (int s = 0; s < count; ++s) {
            for (const int stage : code.stages) {
                run_stage(stage, sub_dt);
            }
            cells_launch(code.gather, {&sub_dt});
            swap_fields(code);
        }
        for (const int field : term.writes) {
            const FieldDecl& decl = law.fields[static_cast<std::size_t>(field)];
            if (decl.clamp) {
                int f = field;
                float lo = static_cast<float>(decl.lo);
                float hi = static_cast<float>(decl.hi);
                cells_launch("k_clamp", {&f, &lo, &hi});
            }
            if (check) {
                ranges.push_back(RangeCheck{.term = index, .field = field, .slot = add_range(field, decl)});
            }
        }
        flow_bounds(equations, dt, count, law, shape.cell, flowing);
        for (const Equation& eq : equations) {
            dirty[static_cast<std::size_t>(eq.target)] = 1;
        }
        for (ConserveCheck& c : mine) {
            const std::vector<int>& group = term.conserves[c.group];
            for (std::size_t i = 0; i < group.size(); ++i) {
                const int slot = add_total(group[i]);
                if (i == 0) {
                    c.first_after = slot;
                }
            }
            conserves.push_back(c);
        }
        return true;
    }

    void apply_term(std::size_t index, Term& term, double dt, bool respect, bool learned, Law& law, std::vector<Violation>& violations,
        std::vector<int>& substeps)
    {
        TermProfile& p = profile_.terms[index];
        p = TermProfile{.name = term.name};
        const int start = event();
        const int launches = profile_.launches;
        int count = 1;
        const ProgramCode& learned_code = lay.learned[index];
        bool done = false;
        if (learned && !term.learned.empty() && !learned_code.gather.empty()) {
            // GUARDED. Snapshot what it writes, run the learned scheme, judge its own contracts now.
            const std::size_t bytes = 4 * static_cast<std::size_t>(cells);
            for (std::size_t w = 0; w < term.writes.size(); ++w) {
                dev->api.cuMemcpyDtoDAsync(snapshot.ptr() + w * bytes, base(term.writes[w]), bytes, dev->stream);
            }
            const std::vector<Bounds> flowing_before = flowing;
            const int first_record = next_record;
            std::vector<ConserveCheck> conserves;
            std::vector<RangeCheck> ranges;
            std::vector<Violation> broken;
            const bool ran = run_program(index, term, term.learned, learned_code, dt, respect, law, true, conserves, ranges, broken, count);
            if (ran) {
                const std::vector<std::uint32_t> words = read_records(next_record);
                judge(law, words, conserves, ranges, broken);
            }
            if (broken.empty()) {
                p.learned = ran;
                p.idle = !ran;
                done = true;
            } else {
                // Undo it: the snapshot back, the records it used free again, the bounds as they were.
                for (std::size_t w = 0; w < term.writes.size(); ++w) {
                    dev->api.cuMemcpyDtoDAsync(base(term.writes[w]), snapshot.ptr() + w * bytes, bytes, dev->stream);
                }
                const std::size_t clear = static_cast<std::size_t>(next_record - first_record);
                dev->api.cuMemsetD32Async(R.ptr() + 4ull * static_cast<unsigned long long>(first_record), 0, clear, dev->stream);
                next_record = first_record;
                flowing = flowing_before;
                p.fell_back = true;
                ++profile_.fallbacks;
            }
        }
        if (!done) {
            const bool ran = run_program(index, term, term.equations, lay.terms[index], dt, respect, law, checking, conserve_checks,
                range_checks, violations, count);
            p.idle = !ran;
        }
        substeps[index] = count;
        p.substeps = count;
        p.launches = profile_.launches - launches;
        term_events[index] = {start, event()};
    }

    void run_derives(Law& law, bool all, bool check)
    {
        std::vector<char> decided(law.fields.size(), 0);
        std::vector<char> needed(law.fields.size(), 0);
        float dt = 0.0f;
        for (std::size_t d = 0; d < law.derives.size(); ++d) {
            Derive& derive = law.derives[d];
            const auto t = static_cast<std::size_t>(derive.target);
            if (!decided[t]) {
                decided[t] = 1;
                bool nothing = true;
                bool moved = false;
                for (const Derive& part : law.derives) {
                    if (part.target == derive.target) {
                        nothing = nothing && always_zero(*part.rhs, zero, law.param_values);
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
            const ProgramCode& code = lay.derives[d];
            for (const int stage : code.stages) {
                run_stage(stage, dt);
            }
            cells_launch(code.gather, {&dt});
            swap_fields(code);
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

    // ---- the lattice, between CPU and GPU ---------------------------------------------------------------------------

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

    bool ringed() const { return grid.ox != 0 || grid.oy != 0 || grid.oz != 0; }

    // A box of one field from the host's linear lattice into the ring.
    void upload_box(int field, const float* host, const CellBox& b)
    {
        const int bx = b.x1 - b.x0;
        const int by = b.y1 - b.y0;
        const int bz = b.z1 - b.z0;
        if (bx <= 0 || by <= 0 || bz <= 0) {
            return;
        }
        const std::size_t count = static_cast<std::size_t>(bx) * static_cast<std::size_t>(by) * static_cast<std::size_t>(bz);
        std::vector<float> packed(count);
        for (int z = 0; z < bz; ++z) {
            for (int y = 0; y < by; ++y) {
                std::memcpy(packed.data() + (static_cast<std::size_t>(z) * by + y) * bx, host + shape.index(b.x0, b.y0 + y, b.z0 + z),
                    static_cast<std::size_t>(bx) * 4);
            }
        }
        dev->api.cuMemcpyHtoDAsync(L.ptr(), packed.data(), count * 4, dev->stream);
        int f = field;
        int x0 = b.x0;
        int y0 = b.y0;
        int z0 = b.z0;
        int ux = bx;
        int uy = by;
        int uz = bz;
        launch(fn("k_scatter"), (static_cast<unsigned>(bx) + kBlockX - 1) / kBlockX, (static_cast<unsigned>(by) + kBlockY - 1) / kBlockY,
            static_cast<unsigned>(bz), kBlockX, kBlockY, 1, 0, {&f, &x0, &y0, &z0, &ux, &uy, &uz});
        dev->api.cuStreamSynchronize(dev->stream); // `packed` is about to go
        profile_.upload_bytes += count * 4;
    }

    void upload_field(int field, const float* host)
    {
        if (!ringed()) {
            dev->api.cuMemcpyHtoDAsync(base(field), host,
                static_cast<std::size_t>(cells) * 4, dev->stream);
            profile_.upload_bytes += static_cast<std::size_t>(cells) * 4;
            return;
        }
        upload_box(field, host, CellBox{0, 0, 0, shape.nx, shape.ny, shape.nz});
    }

    void upload(const FieldSet& fields, FieldEdits& edits, const Law& law, bool derives_follow)
    {
        const int start = event();
        known_zero.resize(fields.count(), 0);
        known.resize(fields.count());
        dirty.resize(fields.count(), 1);
        const std::vector<char> was_zero = known_zero;
        // The window moved: the ring's origin moves with it, and only what arrived goes up.
        if (!edits.all && edits.shifted()) {
            auto wrap = [](int v, int n) { return ((v % n) + n) % n; };
            grid.ox = wrap(grid.ox + edits.shift[0], shape.nx);
            grid.oy = wrap(grid.oy + edits.shift[1], shape.ny);
            grid.oz = wrap(grid.oz + edits.shift[2], shape.nz);
            write_grid();
            std::fill(dirty.begin(), dirty.end(), 1);
            std::vector<int> going;
            for (std::size_t id = 0; id < fields.count(); ++id) {
                if (!law.fields[id].derived) {
                    going.push_back(static_cast<int>(id));
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < static_cast<int>(going.size()); ++k) {
                look_over(going[static_cast<std::size_t>(k)], fields.values(going[static_cast<std::size_t>(k)]).data());
            }
            for (const int field : going) {
                const std::size_t id = static_cast<std::size_t>(field);
                if (uploaded_once && was_zero[id] && known_zero[id]) {
                    continue; // zero on the card before, zero here now: the wrapped cells are zero too
                }
                for (const CellBox& b : edits.boxes) {
                    upload_box(field, fields.values(field).data(), b);
                }
            }
            edits.shift[0] = edits.shift[1] = edits.shift[2] = 0;
            edits.boxes.clear();
        }
        std::vector<int> going;
        for (std::size_t id = 0; id < fields.count(); ++id) {
            const std::vector<char>* marks = !edits.all && id < edits.planes.size() ? &edits.planes[id] : nullptr;
            if (!edits.all && (marks == nullptr || marks->empty())) {
                continue;
            }
            if (derives_follow && edits.all && law.fields[id].derived && uploaded_once) {
                continue;
            }
            going.push_back(static_cast<int>(id));
        }
#pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < static_cast<int>(going.size()); ++k) {
            look_over(going[static_cast<std::size_t>(k)], fields.values(going[static_cast<std::size_t>(k)]).data());
        }
        for (const int field : going) {
            const std::size_t id = static_cast<std::size_t>(field);
            if (uploaded_once && was_zero[id] && known_zero[id]) {
                continue;
            }
            dirty[id] = 1;
            const std::vector<char>* marks = !edits.all && id < edits.planes.size() ? &edits.planes[id] : nullptr;
            if (marks == nullptr) {
                upload_field(field, fields.values(field).data());
                continue;
            }
            for (int z = 0; z < shape.nz; ++z) {
                if (!(*marks)[static_cast<std::size_t>(z)]) {
                    continue;
                }
                int end = z + 1;
                while (end < shape.nz && (*marks)[static_cast<std::size_t>(end)]) {
                    ++end;
                }
                upload_box(field, fields.values(field).data(), CellBox{0, 0, z, shape.nx, shape.ny, end});
                z = end;
            }
        }
        edits.all = false;
        edits.planes.clear();
        edits.shift[0] = edits.shift[1] = edits.shift[2] = 0;
        edits.boxes.clear();
        uploaded_once = true;
        zero = known_zero;
        bool params_changed = false;
        std::vector<float> now(law.param_values.size());
        for (std::size_t p = 0; p < law.param_values.size(); ++p) {
            now[p] = static_cast<float>(law.param_values[p]);
            if (!(params[p] == now[p])) {
                params_changed = true;
            }
        }
        if (params_changed) {
            params = now;
            if (!params.empty()) {
                dev->api.cuMemcpyHtoDAsync(param_symbol, params.data(), params.size() * 4, dev->stream);
                dev->api.cuStreamSynchronize(dev->stream);
            }
            std::fill(dirty.begin(), dirty.end(), 1); // what the derives worked out, they worked out with the old ones
        }
        upload_nets(law, false);
        flowing = known;
        profile_.upload_ms = elapsed(start, event());
    }

    void settle_zero_flags(const Law& law)
    {
        bool grew = true;
        while (grew) {
            grew = false;
            for (const Term& term : law.terms) {
                for (const std::vector<Equation>* eqs : {&term.equations, &term.learned}) {
                    for (const Equation& eq : *eqs) {
                        if (zero[static_cast<std::size_t>(eq.target)] && !always_zero(*eq.rhs, zero, law.param_values)) {
                            zero[static_cast<std::size_t>(eq.target)] = 0;
                            grew = true;
                        }
                    }
                }
            }
        }
    }

    // What the kernels said changed comes back (through pinned memory, a group of fields at a time); a field no cell of
    // which moved is not copied anywhere.
    void download(const Law& law, FieldSet& fields, std::vector<char>& changed_out)
    {
        const int start = event();
        dev->api.cuMemcpyDtoHAsync(changed_words.data(), C.ptr(), changed_words.size() * 4, dev->stream);
        dev->api.cuStreamSynchronize(dev->stream);
        touched.assign(fields.count(), 0);
        std::vector<int> moved;
        for (std::size_t id = 0; id < fields.count(); ++id) {
            touched[id] = changed_words[id] != 0 ? 1 : 0;
            if (touched[id]) {
                moved.push_back(static_cast<int>(id));
            }
        }
        changed_out = touched;
        if (lazy) {
            // The bounds come back, not the cells: 8 bytes a field that moved. Between measurements an integrated field
            // keeps the bounds flow_bounds carried through the terms, substep by substep (interval arithmetic, padded
            // for float rounding, not clipped to an unclamped range: as wide as the cells or wider, so a term with work
            // is not skipped; gate cuda_parity grows a field past a threshold to hold it), measured every kTightenEvery
            // steps; a derived field, which flow_bounds does not follow, is measured every step.
            on_card.resize(fields.count(), 0);
            const bool tighten = lazy_steps++ % kTightenEvery == 0;
            std::vector<int> measured;
            for (int field : moved) {
                const auto f = static_cast<std::size_t>(field);
                on_card[f] = 1;
                if (tighten || law.fields[f].derived) {
                    measured.push_back(field);
                } else {
                    known[f] = flowing[f];
                    known_zero[f] = flowing[f].lo == 0.0 && flowing[f].hi == 0.0 ? 1 : 0;
                }
            }
            if (measured.empty()) {
                profile_.download_ms = elapsed(start, event());
                return;
            }
            const Api& a = dev->api;
            a.cuMemsetD32Async(bounds_keys.ptr(), 0xffffffffu, static_cast<std::size_t>(fields_n), dev->stream);
            a.cuMemsetD32Async(bounds_keys.ptr() + 4ull * static_cast<std::size_t>(fields_n), 0u, static_cast<std::size_t>(fields_n), dev->stream);
            CUdeviceptr out = bounds_keys.ptr();
            int count = fields_n;
            for (int field : measured) {
                launch(fn("k_bounds"), static_cast<unsigned>(std::max(dev->sms, 1) * 8), 1, 1, 256, 1, 1, 0, {&field, &out, &count});
                ++profile_.launches;
            }
            std::vector<std::uint32_t> keys(2 * static_cast<std::size_t>(fields_n));
            a.cuMemcpyDtoHAsync(keys.data(), bounds_keys.ptr(), keys.size() * 4, dev->stream);
            a.cuStreamSynchronize(dev->stream);
            const auto value = [](std::uint32_t k) {
                const std::uint32_t b = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
                float f = 0.0f;
                std::memcpy(&f, &b, 4);
                return f;
            };
            for (int field : measured) {
                const auto f = static_cast<std::size_t>(field);
                const std::uint32_t lo_key = keys[f];
                const std::uint32_t hi_key = keys[static_cast<std::size_t>(fields_n) + f];
                // Every cell NaN (k_bounds skips them): what look_over finds over the same cells.
                const bool none = lo_key == 0xffffffffu && hi_key == 0u;
                const float lo = none ? std::numeric_limits<float>::infinity() : value(lo_key);
                const float hi = none ? -std::numeric_limits<float>::infinity() : value(hi_key);
                known[f] = Bounds{lo, hi};
                known_zero[f] = lo == 0.0f && hi == 0.0f ? 1 : 0;
            }
            profile_.download_bytes += keys.size() * 4;
            profile_.download_ms = elapsed(start, event());
            return;
        }
        copy_back(fields, moved, true);
        profile_.download_ms = elapsed(start, event());
    }

    void set_lazy(bool on) override
    {
        lazy = on;
    }

    void fetch(FieldSet& fields, const std::vector<char>& which) override
    {
        const std::lock_guard<std::mutex> lock(device_mutex());
        dev->bind();
        std::vector<int> wanted;
        for (std::size_t f = 0; f < which.size() && f < on_card.size(); ++f) {
            if (which[f] && on_card[f]) {
                wanted.push_back(static_cast<int>(f));
                on_card[f] = 0;
            }
        }
        copy_back(fields, wanted, false);
    }

    // These fields' cells from the card into `fields` (through pinned memory, a group of fields at a time); with
    // `look`, their bounds from the copies.
    void copy_back(FieldSet& fields, const std::vector<int>& moved, bool look)
    {
        const std::size_t bytes = static_cast<std::size_t>(cells) * 4;
        for (std::size_t first = 0; first < moved.size(); first += kGroup) {
            const std::size_t end = std::min(moved.size(), first + kGroup);
            for (std::size_t k = first; k < end; ++k) {
                const int field = moved[k];
                char* to = static_cast<char*>(pinned.data()) + (k - first) * bytes;
                if (ringed()) {
                    int f = field;
                    cells_launch("k_linearize", {&f});
                    dev->api.cuMemcpyDtoHAsync(to, L.ptr(), bytes, dev->stream);
                } else {
                    dev->api.cuMemcpyDtoHAsync(to, base(field), bytes, dev->stream);
                }
            }
            dev->api.cuStreamSynchronize(dev->stream);
#pragma omp parallel for schedule(dynamic)
            for (int k = static_cast<int>(first); k < static_cast<int>(end); ++k) {
                const int field = moved[static_cast<std::size_t>(k)];
                const float* from = reinterpret_cast<const float*>(static_cast<char*>(pinned.data()) + (static_cast<std::size_t>(k) - first) * bytes);
                std::memcpy(fields.values(field).data(), from, bytes);
                if (look) {
                    look_over(field, from);
                }
            }
            profile_.download_bytes += (end - first) * bytes;
        }
    }

    // ---- LawBackend ------------------------------------------------------------------------------------------------

    void begin_step(const Law& law)
    {
        dev->bind();
        profile_ = StepProfile{.backend = "CUDA"};
        profile_.terms.resize(law.terms.size());
        term_events.assign(law.terms.size(), {-1, -1});
        next_event = 0;
        next_record = 0;
        next_slot = 0; // step() and derive() end in a sync: every queued table copy has read its slot
        dev->api.cuMemsetD32Async(R.ptr(), 0, static_cast<std::size_t>(record_words), dev->stream);
        conserve_checks.clear();
        range_checks.clear();
    }

    void step(double dt, bool respect_max_dt, bool learned, Law& law, FieldSet& fields, FieldEdits& edits,
        std::vector<char>& changed, std::vector<Violation>& violations, std::vector<int>& substeps) override
    {
        const std::lock_guard<std::mutex> lock(device_mutex());
        const auto wall = std::chrono::steady_clock::now();
        begin_step(law);
        const int all_start = event();
        checking = steps_taken++ % check_every == 0;
        upload(fields, edits, law, false);
        settle_zero_flags(law);
        substeps.assign(law.terms.size(), 0);
        for (std::size_t t = 0; t < law.terms.size(); ++t) {
            apply_term(t, law.terms[t], dt, respect_max_dt, learned, law, violations, substeps);
        }
        const int derive_start = event();
        run_derives(law, false, true);
        const int derive_end = event();
        download(law, fields, changed);
        dev->api.cuMemsetD32Async(C.ptr(), 0, static_cast<std::size_t>(fields_n), dev->stream);
        const std::vector<std::uint32_t> words = read_records(next_record);
        judge(law, words, conserve_checks, range_checks, violations);
        std::fill(dirty.begin(), dirty.end(), 0);
        const int all_end = event();
        dev->api.cuEventSynchronize(events[static_cast<std::size_t>(std::max(all_end, 0))]);
        for (std::size_t t = 0; t < law.terms.size(); ++t) {
            profile_.terms[t].ms = elapsed(term_events[t].first, term_events[t].second);
        }
        profile_.derive_ms = elapsed(derive_start, derive_end);
        profile_.gpu_ms = elapsed(all_start, all_end);
        profile_.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
    }

    void derive(Law& law, FieldSet& fields, FieldEdits& edits, std::vector<char>& changed) override
    {
        const std::lock_guard<std::mutex> lock(device_mutex());
        begin_step(law);
        upload(fields, edits, law, true);
        run_derives(law, true, false);
        download(law, fields, changed);
        std::fill(dirty.begin(), dirty.end(), 0);
        dev->api.cuMemsetD32Async(C.ptr(), 0, static_cast<std::size_t>(fields_n), dev->stream);
        dev->api.cuStreamSynchronize(dev->stream);
    }

    void check_contracts_every(int steps) override { check_every = std::max(steps, 1); }
    [[nodiscard]] const std::string& adapter() const override { return name; }
    [[nodiscard]] const char* api() const override { return "CUDA"; }
    [[nodiscard]] const StepProfile& profile() const override { return profile_; }

    ~Impl() override
    {
        if (dev != nullptr && dev->ok) {
            dev->bind();
            dev->api.cuStreamSynchronize(dev->stream);
        }
    }
};

} // namespace

std::shared_ptr<LawBackend> CudaSolver::create(Law& law, const GridShape& shape, std::string& why)
{
    const std::lock_guard<std::mutex> lock(device_mutex());
    auto impl = std::make_shared<Impl>();
    impl->shape = shape;
    if (!impl->build(law, why)) {
        return nullptr;
    }
    return impl;
}

std::string CudaSolver::unavailable()
{
    const cuda::Device& d = cuda::device();
    return d.ok ? std::string{} : d.why;
}

std::string CudaSolver::source(const Law& law)
{
    const ir::Module m = ir::lower(law);
    cuda::LawLayout layout;
    return cuda::generate(m, law, layout);
}

} // namespace fieldengine::field

#include "field/ir/LawIR.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace fieldengine::field::ir {

const char* op_name(Op op)
{
    switch (op) {
    case Op::Const: return "const";
    case Op::Param: return "param";
    case Op::Cell: return "h";
    case Op::Rim: return "rim";
    case Op::Field: return "field";
    case Op::Neg: return "neg";
    case Op::Add: return "add";
    case Op::Sub: return "sub";
    case Op::Mul: return "mul";
    case Op::Div: return "div";
    case Op::Pow: return "pow";
    case Op::Min: return "min";
    case Op::Max: return "max";
    case Op::Abs: return "abs";
    case Op::Sqrt: return "sqrt";
    case Op::Exp: return "exp";
    case Op::Log: return "log";
    case Op::Step: return "step";
    case Op::Pos: return "pos";
    case Op::Clamp: return "clamp";
    case Op::Lap: return "lap";
    case Op::Gx: return "gx";
    case Op::Gy: return "gy";
    case Op::Gz: return "gz";
    case Op::Diffuse: return "diffuse";
    case Op::Adv: return "adv";
    case Op::Distance: return "distance";
    case Op::Poisson: return "poisson";
    case Op::MaxOf: return "maxof";
    case Op::MinOf: return "minof";
    case Op::SumOf: return "sumof";
    case Op::Fall: return "fall";
    case Op::Spread: return "spread";
    case Op::Topple: return "topple";
    case Op::Shift: return "shift";
    case Op::Local: return "local";
    case Op::Mlp: return "mlp";
    case Op::Flux: return "flux";
    case Op::LDiffuse: return "ldiffuse";
    }
    return "?";
}

bool Module::is_stage(int id) const
{
    const Inst& in = insts[static_cast<std::size_t>(id)];
    return in.global || in.net || materialized[static_cast<std::size_t>(id)] != 0;
}

namespace {

bool pure_arithmetic(Op op)
{
    switch (op) {
    case Op::Neg: case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Pow: case Op::Min: case Op::Max:
    case Op::Abs: case Op::Sqrt: case Op::Exp: case Op::Log: case Op::Step: case Op::Pos: case Op::Clamp:
        return true;
    default:
        return false;
    }
}

// What the CPU's scalar path computes (Law.cpp: in double).
double fold(Op op, const std::vector<double>& a)
{
    switch (op) {
    case Op::Neg: return -a[0];
    case Op::Abs: return std::abs(a[0]);
    case Op::Sqrt: return std::sqrt(std::max(0.0, a[0]));
    case Op::Exp: return std::exp(std::min(a[0], 80.0));
    case Op::Log: return std::log(std::max(a[0], 1e-30));
    case Op::Step: return a[0] > 0.0 ? 1.0 : 0.0;
    case Op::Pos: return std::max(a[0], 0.0);
    case Op::Add: return a[0] + a[1];
    case Op::Sub: return a[0] - a[1];
    case Op::Mul: return a[0] * a[1];
    case Op::Div: return a[1] == 0.0 ? 0.0 : a[0] / a[1];
    case Op::Pow: return std::pow(a[0], a[1]);
    case Op::Min: return std::min(a[0], a[1]);
    case Op::Max: return std::max(a[0], a[1]);
    case Op::Clamp: return std::clamp(a[0], a[1], std::max(a[1], a[2]));
    default: return 0.0;
    }
}

// How many points of each argument one point of the op reads.
int points(Op op)
{
    switch (op) {
    case Op::Lap: case Op::Diffuse: return 7;
    case Op::Gx: case Op::Gy: case Op::Gz: return 2;
    case Op::Adv: return 7;
    case Op::Fall: return 3;
    case Op::Spread: return 5;
    case Op::Topple: return 13;
    default: return 1;
    }
}

class Lowering {
public:
    explicit Lowering(Module& m) : m_(m) {}

    int lower(const Node& n)
    {
        if (const auto found = memo_.find(&n); found != memo_.end()) {
            return found->second;
        }
        Inst in;
        in.op = n.op;
        in.number = n.number;
        in.index = n.index;
        in.iterations = n.iterations;
        if (n.op == Op::Shift) {
            in.args.push_back(lower(*n.args[0]));
            for (int a = 0; a < 3; ++a) {
                in.offset[a] = static_cast<int>(n.args[static_cast<std::size_t>(a) + 1]->number);
            }
        } else {
            for (const std::shared_ptr<Node>& arg : n.args) {
                in.args.push_back(lower(*arg));
            }
        }
        const int id = simplify(in);
        memo_.emplace(&n, id);
        return id;
    }

private:
    [[nodiscard]] const Inst& at(int id) const { return m_.insts[static_cast<std::size_t>(id)]; }
    [[nodiscard]] bool is_const(int id, double v) const { return at(id).op == Op::Const && at(id).number == v; }

    int constant(double v)
    {
        Inst c;
        c.op = Op::Const;
        c.number = v;
        return add(c);
    }

    int simplify(Inst in)
    {
        // Every argument a number: the CPU would compute one number, so this is that number.
        if (pure_arithmetic(in.op) &&
            std::all_of(in.args.begin(), in.args.end(), [&](int a) { return at(a).op == Op::Const; })) {
            std::vector<double> v;
            for (const int a : in.args) {
                v.push_back(at(a).number);
            }
            ++m_.folded;
            return constant(fold(in.op, v));
        }
        switch (in.op) {
        case Op::Mul:
            // Zero times anything is zero on the CPU (it never looks at the anything): the same here.
            if (is_const(in.args[0], 0.0) || is_const(in.args[1], 0.0)) {
                ++m_.folded;
                return constant(0.0);
            }
            if (is_const(in.args[0], 1.0)) {
                ++m_.folded;
                return in.args[1];
            }
            if (is_const(in.args[1], 1.0)) {
                ++m_.folded;
                return in.args[0];
            }
            break;
        case Op::Add:
            if (is_const(in.args[0], 0.0)) {
                ++m_.folded;
                return in.args[1];
            }
            if (is_const(in.args[1], 0.0)) {
                ++m_.folded;
                return in.args[0];
            }
            break;
        case Op::Sub:
            if (is_const(in.args[1], 0.0)) {
                ++m_.folded;
                return in.args[0];
            }
            break;
        case Op::Div:
            if (is_const(in.args[1], 1.0)) {
                ++m_.folded;
                return in.args[0];
            }
            if (is_const(in.args[0], 0.0)) {
                ++m_.folded;
                return constant(0.0);
            }
            break;
        case Op::Neg:
            if (at(in.args[0]).op == Op::Neg) {
                ++m_.folded;
                return at(in.args[0]).args[0];
            }
            break;
        case Op::Shift:
            if (at(in.args[0]).shape == Shape::Scalar || (in.offset[0] == 0 && in.offset[1] == 0 && in.offset[2] == 0)) {
                ++m_.folded;
                return in.args[0];
            }
            break;
        case Op::Lap: case Op::Gx: case Op::Gy: case Op::Gz:
            if (at(in.args[0]).shape == Shape::Scalar) {
                ++m_.folded;
                return constant(0.0); // nothing varies: no gradient, no curvature
            }
            break;
        case Op::Diffuse: case Op::LDiffuse:
            if (at(in.args[1]).shape == Shape::Scalar) {
                ++m_.folded;
                return constant(0.0);
            }
            break;
        case Op::Adv: case Op::Fall: case Op::Spread: case Op::Topple:
            if (at(in.args[0]).shape == Shape::Scalar) {
                ++m_.folded;
                return constant(0.0);
            }
            break;
        default:
            break;
        }
        return add(in);
    }

    int add(Inst in)
    {
        switch (in.op) {
        case Op::Const: case Op::Param: case Op::Cell: case Op::Local: case Op::MaxOf: case Op::MinOf: case Op::SumOf:
            in.shape = Shape::Scalar;
            break;
        case Op::Field: case Op::Rim: case Op::Distance: case Op::Poisson: case Op::Fall: case Op::Spread: case Op::Topple:
        case Op::Flux: case Op::LDiffuse: case Op::Lap: case Op::Gx: case Op::Gy: case Op::Gz: case Op::Diffuse: case Op::Adv:
            in.shape = Shape::Cell;
            break;
        default:
            in.shape = std::any_of(in.args.begin(), in.args.end(), [&](int a) { return at(a).shape == Shape::Cell; })
                ? Shape::Cell
                : Shape::Scalar;
            break;
        }
        switch (in.op) {
        case Op::Lap: case Op::Gx: case Op::Gy: case Op::Gz: case Op::Diffuse: case Op::Adv: case Op::Fall: case Op::Spread:
        case Op::Topple: case Op::Flux:
            in.radius = 1;
            break;
        case Op::LDiffuse:
            in.radius = 2;
            break;
        case Op::Shift:
            in.radius = std::max({std::abs(in.offset[0]), std::abs(in.offset[1]), std::abs(in.offset[2])});
            break;
        default:
            break;
        }
        in.global = in.op == Op::Distance || in.op == Op::Poisson || in.op == Op::MaxOf || in.op == Op::MinOf || in.op == Op::SumOf;
        in.net = in.op == Op::Mlp || in.op == Op::Flux || in.op == Op::LDiffuse;

        std::ostringstream key;
        key << static_cast<int>(in.op) << '|' << std::hexfloat << in.number << '|' << in.index << '|' << in.iterations << '|'
            << in.offset[0] << ',' << in.offset[1] << ',' << in.offset[2];
        for (const int a : in.args) {
            key << '|' << a;
        }
        const auto [where, inserted] = table_.emplace(key.str(), static_cast<int>(m_.insts.size()));
        if (!inserted) {
            ++m_.shared;
            return where->second;
        }
        m_.insts.push_back(std::move(in));
        return where->second;
    }

    Module& m_;
    std::map<const Node*, int> memo_;
    std::map<std::string, int> table_;
};

// Whether a stage is needed, and the cost of a point of each instruction, everything not a stage inlined.
void plan_stages(Module& m)
{
    m.materialized.assign(m.insts.size(), 0);
    for (std::size_t id = 0; id < m.insts.size(); ++id) {
        Inst& in = m.insts[id];
        const int per = points(in.op);
        int cost = 1;
        for (const int a : in.args) {
            Inst& arg = m.insts[static_cast<std::size_t>(a)];
            const bool stored = m.is_stage(a);
            int c = stored ? 1 : arg.cost;
            // A stencil reads its argument at several points; past the budget the argument is stored once instead.
            if (in.radius > 0 && !stored && arg.shape == Shape::Cell && arg.op != Op::Field && c * per > kInlineBudget) {
                m.materialized[static_cast<std::size_t>(a)] = 1;
                c = 1;
            }
            cost += c * per;
            ++arg.uses;
        }
        in.cost = std::min(cost, 1 << 20);
    }
}

void collect_stages(const Module& m, int id, std::set<int>& out, std::set<int>& seen)
{
    if (!seen.insert(id).second) {
        return;
    }
    for (const int a : m.insts[static_cast<std::size_t>(id)].args) {
        collect_stages(m, a, out, seen);
    }
    if (m.is_stage(id) && m.insts[static_cast<std::size_t>(id)].shape == Shape::Cell) {
        out.insert(id);
    }
    if (m.insts[static_cast<std::size_t>(id)].global) {
        out.insert(id); // a reduction is one number, but still a pass over the lattice
    }
}

// How far from its cell, through what the gather kernel inlines, each field is read.
void reach(const Module& m, int id, int offset, bool root, std::map<int, int>& out)
{
    const Inst& in = m.insts[static_cast<std::size_t>(id)];
    if (!root && m.is_stage(id)) {
        return; // stored before the kernel ran: it read the state as it was
    }
    if (in.op == Op::Field) {
        int& r = out[in.index];
        r = std::max(r, offset);
        return;
    }
    if (in.net || in.global) {
        return;
    }
    for (const int a : in.args) {
        reach(m, a, offset + in.radius, false, out);
    }
}

Program make_program(Module& m, Lowering& low, const std::string& name, const std::vector<field::Equation>& equations,
    const Node* max_dt)
{
    Program p;
    p.name = name;
    std::set<int> writes;
    for (const field::Equation& eq : equations) {
        p.equations.push_back(Equation{eq.target, low.lower(*eq.rhs)});
        writes.insert(eq.target);
    }
    if (max_dt != nullptr) {
        p.max_dt = low.lower(*max_dt);
    }
    p.writes.assign(writes.begin(), writes.end());
    return p;
}

void finish_program(const Module& m, Program& p)
{
    std::set<int> stages;
    std::set<int> seen;
    for (const Equation& eq : p.equations) {
        // An equation that IS a stage (a whole flux) still needs it computed first.
        collect_stages(m, eq.value, stages, seen);
    }
    p.stages.assign(stages.begin(), stages.end());
    if (p.max_dt >= 0) {
        std::set<int> limit;
        std::set<int> seen_limit;
        collect_stages(m, p.max_dt, limit, seen_limit);
        p.limit_stages.assign(limit.begin(), limit.end());
    }
    std::set<int> reads;
    std::function<void(int)> walk = [&](int id) {
        const Inst& in = m.insts[static_cast<std::size_t>(id)];
        if (in.op == Op::Field) {
            reads.insert(in.index);
        }
        for (const int a : in.args) {
            walk(a);
        }
    };
    for (const Equation& eq : p.equations) {
        walk(eq.value);
    }
    p.reads.assign(reads.begin(), reads.end());
    std::map<int, int> offsets;
    for (const Equation& eq : p.equations) {
        reach(m, eq.value, 0, true, offsets);
    }
    p.in_place = true;
    for (const int t : p.writes) {
        const auto found = offsets.find(t);
        if (found != offsets.end() && found->second > 0) {
            p.in_place = false; // another cell's thread may already have written what this one reads
        }
    }
}

void json_number(std::ostream& out, double v)
{
    if (!std::isfinite(v)) {
        out << (v > 0 ? "1e300" : (v < 0 ? "-1e300" : "0"));
        return;
    }
    char text[40];
    std::snprintf(text, sizeof(text), "%.17g", v);
    out << text;
}

void json_string(std::ostream& out, const std::string& s)
{
    out << '"';
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out << ' ';
        } else {
            out << c;
        }
    }
    out << '"';
}

void json_ints(std::ostream& out, const std::vector<int>& v)
{
    out << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        out << (i ? "," : "") << v[i];
    }
    out << ']';
}

void json_program(std::ostream& out, const Program& p)
{
    out << "{\"name\":";
    json_string(out, p.name);
    out << ",\"equations\":[";
    for (std::size_t e = 0; e < p.equations.size(); ++e) {
        out << (e ? "," : "") << '[' << p.equations[e].target << ',' << p.equations[e].value << ']';
    }
    out << "],\"stages\":";
    json_ints(out, p.stages);
    out << ",\"limit_stages\":";
    json_ints(out, p.limit_stages);
    out << ",\"in_place\":" << (p.in_place ? "true" : "false") << ",\"max_dt\":" << p.max_dt << ",\"reads\":";
    json_ints(out, p.reads);
    out << ",\"writes\":";
    json_ints(out, p.writes);
    out << ",\"conserves\":[";
    for (std::size_t g = 0; g < p.conserves.size(); ++g) {
        out << (g ? "," : "");
        json_ints(out, p.conserves[g]);
    }
    out << "],\"accumulate\":" << (p.accumulate ? "true" : "false") << '}';
}

} // namespace

Module lower(const Law& law)
{
    Module m;
    Lowering low(m);
    for (const Term& term : law.terms) {
        m.terms.push_back(make_program(m, low, term.name, term.equations, term.max_dt.get()));
        m.terms.back().conserves = term.conserves;
        m.learned.push_back(make_program(m, low, term.name + "~learned", term.learned, term.learned_max_dt.get()));
        m.learned.back().conserves = term.conserves;
    }
    std::set<int> started;
    for (const Derive& derive : law.derives) {
        Program p = make_program(m, low, "derive " + law.fields[static_cast<std::size_t>(derive.target)].name,
            {field::Equation{.target = derive.target, .rhs = derive.rhs}}, nullptr);
        p.accumulate = derive.accumulate && started.contains(derive.target);
        started.insert(derive.target);
        m.derives.push_back(std::move(p));
    }
    plan_stages(m);
    for (Program& p : m.terms) {
        finish_program(m, p);
    }
    for (Program& p : m.learned) {
        finish_program(m, p);
    }
    for (Program& p : m.derives) {
        finish_program(m, p);
        p.in_place = false; // a derive replaces its field: its own cell's old value is not an input of the new one
    }
    return m;
}

std::string to_json(const Module& m, const Law& law)
{
    std::ostringstream out;
    out << "{\"version\":1,\"fields\":[";
    for (std::size_t f = 0; f < law.fields.size(); ++f) {
        const FieldDecl& d = law.fields[f];
        out << (f ? "," : "") << "{\"name\":";
        json_string(out, d.name);
        out << ",\"init\":";
        json_number(out, d.init);
        out << ",\"lo\":";
        json_number(out, d.lo);
        out << ",\"hi\":";
        json_number(out, d.hi);
        out << ",\"clamp\":" << (d.clamp ? "true" : "false") << ",\"static\":" << (d.is_static ? "true" : "false")
            << ",\"derived\":" << (d.derived ? "true" : "false") << '}';
    }
    out << "],\"params\":[";
    for (std::size_t p = 0; p < law.param_names.size(); ++p) {
        out << (p ? "," : "") << "{\"name\":";
        json_string(out, law.param_names[p]);
        out << ",\"value\":";
        json_number(out, law.param_values[p]);
        out << '}';
    }
    out << "],\"nets\":[";
    for (std::size_t n = 0; n < law.nets.size(); ++n) {
        const NetDecl& net = law.nets[n];
        out << (n ? "," : "") << "{\"name\":";
        json_string(out, net.name);
        out << ",\"file\":";
        json_string(out, net.file);
        out << ",\"sizes\":";
        json_ints(out, net.sizes);
        out << ",\"act\":\"" << to_string(net.act) << "\",\"half\":" << (net.half ? "true" : "false")
            << ",\"loaded\":" << (net.loaded ? "true" : "false") << '}';
    }
    out << "],\"insts\":[";
    for (std::size_t i = 0; i < m.insts.size(); ++i) {
        const Inst& in = m.insts[i];
        out << (i ? ",\n" : "\n") << "{\"op\":\"" << op_name(in.op) << "\",\"shape\":\"" << (in.shape == Shape::Cell ? "cell" : "scalar")
            << "\",\"number\":";
        json_number(out, in.number);
        out << ",\"index\":" << in.index << ",\"iterations\":" << in.iterations << ",\"offset\":[" << in.offset[0] << ','
            << in.offset[1] << ',' << in.offset[2] << "],\"args\":";
        json_ints(out, in.args);
        out << ",\"radius\":" << in.radius << ",\"stage\":" << (m.is_stage(static_cast<int>(i)) ? "true" : "false") << '}';
    }
    out << "],\n\"terms\":[";
    for (std::size_t t = 0; t < m.terms.size(); ++t) {
        out << (t ? ",\n" : "\n");
        json_program(out, m.terms[t]);
    }
    out << "],\n\"learned\":[";
    for (std::size_t t = 0; t < m.learned.size(); ++t) {
        out << (t ? ",\n" : "\n");
        json_program(out, m.learned[t]);
    }
    out << "],\n\"derives\":[";
    for (std::size_t d = 0; d < m.derives.size(); ++d) {
        out << (d ? ",\n" : "\n");
        json_program(out, m.derives[d]);
    }
    out << "]}\n";
    return out.str();
}

std::string describe(const Module& m, const Law& law)
{
    std::ostringstream out;
    out << m.insts.size() << " instructions (" << m.shared << " found twice and kept once, " << m.folded
        << " folded away)\n";
    for (std::size_t i = 0; i < m.insts.size(); ++i) {
        const Inst& in = m.insts[i];
        out << "  %" << i << " = " << op_name(in.op);
        if (in.op == Op::Const) {
            out << ' ' << in.number;
        } else if (in.op == Op::Field) {
            out << ' ' << law.fields[static_cast<std::size_t>(in.index)].name;
        } else if (in.op == Op::Param) {
            out << ' ' << law.param_names[static_cast<std::size_t>(in.index)];
        } else if (in.net) {
            out << ' ' << law.nets[static_cast<std::size_t>(in.index)].name;
        }
        for (const int a : in.args) {
            out << " %" << a;
        }
        if (in.op == Op::Shift) {
            out << " (" << in.offset[0] << ',' << in.offset[1] << ',' << in.offset[2] << ')';
        }
        out << "   [" << (in.shape == Shape::Cell ? "cell" : "scalar");
        if (in.radius > 0) {
            out << ", reads " << in.radius << " away";
        }
        if (m.is_stage(static_cast<int>(i))) {
            out << (in.global ? ", global" : in.net ? ", net" : ", stored");
        }
        out << ", cost " << in.cost << "]\n";
    }
    auto program = [&](const Program& p) {
        if (p.equations.empty()) {
            return;
        }
        out << p.name << ": " << p.equations.size() << " equation(s), " << p.stages.size() << " stage(s), "
            << (p.in_place ? "one kernel (in place)" : "one kernel (ping-pong buffers)") << "\n";
        for (const int s : p.stages) {
            out << "    stage %" << s << " " << op_name(m.insts[static_cast<std::size_t>(s)].op) << "\n";
        }
        for (const Equation& eq : p.equations) {
            out << "    d " << law.fields[static_cast<std::size_t>(eq.target)].name << " = %" << eq.value << "\n";
        }
        if (p.max_dt >= 0) {
            out << "    max_dt = %" << p.max_dt << "\n";
        }
    };
    for (const Program& p : m.terms) {
        program(p);
    }
    for (const Program& p : m.learned) {
        program(p);
    }
    for (const Program& p : m.derives) {
        program(p);
    }
    return out.str();
}

} // namespace fieldengine::field::ir

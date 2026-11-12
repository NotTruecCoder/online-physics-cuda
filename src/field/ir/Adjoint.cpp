#include "field/ir/Adjoint.hpp"

#include "field/ir/LawIR.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace fieldengine::field::ir {

namespace {

std::string number(double v)
{
    char text[40];
    std::snprintf(text, sizeof(text), "%.17g", std::abs(v));
    std::string s = text;
    return v < 0.0 ? "(0 - " + s + ")" : s;
}

} // namespace

std::string adjoint_law(const Law& law, const std::string& term, std::string& error)
{
    const Module m = lower(law);
    std::size_t t = 0;
    if (!term.empty()) {
        while (t < m.terms.size() && m.terms[t].name != term) {
            ++t;
        }
    }
    if (t >= m.terms.size()) {
        error = "no term '" + term + "'";
        return {};
    }
    const Program& p = m.terms[t];
    if (p.equations.empty()) {
        error = "term '" + p.name + "' has no equations";
        return {};
    }
    auto at = [&](int id) -> const Inst& { return m.insts[static_cast<std::size_t>(id)]; };

    // What the term's equations reach.
    std::set<int> reached;
    std::function<void(int)> reach = [&](int id) {
        if (reached.insert(id).second) {
            for (const int a : at(id).args) {
                reach(a);
            }
        }
    };
    for (const Equation& eq : p.equations) {
        reach(eq.value);
    }

    // A value as the adjoint law writes it: leaves by name, the rest by their let.
    auto prim = [&](int id) -> std::string {
        const Inst& in = at(id);
        switch (in.op) {
        case Op::Const: return number(in.number);
        case Op::Param: return law.param_names[static_cast<std::size_t>(in.index)];
        case Op::Field: return law.fields[static_cast<std::size_t>(in.index)].name;
        case Op::Cell: return "h";
        case Op::Dt: return "dt";
        case Op::Rim: return "rim";
        default: return "pv__" + std::to_string(id);
        }
    };
    auto offsets = [](const Inst& in) {
        return std::to_string(in.offset[0]) + ", " + std::to_string(in.offset[1]) + ", " + std::to_string(in.offset[2]);
    };

    std::ostringstream lets;
    for (const int id : reached) {
        const Inst& in = at(id);
        auto a = [&](std::size_t k) { return prim(in.args[k]); };
        std::string e;
        switch (in.op) {
        case Op::Const: case Op::Param: case Op::Field: case Op::Cell: case Op::Dt: case Op::Rim: continue;
        case Op::Neg: e = "(0 - " + a(0) + ")"; break;
        case Op::Add: e = "(" + a(0) + " + " + a(1) + ")"; break;
        case Op::Sub: e = "(" + a(0) + " - " + a(1) + ")"; break;
        case Op::Mul: e = "(" + a(0) + " * " + a(1) + ")"; break;
        case Op::Div: e = "(" + a(0) + " / " + a(1) + ")"; break;
        case Op::Pow: e = "(" + a(0) + " ^ " + a(1) + ")"; break;
        case Op::Min: e = "min(" + a(0) + ", " + a(1) + ")"; break;
        case Op::Max: e = "max(" + a(0) + ", " + a(1) + ")"; break;
        case Op::Clamp: e = "clamp(" + a(0) + ", " + a(1) + ", " + a(2) + ")"; break;
        case Op::Diffuse: e = "diffuse(" + a(0) + ", " + a(1) + ")"; break;
        case Op::Shift: e = "shift(" + a(0) + ", " + offsets(in) + ")"; break;
        case Op::ShiftT: e = "shift_t(" + a(0) + ", " + offsets(in) + ")"; break;
        case Op::Abs: case Op::Sqrt: case Op::Exp: case Op::Log: case Op::Step: case Op::Pos: case Op::Lap: case Op::Gx:
        case Op::Gy: case Op::Gz: case Op::GxT: case Op::GyT: case Op::GzT:
            e = std::string(op_name(in.op)) + "(" + a(0) + ")";
            break;
        default:
            error = "term '" + p.name + "': " + op_name(in.op) + " has no adjoint here";
            return {};
        }
        lets << "  let pv__" << id << " = " << e << "\n";
    }

    // The reverse sweep: seeds from the equations, then every instruction from the highest id down.
    std::map<int, std::vector<std::string>> parts;
    for (const Equation& eq : p.equations) {
        parts[eq.value].push_back(law.fields[static_cast<std::size_t>(eq.target)].name + "__adj");
    }
    std::map<int, std::string> field_adj; // field index -> its adjoint's let
    std::map<int, std::string> param_adj;
    for (auto it = reached.rbegin(); it != reached.rend(); ++it) {
        const int id = *it;
        const auto found = parts.find(id);
        if (found == parts.end()) {
            continue;
        }
        const Inst& in = at(id);
        const std::string A = "av__" + std::to_string(id);
        std::string sum;
        for (const std::string& part : found->second) {
            sum += (sum.empty() ? "" : " + ") + part;
        }
        lets << "  let " << A << " = " << sum << "\n";
        auto give = [&](std::size_t k, const std::string& e) {
            const Op to = at(in.args[k]).op;
            if (to != Op::Const && to != Op::Cell && to != Op::Dt && to != Op::Rim) { // nothing to differentiate in those
                parts[in.args[k]].push_back(e);
            }
        };
        auto a = [&](std::size_t k) { return prim(in.args[k]); };
        const std::string out = prim(id);
        switch (in.op) {
        case Op::Field: field_adj[in.index] = A; break;
        case Op::Param: param_adj[in.index] = A; break;
        case Op::Const: case Op::Cell: case Op::Dt: case Op::Rim: case Op::Step: break;
        case Op::Neg: give(0, "(0 - " + A + ")"); break;
        case Op::Add: give(0, A); give(1, A); break;
        case Op::Sub: give(0, A); give(1, "(0 - " + A + ")"); break;
        case Op::Mul: give(0, "(" + A + " * " + a(1) + ")"); give(1, "(" + A + " * " + a(0) + ")"); break;
        case Op::Div: give(0, "(" + A + " / " + a(1) + ")"); give(1, "(0 - " + A + " * " + out + " / " + a(1) + ")"); break;
        case Op::Pow:
            give(0, "(" + A + " * " + a(1) + " * (" + a(0) + " ^ (" + a(1) + " - 1)))");
            give(1, "(" + A + " * " + out + " * log(" + a(0) + "))");
            break;
        case Op::Min:
            give(0, "(" + A + " * (1 - step(" + a(0) + " - " + a(1) + ")))");
            give(1, "(" + A + " * step(" + a(0) + " - " + a(1) + "))");
            break;
        case Op::Max:
            give(0, "(" + A + " * (1 - step(" + a(1) + " - " + a(0) + ")))");
            give(1, "(" + A + " * step(" + a(1) + " - " + a(0) + "))");
            break;
        case Op::Abs: give(0, "(" + A + " * (step(" + a(0) + ") - step(0 - " + a(0) + ")))"); break;
        case Op::Sqrt: give(0, "(" + A + " * step(" + a(0) + ") * 0.5 / " + out + ")"); break;
        case Op::Exp: give(0, "(" + A + " * " + out + " * step(80 - " + a(0) + "))"); break;
        case Op::Log: give(0, "(" + A + " * step(" + a(0) + " - 1e-30) / " + a(0) + ")"); break;
        case Op::Pos: give(0, "(" + A + " * step(" + a(0) + "))"); break;
        case Op::Clamp: {
            const std::string top = "max(" + a(1) + ", " + a(2) + ")";
            give(0, "(" + A + " * step(" + a(0) + " - " + a(1) + ") * step(" + top + " - " + a(0) + "))");
            give(1, "(" + A + " * (1 - step(" + a(0) + " - " + a(1) + ")))");
            give(2, "(" + A + " * step(" + a(0) + " - " + top + ") * step(" + a(2) + " - " + a(1) + "))");
            break;
        }
        case Op::Shift: give(0, "shift_t(" + A + ", " + offsets(in) + ")"); break;
        case Op::ShiftT: give(0, "shift(" + A + ", " + offsets(in) + ")"); break;
        case Op::Lap: give(0, "lap(" + A + ")"); break;
        case Op::Gx: give(0, "gx_t(" + A + ")"); break;
        case Op::Gy: give(0, "gy_t(" + A + ")"); break;
        case Op::Gz: give(0, "gz_t(" + A + ")"); break;
        case Op::GxT: give(0, "gx(" + A + ")"); break;
        case Op::GyT: give(0, "gy(" + A + ")"); break;
        case Op::GzT: give(0, "gz(" + A + ")"); break;
        case Op::Diffuse:
            // Symmetric in F (the flux form): its adjoint is itself. In a k that is one number: k's derivative is the
            // unit-k diffusion, where k > 0 (a face of k <= 0 conducts nothing).
            if (at(in.args[0]).shape != Shape::Scalar) {
                error = "term '" + p.name + "': diffuse whose k is a field has no adjoint here";
                return {};
            }
            give(1, "diffuse(" + a(0) + ", " + A + ")");
            give(0, "(" + A + " * step(" + a(0) + ") * diffuse(1, " + a(1) + "))");
            break;
        default:
            error = "term '" + p.name + "': " + op_name(in.op) + " has no adjoint here";
            return {};
        }
    }

    // The law: the primal's fields as static inputs, an adjoint field for each, the params.
    std::set<int> inputs(p.reads.begin(), p.reads.end());
    inputs.insert(p.writes.begin(), p.writes.end());
    std::ostringstream out;
    out << "# The adjoint of term '" << p.name << "' (generated by lawc --adjoint from the IR: ir/Adjoint.cpp). One step of\n"
        << "# it is one vector-Jacobian product of one step of the term: F__adj + dt * (dR/dF)^T A for every field, and for a\n"
        << "# param p the field p__adj, whose total over the lattice is dL/dp. The primal fields are the state the step began from.\n";
    for (const int f : inputs) {
        const FieldDecl& d = law.fields[static_cast<std::size_t>(f)];
        char init[40];
        std::snprintf(init, sizeof(init), "%.17g", d.init);
        out << "field " << d.name << " init " << init << " range -1e30 1e30 static\n";
    }
    for (const int f : inputs) {
        out << "field " << law.fields[static_cast<std::size_t>(f)].name << "__adj init 0 range -1e30 1e30\n";
    }
    for (const auto& [index, name] : param_adj) {
        out << "field " << law.param_names[static_cast<std::size_t>(index)] << "__adj init 0 range -1e30 1e30\n";
    }
    for (std::size_t i = 0; i < law.param_names.size(); ++i) {
        out << "param " << law.param_names[i] << " = " << number(law.param_values[i]) << "\n";
    }
    out << "\nterm adjoint\n" << lets.str();
    for (const auto& [index, name] : field_adj) {
        out << "  d " << law.fields[static_cast<std::size_t>(index)].name << "__adj = " << name << "\n";
    }
    for (const auto& [index, name] : param_adj) {
        out << "  d " << law.param_names[static_cast<std::size_t>(index)] << "__adj = " << name << "\n";
    }
    return out.str();
}

} // namespace fieldengine::field::ir

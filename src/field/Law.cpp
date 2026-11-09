#include "field/LawNode.hpp"

#include "field/Parallel.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace fieldengine::field {

namespace {

using NodePtr = std::shared_ptr<Node>;

NodePtr make(Op op, std::vector<NodePtr> args = {})
{
    auto node = std::make_shared<Node>();
    node->op = op;
    node->args = std::move(args);
    return node;
}

NodePtr constant(double value)
{
    auto node = make(Op::Const);
    node->number = value;
    return node;
}

void collect_reads(const Node& node, std::set<int>& out)
{
    if (node.op == Op::Field) {
        out.insert(node.index);
    }
    for (const NodePtr& arg : node.args) {
        collect_reads(*arg, out);
    }
}

// ---- tokenizer ---------------------------------------------------------------------------------------

struct Token {
    enum Kind { Number, Name, Symbol, End } kind = End;
    std::string text;
    double number = 0.0;
};

std::vector<Token> tokenize(std::string_view line, std::string& error)
{
    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < line.size()) {
        const char c = line[i];
        if (c == '#') {
            break;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < line.size() &&
                                                               std::isdigit(static_cast<unsigned char>(line[i + 1])))) {
            std::size_t j = i;
            while (j < line.size() && (std::isdigit(static_cast<unsigned char>(line[j])) || line[j] == '.')) {
                ++j;
            }
            if (j < line.size() && (line[j] == 'e' || line[j] == 'E')) {
                std::size_t k = j + 1;
                if (k < line.size() && (line[k] == '+' || line[k] == '-')) {
                    ++k;
                }
                if (k < line.size() && std::isdigit(static_cast<unsigned char>(line[k]))) {
                    j = k;
                    while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) {
                        ++j;
                    }
                }
            }
            Token token{.kind = Token::Number, .text = std::string(line.substr(i, j - i))};
            try {
                token.number = std::stod(token.text);
            } catch (...) {
                error = "bad number '" + token.text + "'";
                return {};
            }
            tokens.push_back(token);
            i = j;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            while (j < line.size() && (std::isalnum(static_cast<unsigned char>(line[j])) || line[j] == '_')) {
                ++j;
            }
            tokens.push_back(Token{.kind = Token::Name, .text = std::string(line.substr(i, j - i))});
            i = j;
            continue;
        }
        if (std::string_view("+-*/^(),=").find(c) != std::string_view::npos) {
            tokens.push_back(Token{.kind = Token::Symbol, .text = std::string(1, c)});
            ++i;
            continue;
        }
        error = std::string("unexpected character '") + c + "'";
        return {};
    }
    tokens.push_back(Token{});
    return tokens;
}

// ---- expression parser ------------------------------------------------------------------------------

struct FunctionSpec {
    Op op;
    int min_args;
    int max_args;
};

const std::map<std::string, FunctionSpec, std::less<>>& functions()
{
    static const std::map<std::string, FunctionSpec, std::less<>> table = {
        {"abs", {Op::Abs, 1, 1}},       {"sqrt", {Op::Sqrt, 1, 1}},       {"exp", {Op::Exp, 1, 1}},
        {"log", {Op::Log, 1, 1}},       {"step", {Op::Step, 1, 1}},       {"pos", {Op::Pos, 1, 1}},
        {"min", {Op::Min, 2, 2}},       {"max", {Op::Max, 2, 2}},         {"clamp", {Op::Clamp, 3, 3}},
        {"lap", {Op::Lap, 1, 1}},       {"gx", {Op::Gx, 1, 1}},           {"gy", {Op::Gy, 1, 1}},
        {"gz", {Op::Gz, 1, 1}},         {"diffuse", {Op::Diffuse, 2, 2}}, {"adv", {Op::Adv, 4, 4}},
        {"distance", {Op::Distance, 1, 1}}, {"poisson", {Op::Poisson, 1, 2}},
        {"maxof", {Op::MaxOf, 1, 1}},   {"minof", {Op::MinOf, 1, 1}},     {"sumof", {Op::SumOf, 1, 1}},
        {"fall", {Op::Fall, 3, 3}},     {"spread", {Op::Spread, 3, 3}},   {"topple", {Op::Topple, 3, 3}},
        {"shift", {Op::Shift, 4, 4}},
        // The first argument of these is a net's name, not an expression.
        {"mlp", {Op::Mlp, 2, 33}},      {"flux", {Op::Flux, 2, 9}},       {"ldiffuse", {Op::LDiffuse, 3, 3}},
    };
    return table;
}

class Parser {
public:
    Parser(const std::vector<Token>& tokens, std::size_t start, const Law& law,
        const std::vector<std::string>* locals = nullptr)
        : tokens_(tokens), pos_(start), law_(law), locals_(locals)
    {
    }

    NodePtr expression()
    {
        NodePtr left = product();
        while (ok() && (peek("+") || peek("-"))) {
            const bool add = take().text == "+";
            NodePtr right = product();
            left = make(add ? Op::Add : Op::Sub, {left, right});
        }
        return left;
    }

    [[nodiscard]] bool at_end() const { return tokens_[pos_].kind == Token::End; }
    [[nodiscard]] bool ok() const { return error_.empty(); }
    [[nodiscard]] const std::string& error() const { return error_; }

private:
    NodePtr product()
    {
        NodePtr left = unary();
        while (ok() && (peek("*") || peek("/"))) {
            const bool mul = take().text == "*";
            NodePtr right = unary();
            left = make(mul ? Op::Mul : Op::Div, {left, right});
        }
        return left;
    }

    NodePtr unary()
    {
        if (peek("-")) {
            take();
            return make(Op::Neg, {unary()});
        }
        if (peek("+")) {
            take();
            return unary();
        }
        return power();
    }

    NodePtr power()
    {
        NodePtr base = primary();
        if (ok() && peek("^")) {
            take();
            NodePtr exponent = unary(); // right-associative
            return make(Op::Pow, {base, exponent});
        }
        return base;
    }

    NodePtr primary()
    {
        const Token& token = tokens_[pos_];
        if (token.kind == Token::Number) {
            ++pos_;
            return constant(token.number);
        }
        if (token.kind == Token::Symbol && token.text == "(") {
            ++pos_;
            NodePtr inner = expression();
            expect(")");
            return inner;
        }
        if (token.kind == Token::Name) {
            ++pos_;
            if (peek("(")) {
                return call(token.text);
            }
            return name(token.text);
        }
        fail(token.kind == Token::End ? "expression ends too early" : "unexpected '" + token.text + "'");
        return constant(0.0);
    }

    NodePtr call(const std::string& function)
    {
        take(); // (
        std::vector<NodePtr> args;
        int net = -1;
        if (function == "mlp" || function == "flux" || function == "ldiffuse") {
            const Token& name = tokens_[pos_];
            net = name.kind == Token::Name ? law_.net_id(name.text) : -1;
            if (net < 0) {
                fail("'" + function + "' takes the name of a declared net first");
                return constant(0.0);
            }
            ++pos_;
            args.push_back(constant(0.0)); // its place: the net is the node's index
            if (!peek(",")) {
                fail("expected ',' after the net's name");
                return constant(0.0);
            }
            take();
        }
        if (!peek(")")) {
            args.push_back(expression());
            while (ok() && peek(",")) {
                take();
                args.push_back(expression());
            }
        }
        expect(")");
        const auto found = functions().find(function);
        if (found == functions().end()) {
            fail("unknown function '" + function + "'");
            return constant(0.0);
        }
        const FunctionSpec& spec = found->second;
        const int count = static_cast<int>(args.size());
        if (count < spec.min_args || count > spec.max_args) {
            fail("'" + function + "' takes " + std::to_string(spec.min_args) +
                (spec.max_args != spec.min_args ? "-" + std::to_string(spec.max_args) : "") + " argument(s), got " +
                std::to_string(count));
            return constant(0.0);
        }
        NodePtr node = make(spec.op);
        if (net >= 0) {
            args.erase(args.begin());
            node->index = net;
            const NetDecl& decl = law_.nets[static_cast<std::size_t>(net)];
            const int count_in = static_cast<int>(args.size());
            const int wanted = spec.op == Op::Mlp ? count_in : (spec.op == Op::Flux ? 2 * count_in : 5);
            if (decl.inputs() != wanted) {
                fail("net '" + decl.name + "' takes " + std::to_string(decl.inputs()) + " inputs; " + function + " here gives it " +
                    std::to_string(wanted));
                return constant(0.0);
            }
            if (spec.op == Op::LDiffuse && decl.outputs() != 4) {
                fail("ldiffuse's net gives 4 weights (one per cell of the face's window), '" + decl.name + "' gives " +
                    std::to_string(decl.outputs()));
                return constant(0.0);
            }
            const std::size_t moved = spec.op == Op::LDiffuse ? 1 : 0;
            if ((spec.op == Op::Flux || spec.op == Op::LDiffuse) && args[moved]->op != Op::Field) {
                fail(function + " moves a field: its " + (moved == 0 ? "first" : "second") + " argument after the net is a field's name");
                return constant(0.0);
            }
            if (spec.op == Op::Flux) {
                // What the limiter keeps it within: the moved field's declared range (appended, so the IR carries it).
                const FieldDecl& f = law_.fields[static_cast<std::size_t>(args[0]->index)];
                args.push_back(constant(f.lo));
                args.push_back(constant(f.hi));
            }
            node->args = std::move(args);
            return node;
        }
        if (spec.op == Op::Poisson) {
            node->args = {args[0]};
            node->iterations = 24;
            if (count == 2) {
                if (args[1]->op != Op::Const || args[1]->number < 1.0) {
                    fail("poisson's iteration count must be a positive number");
                } else {
                    node->iterations = static_cast<int>(args[1]->number);
                }
            }
            return node;
        }
        if (spec.op == Op::Shift) {
            // The offsets are whole cells, fixed when the law is written: shift(F, 0, 0, -1) is "F
            // in the cell below".
            for (std::size_t a = 1; a < 4; ++a) {
                double sign = 1.0;
                const Node* arg = args[a].get();
                if (arg->op == Op::Neg) {
                    sign = -1.0;
                    arg = arg->args[0].get();
                }
                if (arg->op != Op::Const || arg->number != std::floor(arg->number) || std::abs(arg->number) > 8.0) {
                    fail("shift's offsets must be whole numbers of cells (at most 8)");
                    return constant(0.0);
                }
                args[a] = constant(sign * arg->number);
            }
        }
        node->args = std::move(args);
        return node;
    }

    NodePtr name(const std::string& text)
    {
        if (text == "h") {
            return make(Op::Cell);
        }
        if (text == "rim") {
            return make(Op::Rim);
        }
        if (text == "pi") {
            return constant(3.14159265358979323846);
        }
        if (locals_ != nullptr) {
            const auto found = std::find(locals_->begin(), locals_->end(), text);
            if (found != locals_->end()) {
                auto node = make(Op::Local);
                node->index = static_cast<int>(found - locals_->begin());
                return node;
            }
        }
        if (const int param = law_.param_id(text); param >= 0) {
            auto node = make(Op::Param);
            node->index = param;
            return node;
        }
        if (const int field = law_.field_id(text); field >= 0) {
            auto node = make(Op::Field);
            node->index = field;
            return node;
        }
        fail("unknown name '" + text + "' (not a field, param, h or pi)");
        return constant(0.0);
    }

    [[nodiscard]] bool peek(std::string_view symbol) const
    {
        return tokens_[pos_].kind == Token::Symbol && tokens_[pos_].text == symbol;
    }
    const Token& take() { return tokens_[pos_++]; }
    void expect(std::string_view symbol)
    {
        if (!peek(symbol)) {
            fail("expected '" + std::string(symbol) + "'");
            return;
        }
        ++pos_;
    }
    void fail(const std::string& message)
    {
        if (error_.empty()) {
            error_ = message;
        }
    }

    const std::vector<Token>& tokens_;
    std::size_t pos_ = 0;
    const Law& law_;
    const std::vector<std::string>* locals_ = nullptr;
    std::string error_;
};

// ---- lattice helpers ---------------------------------------------------------------------------------

struct Lattice {
    int nx, ny, nz;
    std::size_t sx, sy, sz; // index strides
};

Lattice lattice_of(const GridShape& shape)
{
    return Lattice{shape.nx, shape.ny, shape.nz, 1, static_cast<std::size_t>(shape.nx),
        static_cast<std::size_t>(shape.nx) * static_cast<std::size_t>(shape.ny)};
}

// 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher), in place over a strided row.
void edt_1d(float* data, std::size_t count, std::size_t stride, std::vector<float>& f, std::vector<float>& d,
    std::vector<int>& v, std::vector<float>& z)
{
    const float inf = std::numeric_limits<float>::infinity();
    f.resize(count);
    d.resize(count);
    v.resize(count);
    z.resize(count + 1);
    for (std::size_t q = 0; q < count; ++q) {
        f[q] = data[q * stride];
    }
    int k = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    for (int q = 1; q < static_cast<int>(count); ++q) {
        if (f[static_cast<std::size_t>(q)] == inf) {
            continue;
        }
        while (true) {
            const int p = v[static_cast<std::size_t>(k)];
            if (f[static_cast<std::size_t>(p)] == inf) {
                // An unreachable parabola never wins; replace it outright.
                v[static_cast<std::size_t>(k)] = q;
                z[static_cast<std::size_t>(k)] = -inf;
                z[static_cast<std::size_t>(k) + 1] = inf;
                break;
            }
            const float s = ((f[static_cast<std::size_t>(q)] + static_cast<float>(q * q)) -
                                (f[static_cast<std::size_t>(p)] + static_cast<float>(p * p))) /
                static_cast<float>(2 * q - 2 * p);
            if (s <= z[static_cast<std::size_t>(k)]) {
                if (k == 0) {
                    v[0] = q;
                    z[0] = -inf;
                    z[1] = inf;
                    break;
                }
                --k;
                continue;
            }
            ++k;
            v[static_cast<std::size_t>(k)] = q;
            z[static_cast<std::size_t>(k)] = s;
            z[static_cast<std::size_t>(k) + 1] = inf;
            break;
        }
    }
    k = 0;
    for (int q = 0; q < static_cast<int>(count); ++q) {
        while (z[static_cast<std::size_t>(k) + 1] < static_cast<float>(q)) {
            ++k;
        }
        const int p = v[static_cast<std::size_t>(k)];
        const float fp = f[static_cast<std::size_t>(p)];
        d[static_cast<std::size_t>(q)] = fp == inf ? inf : static_cast<float>((q - p) * (q - p)) + fp;
    }
    for (std::size_t q = 0; q < count; ++q) {
        data[q * stride] = d[q];
    }
}

// Squared distance (in cells) from every cell to the nearest cell where `feature` is true.
void squared_distance(const Lattice& l, const std::vector<char>& feature, std::vector<float>& out)
{
    const float inf = std::numeric_limits<float>::infinity();
    const std::size_t n = feature.size();
    out.assign(n, inf);
    for (std::size_t i = 0; i < n; ++i) {
        if (feature[i]) {
            out[i] = 0.0f;
        }
    }
    // Every row of a pass is independent of the others: rows go to threads, each with its own scratch.
    auto pass = [&](int outer, int inner, std::size_t outer_stride, std::size_t inner_stride, std::size_t count,
                    std::size_t stride) {
#pragma omp parallel
        {
            std::vector<float> f;
            std::vector<float> d;
            std::vector<int> v;
            std::vector<float> z;
#pragma omp for schedule(static)
            for (int a = 0; a < outer; ++a) {
                for (int b = 0; b < inner; ++b) {
                    edt_1d(out.data() + static_cast<std::size_t>(a) * outer_stride +
                            static_cast<std::size_t>(b) * inner_stride,
                        count, stride, f, d, v, z);
                }
            }
        }
    };
    pass(l.nz, l.ny, l.sz, l.sy, static_cast<std::size_t>(l.nx), l.sx);
    if (l.ny > 1) {
        pass(l.nz, l.nx, l.sz, l.sx, static_cast<std::size_t>(l.ny), l.sy);
    }
    if (l.nz > 1) {
        pass(l.ny, l.nx, l.sy, l.sx, static_cast<std::size_t>(l.nz), l.sz);
    }
}

std::uint64_t bits_key(const Value& v, std::size_t n, std::uint64_t salt)
{
    std::uint64_t hash = 1469598103934665603ull ^ salt;
    if (v.is_scalar()) {
        const double s = v.scalar;
        const auto* p = reinterpret_cast<const unsigned char*>(&s);
        for (std::size_t i = 0; i < sizeof(s); ++i) {
            hash = (hash ^ p[i]) * 1099511628211ull;
        }
        return hash;
    }
    const auto* p = reinterpret_cast<const std::uint32_t*>(v.array);
    for (std::size_t i = 0; i < n; ++i) {
        hash = (hash ^ p[i]) * 1099511628211ull;
    }
    return hash;
}

template <typename Fn>
void for_each_cell(const Lattice& l, Fn&& fn)
{
    parallel_cells(l.nx, l.ny, l.nz, std::forward<Fn>(fn));
}

// AN ARRAY OF ZEROS IS THE SCALAR 0. What a term computes as nothing — stone below its melting point
// melts at rate 0 in every cell — then costs nothing downstream, exactly like a field nobody poured
// into: zero propagates through products and an idle term stops at its first factor.
Value settle(std::vector<float>& buffer)
{
    for (const float v : buffer) {
        if (v != 0.0f) {
            return Value{.array = buffer.data()};
        }
    }
    return Value{.scalar = 0.0};
}

// Every cell holds the same value: differences, gradients and fluxes of it are exactly zero.
bool uniform(const Value& v, std::size_t n)
{
    if (v.is_scalar()) {
        return true;
    }
    const float first = v.array[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (v.array[i] != first) {
            return false;
        }
    }
    return true;
}

// Per z-plane: can anything move across it? A plane that is uniform, between neighbours uniform at the same
// value, has only exact zeros to compute. The sky and the deep rock are most of a world, and heat, scent and
// mana mostly live in a few planes near the ground.
std::vector<char> active_planes(const float* v, const Lattice& l)
{
    std::vector<char> flat(static_cast<std::size_t>(l.nz), 1);
    std::vector<float> level(static_cast<std::size_t>(l.nz), 0.0f);
    for (int z = 0; z < l.nz; ++z) {
        const std::size_t first = static_cast<std::size_t>(z) * l.sz;
        const float value = v[first];
        level[static_cast<std::size_t>(z)] = value;
        for (std::size_t i = first + 1; i < first + l.sz; ++i) {
            if (v[i] != value) {
                flat[static_cast<std::size_t>(z)] = 0;
                break;
            }
        }
    }
    std::vector<char> active(static_cast<std::size_t>(l.nz), 0);
    for (int z = 0; z < l.nz; ++z) {
        const auto s = static_cast<std::size_t>(z);
        bool moves = !flat[s];
        if (z > 0) {
            moves = moves || !flat[s - 1] || level[s - 1] != level[s];
        }
        if (z + 1 < l.nz) {
            moves = moves || !flat[s + 1] || level[s + 1] != level[s];
        }
        active[s] = moves ? 1 : 0;
    }
    return active;
}

template <typename Fn>
Value unary_op(Node& node, const EvalContext& context, Fn fn)
{
    const Value a = evaluate(*node.args[0], context);
    if (a.is_scalar()) {
        return Value{.scalar = fn(a.scalar)};
    }
    const std::size_t n = context.fields->shape().cells();
    node.buffer.resize(n);
    parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            node.buffer[i] = static_cast<float>(fn(static_cast<double>(a.array[i])));
        }
    });
    return settle(node.buffer);
}

template <typename Fn>
Value binary_op(Node& node, const EvalContext& context, Fn fn)
{
    const Value a = evaluate(*node.args[0], context);
    const Value b = evaluate(*node.args[1], context);
    if (a.is_scalar() && b.is_scalar()) {
        return Value{.scalar = fn(a.scalar, b.scalar)};
    }
    const std::size_t n = context.fields->shape().cells();
    node.buffer.resize(n);
    // The scalar/array decision is made once, outside the loop, so the loop body is just the op.
    parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
        if (a.is_scalar()) {
            const double s = a.scalar;
            for (std::size_t i = begin; i < end; ++i) {
                node.buffer[i] = static_cast<float>(fn(s, static_cast<double>(b.array[i])));
            }
        } else if (b.is_scalar()) {
            const double s = b.scalar;
            for (std::size_t i = begin; i < end; ++i) {
                node.buffer[i] = static_cast<float>(fn(static_cast<double>(a.array[i]), s));
            }
        } else {
            for (std::size_t i = begin; i < end; ++i) {
                node.buffer[i] =
                    static_cast<float>(fn(static_cast<double>(a.array[i]), static_cast<double>(b.array[i])));
            }
        }
    });
    return settle(node.buffer);
}

} // namespace

// ---- evaluation --------------------------------------------------------------------------------------

namespace {
Value evaluate_node(Node& node, const EvalContext& context);
} // namespace

namespace {

// ---- the network ops' faces (the CUDA kernels do the same arithmetic, in the same order) ---------------------------

float limit_face(float r, float from_i, float from_j, double lo, double hi, double dt, double h)
{
    if (dt <= 0.0) {
        return r;
    }
    const float per = static_cast<float>(0.999 * h / (6.0 * dt)); // a margin: six faces' rounding stays inside
    const bool has_lo = lo > -1e29;
    const bool has_hi = hi < 1e29;
    if (r > 0.0f) { // i gives to j: i keeps its least, j gets no more than its most
        if (has_lo) r = std::min(r, std::max(0.0f, (from_i - static_cast<float>(lo)) * per));
        if (has_hi) r = std::min(r, std::max(0.0f, (static_cast<float>(hi) - from_j) * per));
    } else if (r < 0.0f) {
        if (has_lo) r = std::max(r, -std::max(0.0f, (from_j - static_cast<float>(lo)) * per));
        if (has_hi) r = std::max(r, -std::max(0.0f, (static_cast<float>(hi) - from_i) * per));
    }
    return r;
}

double face_flux(const NetDecl& net, const std::vector<Value>& v, std::size_t i, std::size_t j, double lo, double hi, double dt,
    double h, std::vector<float>& xij, std::vector<float>& xji, std::vector<float>& y, std::vector<float>& scratch)
{
    const std::size_t m = v.size();
    for (std::size_t k = 0; k < m; ++k) {
        xij[k] = static_cast<float>(v[k].at(i));
        xij[m + k] = static_cast<float>(v[k].at(j));
        xji[k] = xij[m + k];
        xji[m + k] = xij[k];
    }
    net_forward(net, xij.data(), y.data(), scratch.data());
    const float gij = y[0];
    net_forward(net, xji.data(), y.data(), scratch.data());
    const float gji = y[0];
    const float r = 0.5f * (gij - gji);
    return limit_face(r, xij[0], xij[m], lo, hi, dt, h);
}

// The raw flux i -> j from one side's view of the window w[0..3] = F[i-1] F[i] F[j] F[j+1].
float learned_raw(const NetDecl& net, const float* w, float kf, float dt, float h, std::vector<float>& y, std::vector<float>& scratch)
{
    const float centre = 0.5f * (w[1] + w[2]);
    float spread = 1e-6f;
    for (int a = 0; a < 4; ++a) {
        spread = std::max(spread, std::abs(w[a] - centre));
    }
    float x[5];
    for (int a = 0; a < 4; ++a) {
        x[a] = (w[a] - centre) / spread;
    }
    x[4] = kf * dt / (h * h);
    net_forward(net, x, y.data(), scratch.data());
    const float mean = 0.25f * (y[0] + y[1] + y[2] + y[3]);
    const float base[4] = {0.0f, -1.0f, 1.0f, 0.0f};
    float acc = 0.0f;
    for (int a = 0; a < 4; ++a) {
        acc += (base[a] + (y[static_cast<std::size_t>(a)] - mean)) * w[a];
    }
    return -kf / h * acc;
}

double learned_face(const NetDecl& net, const float* w, float kf, float dt, float h, std::vector<float>& y, std::vector<float>& scratch)
{
    const float mirror[4] = {w[3], w[2], w[1], w[0]};
    const float ij = learned_raw(net, w, kf, dt, h, y, scratch);
    const float ji = learned_raw(net, mirror, kf, dt, h, y, scratch);
    return 0.5f * (ij - ji);
}

} // namespace

Value evaluate(Node& node, const EvalContext& context)
{
    if (node.shared && context.epoch != 0) {
        if (node.epoch == context.epoch) {
            return node.cached;
        }
        node.cached = evaluate_node(node, context);
        node.epoch = context.epoch;
        return node.cached;
    }
    return evaluate_node(node, context);
}

namespace {

Value evaluate_node(Node& node, const EvalContext& context)
{
    const GridShape& shape = context.fields->shape();
    const std::size_t n = shape.cells();
    const double h = shape.cell;
    const Lattice l = lattice_of(shape);

    switch (node.op) {
    case Op::Const:
        return Value{.scalar = node.number};
    case Op::Param:
        return Value{.scalar = (*context.params)[static_cast<std::size_t>(node.index)]};
    case Op::Cell:
        return Value{.scalar = h};
    case Op::Rim:
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            node.buffer[i] = x == 0 || y == 0 || z == 0 || x + 1 == l.nx || y + 1 == l.ny || z + 1 == l.nz ? 1.0f : 0.0f;
        });
        return settle(node.buffer);
    case Op::Local:
        return Value{.scalar = 0.0}; // a creature's own state: only meaningful at a point (evaluate_at)
    case Op::Field:
        if (context.zero != nullptr && (*context.zero)[static_cast<std::size_t>(node.index)]) {
            return Value{.scalar = 0.0};
        }
        return Value{.array = context.fields->values(node.index).data()};
    case Op::Neg:
        return unary_op(node, context, [](double a) { return -a; });
    case Op::Abs:
        return unary_op(node, context, [](double a) { return std::abs(a); });
    case Op::Sqrt:
        return unary_op(node, context, [](double a) { return std::sqrt(std::max(0.0, a)); });
    case Op::Exp:
        return unary_op(node, context, [](double a) { return std::exp(std::min(a, 80.0)); });
    case Op::Log:
        return unary_op(node, context, [](double a) { return std::log(std::max(a, 1e-30)); });
    case Op::Step:
        return unary_op(node, context, [](double a) { return a > 0.0 ? 1.0 : 0.0; });
    case Op::Pos:
        return unary_op(node, context, [](double a) { return std::max(a, 0.0); });
    case Op::Add:
        return binary_op(node, context, [](double a, double b) { return a + b; });
    case Op::Sub:
        return binary_op(node, context, [](double a, double b) { return a - b; });
    case Op::Mul: {
        // Zero times anything is zero, without looking at the anything: this is what makes an idle
        // term (mana nobody cast) free. The cheaper factor goes first, so `-(a long sum) * (T -
        // ambient)` in a world at ambient costs one subtraction, not the long sum.
        const bool swap = node.args[1]->cost < node.args[0]->cost;
        Node& first = *node.args[swap ? 1 : 0];
        Node& second = *node.args[swap ? 0 : 1];
        const Value fa = evaluate(first, context);
        if (fa.is_scalar() && fa.scalar == 0.0) {
            return Value{.scalar = 0.0};
        }
        const Value fb = evaluate(second, context);
        if (fb.is_scalar() && fb.scalar == 0.0) {
            return Value{.scalar = 0.0};
        }
        const Value a = swap ? fb : fa; // a * b keeps its order, so the bits do not depend on the swap
        const Value b = swap ? fa : fb;
        if (a.is_scalar() && b.is_scalar()) {
            return Value{.scalar = a.scalar * b.scalar};
        }
        node.buffer.resize(n);
        parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
            if (a.is_scalar()) {
                const double s = a.scalar;
                for (std::size_t i = begin; i < end; ++i) {
                    node.buffer[i] = static_cast<float>(s * b.array[i]);
                }
            } else if (b.is_scalar()) {
                const double s = b.scalar;
                for (std::size_t i = begin; i < end; ++i) {
                    node.buffer[i] = static_cast<float>(a.array[i] * s);
                }
            } else {
                for (std::size_t i = begin; i < end; ++i) {
                    node.buffer[i] = static_cast<float>(static_cast<double>(a.array[i]) * b.array[i]);
                }
            }
        });
        return settle(node.buffer);
    }
    case Op::Div:
        // Division by zero is a law bug, not a crash: it yields 0 and the range contract on whatever
        // it feeds is what notices.
        return binary_op(node, context, [](double a, double b) { return b == 0.0 ? 0.0 : a / b; });
    case Op::Pow:
        return binary_op(node, context, [](double a, double b) { return std::pow(a, b); });
    case Op::Min:
        return binary_op(node, context, [](double a, double b) { return std::min(a, b); });
    case Op::Max:
        return binary_op(node, context, [](double a, double b) { return std::max(a, b); });
    case Op::Clamp: {
        const Value a = evaluate(*node.args[0], context);
        const Value lo = evaluate(*node.args[1], context);
        const Value hi = evaluate(*node.args[2], context);
        if (a.is_scalar() && lo.is_scalar() && hi.is_scalar()) {
            return Value{.scalar = std::clamp(a.scalar, lo.scalar, std::max(lo.scalar, hi.scalar))};
        }
        node.buffer.resize(n);
        parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const double low = lo.at(i);
                node.buffer[i] = static_cast<float>(std::clamp(a.at(i), low, std::max(low, hi.at(i))));
            }
        });
        return settle(node.buffer);
    }
    case Op::Shift: {
        // The value a few cells away: what lies below, above, beside. Clamped at the edge, like every
        // stencil here (the edge cell repeats).
        const Value a = evaluate(*node.args[0], context);
        if (a.is_scalar()) {
            return a;
        }
        const int dx = static_cast<int>(node.args[1]->number);
        const int dy = static_cast<int>(node.args[2]->number);
        const int dz = static_cast<int>(node.args[3]->number);
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            const int sx = std::clamp(x + dx, 0, l.nx - 1);
            const int sy = std::clamp(y + dy, 0, l.ny - 1);
            const int sz = std::clamp(z + dz, 0, l.nz - 1);
            node.buffer[i] = a.array[static_cast<std::size_t>(sz) * l.sz + static_cast<std::size_t>(sy) * l.sy +
                static_cast<std::size_t>(sx)];
        });
        return settle(node.buffer);
    }
    case Op::Lap:
    case Op::Gx:
    case Op::Gy:
    case Op::Gz: {
        const Value a = evaluate(*node.args[0], context);
        if (uniform(a, n)) {
            return Value{.scalar = 0.0};
        }
        node.buffer.resize(n);
        const float* v = a.array;
        const Op op = node.op;
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            const std::size_t xm = x > 0 ? i - l.sx : i;
            const std::size_t xp = x + 1 < l.nx ? i + l.sx : i;
            const std::size_t ym = y > 0 ? i - l.sy : i;
            const std::size_t yp = y + 1 < l.ny ? i + l.sy : i;
            const std::size_t zm = z > 0 ? i - l.sz : i;
            const std::size_t zp = z + 1 < l.nz ? i + l.sz : i;
            double out = 0.0;
            switch (op) {
            case Op::Lap:
                out = (static_cast<double>(v[xm]) + v[xp] + v[ym] + v[yp] + v[zm] + v[zp] - 6.0 * v[i]) / (h * h);
                break;
            case Op::Gx:
                out = (static_cast<double>(v[xp]) - v[xm]) / (h * static_cast<double>((xp - xm) / l.sx));
                break;
            case Op::Gy:
                out = l.ny > 1 ? (static_cast<double>(v[yp]) - v[ym]) / (h * static_cast<double>((yp - ym) / l.sy))
                               : 0.0;
                break;
            default:
                out = l.nz > 1 ? (static_cast<double>(v[zp]) - v[zm]) / (h * static_cast<double>((zp - zm) / l.sz))
                               : 0.0;
                break;
            }
            node.buffer[i] = static_cast<float>(std::isfinite(out) ? out : 0.0);
        });
        return settle(node.buffer);
    }
    case Op::Diffuse: {
        const Value k = evaluate(*node.args[0], context);
        const Value f = evaluate(*node.args[1], context);
        if (uniform(f, n)) {
            return Value{.scalar = 0.0}; // nothing to even out: every flux is k * 0
        }
        node.buffer.resize(n);
        const float* v = f.array;
        // A face conducts with the harmonic mean of its two cells: an insulator on either side
        // stops the flow, and the flux leaving one cell is exactly the flux entering the other.
        auto face = [&](std::size_t a, std::size_t b) {
            const double ka = k.at(a);
            const double kb = k.at(b);
            const double kf = (ka + kb) > 0.0 ? 2.0 * ka * kb / (ka + kb) : 0.0;
            return kf * (static_cast<double>(v[b]) - v[a]);
        };
        const std::vector<char> active = active_planes(v, l);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            if (!active[static_cast<std::size_t>(z)]) {
                node.buffer[i] = 0.0f;
                return;
            }
            double acc = 0.0;
            if (x > 0) acc += face(i, i - l.sx);
            if (x + 1 < l.nx) acc += face(i, i + l.sx);
            if (y > 0) acc += face(i, i - l.sy);
            if (y + 1 < l.ny) acc += face(i, i + l.sy);
            if (z > 0) acc += face(i, i - l.sz);
            if (z + 1 < l.nz) acc += face(i, i + l.sz);
            node.buffer[i] = static_cast<float>(acc / (h * h));
        });
        return settle(node.buffer);
    }
    case Op::Adv: {
        const Value f = evaluate(*node.args[0], context);
        const Value u = evaluate(*node.args[1], context);
        const Value vv = evaluate(*node.args[2], context);
        const Value w = evaluate(*node.args[3], context);
        if (uniform(f, n)) {
            return Value{.scalar = 0.0};
        }
        node.buffer.resize(n);
        const float* v = f.array;
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            auto upwind = [&](double speed, int c, int count, std::size_t stride) {
                if (speed > 0.0) {
                    return c > 0 ? speed * (static_cast<double>(v[i]) - v[i - stride]) / h : 0.0;
                }
                return c + 1 < count ? speed * (static_cast<double>(v[i + stride]) - v[i]) / h : 0.0;
            };
            const double out = upwind(u.at(i), x, l.nx, l.sx) + upwind(vv.at(i), y, l.ny, l.sy) +
                upwind(w.at(i), z, l.nz, l.sz);
            node.buffer[i] = static_cast<float>(-out);
        });
        return settle(node.buffer);
    }
    case Op::Distance: {
        const Value a = evaluate(*node.args[0], context);
        const std::uint64_t key = bits_key(a, n, 0xd157);
        if (node.cache_valid && node.cache_key == key && node.buffer.size() == n) {
            return Value{.array = node.buffer.data()};
        }
        node.cache_key = key;
        node.cache_valid = true;
        node.buffer.resize(n);
        const double far = (l.nx + l.ny + l.nz) * h;
        std::vector<char> inside(n, 0);
        bool any_inside = false;
        bool any_outside = false;
        for (std::size_t i = 0; i < n; ++i) {
            inside[i] = a.at(i) >= 0.5 ? 1 : 0;
            any_inside = any_inside || inside[i];
            any_outside = any_outside || !inside[i];
        }
        std::vector<float> to_inside;
        std::vector<float> to_outside;
        if (any_inside) {
            squared_distance(l, inside, to_inside);
        }
        if (any_outside) {
            std::vector<char> outside(n);
            for (std::size_t i = 0; i < n; ++i) {
                outside[i] = inside[i] ? 0 : 1;
            }
            squared_distance(l, outside, to_outside);
        }
        // BEYOND THE LATTICE IS AIR. Rock cut off by the lattice edge has a face there, and a ray
        // entering through that face hits it — so the distance field must know the face too, or the
        // body walks into what the eye sees as a wall. The nearest out-of-lattice cell is straight
        // across the nearest border plane, so the min over axes is exact. (An axis of size 1 is a 2-D
        // world's thickness, not a border.)
        auto border = [&](int x, int y, int z) {
            double best = std::numeric_limits<double>::infinity();
            if (l.nx > 1) best = std::min({best, static_cast<double>(x + 1), static_cast<double>(l.nx - x)});
            if (l.ny > 1) best = std::min({best, static_cast<double>(y + 1), static_cast<double>(l.ny - y)});
            if (l.nz > 1) best = std::min({best, static_cast<double>(z + 1), static_cast<double>(l.nz - z)});
            return best;
        };
        // The surface lies half a cell from the centre of the last solid cell.
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            double d = 0.0;
            if (inside[i]) {
                const double out_cells =
                    std::min(any_outside ? std::sqrt(static_cast<double>(to_outside[i])) : far / h, border(x, y, z));
                d = -(out_cells - 0.5) * h;
            } else {
                d = any_inside ? (std::sqrt(static_cast<double>(to_inside[i])) - 0.5) * h : far;
            }
            node.buffer[i] = static_cast<float>(d);
        });
        return Value{.array = node.buffer.data()};
    }
    case Op::Poisson: {
        const Value s = evaluate(*node.args[0], context);
        // An unchanged source after a full set of sweeps: the warm start already IS the answer to
        // within what one more set would change, so the solve is skipped. (Not exact for a source
        // that changed once and then stopped — the next steps keep sweeping until it repeats.)
        const std::uint64_t key = bits_key(s, n, 0x9015 + static_cast<std::uint64_t>(node.iterations));
        if (node.cache_valid && node.cache_key == key && node.buffer.size() == n) {
            return Value{.array = node.buffer.data()};
        }
        node.cache_key = key;
        node.cache_valid = true;
        // Warm start: the previous solution is the first guess, so a few sweeps per step track a
        // slowly moving mass. phi = 0 outside the lattice (an isolated world).
        if (node.buffer.size() != n) {
            node.buffer.assign(n, 0.0f);
        }
        node.scratch.resize(n);
        int axes = 0;
        axes += l.nx > 1 ? 1 : 0;
        axes += l.ny > 1 ? 1 : 0;
        axes += l.nz > 1 ? 1 : 0;
        const double denom = 2.0 * std::max(axes, 1);
        for (int it = 0; it < node.iterations; ++it) {
            const float* phi = node.buffer.data();
            for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
                double sum = 0.0;
                if (l.nx > 1) {
                    sum += x > 0 ? phi[i - l.sx] : 0.0;
                    sum += x + 1 < l.nx ? phi[i + l.sx] : 0.0;
                }
                if (l.ny > 1) {
                    sum += y > 0 ? phi[i - l.sy] : 0.0;
                    sum += y + 1 < l.ny ? phi[i + l.sy] : 0.0;
                }
                if (l.nz > 1) {
                    sum += z > 0 ? phi[i - l.sz] : 0.0;
                    sum += z + 1 < l.nz ? phi[i + l.sz] : 0.0;
                }
                node.scratch[i] = static_cast<float>((sum - h * h * s.at(i)) / denom);
            });
            node.buffer.swap(node.scratch);
        }
        return Value{.array = node.buffer.data()};
    }
    case Op::Fall: {
        const Value f = evaluate(*node.args[0], context);
        const Value b = evaluate(*node.args[1], context);
        const Value speed = evaluate(*node.args[2], context);
        if (f.is_scalar() || l.nz < 2) {
            return Value{.scalar = 0.0};
        }
        // The flux from a cell into the one below it: its content times how much room is there. Each
        // flux leaves one cell and enters the other, so the op only moves F, never makes it. Written
        // as two gathers (the fluxes, then what each cell gains and loses) so cells are independent.
        node.wide.assign(n, 0.0);
        for_each_cell(l, [&](std::size_t i, int, int, int z) {
            if (z == 0) {
                return;
            }
            const std::size_t below = i - l.sz;
            const double free = std::clamp(1.0 - b.at(below) - static_cast<double>(f.array[below]), 0.0, 1.0);
            node.wide[i] = std::max(0.0, speed.at(i)) / h * std::max(0.0, static_cast<double>(f.array[i])) * free;
        });
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int, int, int z) {
            float out = static_cast<float>(0.0 - node.wide[i]);
            if (z + 1 < l.nz) {
                out = static_cast<float>(out + node.wide[i + l.sz]);
            }
            node.buffer[i] = out;
        });
        return settle(node.buffer);
    }
    case Op::Spread: {
        const Value f = evaluate(*node.args[0], context);
        const Value b = evaluate(*node.args[1], context);
        const Value k = evaluate(*node.args[2], context);
        if (uniform(f, n)) {
            return Value{.scalar = 0.0};
        }
        // Sideways only, downhill only, and only into room that is there: the flux between two
        // neighbours is k * (difference) * (free space on the receiving side), antisymmetric, so a
        // puddle levels out without climbing walls or leaking into rock.
        auto flux = [&](std::size_t a, std::size_t c) {
            const double fa = f.array[a];
            const double fc = f.array[c];
            const std::size_t r = fa > fc ? c : a; // the side the flow goes into
            const double free = std::clamp(1.0 - b.at(r) - static_cast<double>(f.array[r]), 0.0, 1.0);
            const double kf = 0.5 * (std::max(0.0, k.at(a)) + std::max(0.0, k.at(c)));
            return kf / (h * h) * (fa - fc) * free; // from a to c
        };
        node.wide.assign(2 * n, 0.0); // [0, n): across the +x face of each cell; [n, 2n): the +y face
        for_each_cell(l, [&](std::size_t i, int x, int y, int) {
            if (x + 1 < l.nx) node.wide[i] = flux(i, i + l.sx);
            if (y + 1 < l.ny) node.wide[n + i] = flux(i, i + l.sy);
        });
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int) {
            float out = 0.0f;
            if (y > 0) out = static_cast<float>(out + node.wide[n + i - l.sy]);
            if (x > 0) out = static_cast<float>(out + node.wide[i - l.sx]);
            out = static_cast<float>(out - node.wide[i]);
            out = static_cast<float>(out - node.wide[n + i]);
            node.buffer[i] = out;
        });
        return settle(node.buffer);
    }
    case Op::Topple: {
        const Value f = evaluate(*node.args[0], context);
        const Value b = evaluate(*node.args[1], context);
        const Value rate = evaluate(*node.args[2], context);
        if (f.is_scalar() || l.nz < 2) {
            return Value{.scalar = 0.0};
        }
        // A pile, not a column: what rests on something (so cannot fall straight) slides off any
        // side where both the cell beside it and the cell below THAT are open — diagonally down, one
        // cell at a time. Repeated, that is a 45-degree slope: rubble heaps instead of standing in
        // the shape of the wall it used to be. Each flux leaves one cell and enters another.
        auto room = [&](std::size_t j) {
            return std::clamp(1.0 - b.at(j) - static_cast<double>(f.array[j]), 0.0, 1.0);
        };
        // wide[4 i + s]: what cell i sends down its side s (-x, +x, -y, +y).
        node.wide.assign(4 * n, 0.0);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            const double here = f.array[i];
            if (here <= 0.0 || z == 0) {
                return;
            }
            if (room(i - l.sz) > 0.01) {
                return; // it can fall straight down; fall() handles that
            }
            const std::size_t sides[4] = {
                x > 0 ? i - l.sx : i, x + 1 < l.nx ? i + l.sx : i, y > 0 ? i - l.sy : i, y + 1 < l.ny ? i + l.sy : i};
            for (std::size_t s = 0; s < 4; ++s) {
                const std::size_t j = sides[s];
                if (j == i) {
                    continue;
                }
                node.wide[4 * i + s] = 0.25 * std::max(0.0, rate.at(i)) / h * here * std::min(room(j), room(j - l.sz));
            }
        });
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            double out = 0.0;
            for (std::size_t s = 0; s < 4; ++s) {
                out -= node.wide[4 * i + s];
            }
            // What lands here slid off the cell above-and-beside, toward this column.
            if (z + 1 < l.nz) {
                const std::size_t up = i + l.sz;
                if (x + 1 < l.nx) out += node.wide[4 * (up + l.sx) + 0]; // it slid -x
                if (x > 0) out += node.wide[4 * (up - l.sx) + 1];        // it slid +x
                if (y + 1 < l.ny) out += node.wide[4 * (up + l.sy) + 2]; // it slid -y
                if (y > 0) out += node.wide[4 * (up - l.sy) + 3];        // it slid +y
            }
            node.buffer[i] = static_cast<float>(out);
        });
        return settle(node.buffer);
    }
    case Op::Mlp: {
        if (context.nets == nullptr) {
            return Value{.scalar = 0.0};
        }
        const NetDecl& net = (*context.nets)[static_cast<std::size_t>(node.index)];
        std::vector<Value> in;
        bool scalar = true;
        for (const NodePtr& arg : node.args) {
            in.push_back(evaluate(*arg, context));
            scalar = scalar && in.back().is_scalar();
        }
        const std::size_t width = static_cast<std::size_t>(net.widest());
        if (scalar) {
            std::vector<float> x(in.size());
            std::vector<float> y(static_cast<std::size_t>(net.outputs()));
            std::vector<float> scratch(2 * width);
            for (std::size_t k = 0; k < in.size(); ++k) {
                x[k] = static_cast<float>(in[k].scalar);
            }
            net_forward(net, x.data(), y.data(), scratch.data());
            return Value{.scalar = y[0]};
        }
        node.buffer.resize(n);
        parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
            std::vector<float> x(in.size());
            std::vector<float> y(static_cast<std::size_t>(net.outputs()));
            std::vector<float> scratch(2 * width);
            for (std::size_t i = begin; i < end; ++i) {
                for (std::size_t k = 0; k < in.size(); ++k) {
                    x[k] = static_cast<float>(in[k].at(i));
                }
                net_forward(net, x.data(), y.data(), scratch.data());
                node.buffer[i] = y[0];
            }
        });
        return settle(node.buffer);
    }
    case Op::Flux: {
        // flux(NET, F, a...): across each face the net sees both sides, both ways round; the flux is half the difference,
        // so what leaves one cell is to the bit what enters the other. Then it is limited, the same on both sides of the
        // face, so in one step of dt no cell goes below F's least or above its most.
        if (context.nets == nullptr) {
            return Value{.scalar = 0.0};
        }
        const NetDecl& net = (*context.nets)[static_cast<std::size_t>(node.index)];
        const std::size_t m = node.args.size() - 2; // F and its companions; then lo, hi
        std::vector<Value> v;
        bool flat = true;
        for (std::size_t k = 0; k < m; ++k) {
            v.push_back(evaluate(*node.args[k], context));
            flat = flat && uniform(v.back(), n);
        }
        if (flat) {
            return Value{.scalar = 0.0}; // both sides of every face alike: the two ways round cancel
        }
        const double lo = node.args[m]->number;
        const double hi = node.args[m + 1]->number;
        const double dt = context.dt;
        const std::size_t width = static_cast<std::size_t>(net.widest());
        node.wide.assign(3 * n, 0.0); // the flux from each cell through its +x, +y, +z face
        parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
            std::vector<float> xij(2 * m);
            std::vector<float> xji(2 * m);
            std::vector<float> y(static_cast<std::size_t>(net.outputs()));
            std::vector<float> scratch(2 * width);
            for (std::size_t i = begin; i < end; ++i) {
                const int x = static_cast<int>(i % static_cast<std::size_t>(l.nx));
                const int yy = static_cast<int>((i / static_cast<std::size_t>(l.nx)) % static_cast<std::size_t>(l.ny));
                const int z = static_cast<int>(i / l.sz);
                const bool open[3] = {x + 1 < l.nx, yy + 1 < l.ny, z + 1 < l.nz};
                const std::size_t step[3] = {l.sx, l.sy, l.sz};
                for (int a = 0; a < 3; ++a) {
                    if (!open[a]) {
                        continue;
                    }
                    const std::size_t j = i + step[a];
                    node.wide[3 * i + static_cast<std::size_t>(a)] = face_flux(net, v, i, j, lo, hi, dt, h, xij, xji, y, scratch);
                }
            }
        });
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            float out = 0.0f;
            if (x > 0) out += static_cast<float>(node.wide[3 * (i - l.sx) + 0]);
            if (x + 1 < l.nx) out -= static_cast<float>(node.wide[3 * i + 0]);
            if (y > 0) out += static_cast<float>(node.wide[3 * (i - l.sy) + 1]);
            if (y + 1 < l.ny) out -= static_cast<float>(node.wide[3 * i + 1]);
            if (z > 0) out += static_cast<float>(node.wide[3 * (i - l.sz) + 2]);
            if (z + 1 < l.nz) out -= static_cast<float>(node.wide[3 * i + 2]);
            node.buffer[i] = out / static_cast<float>(h);
        });
        return settle(node.buffer);
    }
    case Op::LDiffuse: {
        // ldiffuse(NET, k, F): the flux through a face is interpolated from the four cells across it, F[i-1] F[i] | F[j]
        // F[j+1], with weights the net gives from what it sees there (scaled to the local spread) and from how big the
        // step is against the face's diffusion time. Weights sum to zero (a flat field does not flow) and start from the
        // classical [0 -1 1 0]: an untrained net is diffuse() exactly.
        if (context.nets == nullptr) {
            return Value{.scalar = 0.0};
        }
        const NetDecl& net = (*context.nets)[static_cast<std::size_t>(node.index)];
        const Value k = evaluate(*node.args[0], context);
        const Value f = evaluate(*node.args[1], context);
        if (uniform(f, n)) {
            return Value{.scalar = 0.0};
        }
        const std::size_t width = static_cast<std::size_t>(net.widest());
        node.wide.assign(3 * n, 0.0);
        parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
            std::vector<float> y(4);
            std::vector<float> scratch(2 * width);
            for (std::size_t i = begin; i < end; ++i) {
                const int c[3] = {static_cast<int>(i % static_cast<std::size_t>(l.nx)),
                    static_cast<int>((i / static_cast<std::size_t>(l.nx)) % static_cast<std::size_t>(l.ny)), static_cast<int>(i / l.sz)};
                const int count[3] = {l.nx, l.ny, l.nz};
                const std::size_t step[3] = {l.sx, l.sy, l.sz};
                for (int a = 0; a < 3; ++a) {
                    if (c[a] + 1 >= count[a]) {
                        continue;
                    }
                    const std::size_t j = i + step[a];
                    const std::size_t im = c[a] > 0 ? i - step[a] : i;
                    const std::size_t jp = c[a] + 2 < count[a] ? j + step[a] : j;
                    const float window[4] = {f.array[im], f.array[i], f.array[j], f.array[jp]};
                    const double ka = k.at(i);
                    const double kb = k.at(j);
                    const float kf = static_cast<float>((ka + kb) > 0.0 ? 2.0 * ka * kb / (ka + kb) : 0.0);
                    node.wide[3 * i + static_cast<std::size_t>(a)] = learned_face(net, window, kf, static_cast<float>(context.dt),
                        static_cast<float>(h), y, scratch);
                }
            }
        });
        node.buffer.resize(n);
        for_each_cell(l, [&](std::size_t i, int x, int y, int z) {
            float out = 0.0f;
            if (x > 0) out += static_cast<float>(node.wide[3 * (i - l.sx) + 0]);
            if (x + 1 < l.nx) out -= static_cast<float>(node.wide[3 * i + 0]);
            if (y > 0) out += static_cast<float>(node.wide[3 * (i - l.sy) + 1]);
            if (y + 1 < l.ny) out -= static_cast<float>(node.wide[3 * i + 1]);
            if (z > 0) out += static_cast<float>(node.wide[3 * (i - l.sz) + 2]);
            if (z + 1 < l.nz) out -= static_cast<float>(node.wide[3 * i + 2]);
            node.buffer[i] = out / static_cast<float>(h);
        });
        return settle(node.buffer);
    }
    case Op::MaxOf:
    case Op::MinOf:
    case Op::SumOf: {
        const Value a = evaluate(*node.args[0], context);
        if (a.is_scalar()) {
            return Value{.scalar = node.op == Op::SumOf ? a.scalar * static_cast<double>(n) : a.scalar};
        }
        double acc = node.op == Op::SumOf ? 0.0 : static_cast<double>(a.array[0]);
        for (std::size_t i = 0; i < n; ++i) {
            const double value = a.array[i];
            if (node.op == Op::SumOf) {
                acc += value;
            } else if (node.op == Op::MaxOf) {
                acc = std::max(acc, value);
            } else {
                acc = std::min(acc, value);
            }
        }
        return Value{.scalar = acc};
    }
    }
    return Value{};
}

} // namespace

std::optional<double> evaluate_scalar(Node& node, const EvalContext& context)
{
    const Value value = evaluate(node, context);
    if (!value.is_scalar()) {
        return std::nullopt;
    }
    return value.scalar;
}

// ---- law ---------------------------------------------------------------------------------------------

namespace {

// A deep copy that keeps the sharing: a subexpression shared by two equations of a term is still one
// node in the copy (`seen` maps originals to copies).
NodePtr clone_node(const NodePtr& node, std::map<const Node*, NodePtr>& seen)
{
    if (!node) {
        return nullptr;
    }
    if (const auto found = seen.find(node.get()); found != seen.end()) {
        return found->second;
    }
    auto copy = std::make_shared<Node>();
    copy->op = node->op;
    copy->number = node->number;
    copy->index = node->index;
    copy->iterations = node->iterations;
    copy->cost = node->cost;
    copy->shared = node->shared;
    for (const NodePtr& arg : node->args) {
        copy->args.push_back(clone_node(arg, seen));
    }
    seen.emplace(node.get(), copy);
    return copy;
}

// COMMON SUBEXPRESSIONS ARE COMPUTED ONCE. Within one term (or one derive), structurally identical
// subtrees become one node; the evaluator keeps a shared node's value for the epoch of one gather, in
// which every equation reads the same pre-state, so sharing changes no value — only the work.
NodePtr intern(const NodePtr& node, std::map<std::string, NodePtr>& table)
{
    for (NodePtr& arg : node->args) {
        arg = intern(arg, table);
    }
    std::ostringstream key;
    key << static_cast<int>(node->op) << '|' << std::hexfloat << node->number << '|' << node->index << '|' << node->iterations;
    for (const NodePtr& arg : node->args) {
        key << '|' << arg.get();
    }
    node->cost = 1;
    for (const NodePtr& arg : node->args) {
        node->cost += arg->cost;
    }
    const auto [where, inserted] = table.emplace(key.str(), node);
    if (!inserted) {
        // Leaves are free to read; only computed values are worth keeping.
        where->second->shared = !where->second->args.empty();
    }
    return where->second;
}

} // namespace

std::optional<LawSource> read_law_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::ostringstream text;
    text << in.rdbuf();
    return LawSource{.name = path, .text = text.str()};
}

Law clone_law(const Law& law)
{
    Law copy = law;
    for (Term& term : copy.terms) {
        std::map<const Node*, NodePtr> seen;
        for (Equation& equation : term.equations) {
            equation.rhs = clone_node(equation.rhs, seen);
        }
        term.max_dt = clone_node(term.max_dt, seen);
        for (Equation& equation : term.learned) {
            equation.rhs = clone_node(equation.rhs, seen);
        }
        term.learned_max_dt = clone_node(term.learned_max_dt, seen);
    }
    for (Derive& derive : copy.derives) {
        std::map<const Node*, NodePtr> seen;
        derive.rhs = clone_node(derive.rhs, seen);
    }
    for (CreatureDef& creature : copy.creatures) {
        std::map<const Node*, NodePtr> seen;
        for (NodePtr& rate : creature.rates) {
            rate = clone_node(rate, seen);
        }
        creature.seek = clone_node(creature.seek, seen);
        creature.hurt = clone_node(creature.hurt, seen);
        creature.breed = clone_node(creature.breed, seen);
        creature.plan = clone_node(creature.plan, seen);
        for (CreatureFlow& flow : creature.flows) {
            flow.rate = clone_node(flow.rate, seen);
        }
    }
    return copy;
}

int Law::field_id(std::string_view name) const
{
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (fields[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int Law::net_id(std::string_view name) const
{
    for (std::size_t i = 0; i < nets.size(); ++i) {
        if (nets[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int Law::creature_id(std::string_view name) const
{
    for (std::size_t i = 0; i < creatures.size(); ++i) {
        if (creatures[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int Law::param_id(std::string_view name) const
{
    for (std::size_t i = 0; i < param_names.size(); ++i) {
        if (param_names[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

namespace {

struct Line {
    std::string where;
    std::vector<Token> tokens;
};

bool is_reserved(std::string_view name)
{
    static const std::set<std::string, std::less<>> reserved = {
        "field", "param", "term", "derive", "creature", "show", "d", "conserves", "max_dt", "h", "pi", "inf",
        "init", "range", "unit", "static", "clamp", "net", "learned", "plan", "observe"};
    return reserved.contains(name) || functions().contains(name);
}

bool parse_bound(const std::vector<Token>& t, std::size_t& i, double& out)
{
    double sign = 1.0;
    if (t[i].kind == Token::Symbol && t[i].text == "-") {
        sign = -1.0;
        ++i;
    }
    if (t[i].kind == Token::Number) {
        out = sign * t[i].number;
        ++i;
        return true;
    }
    if (t[i].kind == Token::Name && t[i].text == "inf") {
        out = sign * 1e30;
        ++i;
        return true;
    }
    return false;
}

} // namespace

CompileResult compile_law(const std::vector<LawSource>& sources)
{
    CompileResult result;
    Law law;
    std::vector<Line> lines;

    // Tokenize everything first.
    for (const LawSource& source : sources) {
        std::istringstream stream(source.text);
        std::string raw;
        int number = 0;
        while (std::getline(stream, raw)) {
            ++number;
            const std::string where = source.name + ":" + std::to_string(number);
            std::string error;
            std::vector<Token> tokens = tokenize(raw, error);
            if (!error.empty()) {
                result.errors.push_back(where + ": " + error);
                continue;
            }
            if (tokens.size() > 1) {
                lines.push_back(Line{.where = where, .tokens = std::move(tokens)});
            }
        }
    }

    // Pass 1: declarations, so a term may use a field declared further down (or in another source),
    // and a creature's equations may use a state declared on a later line of its block.
    CreatureDef* declaring = nullptr;
    for (const Line& line : lines) {
        const std::vector<Token>& t = line.tokens;
        if (t[0].kind != Token::Name) {
            continue;
        }
        if (t[0].text == "field" || t[0].text == "param" || t[0].text == "term" || t[0].text == "derive" ||
            t[0].text == "show" || t[0].text == "net") {
            declaring = nullptr;
        }
        if (t[0].text == "net") {
            // net NAME in N hidden H... out M act A [half] [file PATH]
            if (t[1].kind != Token::Name || is_reserved(t[1].text) || law.net_id(t[1].text) >= 0) {
                result.errors.push_back(line.where + ": expected 'net NAME in N hidden H... out M act tanh|relu|silu [half] [file PATH]'");
                continue;
            }
            NetDecl net{.name = t[1].text, .where = line.where};
            net.file = "assets/nets/" + net.name + ".net";
            std::vector<int> hidden;
            int in = 0;
            int out = 0;
            bool bad = false;
            std::size_t i = 2;
            std::string section;
            while (t[i].kind != Token::End && !bad) {
                const Token& k = t[i++];
                if (k.kind == Token::Number) {
                    if (section == "in") {
                        in = static_cast<int>(k.number);
                    } else if (section == "hidden") {
                        hidden.push_back(static_cast<int>(k.number));
                    } else if (section == "out") {
                        out = static_cast<int>(k.number);
                    } else {
                        bad = true;
                    }
                    continue;
                }
                section = k.text;
                if (section == "act") {
                    bad = t[i].kind != Token::Name || !parse_activation(t[i].text, net.act);
                    ++i;
                } else if (section == "half") {
                    net.half = true;
                } else if (section == "file") {
                    // A path: everything up to the end or the next keyword, glued back together.
                    std::string path;
                    while (t[i].kind != Token::End && !(t[i].kind == Token::Name && (t[i].text == "half" || t[i].text == "act"))) {
                        path += t[i].kind == Token::Number ? std::to_string(static_cast<long long>(t[i].number)) : t[i].text;
                        ++i;
                    }
                    bad = path.empty();
                    net.file = path;
                } else if (section != "in" && section != "hidden" && section != "out") {
                    bad = true;
                }
            }
            if (bad || in < 1 || out < 1 || in > 64 || out > 64 || hidden.size() > 6 ||
                std::any_of(hidden.begin(), hidden.end(), [](int h) { return h < 1 || h > 128; })) {
                result.errors.push_back(line.where + ": malformed net (in 1..64, hidden up to 6 layers of 1..128, out 1..64)");
                continue;
            }
            net.sizes.push_back(in);
            net.sizes.insert(net.sizes.end(), hidden.begin(), hidden.end());
            net.sizes.push_back(out);
            init_net(net);
            std::string why;
            if (!load_net(net, why) && why.rfind("no weights at", 0) != 0) {
                result.errors.push_back(line.where + ": " + why);
                continue;
            }
            law.nets.push_back(std::move(net));
            continue;
        }
        if (t[0].text == "creature") {
            declaring = nullptr;
            if (t[1].kind != Token::Name || t[2].kind != Token::End || is_reserved(t[1].text)) {
                result.errors.push_back(line.where + ": expected 'creature NAME'");
                continue;
            }
            if (law.creature_id(t[1].text) >= 0) {
                result.errors.push_back(line.where + ": creature '" + t[1].text + "' declared twice");
                continue;
            }
            CreatureDef def{.name = t[1].text, .where = line.where};
            def.locals = {"hp", "eaten", "age"};
            def.init = {0.0, 0.0, 0.0};
            def.lo = {-1e30, -1e30, -1e30};
            def.hi = {1e30, 1e30, 1e30};
            def.rates.assign(3, nullptr);
            law.creatures.push_back(std::move(def));
            declaring = &law.creatures.back();
            continue;
        }
        if (t[0].text == "state" && declaring != nullptr) {
            if (t[1].kind != Token::Name || is_reserved(t[1].text)) {
                result.errors.push_back(line.where + ": expected 'state NAME init V range LO HI'");
                continue;
            }
            if (std::find(declaring->locals.begin(), declaring->locals.end(), t[1].text) != declaring->locals.end()) {
                result.errors.push_back(line.where + ": '" + t[1].text + "' is already a state of " + declaring->name);
                continue;
            }
            double init = 0.0;
            double lo = -1e30;
            double hi = 1e30;
            std::size_t i = 2;
            bool bad = false;
            while (t[i].kind != Token::End && !bad) {
                const std::string key = t[i].text;
                ++i;
                if (key == "init") {
                    bad = !parse_bound(t, i, init);
                } else if (key == "range") {
                    bad = !parse_bound(t, i, lo) || !parse_bound(t, i, hi) || lo > hi;
                } else {
                    bad = true;
                }
            }
            if (bad) {
                result.errors.push_back(line.where + ": malformed state (expected init V, range LO HI)");
                continue;
            }
            declaring->locals.push_back(t[1].text);
            declaring->init.push_back(init);
            declaring->lo.push_back(lo);
            declaring->hi.push_back(hi);
            declaring->rates.push_back(nullptr);
            continue;
        }
        if (t[0].text == "field") {
            if (t[1].kind != Token::Name || is_reserved(t[1].text)) {
                result.errors.push_back(line.where + ": 'field' needs a name that is not a keyword");
                continue;
            }
            if (law.field_id(t[1].text) >= 0) {
                result.errors.push_back(line.where + ": field '" + t[1].text + "' declared twice");
                continue;
            }
            FieldDecl decl{.name = t[1].text};
            std::size_t i = 2;
            bool bad = false;
            while (t[i].kind != Token::End && !bad) {
                const std::string& key = t[i].text;
                ++i;
                if (key == "init") {
                    bad = !parse_bound(t, i, decl.init);
                } else if (key == "range") {
                    bad = !parse_bound(t, i, decl.lo) || !parse_bound(t, i, decl.hi) || decl.lo > decl.hi;
                } else if (key == "unit") {
                    bad = t[i].kind != Token::Name;
                    decl.unit = t[i].text;
                    ++i;
                } else if (key == "static") {
                    decl.is_static = true;
                } else if (key == "clamp") {
                    decl.clamp = true;
                } else {
                    bad = true;
                }
            }
            if (bad) {
                result.errors.push_back(line.where + ": malformed field declaration (expected init N, range LO HI, "
                                                     "unit U, static, clamp)");
                continue;
            }
            law.fields.push_back(decl);
        } else if (t[0].text == "derive" && t[1].kind == Token::Name && law.field_id(t[1].text) < 0 &&
            !is_reserved(t[1].text)) {
            law.fields.push_back(FieldDecl{.name = t[1].text, .derived = true});
        }
    }

    // Pass 2: params (in order: a param may use earlier params), terms, derives and creatures.
    Term* current = nullptr;
    CreatureDef* creature = nullptr;
    std::set<int> derived_targets;
    for (const Line& line : lines) {
        const std::vector<Token>& t = line.tokens;
        auto fail = [&](const std::string& message) { result.errors.push_back(line.where + ": " + message); };
        if (t[0].kind != Token::Name) {
            fail("a line starts with a keyword (field, param, term, derive, creature, d, conserves, max_dt)");
            continue;
        }
        const std::string& keyword = t[0].text;
        if (keyword == "field" || keyword == "param" || keyword == "term" || keyword == "derive" || keyword == "show" ||
            keyword == "net") {
            creature = nullptr;
        }
        if (keyword == "net") {
            current = nullptr;
            continue; // declared in pass 1
        }
        if (keyword == "show") {
            current = nullptr;
            FieldVisual visual;
            visual.field = t[1].kind == Token::Name ? law.field_id(t[1].text) : -1;
            bool bad = visual.field < 0;
            std::size_t i = 2;
            bool has_colour = false;
            while (!bad && t[i].kind != Token::End) {
                const std::string key = t[i].text;
                ++i;
                if (key == "colour") {
                    double c[3] = {0.0, 0.0, 0.0};
                    bad = !parse_bound(t, i, c[0]) || !parse_bound(t, i, c[1]) || !parse_bound(t, i, c[2]);
                    for (int k = 0; k < 3; ++k) {
                        visual.colour[k] = static_cast<float>(c[k]);
                    }
                    has_colour = true;
                } else if (key == "above") {
                    bad = !parse_bound(t, i, visual.above);
                } else if (key == "full") {
                    bad = !parse_bound(t, i, visual.full);
                } else if (key == "alpha") {
                    bad = !parse_bound(t, i, visual.alpha);
                } else {
                    bad = true;
                }
            }
            if (bad || !has_colour || visual.full <= visual.above) {
                fail("expected 'show FIELD colour R G B above LO full HI [alpha A]' with a declared field and HI > LO");
                continue;
            }
            law.visuals.push_back(visual);
            continue;
        }
        if (keyword == "creature") {
            current = nullptr;
            const int id = t[1].kind == Token::Name ? law.creature_id(t[1].text) : -1;
            creature = id >= 0 ? &law.creatures[static_cast<std::size_t>(id)] : nullptr;
            continue;
        }
        if (creature != nullptr) {
            // Inside a creature block every line is about that creature.
            auto parse_expression = [&](std::size_t start) -> NodePtr {
                Parser parser(t, start, law, &creature->locals);
                NodePtr expr = parser.expression();
                if (!parser.ok() || !parser.at_end()) {
                    fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                    return nullptr;
                }
                return expr;
            };
            if (keyword == "state") {
                continue; // declared in pass 1
            }
            if (keyword == "radius" || keyword == "speed" || keyword == "hp" || keyword == "wander" ||
                keyword == "sense" || keyword == "heed") {
                // Key-value pairs, several to a line.
                for (std::size_t i = 0; t[i].kind != Token::End;) {
                    const std::string key = t[i].text;
                    double value = 0.0;
                    ++i;
                    if (!parse_bound(t, i, value)) {
                        fail("expected a number after '" + key + "'");
                        break;
                    }
                    if (key == "radius" && value > 0.05 && value < 2.0) {
                        creature->radius = value;
                    } else if (key == "speed" && value >= 0.0) {
                        creature->speed = value;
                    } else if (key == "hp" && value > 0.0) {
                        creature->hp = value;
                    } else if (key == "wander" && value >= 0.0 && value <= 1.0) {
                        creature->wander = value;
                    } else if (key == "sense" && value > 0.0 && value <= 16.0) {
                        creature->sense = value;
                    } else if (key == "heed" && value > 0.0) {
                        creature->heed = value;
                    } else {
                        fail("'" + key + "' is not a creature property here (radius 0.05..2, speed, hp > 0, wander 0..1, sense 0..16, heed > 0)");
                        break;
                    }
                }
                continue;
            }
            if (keyword == "d") {
                const auto found = t[1].kind == Token::Name
                    ? std::find(creature->locals.begin(), creature->locals.end(), t[1].text)
                    : creature->locals.end();
                if (found == creature->locals.end() || !(t[2].kind == Token::Symbol && t[2].text == "=")) {
                    fail("expected 'd STATE = expression' with a state of " + creature->name);
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(found - creature->locals.begin());
                if (index < static_cast<std::size_t>(kFirstState)) {
                    fail("'" + t[1].text + "' is kept by the engine; give the creature a state of its own");
                    continue;
                }
                creature->rates[index] = parse_expression(3);
                continue;
            }
            if (keyword == "colour") {
                double c[3] = {0.0, 0.0, 0.0};
                std::size_t i = 1;
                if (!parse_bound(t, i, c[0]) || !parse_bound(t, i, c[1]) || !parse_bound(t, i, c[2])) {
                    fail("expected 'colour R G B'");
                    continue;
                }
                for (int k = 0; k < 3; ++k) {
                    creature->colour[k] = static_cast<float>(c[k]);
                }
                continue;
            }
            if (keyword == "fights") {
                // fights WEAPON [skill N]: what it holds and how much of that it knows.
                creature->fights = t[1].text; // t[0] is the word itself
                if (t[1].kind != Token::End && t[2].kind != Token::End && t[2].text == "skill" && t[3].kind != Token::End) {
                    creature->skill = std::atoi(t[3].text.c_str());
                    creature->riled = t[4].kind != Token::End && t[4].text == "riled";
                } else if (t[1].kind != Token::End && t[2].kind != Token::End && t[2].text == "riled") {
                    creature->riled = true;
                }
                continue;
            }
            if (keyword == "prey") {
                // prey NAME...: what it goes for.
                creature->prey.clear();
                for (std::size_t k = 1; k < t.size() && t[k].kind != Token::End; ++k) {
                    creature->prey.push_back(t[k].text);
                }
                continue;
            }
            if (keyword == "skin") {
                creature->skin = t[1].text;
                continue;
            }
            if (keyword == "event") {
                const auto local = [&](const Token& name) {
                    const auto found = name.kind == Token::Name ? std::find(creature->locals.begin(), creature->locals.end(), name.text) : creature->locals.end();
                    return found == creature->locals.end() ? -1 : static_cast<int>(found - creature->locals.begin());
                };
                CreatureEvent e;
                e.name = t[1].text;
                e.state = local(t[2]);
                e.above = t[3].text == "above";
                std::size_t i = 4;
                if (e.state < 0 || (t[3].text != "above" && t[3].text != "below") || !parse_bound(t, i, e.line) || t[i].text != "weight") {
                    fail("expected 'event NAME STATE above|below X weight W [sets STATE V]'");
                    continue;
                }
                ++i;
                if (!parse_bound(t, i, e.weight)) {
                    fail("expected a number after 'weight'");
                    continue;
                }
                if (t[i].kind != Token::End) {
                    if (t[i].text != "sets" || t[i + 1].kind == Token::End) {
                        fail("expected 'sets STATE V' at the end of an event");
                        continue;
                    }
                    e.sets = local(t[i + 1]);
                    i += 2;
                    if (e.sets < 0 || !parse_bound(t, i, e.to)) {
                        fail("expected 'sets STATE V' with a state of " + creature->name);
                        continue;
                    }
                }
                creature->events.push_back(std::move(e));
                continue;
            }
            if (keyword == "plan") {
                if (!(t[1].kind == Token::Symbol && t[1].text == "=")) {
                    fail("expected 'plan = cost' (what its world model plans to keep low)");
                    continue;
                }
                creature->plan = parse_expression(2);
                continue;
            }
            if (keyword == "observe") {
                // observe FIELD... [horizon N] [samples N] [dt S]
                for (std::size_t i = 1; t[i].kind != Token::End; ++i) {
                    const std::string& key = t[i].text;
                    if ((key == "horizon" || key == "samples" || key == "dt") && t[i + 1].kind == Token::Number) {
                        const double v = t[i + 1].number;
                        if (key == "horizon" && v >= 1 && v <= 64) creature->horizon = static_cast<int>(v);
                        else if (key == "samples" && v >= 8 && v <= 1024) creature->samples = static_cast<int>(v);
                        else if (key == "dt" && v > 0.0 && v <= 2.0) creature->plan_dt = v;
                        else fail("observe: " + key + " out of range (horizon 1..64, samples 8..1024, dt 0..2)");
                        ++i;
                        continue;
                    }
                    const int field = t[i].kind == Token::Name ? law.field_id(key) : -1;
                    if (field < 0) {
                        fail("observe: '" + key + "' is not a field (or horizon N, samples N, dt S)");
                        break;
                    }
                    creature->observe.push_back(field);
                }
                continue;
            }
            if (keyword == "seek" || keyword == "hurt" || keyword == "breed") {
                if (!(t[1].kind == Token::Symbol && t[1].text == "=")) {
                    fail("expected '" + keyword + " = expression'");
                    continue;
                }
                (keyword == "seek" ? creature->seek : keyword == "hurt" ? creature->hurt : creature->breed) =
                    parse_expression(2);
                continue;
            }
            if (keyword == "eat" || keyword == "emit") {
                const int field = t[1].kind == Token::Name ? law.field_id(t[1].text) : -1;
                if (field < 0 || !(t[2].kind == Token::Symbol && t[2].text == "=")) {
                    fail("expected '" + keyword + " FIELD = rate' with a declared field");
                    continue;
                }
                const FieldDecl& decl = law.fields[static_cast<std::size_t>(field)];
                if (decl.derived || decl.is_static) {
                    fail("a creature cannot " + keyword + " '" + decl.name + "': it is " +
                        (decl.derived ? "derived" : "static"));
                    continue;
                }
                creature->flows.push_back(CreatureFlow{.field = field, .eat = keyword == "eat", .rate = parse_expression(3)});
                continue;
            }
            fail("unknown creature line '" + keyword + "' (radius/speed/hp/wander/sense/heed, state, d, seek, hurt, breed, eat, emit, colour)");
            continue;
        }
        if (keyword == "field") {
            current = nullptr;
            continue;
        }
        if (keyword == "param") {
            current = nullptr;
            if (t[1].kind != Token::Name || is_reserved(t[1].text) || !(t[2].kind == Token::Symbol && t[2].text == "=")) {
                fail("expected 'param NAME = expression'");
                continue;
            }
            if (law.param_id(t[1].text) >= 0 || law.field_id(t[1].text) >= 0) {
                fail("'" + t[1].text + "' is already a param or a field");
                continue;
            }
            Parser parser(t, 3, law);
            NodePtr expr = parser.expression();
            if (!parser.ok() || !parser.at_end()) {
                fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                continue;
            }
            if (!always_scalar(*expr)) {
                fail("a param cannot read a field");
                continue;
            }
            // Params are evaluated now, against an empty one-cell lattice (h = 1 is not meaningful
            // here, so a param that uses h is refused).
            std::set<int> reads;
            collect_reads(*expr, reads);
            FieldSet dummy(GridShape{1, 1, 1, 1.0});
            EvalContext context{.fields = &dummy, .params = &law.param_values};
            const std::optional<double> value = evaluate_scalar(*expr, context);
            law.param_names.push_back(t[1].text);
            law.param_values.push_back(value.value_or(0.0));
            continue;
        }
        if (keyword == "term") {
            if (t[1].kind != Token::Name || t[2].kind != Token::End) {
                fail("expected 'term NAME'");
                current = nullptr;
                continue;
            }
            for (const Term& existing : law.terms) {
                if (existing.name == t[1].text) {
                    fail("term '" + t[1].text + "' declared twice");
                }
            }
            law.terms.push_back(Term{.name = t[1].text, .where = line.where});
            current = &law.terms.back();
            continue;
        }
        if (keyword == "derive") {
            current = nullptr;
            const int target = t[1].kind == Token::Name ? law.field_id(t[1].text) : -1;
            const bool accumulate = t[2].kind == Token::Symbol && t[2].text == "+" &&
                t[3].kind == Token::Symbol && t[3].text == "=";
            if (target < 0 || !(accumulate || (t[2].kind == Token::Symbol && t[2].text == "="))) {
                fail("expected 'derive FIELD = expression' or 'derive FIELD += expression'");
                continue;
            }
            if (!accumulate && derived_targets.contains(target)) {
                fail("field '" + t[1].text + "' is derived twice with '=' (use '+=' to add to it)");
                continue;
            }
            Parser parser(t, accumulate ? 4 : 3, law);
            NodePtr expr = parser.expression();
            if (!parser.ok() || !parser.at_end()) {
                fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                continue;
            }
            law.fields[static_cast<std::size_t>(target)].derived = true;
            if (!accumulate) {
                derived_targets.insert(target);
            }
            std::set<int> reads;
            collect_reads(*expr, reads);
            law.derives.push_back(Derive{.target = target, .accumulate = accumulate, .where = line.where,
                .rhs = expr, .reads = {reads.begin(), reads.end()}});
            continue;
        }
        if (keyword == "learned") {
            if (current == nullptr) {
                fail("'learned' outside a term");
                continue;
            }
            if (t[1].kind == Token::Name && t[1].text == "d") {
                const int target = t[2].kind == Token::Name ? law.field_id(t[2].text) : -1;
                if (target < 0 || !(t[3].kind == Token::Symbol && t[3].text == "=")) {
                    fail("expected 'learned d FIELD = expression'");
                    continue;
                }
                Parser parser(t, 4, law);
                NodePtr expr = parser.expression();
                if (!parser.ok() || !parser.at_end()) {
                    fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                    continue;
                }
                current->learned.push_back(Equation{.target = target, .rhs = expr});
            } else if (t[1].kind == Token::Name && t[1].text == "max_dt" && t[2].kind == Token::Symbol && t[2].text == "=") {
                Parser parser(t, 3, law);
                NodePtr expr = parser.expression();
                if (!parser.ok() || !parser.at_end() || !always_scalar(*expr)) {
                    fail(parser.ok() ? "learned max_dt must be a number" : parser.error());
                    continue;
                }
                current->learned_max_dt = expr;
            } else {
                fail("expected 'learned d FIELD = expression' or 'learned max_dt = expression'");
            }
            continue;
        }
        if (keyword == "d" || keyword == "conserves" || keyword == "max_dt") {
            if (current == nullptr) {
                fail("'" + keyword + "' outside a term");
                continue;
            }
            if (keyword == "d") {
                const int target = t[1].kind == Token::Name ? law.field_id(t[1].text) : -1;
                if (target < 0 || !(t[2].kind == Token::Symbol && t[2].text == "=")) {
                    fail("expected 'd FIELD = expression' with a declared field");
                    continue;
                }
                Parser parser(t, 3, law);
                NodePtr expr = parser.expression();
                if (!parser.ok() || !parser.at_end()) {
                    fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                    continue;
                }
                current->equations.push_back(Equation{.target = target, .rhs = expr});
            } else if (keyword == "conserves") {
                // `a + b + c` joins fields into one total; a bare name after another starts a group.
                bool join = false;
                for (std::size_t i = 1; t[i].kind != Token::End; ++i) {
                    if (t[i].kind == Token::Symbol && t[i].text == "+") {
                        join = true;
                        continue;
                    }
                    const int field = t[i].kind == Token::Name ? law.field_id(t[i].text) : -1;
                    if (field < 0) {
                        fail("conserves: '" + t[i].text + "' is not a field");
                        continue;
                    }
                    if (join && !current->conserves.empty()) {
                        current->conserves.back().push_back(field);
                    } else {
                        current->conserves.push_back({field});
                    }
                    join = false;
                }
            } else {
                if (!(t[1].kind == Token::Symbol && t[1].text == "=")) {
                    fail("expected 'max_dt = expression'");
                    continue;
                }
                Parser parser(t, 2, law);
                NodePtr expr = parser.expression();
                if (!parser.ok() || !parser.at_end()) {
                    fail(parser.ok() ? "trailing tokens after the expression" : parser.error());
                    continue;
                }
                if (!always_scalar(*expr)) {
                    fail("max_dt must be a number: use maxof/minof/sumof to reduce a field");
                    continue;
                }
                current->max_dt = expr;
            }
            continue;
        }
        if (keyword == "derive" || keyword == "field") {
            continue;
        }
        fail("unknown keyword '" + keyword + "'");
    }

    // Validation that needs the whole law: what each term reads and writes, and contracts that would
    // otherwise pass vacuously.
    for (Term& term : law.terms) {
        std::map<std::string, NodePtr> table;
        for (Equation& equation : term.equations) {
            equation.rhs = intern(equation.rhs, table);
        }
        std::map<std::string, NodePtr> learned_table;
        for (Equation& equation : term.learned) {
            equation.rhs = intern(equation.rhs, learned_table);
        }
        std::set<int> reads;
        std::set<int> writes;
        for (const Equation& equation : term.equations) {
            collect_reads(*equation.rhs, reads);
            writes.insert(equation.target);
            const FieldDecl& decl = law.fields[static_cast<std::size_t>(equation.target)];
            if (decl.is_static) {
                result.errors.push_back(term.where + ": term '" + term.name + "' writes static field '" + decl.name + "'");
            }
            if (decl.derived) {
                result.errors.push_back(
                    term.where + ": term '" + term.name + "' writes derived field '" + decl.name + "'");
            }
        }
        if (term.equations.empty()) {
            result.errors.push_back(term.where + ": term '" + term.name + "' has no equations");
        }
        for (const Equation& equation : term.learned) {
            collect_reads(*equation.rhs, reads);
            if (!writes.contains(equation.target)) {
                // The classical equations are the fallback: they must cover whatever the learned ones write.
                result.errors.push_back(term.where + ": term '" + term.name + "' has a learned equation for '" +
                    law.fields[static_cast<std::size_t>(equation.target)].name + "', which its own equations do not write");
            }
        }
        for (const std::vector<int>& group : term.conserves) {
            bool touched = false;
            std::string names;
            for (const int conserved : group) {
                touched = touched || writes.contains(conserved);
                names += (names.empty() ? "" : " + ") + law.fields[static_cast<std::size_t>(conserved)].name;
            }
            if (!touched) {
                // A conservation promise about fields the term never touches is true and worthless.
                result.errors.push_back(term.where + ": term '" + term.name + "' conserves '" + names +
                    "' but never writes it");
            }
        }
        term.reads.assign(reads.begin(), reads.end());
        term.writes.assign(writes.begin(), writes.end());
    }

    // Group derives per field (in order of first appearance), the '=' before the '+='s.
    {
        std::vector<int> order;
        for (const Derive& derive : law.derives) {
            if (std::find(order.begin(), order.end(), derive.target) == order.end()) {
                order.push_back(derive.target);
            }
        }
        std::vector<Derive> grouped;
        for (const int target : order) {
            for (int pass = 0; pass < 2; ++pass) {
                for (const Derive& derive : law.derives) {
                    if (derive.target == target && derive.accumulate == (pass == 1)) {
                        grouped.push_back(derive);
                    }
                }
            }
        }
        law.derives = std::move(grouped);
    }

    for (Derive& derive : law.derives) {
        std::map<std::string, NodePtr> table;
        derive.rhs = intern(derive.rhs, table);
    }

    // A creature reads the fields where it stands: whole-lattice ops have no meaning at a point.
    for (CreatureDef& def : law.creatures) {
        std::set<int> reads;
        std::vector<const Node*> roots;
        for (const NodePtr& rate : def.rates) {
            roots.push_back(rate.get());
        }
        roots.push_back(def.seek.get());
        roots.push_back(def.hurt.get());
        roots.push_back(def.breed.get());
        roots.push_back(def.plan.get());
        for (const CreatureFlow& flow : def.flows) {
            roots.push_back(flow.rate.get());
        }
        std::string lattice_op;
        std::function<void(const Node&)> check = [&](const Node& node) {
            switch (node.op) {
            case Op::Lap: case Op::Gx: case Op::Gy: case Op::Gz: case Op::Diffuse: case Op::Adv:
            case Op::Distance: case Op::Poisson: case Op::MaxOf: case Op::MinOf: case Op::SumOf:
            case Op::Fall: case Op::Spread: case Op::Topple: case Op::Shift: case Op::Flux: case Op::LDiffuse:
                lattice_op = "a lattice op";
                break;
            default:
                break;
            }
            for (const NodePtr& arg : node.args) {
                check(*arg);
            }
        };
        for (const Node* root : roots) {
            if (root != nullptr) {
                check(*root);
                collect_reads(*root, reads);
            }
        }
        if (!lattice_op.empty()) {
            result.errors.push_back(def.where + ": creature '" + def.name +
                "' uses " + lattice_op + " (lap, gx, diffuse, distance, shift, ...): a creature reads fields where it "
                "stands; derive a field with the op and read that");
        }
        def.reads.assign(reads.begin(), reads.end());
    }

    if (result.errors.empty()) {
        result.law = std::move(law);
    }
    return result;
}

// ---- at a point (creatures) --------------------------------------------------------------------------

double evaluate_at(const Node& node, const PointContext& c)
{
    auto arg = [&](std::size_t i) { return evaluate_at(*node.args[i], c); };
    switch (node.op) {
    case Op::Const:
        return node.number;
    case Op::Param:
        return (*c.params)[static_cast<std::size_t>(node.index)];
    case Op::Cell:
        return c.fields->shape().cell;
    case Op::Rim:
        return 0.0; // a creature is never on the rim as far as it can tell
    case Op::Local:
        return c.locals != nullptr && static_cast<std::size_t>(node.index) < c.locals->size()
            ? (*c.locals)[static_cast<std::size_t>(node.index)]
            : 0.0;
    case Op::Field:
        return static_cast<double>(c.fields->sample(node.index, c.x, c.y, c.z));
    case Op::Neg:
        return -arg(0);
    case Op::Abs:
        return std::abs(arg(0));
    case Op::Sqrt:
        return std::sqrt(std::max(0.0, arg(0)));
    case Op::Exp:
        return std::exp(std::min(arg(0), 80.0));
    case Op::Log:
        return std::log(std::max(arg(0), 1e-30));
    case Op::Step:
        return arg(0) > 0.0 ? 1.0 : 0.0;
    case Op::Pos:
        return std::max(arg(0), 0.0);
    case Op::Add:
        return arg(0) + arg(1);
    case Op::Sub:
        return arg(0) - arg(1);
    case Op::Mul:
        return arg(0) * arg(1);
    case Op::Div: {
        const double b = arg(1);
        return b == 0.0 ? 0.0 : arg(0) / b;
    }
    case Op::Pow:
        return std::pow(arg(0), arg(1));
    case Op::Min:
        return std::min(arg(0), arg(1));
    case Op::Max:
        return std::max(arg(0), arg(1));
    case Op::Clamp: {
        const double lo = arg(1);
        return std::clamp(arg(0), lo, std::max(lo, arg(2)));
    }
    case Op::Mlp: {
        if (c.nets == nullptr) {
            return 0.0;
        }
        const NetDecl& net = (*c.nets)[static_cast<std::size_t>(node.index)];
        std::vector<float> x(node.args.size());
        for (std::size_t k = 0; k < node.args.size(); ++k) {
            x[k] = static_cast<float>(arg(k));
        }
        std::vector<float> y(static_cast<std::size_t>(net.outputs()));
        std::vector<float> scratch(2 * static_cast<std::size_t>(net.widest()));
        net_forward(net, x.data(), y.data(), scratch.data());
        return y[0];
    }
    default:
        return 0.0; // lattice ops are refused in creature expressions at compile time
    }
}

// ---- the law as mathematics --------------------------------------------------------------------------

namespace {

int precedence(const Node& node)
{
    switch (node.op) {
    case Op::Add:
    case Op::Sub:
        return 1;
    case Op::Mul:
    case Op::Div:
        return 2;
    case Op::Neg:
        return 3;
    case Op::Pow:
        return 4;
    case Op::Const:
        return node.number < 0.0 ? 3 : 5;
    default:
        return 5;
    }
}

std::string number_text(double v)
{
    if (v == 3.14159265358979323846) {
        return "π";
    }
    std::ostringstream out;
    out << std::setprecision(4) << v;
    std::string text = out.str();
    // 1e-05 reads better as 1·10⁻⁵.
    if (const std::size_t e = text.find('e'); e != std::string::npos) {
        static const char* const superscript[] = {"⁰", "¹", "²", "³", "⁴", "⁵",
            "⁶", "⁷", "⁸", "⁹"};
        std::string mantissa = text.substr(0, e);
        std::string exponent = text.substr(e + 1);
        std::string sup;
        for (const char ch : exponent) {
            if (ch == '-') {
                sup += "⁻";
            } else if (ch >= '0' && ch <= '9') {
                if (!(sup.empty() || sup == "⁻") || ch != '0') {
                    sup += superscript[ch - '0'];
                }
            }
        }
        text = (mantissa == "1" ? "" : mantissa + "·") + "10" + sup;
    }
    return text;
}

class MathPrinter {
public:
    MathPrinter(const Law& law, const std::vector<std::string>* locals) : law_(law), locals_(locals) {}

    std::string print(const Node& node) const
    {
        auto a = [&](std::size_t i) { return print(*node.args[i]); };
        auto wrap = [&](std::size_t i, int min_precedence) {
            const Node& child = *node.args[i];
            const std::string text = print(child);
            return precedence(child) < min_precedence ? "(" + text + ")" : text;
        };
        auto field_name = [&](const Node& n) -> std::string {
            if (n.op == Op::Field) {
                return law_.fields[static_cast<std::size_t>(n.index)].name;
            }
            return "(" + print(n) + ")";
        };
        switch (node.op) {
        case Op::Const:
            return number_text(node.number);
        case Op::Param:
            return law_.param_names[static_cast<std::size_t>(node.index)];
        case Op::Cell:
            return "h";
        case Op::Rim:
            return "rim";
        case Op::Local:
            return locals_ != nullptr && static_cast<std::size_t>(node.index) < locals_->size()
                ? (*locals_)[static_cast<std::size_t>(node.index)]
                : "?";
        case Op::Field:
            return law_.fields[static_cast<std::size_t>(node.index)].name;
        case Op::Neg:
            return "−" + wrap(0, 3);
        case Op::Add: {
            // a + (−b) prints as a − b.
            const Node& right = *node.args[1];
            if (right.op == Op::Neg) {
                return a(0) + " − " + (precedence(*right.args[0]) < 2 ? "(" + print(*right.args[0]) + ")"
                                                                          : print(*right.args[0]));
            }
            return a(0) + " + " + a(1);
        }
        case Op::Sub:
            return a(0) + " − " + wrap(1, 2);
        case Op::Mul:
            return wrap(0, 2) + "·" + wrap(1, 3);
        case Op::Div:
            return wrap(0, 2) + "/" + wrap(1, 3);
        case Op::Pow: {
            const Node& e = *node.args[1];
            if (e.op == Op::Const && e.number == 2.0) {
                return wrap(0, 5) + "²";
            }
            if (e.op == Op::Const && e.number == 3.0) {
                return wrap(0, 5) + "³";
            }
            return wrap(0, 5) + "^" + wrap(1, 5);
        }
        case Op::Abs:
            return "|" + a(0) + "|";
        case Op::Sqrt:
            return "√(" + a(0) + ")";
        case Op::Exp:
            return "exp(" + a(0) + ")";
        case Op::Log:
            return "ln(" + a(0) + ")";
        case Op::Step:
            return "H(" + a(0) + ")";
        case Op::Pos:
            return "[" + a(0) + "]₊";
        case Op::Min:
            return "min(" + a(0) + ", " + a(1) + ")";
        case Op::Max:
            return "max(" + a(0) + ", " + a(1) + ")";
        case Op::Clamp:
            return "clamp(" + a(0) + ", " + a(1) + ", " + a(2) + ")";
        case Op::Lap:
            return "∇²" + field_name(*node.args[0]);
        case Op::Gx:
            return "∂" + field_name(*node.args[0]) + "/∂x";
        case Op::Gy:
            return "∂" + field_name(*node.args[0]) + "/∂y";
        case Op::Gz:
            return "∂" + field_name(*node.args[0]) + "/∂z";
        case Op::Diffuse:
            return "∇·(" + wrap(0, 3) + "∇" + field_name(*node.args[1]) + ")";
        case Op::Adv:
            return "−(" + a(1) + ", " + a(2) + ", " + a(3) + ")·∇" + field_name(*node.args[0]);
        case Op::Distance:
            return "dist(" + a(0) + " ≥ ½)";
        case Op::Poisson:
            return "∇⁻²(" + a(0) + ")";
        case Op::MaxOf:
            return "maxₓ " + wrap(0, 5);
        case Op::MinOf:
            return "minₓ " + wrap(0, 5);
        case Op::SumOf:
            return "Σₓ " + wrap(0, 5);
        case Op::Fall:
            return "fall↓(" + a(0) + " | " + a(1) + ", " + a(2) + ")";
        case Op::Spread:
            return "spread↔(" + a(0) + " | " + a(1) + ", " + a(2) + ")";
        case Op::Topple:
            return "topple↘(" + a(0) + " | " + a(1) + ", " + a(2) + ")";
        case Op::Mlp: {
            std::string text = law_.nets[static_cast<std::size_t>(node.index)].name + "(";
            for (std::size_t i = 0; i < node.args.size(); ++i) {
                text += (i ? ", " : "") + a(i);
            }
            return text + ")";
        }
        case Op::Flux: {
            std::string text = "−∇·Φ_" + law_.nets[static_cast<std::size_t>(node.index)].name + "(" + field_name(*node.args[0]);
            for (std::size_t i = 1; i + 2 < node.args.size(); ++i) {
                text += ", " + a(i);
            }
            return text + ")";
        }
        case Op::LDiffuse:
            return "∇·(" + wrap(0, 3) + "∇̃_" + law_.nets[static_cast<std::size_t>(node.index)].name + " " + field_name(*node.args[1]) + ")";
        case Op::Shift: {
            auto offset = [&](const char* axis, double d) {
                if (d == 0.0) {
                    return std::string(axis);
                }
                return std::string(axis) + (d > 0.0 ? "+" : "−") + number_text(std::abs(d));
            };
            return field_name(*node.args[0]) + "[" + offset("x", node.args[1]->number) + ", " +
                offset("y", node.args[2]->number) + ", " + offset("z", node.args[3]->number) + "]";
        }
        }
        return "?";
    }

private:
    const Law& law_;
    const std::vector<std::string>* locals_;
};

// A summand of a sum: a top-level Add splits; a top-level Sub splits with the sign carried over.
void summands(const Node& node, bool negate, std::vector<std::pair<bool, const Node*>>& out)
{
    if (node.op == Op::Add) {
        summands(*node.args[0], negate, out);
        summands(*node.args[1], negate, out);
        return;
    }
    if (node.op == Op::Sub) {
        summands(*node.args[0], negate, out);
        summands(*node.args[1], !negate, out);
        return;
    }
    if (node.op == Op::Neg) {
        summands(*node.args[0], !negate, out);
        return;
    }
    out.emplace_back(negate, &node);
}

std::string signed_part(const MathPrinter& printer, const Node& node, bool first)
{
    std::vector<std::pair<bool, const Node*>> parts;
    summands(node, false, parts);
    std::string text;
    static const std::string minus = "−";
    for (const auto& [was_negative, part] : parts) {
        std::string body = printer.print(*part);
        // A product that opens with a minus is a negative summand: - a*b, never + -a*b.
        bool negative = was_negative;
        if (body.rfind(minus, 0) == 0) {
            body.erase(0, minus.size());
            negative = !negative;
        }
        if (text.empty()) {
            text = first ? (negative ? minus : std::string()) + body : (negative ? minus + " " : std::string("+ ")) + body;
        } else {
            text += (negative ? " " + minus + " " : std::string(" + ")) + body;
        }
    }
    return text;
}

} // namespace

std::string to_math(const Node& node, const Law& law, const std::vector<std::string>* locals)
{
    return MathPrinter(law, locals).print(node);
}

std::vector<LawEquation> law_equations(const Law& law)
{
    std::vector<LawEquation> out;
    const MathPrinter printer(law, nullptr);
    // ∂F/∂t: everything every term adds to F, in the order the solver applies it.
    for (std::size_t f = 0; f < law.fields.size(); ++f) {
        if (law.fields[f].derived) {
            continue;
        }
        LawEquation eq{.kind = LawEquation::Kind::Rate, .lhs = "∂" + law.fields[f].name + "/∂t"};
        for (const Term& term : law.terms) {
            for (const Equation& equation : term.equations) {
                if (equation.target == static_cast<int>(f)) {
                    eq.parts.push_back(signed_part(printer, *equation.rhs, eq.parts.empty()));
                    eq.source.push_back(term.name);
                }
            }
        }
        if (!eq.parts.empty()) {
            out.push_back(std::move(eq));
        }
    }
    for (const Derive& derive : law.derives) {
        const std::string& name = law.fields[static_cast<std::size_t>(derive.target)].name;
        if (derive.accumulate && !out.empty() && out.back().kind == LawEquation::Kind::Derived &&
            out.back().lhs == name) {
            out.back().parts.push_back(signed_part(printer, *derive.rhs, false));
            out.back().source.push_back(derive.where);
            continue;
        }
        out.push_back(LawEquation{.kind = LawEquation::Kind::Derived, .lhs = name,
            .parts = {signed_part(printer, *derive.rhs, true)}, .source = {derive.where}});
    }
    for (const CreatureDef& def : law.creatures) {
        const MathPrinter local(law, &def.locals);
        for (std::size_t i = static_cast<std::size_t>(kFirstState); i < def.locals.size(); ++i) {
            if (def.rates[i]) {
                out.push_back(LawEquation{.kind = LawEquation::Kind::Creature,
                    .lhs = def.name + ": d" + def.locals[i] + "/dt", .parts = {signed_part(local, *def.rates[i], true)},
                    .source = {def.name}});
            }
        }
        if (def.seek) {
            out.push_back(LawEquation{.kind = LawEquation::Kind::Creature,
                .lhs = def.name + ": dx/dt",
                .parts = {number_text(def.speed) + "·∇s/(|∇s| + " + number_text(def.heed) + "),   s = " +
                    signed_part(local, *def.seek, true)},
                .source = {def.name}});
        }
        for (const CreatureFlow& flow : def.flows) {
            const std::string& field = law.fields[static_cast<std::size_t>(flow.field)].name;
            out.push_back(LawEquation{.kind = LawEquation::Kind::Creature,
                .lhs = def.name + ": ∂" + field + "/∂t at its cell",
                .parts = {(flow.eat ? "−" : "+") + std::string("(") + signed_part(local, *flow.rate, true) + ")"},
                .source = {def.name}});
        }
        if (def.breed) {
            out.push_back(LawEquation{.kind = LawEquation::Kind::Creature, .lhs = def.name + ": newborns/s",
                .parts = {signed_part(local, *def.breed, true)}, .source = {def.name}});
        }
        if (def.hurt) {
            out.push_back(LawEquation{.kind = LawEquation::Kind::Creature, .lhs = def.name + ": dhp/dt",
                .parts = {"−harm − (" + signed_part(local, *def.hurt, true) + ")"}, .source = {def.name}});
        }
    }
    return out;
}

} // namespace fieldengine::field

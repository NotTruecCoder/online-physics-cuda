#include "field/nn/Net.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fieldengine::field {

const char* to_string(Activation a)
{
    switch (a) {
    case Activation::Tanh:
        return "tanh";
    case Activation::Relu:
        return "relu";
    case Activation::Silu:
        return "silu";
    case Activation::Linear:
        return "linear";
    }
    return "?";
}

bool parse_activation(const std::string& text, Activation& out)
{
    for (const Activation a : {Activation::Tanh, Activation::Relu, Activation::Silu, Activation::Linear}) {
        if (text == to_string(a)) {
            out = a;
            return true;
        }
    }
    return false;
}

int NetDecl::widest() const
{
    int most = 1;
    for (const int s : sizes) {
        most = std::max(most, s);
    }
    return most;
}

std::size_t NetDecl::weight_offset(int layer) const
{
    std::size_t at = 0;
    for (int l = 0; l < layer; ++l) {
        at += static_cast<std::size_t>(sizes[static_cast<std::size_t>(l)] + 1) * static_cast<std::size_t>(sizes[static_cast<std::size_t>(l) + 1]);
    }
    return at;
}

std::size_t NetDecl::parameter_count() const
{
    return weight_offset(layers());
}

float activate(Activation a, float pre)
{
    switch (a) {
    case Activation::Tanh:
        return std::tanh(pre);
    case Activation::Relu:
        return pre > 0.0f ? pre : 0.0f;
    case Activation::Silu:
        return pre / (1.0f + std::exp(-pre));
    case Activation::Linear:
        return pre;
    }
    return pre;
}

float activate_grad(Activation a, float pre)
{
    switch (a) {
    case Activation::Tanh: {
        const float t = std::tanh(pre);
        return 1.0f - t * t;
    }
    case Activation::Relu:
        return pre > 0.0f ? 1.0f : 0.0f;
    case Activation::Silu: {
        const float s = 1.0f / (1.0f + std::exp(-pre));
        return s * (1.0f + pre * (1.0f - s));
    }
    case Activation::Linear:
        return 1.0f;
    }
    return 1.0f;
}

namespace {

void normalise(const NetDecl& net, const float* in, float* x)
{
    const int n = net.inputs();
    for (int i = 0; i < n; ++i) {
        const float shift = i < static_cast<int>(net.in_shift.size()) ? net.in_shift[static_cast<std::size_t>(i)] : 0.0f;
        const float scale = i < static_cast<int>(net.in_scale.size()) ? net.in_scale[static_cast<std::size_t>(i)] : 1.0f;
        x[i] = (in[i] - shift) * scale;
    }
}

// One layer: y[o] = b[o] + sum_i W[o][i] x[i], i in order.
void dense(const float* w, int in, int out, const float* x, float* y)
{
    const float* b = w + static_cast<std::size_t>(in) * static_cast<std::size_t>(out);
    for (int o = 0; o < out; ++o) {
        const float* row = w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in);
        float acc = b[o];
        for (int i = 0; i < in; ++i) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

std::uint64_t mix(std::uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

} // namespace

void net_forward(const NetDecl& net, const float* in, float* out, float* scratch)
{
    const int width = net.widest();
    float* a = scratch;
    float* b = scratch + width;
    normalise(net, in, a);
    const int layers = net.layers();
    for (int l = 0; l < layers; ++l) {
        const int ni = net.sizes[static_cast<std::size_t>(l)];
        const int no = net.sizes[static_cast<std::size_t>(l) + 1];
        dense(net.weights.data() + net.weight_offset(l), ni, no, a, b);
        if (l + 1 < layers) {
            for (int o = 0; o < no; ++o) {
                b[o] = activate(net.act, b[o]);
            }
        }
        std::swap(a, b);
    }
    for (int o = 0; o < net.outputs(); ++o) {
        out[o] = a[o] * net.out_scale;
    }
}

void net_forward_taped(const NetDecl& net, const float* in, float* out, float* tape)
{
    std::vector<float> a(static_cast<std::size_t>(net.widest()));
    std::vector<float> next(a.size());
    normalise(net, in, a.data());
    const int layers = net.layers();
    float* pre = tape;
    for (int l = 0; l < layers; ++l) {
        const int ni = net.sizes[static_cast<std::size_t>(l)];
        const int no = net.sizes[static_cast<std::size_t>(l) + 1];
        dense(net.weights.data() + net.weight_offset(l), ni, no, a.data(), pre);
        if (l + 1 < layers) {
            for (int o = 0; o < no; ++o) {
                next[static_cast<std::size_t>(o)] = activate(net.act, pre[o]);
            }
            std::swap(a, next);
        } else {
            for (int o = 0; o < no; ++o) {
                out[o] = pre[o] * net.out_scale;
            }
        }
        pre += no;
    }
}

void net_backward(const NetDecl& net, const float* in, const float* tape, const float* dout, float* grad, float* din,
    float* scratch)
{
    (void)scratch;
    const int layers = net.layers();
    // Where each layer's pre-activations sit on the tape.
    std::vector<std::size_t> at(static_cast<std::size_t>(layers));
    std::size_t sum = 0;
    for (int l = 0; l < layers; ++l) {
        at[static_cast<std::size_t>(l)] = sum;
        sum += static_cast<std::size_t>(net.sizes[static_cast<std::size_t>(l) + 1]);
    }
    std::vector<float> x(static_cast<std::size_t>(net.inputs()));
    normalise(net, in, x.data());
    std::vector<float> delta(static_cast<std::size_t>(net.widest()));
    std::vector<float> below(static_cast<std::size_t>(net.widest()));
    for (int o = 0; o < net.outputs(); ++o) {
        delta[static_cast<std::size_t>(o)] = dout[o] * net.out_scale;
    }
    for (int l = layers - 1; l >= 0; --l) {
        const int ni = net.sizes[static_cast<std::size_t>(l)];
        const int no = net.sizes[static_cast<std::size_t>(l) + 1];
        const float* w = net.weights.data() + net.weight_offset(l);
        float* gw = grad + net.weight_offset(l);
        float* gb = gw + static_cast<std::size_t>(ni) * static_cast<std::size_t>(no);
        // The input of this layer: the activation of the one below, or the normalised input.
        std::vector<float> input(static_cast<std::size_t>(ni));
        if (l == 0) {
            input.assign(x.begin(), x.end());
        } else {
            const float* pre = tape + at[static_cast<std::size_t>(l) - 1];
            for (int i = 0; i < ni; ++i) {
                input[static_cast<std::size_t>(i)] = activate(net.act, pre[i]);
            }
        }
        for (int o = 0; o < no; ++o) {
            const float d = delta[static_cast<std::size_t>(o)];
            gb[o] += d;
            for (int i = 0; i < ni; ++i) {
                gw[static_cast<std::size_t>(o) * static_cast<std::size_t>(ni) + static_cast<std::size_t>(i)] += d * input[static_cast<std::size_t>(i)];
            }
        }
        for (int i = 0; i < ni; ++i) {
            float acc = 0.0f;
            for (int o = 0; o < no; ++o) {
                acc += w[static_cast<std::size_t>(o) * static_cast<std::size_t>(ni) + static_cast<std::size_t>(i)] * delta[static_cast<std::size_t>(o)];
            }
            below[static_cast<std::size_t>(i)] = acc;
        }
        if (l > 0) {
            const float* pre = tape + at[static_cast<std::size_t>(l) - 1];
            for (int i = 0; i < ni; ++i) {
                delta[static_cast<std::size_t>(i)] = below[static_cast<std::size_t>(i)] * activate_grad(net.act, pre[i]);
            }
        } else if (din != nullptr) {
            for (int i = 0; i < ni; ++i) {
                const float scale = i < static_cast<int>(net.in_scale.size()) ? net.in_scale[static_cast<std::size_t>(i)] : 1.0f;
                din[i] = below[static_cast<std::size_t>(i)] * scale;
            }
        }
    }
}

void init_net(NetDecl& net, std::uint64_t seed)
{
    net.weights.assign(net.parameter_count(), 0.0f);
    std::uint64_t h = 1469598103934665603ull ^ seed;
    for (const char c : net.name) {
        h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
    const int layers = net.layers();
    for (int l = 0; l + 1 < layers; ++l) {
        const int ni = net.sizes[static_cast<std::size_t>(l)];
        const int no = net.sizes[static_cast<std::size_t>(l) + 1];
        const float limit = std::sqrt(6.0f / static_cast<float>(ni + no));
        float* w = net.weights.data() + net.weight_offset(l);
        for (int k = 0; k < ni * no; ++k) {
            const std::uint64_t r = mix(h + static_cast<std::uint64_t>(l) * 0x9e3779b97f4a7c15ull + static_cast<std::uint64_t>(k));
            const float u = static_cast<float>(static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0)); // [0, 1)
            w[k] = (2.0f * u - 1.0f) * limit;
        }
    }
    net.in_shift.resize(static_cast<std::size_t>(net.inputs()), 0.0f);
    net.in_scale.resize(static_cast<std::size_t>(net.inputs()), 1.0f);
    net.loaded = false;
}

bool load_net(NetDecl& net, std::string& error)
{
    std::ifstream in(net.file);
    if (!in) {
        error = "no weights at " + net.file;
        return false;
    }
    std::string word;
    std::vector<int> sizes;
    std::vector<float> shift;
    std::vector<float> scale;
    std::vector<float> weights;
    float out_scale = 1.0f;
    Activation act = net.act;
    std::string section;
    while (in >> word) {
        if (word == "net") {
            in >> word; // its name: informative
        } else if (word == "sizes" || word == "in_shift" || word == "in_scale" || word == "weights" || word == "act" ||
            word == "out_scale") {
            section = word;
            if (section == "act") {
                in >> word;
                if (!parse_activation(word, act)) {
                    error = net.file + ": unknown activation '" + word + "'";
                    return false;
                }
                section.clear();
            } else if (section == "out_scale") {
                in >> out_scale;
                section.clear();
            }
        } else {
            std::istringstream number(word);
            double v = 0.0;
            if (!(number >> v)) {
                error = net.file + ": unexpected '" + word + "'";
                return false;
            }
            if (section == "sizes") {
                sizes.push_back(static_cast<int>(v));
            } else if (section == "in_shift") {
                shift.push_back(static_cast<float>(v));
            } else if (section == "in_scale") {
                scale.push_back(static_cast<float>(v));
            } else if (section == "weights") {
                weights.push_back(static_cast<float>(v));
            } else {
                error = net.file + ": a number outside any section";
                return false;
            }
        }
    }
    if (sizes != net.sizes) {
        error = net.file + ": its sizes are not the ones the law declares for '" + net.name + "'";
        return false;
    }
    if (act != net.act) {
        error = net.file + ": trained with another activation than the law declares";
        return false;
    }
    if (weights.size() != net.parameter_count()) {
        error = net.file + ": " + std::to_string(weights.size()) + " weights, the net has " + std::to_string(net.parameter_count());
        return false;
    }
    shift.resize(static_cast<std::size_t>(net.inputs()), 0.0f);
    scale.resize(static_cast<std::size_t>(net.inputs()), 1.0f);
    net.in_shift = std::move(shift);
    net.in_scale = std::move(scale);
    net.out_scale = out_scale;
    net.weights = std::move(weights);
    net.loaded = true;
    return true;
}

bool save_net(const NetDecl& net, const std::string& path, std::string& error)
{
    std::ofstream out(path);
    if (!out) {
        error = "cannot write " + path;
        return false;
    }
    out << "net " << net.name << "\nsizes";
    for (const int s : net.sizes) {
        out << ' ' << s;
    }
    out << "\nact " << to_string(net.act) << "\nin_shift";
    out << std::setprecision(9);
    for (const float v : net.in_shift) {
        out << ' ' << v;
    }
    out << "\nin_scale";
    for (const float v : net.in_scale) {
        out << ' ' << v;
    }
    out << "\nout_scale " << net.out_scale << "\nweights\n";
    for (std::size_t i = 0; i < net.weights.size(); ++i) {
        out << net.weights[i] << ((i + 1) % 8 == 0 ? '\n' : ' ');
    }
    out << "\n";
    return static_cast<bool>(out);
}

} // namespace fieldengine::field

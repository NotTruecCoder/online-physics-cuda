#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A SMALL NETWORK IS PART OF THE LAW. A law names it and uses it like any other function:
//
//   net front_fix  in 4  hidden 32 32  out 1  act tanh  half
//   term conduction
//     d T = diffuse(k_heat, T) + flux(front_fix, T, fuel)
//
// The engine runs it in the same kernel as the rest of the expression (on the CPU reference, in CUDA with tensor
// cores when `half`, and in JAX for training). Weights live in assets/nets/NAME.net, written by the trainers in
// tools/lawjax; with no file the net starts with its last layer at zero, so an untrained correction is exactly none.
namespace fieldengine::field {

enum class Activation { Tanh, Relu, Silu, Linear };

const char* to_string(Activation a);
bool parse_activation(const std::string& text, Activation& out);

struct NetDecl {
    std::string name;
    std::string where;              // "source:line"
    std::string file;               // where its weights are read from (and written to)
    std::vector<int> sizes;         // in, hidden..., out
    Activation act = Activation::Tanh; // every hidden layer; the last one is linear
    bool half = false;              // the GPU may run it on tensor cores (fp16 operands, fp32 accumulate)
    // y = net((x - in_shift) * in_scale) * out_scale: what the trainer learned to feed it.
    std::vector<float> in_shift;
    std::vector<float> in_scale;
    float out_scale = 1.0f;
    // Layer l: W_l (sizes[l+1] x sizes[l], row-major: W[o * in + i]), then b_l (sizes[l+1]).
    std::vector<float> weights;
    bool loaded = false;            // from its file; else initialised here

    [[nodiscard]] int inputs() const { return sizes.empty() ? 0 : sizes.front(); }
    [[nodiscard]] int outputs() const { return sizes.empty() ? 0 : sizes.back(); }
    [[nodiscard]] int layers() const { return sizes.size() < 2 ? 0 : static_cast<int>(sizes.size()) - 1; }
    [[nodiscard]] int widest() const;
    [[nodiscard]] std::size_t weight_offset(int layer) const; // of W_layer; b_layer follows W_layer
    [[nodiscard]] std::size_t parameter_count() const;
};

float activate(Activation a, float pre);
// d act / d pre, from the pre-activation.
float activate_grad(Activation a, float pre);

// One row through the net, in float, in a fixed order: the CUDA scalar path does the same arithmetic, so the two
// agree to the rounding of tanh. `scratch` holds 2 * widest() floats.
void net_forward(const NetDecl& net, const float* in, float* out, float* scratch);

// The same, keeping every layer's pre-activation (for the backward pass): `tape` holds sum(sizes[1..]) floats.
void net_forward_taped(const NetDecl& net, const float* in, float* out, float* tape);
// Back through one row: adds dL/dweights into `grad` (parameter_count() floats) and writes dL/din (raw inputs, through
// the normalisation) if `din` is not null. `dout` is dL/dy of the scaled outputs. `scratch` holds 2 * widest() floats.
void net_backward(const NetDecl& net, const float* in, const float* tape, const float* dout, float* grad, float* din,
    float* scratch);

// Glorot-uniform from a hash of the name and the seed (no RNG state: the same law gives the same net); the last layer
// is zero, so the output is exactly 0 until trained.
void init_net(NetDecl& net, std::uint64_t seed = 0);

// The text format:  net NAME / sizes 4 32 1 / act tanh / in_shift ... / in_scale ... / out_scale S / weights ...
bool load_net(NetDecl& net, std::string& error);
bool save_net(const NetDecl& net, const std::string& path, std::string& error);

} // namespace fieldengine::field

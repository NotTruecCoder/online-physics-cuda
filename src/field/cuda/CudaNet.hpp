#pragma once

#include "field/nn/Net.hpp"

#include <cstdint>
#include <string>
#include <vector>

// NETWORKS AS CUDA CODE. Each net becomes device functions with its sizes as constants (every loop unrolled):
//
//   NAME_fwd(w, x, y)          one row in fp32, in the CPU reference's order (Net.cpp): bit-for-bit up to tanh;
//   NAME_tc(w, frag, xs, ys)   32 rows a warp on tensor cores: mma.sync.m16n8k16 (fp16 operands, fp32 accumulate),
//                              written in inline PTX so no CUDA header is needed. A layer's fp32 accumulators are, register
//                              for register, the next layer's operand fragments: activations never leave the registers.
//
// and, for learning online in kernels (the agents' world model), three kernels with no atomics, so a run repeats to the
// bit: NAME_train_rows (forward and backward of each row fused, the activations and deltas kept), NAME_train_grad (one
// thread a weight, summing the rows in order), NAME_adam.
namespace fieldengine::field::cuda {

struct NetLayout {
    int in = 0;
    int out = 0;
    int widest = 0;
    int image_floats = 0;          // the fp32 image: in_shift, in_scale, out_scale, pad, then W and b per layer
    int weights_at = 0;            // where W_0 starts in the image
    std::vector<int> layer_at;     // where W_l starts in the image
    // Tensor cores: K padded to 16, hidden N to 16, out to 8.
    std::vector<int> k_pad;
    std::vector<int> n_pad;
    std::vector<int> frag_at;      // where layer l's fragments start (in uint2)
    int frag_words = 0;            // uint32 words of all fragments
    bool tensor = false;
};

NetLayout net_layout(const NetDecl& net);

// The device code of one net, its identifier `id` (a C name).
std::string net_device_code(const NetDecl& net, const std::string& id);
// The training kernels of one net (needs net_device_code first in the same source).
std::string net_train_code(const NetDecl& net, const std::string& id);

// The host side of the device layouts.
std::vector<float> net_image(const NetDecl& net);
std::vector<std::uint32_t> net_fragments(const NetDecl& net);
// Back from an image the device trained: weights into the decl.
void net_from_image(NetDecl& net, const std::vector<float>& image);

// What every source with nets starts with: activations, the f16x2 packing and the mma.
const char* net_prelude();

} // namespace fieldengine::field::cuda

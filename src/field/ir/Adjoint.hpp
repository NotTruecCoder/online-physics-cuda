#pragma once

#include "field/Law.hpp"

#include <string>

// REVERSE MODE ON THE IR. One step of a term is U' = U + dt R(U, p). Its vector-Jacobian product, given the cotangent
// A of U' (one field F__adj per field), is
//
//   U_bar = A + dt (dR/dU)^T A        p_bar = dt (dR/dp)^T A        (summed over the cells for a param)
//
// built by one reverse sweep over the term's SSA instructions (highest id first: every user before what it uses), each
// instruction adding its local derivative times its own adjoint to its arguments'. The result is a law, as text: the
// primal values as `let`s, the adjoints as `let`s, and one equation `d F__adj = ...` a field (and a param), so the
// engine's own Euler store F__adj + dt * (...) is the product above. Being a law, it runs wherever laws run (CPU, CUDA,
// the JAX twin) and reads as what it is.
//
// The fields of the primal are static inputs of the adjoint law (the driver writes the state the primal step started
// from); a param p's adjoint is the field p__adj, whose total over the lattice is dL/dp. Stencils go to their transposes
// (shift -> shift_t, gx -> gx_t, lap and diffuse are symmetric). Refused, with the reason: ops with no adjoint here (adv,
// tridiag, the matter-moving ops, distance, poisson, reductions, nets) and diffuse whose k is a field.
namespace fieldengine::field::ir {

// The adjoint law of `term` (the law's first term when empty); empty text and `error` set when it cannot be built.
std::string adjoint_law(const Law& law, const std::string& term, std::string& error);

} // namespace fieldengine::field::ir

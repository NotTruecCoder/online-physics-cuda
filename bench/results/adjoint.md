# Reverse mode in the IR

`lawc --adjoint` generates a term's adjoint law from the IR (one reverse sweep over its SSA instructions); `lawc --vjp` runs the primal forward keeping every state, then the adjoint law backward. Everything below is the engine's own adjoint, on the backend named, against independent references.

## Transposed stencils

<op(a), b> against <a, op_t(b)>, random a and b on 7 x 6 x 5 (every edge):

| backend | op | <op(a), b> | <a, op_t(b)> | relative difference |
|---|---|---|---|---|
| cpu | shift | 15.184842 | 15.184842 | 2.4e-08 |
| cpu | shift | 2.917357 | 2.917358 | 1.4e-07 |
| cpu | gx | 5.574818 | 5.574818 | 6.8e-08 |
| cpu | gy | -10.837739 | -10.837739 | 3.8e-08 |
| cpu | gz | -4.689681 | -4.689683 | 3.4e-07 |
| cuda | shift | 15.184842 | 15.184842 | 2.6e-08 |
| cuda | shift | 2.917357 | 2.917357 | 4.1e-08 |
| cuda | gx | 5.574818 | 5.574818 | 6.8e-08 |
| cuda | gy | -10.837739 | -10.837739 | 3.8e-08 |
| cuda | gz | -4.689681 | -4.689683 | 3.4e-07 |

## Black-Scholes: dV(100)/d(everything) by one backward pass

512 cells, 410 steps. Against jax.grad through the twin (initial state and params) and the formula (vega, rho):

| backend | vega (total of sig__adj) | rho (total of r__adj) | vs jax.grad: dL/dV0, sig, r (relative) | vega, rho vs formula | forward, backward ms |
|---|---|---|---|---|---|
| cpu | 37.51228 | -41.90874 | 3.1e-07, 1.2e-06, 2.4e-06 | 3.1e-04, 4.4e-04 | forward 9.3393 ms, backward 22.4015 ms (card 0 ms) |
| cuda | 37.51228 | -41.90875 | 2.5e-07, 1.2e-06, 2.5e-06 | 3.1e-04, 4.4e-04 | forward 472.958 ms, backward 429.633 ms (card 278.993 ms) |

## Heston (heston.law, explicit): every cell weighed

64 x 21 cells, 328 steps, L = sum(w V_N) with random w. Relative difference from jax.grad:

| backend | dL/dV0 | kappa | theta | rho | r | sig_v (the law's own, var fixed) |
|---|---|---|---|---|---|---|
| cpu | 2.0e-07 | 5.4e-08 | 3.0e-07 | 2.7e-06 | 6.3e-08 | 5.6e-07 |
| cuda | 2.1e-07 | 2.6e-09 | 1.8e-07 | 2.1e-06 | 9.6e-08 | 6.6e-07 |

## Cost

3-D heat (diffusion.law), 128³, 20 steps on CUDA, L = sum(T_N) (conserved, so dL/dT0 is 1 everywhere: the adjoint gives it to 0.0e+00). forward 90.9054 ms, backward 173.396 ms (card 124.155 ms): the backward pass includes uploading each kept state (the primal states are kept on the host; no checkpointing yet).

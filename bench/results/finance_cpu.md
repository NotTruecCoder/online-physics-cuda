# Finance: option pricing PDEs as laws

Put, K = 100, T = 1, sigma = 0.2, r = 0.05; x = ln S on ln K +- 4. Explicit Euler, one substep a step at dt = 0.4 h^2 / sigma^2.

CPU reference only (no card): `python bench/finance.py --cpu`.

## European put against the Black-Scholes formula

| backend | cells | steps | V(100) | error at 100 | order | RMS error, S = 80..120 | order |
|---|---|---|---|---|---|---|---|
| cpu | 64 | 7 | 5.99608 | 4.23e-01 |  | 2.85e-01 |  |
| cpu | 128 | 26 | 5.68344 | 1.10e-01 | 1.94 | 6.69e-02 | 2.09 |
| cpu | 256 | 103 | 5.60122 | 2.77e-02 | 1.99 | 1.95e-02 | 1.78 |
| cpu | 512 | 410 | 5.58047 | 6.94e-03 | 2.00 | 4.85e-03 | 2.00 |

Formula: V(100) = 5.57353.

## American put against a binomial tree

Reference: CRR tree, mean of 20 000 and 20 001 steps (the tree's odd-even wobble), V(100) = 6.09039; the European value is 5.57353.

| backend | cells | V(100) | error |
|---|---|---|---|
| cpu | 256 | 6.11521 | +2.48e-02 |
| cpu | 512 | 6.09665 | +6.26e-03 |

## Greeks at S = 100 (256 cells)

Delta and gamma from the lattice (V_x central, V_xx the three-point stencil, in x = ln S); vega and rho by `jax.grad` through every step of the IR's JAX twin (tools/lawjax).

| greek | engine | formula | relative error |
|---|---|---|---|
| delta | -0.36314 | -0.36317 | 7.3e-05 |
| gamma | 0.01864 | 0.01876 | 6.6e-03 |
| vega | 37.47664 | 37.52403 | 1.3e-03 |
| rho | -41.96334 | -41.89046 | 1.7e-03 |

The JAX twin's price 5.60122 against the engine's 5.60122.

## Heston, European put against the semi-analytic price

kappa 1.5, theta 0.04, sigma_v 0.3, rho -0.9, r 0.025, v0 0.04; x on ln K +- 3, v on [0, 0.6]. Reference (characteristic function) V(100, v0) = 6.42586. `heston.law`: every derivative second order (three-point V_xx, the y flux with exact face variances, second-order upwind drift); `heston_first_order.law`: the scheme it replaced (first-order upwind drift, harmonic face mean). Order: log2 of the ratio of successive errors against the reference. Self-convergence order: log2((V_n - V_2n) / (V_2n - V_4n)), the lattice against itself, which the truncated domain (the same at every lattice) does not enter.

| law | backend | cells along x | cells | steps | V(100, v0) | error | order | self-convergence order | card ms a step |
|---|---|---|---|---|---|---|---|---|---|
| heston | cpu | 32 | 352 | 84 | 6.79836 | +3.72e-01 |  |  |  |
| heston | cpu | 64 | 1344 | 328 | 6.08421 | -3.42e-01 | 0.12 |  |  |
| heston | cpu | 128 | 5504 | 1360 | 6.31416 | -1.12e-01 | 1.61 | 1.63 |  |
| heston_first_order | cpu | 32 | 352 | 84 | 6.30791 | -1.18e-01 |  |  |  |
| heston_first_order | cpu | 64 | 1344 | 328 | 6.30584 | -1.20e-01 | -0.03 |  |  |
| heston_first_order | cpu | 128 | 5504 | 1360 | 6.40457 | -2.13e-02 | 2.50 | -5.58 |  |
| heston, NumPy float64 | cpu | 32 | | | 6.79836 | +3.72e-01 |  | | |
| heston, NumPy float64 | cpu | 64 | | | 6.08421 | -3.42e-01 | 0.12 | | |
| heston, NumPy float64 | cpu | 128 | | | 6.31415 | -1.12e-01 | 1.61 | | |

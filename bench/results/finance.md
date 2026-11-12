# Finance: option pricing PDEs as laws

Put, K = 100, T = 1, sigma = 0.2, r = 0.05; x = ln S on ln K +- 4. Explicit Euler, one substep a step at dt = 0.4 h^2 / sigma^2.

## European put against the Black-Scholes formula

| backend | cells | steps | V(100) | error at 100 | order | RMS error, S = 80..120 | order |
|---|---|---|---|---|---|---|---|
| cpu | 64 | 7 | 5.99608 | 4.23e-01 |  | 2.85e-01 |  |
| cpu | 128 | 26 | 5.68344 | 1.10e-01 | 1.94 | 6.69e-02 | 2.09 |
| cpu | 256 | 103 | 5.60122 | 2.77e-02 | 1.99 | 1.95e-02 | 1.78 |
| cpu | 512 | 410 | 5.58047 | 6.94e-03 | 2.00 | 4.85e-03 | 2.00 |
| cpu | 1024 | 1639 | 5.57527 | 1.74e-03 | 2.00 | 1.08e-03 | 2.17 |
| cuda | 64 | 7 | 5.99608 | 4.23e-01 |  | 2.85e-01 |  |
| cuda | 128 | 26 | 5.68344 | 1.10e-01 | 1.94 | 6.69e-02 | 2.09 |
| cuda | 256 | 103 | 5.60122 | 2.77e-02 | 1.99 | 1.95e-02 | 1.78 |
| cuda | 512 | 410 | 5.58047 | 6.94e-03 | 2.00 | 4.85e-03 | 2.00 |
| cuda | 1024 | 1639 | 5.57527 | 1.74e-03 | 2.00 | 1.08e-03 | 2.17 |

Formula: V(100) = 5.57353.

## American put against a binomial tree

Reference: CRR tree, mean of 20 000 and 20 001 steps (the tree's odd-even wobble), V(100) = 6.09039; the European value is 5.57353.

| backend | cells | V(100) | error |
|---|---|---|---|
| cpu | 256 | 6.11521 | +2.48e-02 |
| cpu | 512 | 6.09665 | +6.26e-03 |
| cpu | 1024 | 6.09195 | +1.56e-03 |
| cuda | 256 | 6.11521 | +2.48e-02 |
| cuda | 512 | 6.09665 | +6.26e-03 |
| cuda | 1024 | 6.09195 | +1.56e-03 |

## Greeks at S = 100 (512 cells)

Delta and gamma from the lattice (V_x central, V_xx the three-point stencil, in x = ln S); vega and rho by `jax.grad` through every step of the IR's JAX twin (tools/lawjax).

| greek | engine | formula | relative error |
|---|---|---|---|
| delta | -0.36316 | -0.36317 | 1.8e-05 |
| gamma | 0.01873 | 0.01876 | 1.7e-03 |
| vega | 37.51223 | 37.52403 | 3.1e-04 |
| rho | -41.90864 | -41.89046 | 4.3e-04 |

The JAX twin's price 5.58047 against the engine's 5.58047.

## Heston, European put against the semi-analytic price

kappa 1.5, theta 0.04, sigma_v 0.3, rho -0.9, r 0.025, v0 0.04; x on ln K +- 3, v on [0, 0.6]. Reference (characteristic function) V(100, v0) = 6.42586. `heston.law`: every derivative second order (three-point V_xx, the y flux with exact face variances, second-order upwind drift); `heston_first_order.law`: the scheme it replaced (first-order upwind drift, harmonic face mean). Order: log2 of the ratio of successive errors against the reference. Self-convergence order: log2((V_n - V_2n) / (V_2n - V_4n)), the lattice against itself, which the truncated domain (the same at every lattice) does not enter.

| law | backend | cells along x | cells | steps | V(100, v0) | error | order | self-convergence order | card ms a step |
|---|---|---|---|---|---|---|---|---|---|
| heston | cuda | 64 | 1344 | 328 | 6.08421 | -3.42e-01 |  |  | 0.296 |
| heston | cuda | 128 | 5504 | 1360 | 6.31416 | -1.12e-01 | 1.61 |  | 0.240 |
| heston | cuda | 256 | 21760 | 5408 | 6.42256 | -3.30e-03 | 5.08 | 1.09 | 0.235 |
| heston | cuda | 512 | 87552 | 21824 | 6.42487 | -9.88e-04 | 1.74 | 5.55 | 0.309 |
| heston | cuda | 1024 | 349184 | 87168 | 6.42393 | -1.93e-03 | -0.97 | 1.29 | 0.699 |
| heston | cuda | 2048 | 1398784 | 349440 | 6.41343 | -1.24e-02 | -2.68 | -3.47 | 1.204 |
| heston_first_order | cuda | 64 | 1344 | 328 | 6.30584 | -1.20e-01 |  |  | 0.752 |
| heston_first_order | cuda | 128 | 5504 | 1360 | 6.40457 | -2.13e-02 | 2.50 |  | 0.763 |
| heston_first_order | cuda | 256 | 21760 | 5408 | 6.42536 | -5.01e-04 | 5.41 | 2.25 | 0.786 |
| heston_first_order | cuda | 512 | 87552 | 21824 | 6.42434 | -1.52e-03 | -1.60 | 4.35 | 0.495 |
| heston, NumPy float64 | cpu | 64 | | | 6.08421 | -3.42e-01 |  | | |
| heston, NumPy float64 | cpu | 128 | | | 6.31415 | -1.12e-01 | 1.61 | | |
| heston, NumPy float64 | cpu | 256 | | | 6.42250 | -3.36e-03 | 5.05 | | |
| heston, NumPy float64 | cpu | 512 | | | 6.42545 | -4.10e-04 | 3.03 | | |

The same at v0 = 0.2, away from var = 0 (reference 12.15236):

| law | cells along x | V(100, 0.2) | error | order |
|---|---|---|---|---|
| heston | 64 | 12.11291 | -3.95e-02 |  |
| heston | 128 | 12.15468 | +2.33e-03 | 4.08 |
| heston | 256 | 12.15283 | +4.69e-04 | 2.31 |
| heston | 512 | 12.15105 | -1.31e-03 | -1.48 |
| heston_first_order | 64 | 12.10473 | -4.76e-02 |  |
| heston_first_order | 128 | 12.12668 | -2.57e-02 | 0.89 |
| heston_first_order | 256 | 12.13823 | -1.41e-02 | 0.86 |
| heston_first_order | 512 | 12.14355 | -8.81e-03 | 0.68 |
| heston, NumPy float64 | 128 | 12.15466 | +2.30e-03 |  |
| heston, NumPy float64 | 256 | 12.15284 | +4.80e-04 | 2.26 |
| heston, NumPy float64 | 512 | 12.15247 | +1.14e-04 | 2.07 |

CPU reference at 128: 6.31416 (CUDA 6.31416).

### A price surface in one run

64 strikes (80 to 120) along the lattice's z, 4 maturities (0.25, 0.5, 0.75, 1) as states dumped on the way: 256 Heston puts from one run of 21824 steps at 512 cells along x. Against the characteristic function: RMS error 2.9e-03 (by maturity 3.1e-03, 3.7e-03, 2.6e-03, 1.9e-03), worst 1.1e-02 at K = 118.1, T = 0.5 (the in-the-money strikes, K > 100, at the shorter maturities are the worst; their error falls about 3x from 256 to 512 cells). Payoffs are averaged over each cell (a kink inside a cell sampled at its centre is a first-order error). Card time a step 1.378 ms for the 64 strikes against 0.314 ms for one: 469.9 ms of card time a strike (its 4 maturities) in the batch, against 6847.5 ms for a strike run alone.

## Heston by ADI (Hundsdorfer-Verwer), implicit through `tridiag_x` / `tridiag_y`

`heston_adi.law`: the same PDE, three-point operators (central drift, a forward difference on the v = 0 row), the mixed derivative explicit, the y and x parts implicit, solved in increments (the delta form) so float32 rounds changes, not prices. Steps = cells along x / 4 (dt ~ h: second order in both). Total card time = steps x card time a step; the explicit scheme's beside it, same cells.

v0 = 0.04, reference 6.42586:

| cells along x | steps | V(100, v0) | error | order | self-convergence order | card ms a step | card ms in all | explicit: steps, card ms in all |
|---|---|---|---|---|---|---|---|---|
| 64 | 16 | 6.33128 | -9.46e-02 |  |  | 0.475 | 8 | 328, 97 |
| 128 | 32 | 6.39569 | -3.02e-02 | 1.65 |  | 0.517 | 17 | 1360, 327 |
| 256 | 64 | 6.42087 | -4.99e-03 | 2.60 | 1.35 | 0.982 | 63 | 5408, 1269 |
| 512 | 128 | 6.42477 | -1.09e-03 | 2.20 | 2.69 | 1.356 | 174 | 21824, 6739 |
| 1024 | 256 | 6.42560 | -2.64e-04 | 2.04 | 2.24 | 3.414 | 874 | 87168, 60942 |
| 2048 | 512 | 6.42580 | -6.38e-05 | 2.05 | 2.04 | 7.945 | 4068 | 349440, 420645 |

v0 = 0.2, reference 12.15236:

| cells along x | steps | V(100, v0) | error | order | self-convergence order | card ms a step | card ms in all | explicit: steps, card ms in all |
|---|---|---|---|---|---|---|---|---|
| 64 | 16 | 12.12693 | -2.54e-02 |  |  | 0.397 | 6 |  |
| 128 | 32 | 12.14654 | -5.82e-03 | 2.13 |  | 0.460 | 15 |  |
| 256 | 64 | 12.15090 | -1.46e-03 | 2.00 | 2.17 | 0.762 | 49 |  |
| 512 | 128 | 12.15199 | -3.63e-04 | 2.00 | 2.00 | 1.393 | 178 |  |
| 1024 | 256 | 12.15226 | -9.80e-05 | 1.89 | 2.05 | 3.277 | 839 |  |
| 2048 | 512 | 12.15233 | -3.22e-05 | 1.60 | 2.01 | 14.389 | 7367 |  |

### A batch: thousands of options in one run

Every z-plane of the lattice is an option: its strike in its payoff, its own kappa, theta, sig_v, rho and r as static fields. 256 cells along x, 64 steps. Card time of a step against the batch (the same option repeated):

| options | cells | card ms a step | card ms an option (all steps) | options a second |
|---|---|---|---|---|
| 1 | 22016 | 1.285 | 82.23 | 12 |
| 16 | 352256 | 2.152 | 8.61 | 116 |
| 256 | 5636096 | 19.569 | 4.89 | 204 |
| 1024 | 22544384 | 89.695 | 5.61 | 178 |

512 different options (strike 80-120, kappa 0.5-3, theta 0.02-0.09, sig_v 0.2-0.6, rho -0.9 to -0.1, r 0-0.05, v0 0.02-0.1, Feller holding) and 4 maturities each, 2048 prices from one run of 64 steps (3853 ms of card time), against the characteristic function: RMS error 3.8e-03 (by maturity 5.7e-03, 3.8e-03, 2.6e-03, 2.3e-03), worst 3.7e-02 (K 119.4, sig_v 0.34, T 1, v0 0.024). The same lattice and step for every option: 256 cells is the accuracy of the convergence table's row.

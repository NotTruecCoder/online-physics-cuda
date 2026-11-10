# Finance: option pricing PDEs as laws

Put, K = 100, T = 1, sigma = 0.2, r = 0.05; x = ln S on ln K +- 4. Explicit Euler, one substep a step at dt = 0.4 h^2 / sigma^2.

## European put against the Black-Scholes formula

| backend | cells | steps | V(100) | error at 100 | RMS error, S = 80..120 | order |
|---|---|---|---|---|---|---|
| cpu | 64 | 7 | 5.99608 | 4.23e-01 | 2.85e-01 | |
| cpu | 128 | 26 | 5.68344 | 1.10e-01 | 6.69e-02 | 2.09 |
| cpu | 256 | 103 | 5.60122 | 2.77e-02 | 1.95e-02 | 1.78 |
| cpu | 512 | 410 | 5.58047 | 6.94e-03 | 4.85e-03 | 2.00 |
| cpu | 1024 | 1639 | 5.57527 | 1.74e-03 | 1.08e-03 | 2.17 |
| cuda | 64 | 7 | 5.99608 | 4.23e-01 | 2.85e-01 | |
| cuda | 128 | 26 | 5.68344 | 1.10e-01 | 6.69e-02 | 2.09 |
| cuda | 256 | 103 | 5.60122 | 2.77e-02 | 1.95e-02 | 1.78 |
| cuda | 512 | 410 | 5.58047 | 6.94e-03 | 4.85e-03 | 2.00 |
| cuda | 1024 | 1639 | 5.57527 | 1.74e-03 | 1.08e-03 | 2.17 |

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

Delta and gamma from the lattice (central differences in x = ln S); vega and rho by `jax.grad` through every step of the IR's JAX twin (tools/lawjax).

| greek | engine | formula | relative error |
|---|---|---|---|
| delta | -0.36316 | -0.36317 | 1.8e-05 |
| gamma | 0.01870 | 0.01876 | 3.2e-03 |
| vega | 37.51223 | 37.52403 | 3.1e-04 |
| rho | -41.90864 | -41.89046 | 4.3e-04 |

The JAX twin's price 5.58047 against the engine's 5.58047.

## Heston, European put against the semi-analytic price

kappa 1.5, theta 0.04, sigma_v 0.3, rho -0.9, r 0.025, v0 0.04; x on ln K +- 3, v on [0, 0.6]. Reference (characteristic function) V(100, v0) = 6.42586.

| backend | cells | steps | V(100, v0) | error | card ms a step |
|---|---|---|---|---|---|
| cuda | 1344 | 328 | 6.47558 | +4.97e-02 | 0.387 |
| cuda | 5504 | 1360 | 6.43757 | +1.17e-02 | 0.315 |
| cuda | 21760 | 5408 | 6.43420 | +8.34e-03 | 0.397 |
| cuda | 87552 | 21824 | 6.42665 | +7.93e-04 | 0.556 |

CPU reference at 128: 6.43757 (CUDA 6.43757).

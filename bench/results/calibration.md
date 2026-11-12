# Calibration: Heston fitted through the PDE by jax.grad

Market: puts on S = 100, r = 0.025, strikes 80-120 (9), maturities 0.25, 0.5, 1: 27 prices from the characteristic function at the true numbers, and the same with noise of a tenth of a cent on each. The pricer: `heston_adi.law` through its JAX twin, the strikes as z-planes, the maturities read on the way; L-BFGS-B on the summed squared error with the gradient by `jax.value_and_grad` through every step. Beside it, the characteristic function itself fitted to the same prices (no lattice: what a perfect pricer would find).

## The gradient

Against central differences of the same float32 loss, at the start:

| lattice | number | jax.grad | central difference | relative difference |
|---|---|---|---|---|
| 128 x 43 | kappa | 1.7354 | 1.7347 | 3.9e-04 |
| 128 x 43 | theta | 242.43 | 242.5 | 3.2e-04 |
| 128 x 43 | sig_v | -6.9763 | -6.9825 | 8.8e-04 |
| 128 x 43 | rho | 8.3742 | 8.3768 | 3.2e-04 |
| 128 x 43 | v0 | -14.779 | -14.868 | 6.0e-03 |
| 256 x 86 | kappa | 1.7357 | 1.7354 | 1.4e-04 |
| 256 x 86 | theta | 240.47 | 240.56 | 3.6e-04 |
| 256 x 86 | sig_v | -7.4212 | -7.4085 | 1.7e-03 |
| 256 x 86 | rho | 8.3206 | 8.3188 | 2.1e-04 |
| 256 x 86 | v0 | -22.484 | -22.612 | 5.6e-03 |

| lattice | cells (all strikes) | steps a year | loss | loss and gradient | compile | lattice's RMS error at the true numbers |
|---|---|---|---|---|---|---|
| 128 x 43 | 49536 | 32 | 0.21 s | 0.81 s (3.9 losses) | 5 s | 2.5e-02 |
| 256 x 86 | 198144 | 64 | 2.44 s | 6.94 s (2.9 losses) | 12 s | 6.2e-03 |

Central differences would cost 10 losses for the five derivatives; reverse mode costs what the table says, whatever the number of parameters (CPU, JAX).

## The fit

Exact prices:

| number | true | start | PDE 128 | PDE 256 | closed form |
|---|---|---|---|---|---|
| kappa | 1.5 | 2.5 | 1.7660 | 1.5644 | 1.5000 |
| theta | 0.04 | 0.06 | 0.0402 | 0.0400 | 0.0400 |
| sig_v | 0.3 | 0.45 | 0.2723 | 0.2932 | 0.3000 |
| rho | -0.7 | -0.3 | -0.8284 | -0.7287 | -0.7000 |
| v0 | 0.05 | 0.03 | 0.0499 | 0.0500 | 0.0500 |
| RMS price error | | | 9.0e-03 | 2.6e-03 | 6.2e-07 |
| iterations (seconds) | | | 42 (56 s) | 46 (408 s) | |

Prices with noise:

| number | true | start | PDE 128 | PDE 256 | closed form |
|---|---|---|---|---|---|
| kappa | 1.5 | 2.5 | 2.0552 | 1.5694 | 1.5046 |
| theta | 0.04 | 0.06 | 0.0404 | 0.0401 | 0.0401 |
| sig_v | 0.3 | 0.45 | 0.2803 | 0.2947 | 0.3015 |
| rho | -0.7 | -0.3 | -0.8532 | -0.7265 | -0.6980 |
| v0 | 0.05 | 0.03 | 0.0503 | 0.0500 | 0.0500 |
| RMS price error | | | 1.2e-02 | 3.0e-03 | 1.0e-03 |
| iterations (seconds) | | | 30 (45 s) | 50 (532 s) | |

What separates a PDE fit from the true numbers is the lattice's pricing error turned into the numbers (rho and kappa move most: the surface is least sensitive to them); the table shows how it shrinks with the lattice.

# Verification: order of convergence
| problem | backend | lattice | L2 error | observed order |
|---|---|---|---|---|
| diffusion | cpu | 16 | 3.279e-03 |  |
| diffusion | cpu | 32 | 7.714e-04 | 2.09 |
| diffusion | cpu | 64 | 1.005e-04 | 2.94 |
| diffusion | cpu | 128 | 2.175e-05 | 2.21 |
| diffusion | cpu | 256 | 5.299e-06 | 2.04 |
| advection | cpu | 64 | 1.134e-01 |  |
| advection | cpu | 128 | 7.910e-02 | 0.52 |
| advection | cpu | 256 | 4.985e-02 | 0.67 |
| advection | cpu | 512 | 2.859e-02 | 0.80 |
| advection | cpu | 1024 | 1.546e-02 | 0.89 |
| diffusion | cuda | 16 | 3.279e-03 |  |
| diffusion | cuda | 32 | 7.714e-04 | 2.09 |
| diffusion | cuda | 64 | 1.005e-04 | 2.94 |
| diffusion | cuda | 128 | 2.178e-05 | 2.21 |
| diffusion | cuda | 256 | 5.316e-06 | 2.03 |
| advection | cuda | 64 | 1.134e-01 |  |
| advection | cuda | 128 | 7.910e-02 | 0.52 |
| advection | cuda | 256 | 4.985e-02 | 0.67 |
| advection | cuda | 512 | 2.859e-02 | 0.80 |
| advection | cuda | 1024 | 1.546e-02 | 0.89 |

Largest relative drift of the conserved total of T (float32, up to 2190 substeps): 1.5e-06

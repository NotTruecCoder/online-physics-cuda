# Ablations: which piece gives which guarantee

Random untrained weights, 20 draws, worst draw. Mirror error: one step of the x-mirrored state, mirrored back, against one step of the state (K). Offset error: one step of T + 1000 K, minus 1000, against one step of T (K). Peak: the largest |T| reached (random weights at twice the explicit step can go unstable; catching that is the guard's job in model 3, not these properties').

| operator | variant | drift of the total (relative) | draws leaving the range | mirror error (K) | offset error (K) | peak abs T |
|---|---|---|---|---|---|---|
| flux | full: flux form, antisymmetric, limited | 6.4e-08 | 0 / 20 | 1.22e-04 |  |  |
| flux | no limiter | 1.7e-07 | 13 / 20 | 1.22e-04 |  |  |
| flux | no antisymmetry | 5.4e-07 | 0 / 20 | 1.42e+02 |  |  |
| flux | no flux form (learned source per cell) | 1.6e+02 | 14 / 20 | 9.48e+01 |  |  |
| ldiffuse | full: flux form, antisymmetric, mean removed | 4.6e-06 |  | 0.00e+00 | 6.10e-04 | 3.42e+06 |
| ldiffuse | no mean removal | 2.4e-03 |  | 0.00e+00 | 9.26e+02 | 1.02e+10 |
| ldiffuse | no antisymmetry | 1.5e-06 |  | 4.33e+01 | 6.10e-04 | 6.5e+06 |

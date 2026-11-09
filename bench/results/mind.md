# Model 1: a world model learned online, planned through with MPPI

64 foragers, episodes of 240 decisions (0.25 s), 30 episodes a world, 5 world seeds; MPPI 256 futures x 12 steps. Cost a decision (the plan line hunger + 2 fear - 0.5 eaten; lower is better) over the last 10 episodes, mean ± std over seeds.

| policy | cost a decision | plan time for 64 agents (ms) |
|---|---|---|
| still | 0.946 ± 0.12 | 0.03 |
| random direction | 0.923 ± 0.12 | 0.04 |
| hand-written seek rule (reads the true gradient) | 0.485 ± 0.067 | 0.05 |
| MPPI, true dynamics (ceiling) | 0.317 ± 0.071 | 21.98 |
| MPPI, learned world model (online) | 0.5 ± 0.038 | 12.08 |

Learning curve (learned MPPI, mean over seeds; oracle for reference): episode 1: 0.712, 6: 0.515, 11: 0.510, 16: 0.544, 21: 0.513, 26: 0.476; oracle mean 0.317.

The model alone on worlds it never saw (another layout), random actions: one-step R² of the change, mean over features 0.0147 ± 0.077 (hunger 0.0871 ± 0.23, fear -1.52 ± 0.76); cost 12 steps ahead open loop, RMSE 0.447 ± 0.082 against 0.29 ± 0.029 for persistence (nothing changes). Backend: CUDA.

The same on the world it learned in (other starts, random actions): one-step R² 0.357 ± 0.039 (hunger 0.361 ± 0.16, fear -1.92 ± 1.1); cost 12 steps ahead RMSE 0.427 ± 0.12 against 0.279 ± 0.036 for persistence.

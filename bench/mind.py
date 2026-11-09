"""MODEL 1, evaluated: runs bench_mind (bench/mind_bench.cpp) and writes the report.

Policies on the forager task (see mind_bench.cpp): still, random, the hand-written seek rule, MPPI with the true
dynamics (the planner's ceiling), MPPI through the learned world model. Cost a decision over the last 10 episodes,
mean ± std over world seeds; and the model alone on unseen worlds (one-step R², cost 12 steps ahead).

    python bench/mind.py [--seeds 5] [--episodes 30]
"""

import argparse
import json
import os
import subprocess

import numpy as np

from common import BUILD, binary, mean_std, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=30)
    a = p.parse_args()
    path = os.path.join(BUILD, "mind.json")
    subprocess.run([binary("bench_mind"), "--seeds", str(a.seeds), "--episodes", str(a.episodes), "--out", path], cwd=BUILD, check=True,
                   stdout=subprocess.DEVNULL)
    with open(path) as f:
        data = json.load(f)
    policies = ["still", "random", "seek", "mppi_oracle", "mppi_learned"]
    label = {"still": "still", "random": "random direction", "seek": "hand-written seek rule (reads the true gradient)",
             "mppi_oracle": "MPPI, true dynamics (ceiling)", "mppi_learned": "MPPI, learned world model (online)"}
    md = [f"# Model 1: a world model learned online, planned through with MPPI\n\n{data['agents']} foragers, episodes of "
          f"{data['episode_steps']} decisions (0.25 s), {data['episodes']} episodes a world, {len(data['seeds'])} world seeds; "
          f"MPPI {data['samples']} futures x {data['horizon']} steps. Cost a decision (the plan line hunger + 2 fear - 0.5 eaten; lower "
          "is better) over the last 10 episodes, mean ± std over seeds.\n\n| policy | cost a decision | plan time for 64 agents (ms) |\n|---|---|---|\n"]
    for pol in policies:
        vals = [s["policies"][pol]["last10"] for s in data["seeds"]]
        ms = np.mean([s["policies"][pol]["plan_ms"] for s in data["seeds"]])
        md.append(f"| {label[pol]} | {mean_std(vals)} | {ms:.2f} |\n")
    curve = np.mean([s["policies"]["mppi_learned"]["per_episode"] for s in data["seeds"]], axis=0)
    oracle = np.mean([s["policies"]["mppi_oracle"]["per_episode"] for s in data["seeds"]], axis=0)
    md.append("\nLearning curve (learned MPPI, mean over seeds; oracle for reference): episode "
              + ", ".join(f"{e + 1}: {curve[e]:.3f}" for e in range(0, len(curve), max(1, len(curve) // 6))) + f"; oracle mean {oracle.mean():.3f}.\n")
    r2 = [s["model"]["r2_mean"] for s in data["seeds"]]
    rh = [s["model"]["r2_hunger"] for s in data["seeds"]]
    rf = [s["model"]["r2_fear"] for s in data["seeds"]]
    cm = [s["model"]["cost_rmse_model"] for s in data["seeds"]]
    cp = [s["model"]["cost_rmse_persist"] for s in data["seeds"]]
    md.append(f"\nThe model alone on worlds it never saw (another layout), random actions: one-step R² of the change, mean over features "
              f"{mean_std(r2)} (hunger {mean_std(rh)}, fear {mean_std(rf)}); cost {data['horizon']} steps ahead open loop, RMSE "
              f"{mean_std(cm)} against {mean_std(cp)} for persistence (nothing changes). Backend: {data['seeds'][0]['model']['backend']}.\n")
    sm = [s["model"]["seen_r2_mean"] for s in data["seeds"]]
    sh_ = [s["model"]["seen_r2_hunger"] for s in data["seeds"]]
    sf = [s["model"]["seen_r2_fear"] for s in data["seeds"]]
    scm = [s["model"]["seen_cost_rmse_model"] for s in data["seeds"]]
    scp = [s["model"]["seen_cost_rmse_persist"] for s in data["seeds"]]
    md.append(f"\nThe same on the world it learned in (other starts, random actions): one-step R² {mean_std(sm)} (hunger {mean_std(sh_)}, "
              f"fear {mean_std(sf)}); cost {data['horizon']} steps ahead RMSE {mean_std(scm)} against {mean_std(scp)} for persistence.\n")
    save("mind", data, "".join(md))
    print("".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

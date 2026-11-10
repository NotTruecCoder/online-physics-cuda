"""PERFORMANCE AND PARITY: what the compiled law costs, against what the card can do, and how close the backends agree.

  stencil    bench_stencil: one conduction term on N^3 (64, 128, 256): a copy kernel (the bandwidth the card reaches),
             a handwritten fused kernel (the least traffic the step can have), and the engine's generated code, with
             and without its contract checks; plus the readback the engine does every step and the CPU reference;
  world law  the shipped law (43 terms, 39 fields) on the sandbox's live window (144 x 144 x 48, 1 M cells): CPU
             reference (OpenMP) against CUDA, per step and the most expensive terms;
  parity     lawc --compare (CPU reference against CUDA after 6 steps of everything at once) and tools/lawjax/parity.py
             (the JAX twin against the CPU reference).

    python bench/perf.py
"""

import json
import os
import re
import subprocess
import sys

from common import BUILD, ROOT, binary, lawc, lawc_json, save, write_scene


def main():
    out = {}
    stencil_path = os.path.join(BUILD, "stencil.json")
    run = subprocess.run([binary("bench_stencil"), "--sizes", "64,128,256", "--steps", "100", "--out", stencil_path], cwd=BUILD,
                         capture_output=True, text=True)
    print(run.stderr)
    with open(stencil_path) as f:
        out["stencil"] = json.load(f)

    world = {}
    for backend in ("cpu", "cuda"):
        j = lawc_json("--scene", "assets/scenes/window.scene", "--steps", "8" if backend == "cpu" else "40", "--warmup", "2" if backend == "cpu" else "5",
                      "--dt", "0.25", "--backend", backend)
        world[backend] = j
        top = sorted(j["terms"].items(), key=lambda kv: -kv[1])[:5]
        print(f"world law on {backend}: median step {j['median_ms']:.2f} ms; most expensive terms {[(k, round(v, 3)) for k, v in top]}")
    out["world"] = world

    # ---- what the learned operators cost on the card, against what they replace ------------------------------------
    learned = {}
    tmp = os.path.join(BUILD, "_perf_tmp")
    os.makedirs(tmp, exist_ok=True)
    heat = write_scene(os.path.join(tmp, "heat.scene"), (1024, 1024, 1, 1.0), ["assets/law/bench/heat.law"], ["fill T 290", "noise T 1000 5"])
    for name, extra in (("heat: classical, 2 substeps", []), ("heat: learned ldiffuse, 1 pass", ["--learned"])):
        j = lawc_json("--scene", heat, "--steps", "30", "--warmup", "5", "--dt", "0.3", "--backend", "cuda", *extra)
        learned[name] = sum(j["terms"].values())
    fine = write_scene(os.path.join(tmp, "fine.scene"), (2048, 2048, 1, 0.5), ["assets/law/bench/front.law"],
                       ["fill fuel 0.8", "noise fuel 0.15 3", "box T 1500 0 0 0 6 2048 1"])
    coarse = write_scene(os.path.join(tmp, "coarse.scene"), (1024, 1024, 1, 1.0), ["assets/law/bench/front.law"],
                         ["fill fuel 0.8", "noise fuel 0.15 3", "box T 1500 0 0 0 3 1024 1"])
    for name, args in (("front: fine 2048²", [fine]), ("front: coarse 1024²", [coarse]),
                       ("front: coarse 1024² + learned flux (tensor cores)", [coarse, "assets/law/bench/front.law", "assets/law/bench/front_fix.law"])):
        j = lawc_json("--scene", *args, "--steps", "30", "--warmup", "5", "--dt", "0.05", "--backend", "cuda")
        learned[name] = sum(j["terms"].values())
    out["learned_cost_ms"] = learned
    print(f"learned operators, card ms a step: {learned}")

    cmp_text = lawc("--scene", "assets/scenes/window.scene", "--steps", "6", "--dt", "0.25", "--compare", "--backend", "cuda")
    m = re.search(r"worst relative difference ([0-9.e+-]+)", cmp_text)
    out["cuda_vs_cpu_worst_relative"] = float(m.group(1)) if m else None
    par = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "lawjax", "parity.py"), "--scene", "assets/scenes/fire_test.scene", "--steps", "5",
                          "--dt", "0.25", "--lawc", binary("lawc")], cwd=BUILD, capture_output=True, text=True)
    m = re.search(r"worst relative difference ([0-9.e+-]+)", par.stdout)
    out["jax_vs_cpu_worst_relative"] = float(m.group(1)) if m else None
    print(f"parity: CUDA vs CPU {out['cuda_vs_cpu_worst_relative']}, JAX vs CPU {out['jax_vs_cpu_worst_relative']}")

    s = out["stencil"]
    md = [f"# Performance and parity\n\nCard: {s['card']} (sm_{s['sm']}). Medians over {s['steps']} steps.\n\n",
          "One conduction term (`d T = diffuse(k, T)`, conserves T, a range) on N³, times on the card:\n\n",
          "| N³ | copy GB/s | handwritten ms (GB/s at 8 B/cell) | engine ms, contracts off | engine ms, contracts on | step, every step read back | step, lazy readback: p50 / p99 / p99.9 | CPU ms | max abs diff engine vs handwritten |\n",
          "|---|---|---|---|---|---|---|---|---|\n"]
    for r in s["sizes"]:
        md.append(f"| {r['n']}³ | {r['copy_gbs']:.0f} | {r['hand_ms']:.3f} ({r['hand_gbs_8b']:.0f}) | {r['engine_card_ms_no_contracts']:.3f} | "
                  f"{r['engine_card_ms']:.3f} | {r['engine_step_ms']:.2f} | {r['lazy_p50_ms']:.2f} / {r['lazy_p99_ms']:.2f} / {r['lazy_p999_ms']:.2f} "
                  f"| {r['cpu_ms']:.1f} | {r['max_abs_diff_K']:.1e} K |\n")
    md.append("\nThe first three are card time (CUDA events); the steps are wall time, the lazy ones over "
              f"{s['sizes'][-1]['tail_steps']}+ steps with contracts off (the latency tail of a solver loop).\n")
    c, g = world["cpu"], world["cuda"]
    md.append(f"\nThe shipped world law (43 terms, 39 fields) on 144 × 144 × 48 (1 M cells), median step: CPU (OpenMP) {c['median_ms']:.1f} ms, "
              f"CUDA {g['median_ms']:.1f} ms (lawc: lazy readback, contracts checked every step) - {c['median_ms'] / g['median_ms']:.0f}x.\n\n")
    md.append("What the learned operators cost on the card a step (sum of the terms' card time), against what they replace:\n\n"
              "| run | card ms a step |\n|---|---|\n")
    for name, ms in learned.items():
        md.append(f"| {name} | {ms:.3f} |\n")
    md.append("\n")
    md.append(f"Parity: CUDA against the CPU reference after 6 steps of the whole law, worst relative difference {out['cuda_vs_cpu_worst_relative']:.1e}; "
              f"the JAX twin against the CPU reference, {out['jax_vs_cpu_worst_relative']:.1e}.\n")
    save("perf", out, "".join(md))
    print("".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

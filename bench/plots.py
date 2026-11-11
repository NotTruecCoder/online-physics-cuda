"""The figures of the README and the report, from bench/results/*.json, as SVG in docs/figures/.

  convergence.svg  error against lattice size: diffusion, advection, Black-Scholes, Heston, with their orders
  pareto.svg       error against card time a step for the learned models and what they replace
  roofline.svg     the stencil in the engine and other tools against the card's bandwidth and compute roofs
  tails.svg        the step's latency distribution (lazy readback), p50 / p99 / p99.9
  profile.svg      where the card's time goes in a step of the shipped world law, term by term

    python bench/plots.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "docs", "figures")

# The reference categorical order (dataviz palette, light mode), text in ink tokens, recessive grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
    "svg.fonttype": "none", "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def load(name):
    with open(os.path.join(RESULTS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print("wrote docs/figures/" + name)


def slope_guide(ax, x, y0, order, label):
    x = np.asarray(x, float)
    ax.plot(x, y0 * (x / x[0]) ** (-order), color=MUTED, lw=1, ls="--", zorder=1)
    ax.annotate(label, (x[-1], y0 * (x[-1] / x[0]) ** (-order)), xytext=(4, -2), textcoords="offset points", color=MUTED, fontsize=8)


def convergence():
    v, f = load("verify"), load("finance")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = [("diffusion (exact solution)", v["diffusion_cuda"]["n"], v["diffusion_cuda"]["l2"]),
            ("upwind advection", v["advection_cuda"]["n"], v["advection_cuda"]["l2"]),
            ("Black-Scholes put (formula)", [r["n"] for r in f["european_cuda"]["rows"]], [r["rms"] for r in f["european_cuda"]["rows"]]),
            ("Heston put (char. function)", [r["nx"] for r in f["heston"]["rows"]], [abs(r["err"]) for r in f["heston"]["rows"]])]
    for i, (name, n, e) in enumerate(rows):
        ax.loglog(n, e, "o-", color=SERIES[i], label=name, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
    slope_guide(ax, [16, 256], 4e-3, 2, "order 2")
    slope_guide(ax, [64, 1024], 1.6e-1, 1, "order 1")
    ax.set_xlabel("cells along x")
    ax.set_ylabel("error (L2 or at the quote)")
    ax.set_title("Generated CUDA converges at the scheme's order")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "convergence.svg")


def pareto():
    m2, m3, perf = load("model2"), load("model3"), load("perf")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    # Model 2: test-IC mean RMSE (learned: mean over seeds) against card time a step.
    tests = [str(t) for t in m2["test"]]
    mean = lambda d: float(np.mean([d[t]["rmse"] for t in tests]))  # noqa: E731
    learned = float(np.mean([mean(m2["results"]["learned"][s]) for s in m2["results"]["learned"]]))
    pts = [("plain coarse", m2["cost"]["coarse plain"]["card_ms"], mean(m2["results"]["plain"]), SERIES[1]),
           ("fitted coarse", m2["cost"]["coarse plain"]["card_ms"], mean(m2["results"]["calibrated"]), SERIES[3]),
           ("coarse + learned flux", m2["cost"]["coarse + learned flux"]["card_ms"], learned, SERIES[0]),
           ("fine lattice (the reference)", m2["cost"]["fine"]["card_ms"], 0.0, SERIES[2])]
    ax = axes[0]
    for name, x, y, c in pts:
        ax.scatter(x, y, s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(6, -11 if name == "plain coarse" else 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("card ms a step")
    ax.set_ylabel("T RMSE on unseen ICs (K)")
    ax.set_title("Model 2: coarse-lattice correction")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-8, 125)
    # Model 3: held-out RMSE against card time.
    res = m3["results"]
    held = "held-out (training generator)"
    cl = float(np.mean(res["classical, 2 substeps"][held]["rmse"]))
    ln = float(np.mean([np.mean(s[held]["rmse"]) for s in res["learned, 1 pass"].values()]))  # mean over seeds
    cost = perf["learned_cost_ms"]
    ax = axes[1]
    for name, x, y, c in [("classical, 2 substeps", cost["heat: classical, 2 substeps"], cl, SERIES[1]),
                          ("learned, 1 pass at 2x step", cost["heat: learned ldiffuse, 1 pass"], ln, SERIES[0])]:
        ax.scatter(x, y, s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK, ha="left" if x < 4 else "right")
    ax.annotate("classical, 1 pass: diverges", (0.3, 0.33), fontsize=8.5, color=INK2)
    ax.set_xlabel("card ms a step")
    ax.set_ylabel("T RMSE, held-out fields (K)")
    ax.set_title("Model 3: learned discretization")
    ax.set_ylim(0, 0.38)
    ax.set_xlim(-0.3, 9.5)
    fig.suptitle("Lower left is better: accuracy against cost, out of sample", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "pareto.svg")


def roofline():
    fw = load("frameworks")
    perf = load("perf")["stencil"]
    copy = max(r["copy_gbs"] for r in perf["sizes"])
    peak = 2 * 3840 * 1.425  # GFLOP/s, FP32 FMA: 3840 cores at the 1.425 GHz boost of an RTX 3060 Laptop
    flops = 14.0  # a cell: 6 differences, 6 adds, a multiply and an add
    ai = flops / 8.0  # at the minimum traffic, one read and one write of a float
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.logspace(-1, 2, 200)
    ax.loglog(x, np.minimum(x * copy, peak), color=INK2, lw=1.5, zorder=2)
    ax.annotate(f"copy bandwidth {copy:.0f} GB/s", (0.12, 0.12 * copy * 1.25), color=INK2, fontsize=8.5, rotation=28)
    ax.annotate(f"FP32 peak {peak / 1000:.1f} TFLOP/s", (12, peak * 1.12), color=INK2, fontsize=8.5)
    rows = fw["sizes"]["256"]["rows"]
    order = ["handwritten CUDA (ping-pong)", "Warp (CUDA)", "engine, lazy readback, contracts off", "Taichi (CUDA)", "PyTorch eager (CUDA)"]
    names = {"handwritten CUDA (ping-pong)": "handwritten CUDA", "engine, lazy readback, contracts off": "this engine",
             "Warp (CUDA)": "Warp", "Taichi (CUDA)": "Taichi", "PyTorch eager (CUDA)": "PyTorch eager"}
    for i, name in enumerate(order):
        r = next((r for r in rows if r["name"] == name and r["ms"]), None)
        if r is None:
            continue
        gflops = flops * 256 ** 3 / (r["ms"] * 1e6)
        ax.scatter(ai, gflops, s=70, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"{names[name]}: {r['ms']:.2f} ms")
    ax.axvline(ai, color=GRID, lw=1, zorder=1)
    ax.set_xlabel("arithmetic intensity (FLOP per byte, at 8 bytes a cell)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("7-point stencil on 256³: every tool is memory-bound")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(0.1, 100)
    save(fig, "roofline.svg")


def tails():
    perf = load("perf")["stencil"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    labels, y = [], []
    for k, r in enumerate(perf["sizes"]):
        for j, (q, c) in enumerate([("lazy_p50_ms", SERIES[0]), ("lazy_p99_ms", SERIES[1]), ("lazy_p999_ms", SERIES[6])]):
            ax.scatter(r[q], k, s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=["p50", "p99", "p99.9"][j] if k == 0 else None)
        ax.plot([r["lazy_p50_ms"], r["lazy_p999_ms"]], [k, k], color=GRID, lw=2, zorder=1)
        labels.append(f"{r['n']}³")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlabel("wall ms a step (lazy readback, contracts off)")
    ax.set_title("Step latency: median and tail")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=3, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "tails.svg")


def profile():
    terms = load("perf")["world"]["cuda"]["terms"]
    items = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)
    top, rest = items[:10], items[10:]
    names = [n.replace("_", " ") for n, _ in top] + [f"{len(rest)} other terms"]
    values = [v for _, v in top] + [sum(v for _, v in rest)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=SERIES[0], height=0.62, zorder=3)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, -3), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlabel("card ms a step")
    total = sum(terms.values())
    ax.set_title(f"World law, 43 terms on 1 M cells: {total:.1f} ms of terms a step")
    ax.grid(axis="y", visible=False)
    save(fig, "profile.svg")


if __name__ == "__main__":
    convergence()
    pareto()
    roofline()
    tails()
    profile()

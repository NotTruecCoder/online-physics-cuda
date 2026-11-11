"""FRAMEWORKS: one stencil, the same step, in the engine and in the tools people would use instead.

The step is bench_stencil's: explicit 7-point heat on N^3 with zero-flux edges (the edge cell repeats), float32,
T' = T + c * sum(neighbour - T). The least traffic it can have is one read and one write of T: 8 bytes a cell. Each
implementation is timed the same way, as the median wall time of a step with the device synchronised after it, after
warm-up (so a launch's overhead counts, as it does for a solver loop):

  engine        the law compiled by the engine (bench_stencil: lazy readback, contracts off; and with contracts)
  handwritten   bench_stencil's fused CUDA kernel, ping-pong buffers
  Taichi        @ti.kernel over an ndrange, CUDA backend
  Warp          @wp.kernel over a 3-D launch
  PyTorch       eager, slicing on a replicate-padded tensor (cuda)
  Numba         @njit(parallel=True) on the CPU (numba.cuda needs the CUDA toolkit's NVVM, which is not assumed here)
  NumPy         one thread, the CPU baseline

Every result is compared with the engine's after the same steps. A framework that is not installed is reported as
such. Writes bench/results/frameworks.{json,md}.

    python bench/frameworks.py [--sizes 128,256] [--steps 50]
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

from common import BUILD, binary, save

C = 0.15  # k dt / h^2: stable (< 1/6)


def start_values(n):
    i = np.arange(n * n * n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = i * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(31)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(29)
    return (290.0 + 1000.0 * (h >> np.uint64(40)).astype(np.float32) / 16777216.0).astype(np.float32).reshape(n, n, n)


def numpy_step(t):
    p = np.pad(t, 1, mode="edge")
    s = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:]
         - np.float32(6.0) * t)
    return (t + np.float32(C) * s).astype(np.float32)


def timed(step_fn, sync, steps, warmup=5):
    for _ in range(warmup):
        step_fn()
    sync()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step_fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def run_torch(t0, steps):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    a = torch.from_numpy(t0).to(dev)
    c = torch.tensor(C, device=dev)

    def step():
        nonlocal a
        p = F.pad(a[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
        s = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:]
             - 6.0 * a)
        a = a + c * s

    ms = timed(step, torch.cuda.synchronize, steps)
    a = torch.from_numpy(t0).to(dev)
    for _ in range(steps):
        step()
    return ms, a.cpu().numpy(), f"torch {torch.__version__}"


def run_taichi(t0, steps):
    import taichi as ti
    ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=True, log_level=ti.ERROR)
    n = t0.shape[0]
    a = ti.field(ti.f32, shape=(n, n, n))
    b = ti.field(ti.f32, shape=(n, n, n))

    @ti.kernel
    def heat(src: ti.template(), dst: ti.template()):
        for z, y, x in src:
            t = src[z, y, x]
            s = 0.0
            s += (src[z, y, x - 1] if x > 0 else t) - t
            s += (src[z, y, x + 1] if x + 1 < n else t) - t
            s += (src[z, y - 1, x] if y > 0 else t) - t
            s += (src[z, y + 1, x] if y + 1 < n else t) - t
            s += (src[z - 1, y, x] if z > 0 else t) - t
            s += (src[z + 1, y, x] if z + 1 < n else t) - t
            dst[z, y, x] = t + C * s

    state = {"flip": False}

    def step():
        if state["flip"]:
            heat(b, a)
        else:
            heat(a, b)
        state["flip"] = not state["flip"]

    a.from_numpy(t0)
    ms = timed(step, ti.sync, steps)
    a.from_numpy(t0)
    state["flip"] = False
    for _ in range(steps):
        step()
    out = (b if state["flip"] else a).to_numpy()
    return ms, out, f"taichi {ti.__version__}"


def run_warp(t0, steps):
    import warp as wp
    wp.config.quiet = True
    wp.init()
    n = t0.shape[0]

    @wp.kernel
    def heat(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32), m: int, c: float):
        z, y, x = wp.tid()
        t = src[z, y, x]
        s = float(0.0)
        s += wp.where(x > 0, src[z, y, wp.max(x - 1, 0)], t) - t
        s += wp.where(x + 1 < m, src[z, y, wp.min(x + 1, m - 1)], t) - t
        s += wp.where(y > 0, src[z, wp.max(y - 1, 0), x], t) - t
        s += wp.where(y + 1 < m, src[z, wp.min(y + 1, m - 1), x], t) - t
        s += wp.where(z > 0, src[wp.max(z - 1, 0), y, x], t) - t
        s += wp.where(z + 1 < m, src[wp.min(z + 1, m - 1), y, x], t) - t
        dst[z, y, x] = t + c * s

    a = wp.array(t0, dtype=wp.float32, device="cuda")
    b = wp.zeros_like(a)
    bufs = [a, b]
    state = {"i": 0}

    def step():
        i = state["i"]
        wp.launch(heat, dim=(n, n, n), inputs=[bufs[i], bufs[1 - i], n, C], device="cuda")
        state["i"] = 1 - i

    ms = timed(step, wp.synchronize, steps)
    bufs[0] = wp.array(t0, dtype=wp.float32, device="cuda")
    state["i"] = 0
    for _ in range(steps):
        step()
    return ms, bufs[state["i"]].numpy(), f"warp {wp.__version__}"


def run_numba(t0, steps):
    import numba
    from numba import njit, prange

    @njit(parallel=True, fastmath=False)
    def heat(src, dst, c):
        n = src.shape[0]
        for z in prange(n):
            for y in range(n):
                for x in range(n):
                    t = src[z, y, x]
                    s = np.float32(0.0)
                    s += (src[z, y, x - 1] if x > 0 else t) - t
                    s += (src[z, y, x + 1] if x + 1 < n else t) - t
                    s += (src[z, y - 1, x] if y > 0 else t) - t
                    s += (src[z, y + 1, x] if y + 1 < n else t) - t
                    s += (src[z - 1, y, x] if z > 0 else t) - t
                    s += (src[z + 1, y, x] if z + 1 < n else t) - t
                    dst[z, y, x] = t + c * s

    a, b = t0.copy(), np.empty_like(t0)
    c = np.float32(C)
    bufs = [a, b]
    state = {"i": 0}

    def step():
        i = state["i"]
        heat(bufs[i], bufs[1 - i], c)
        state["i"] = 1 - i

    ms = timed(step, lambda: None, max(3, steps // 10), warmup=2)
    bufs[0], bufs[1] = t0.copy(), np.empty_like(t0)
    state["i"] = 0
    for _ in range(steps):
        step()
    return ms, bufs[state["i"]], f"numba {numba.__version__}, {numba.config.NUMBA_NUM_THREADS} threads"


def run_numpy(t0, steps):
    state = {"t": t0.copy()}

    def step():
        state["t"] = numpy_step(state["t"])

    ms = timed(step, lambda: None, 3, warmup=1)
    t = t0.copy()
    for _ in range(steps):
        t = numpy_step(t)
    return ms, t, f"numpy {np.__version__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    out = subprocess.run([binary("bench_stencil"), "--sizes", a.sizes, "--steps", str(a.steps)], cwd=BUILD, capture_output=True, text=True, check=True)
    engine = {row["n"]: row for row in json.loads(out.stdout)["sizes"]}

    runners = [("Taichi (CUDA)", run_taichi), ("Warp (CUDA)", run_warp), ("PyTorch eager (CUDA)", run_torch),
               ("Numba parallel (CPU)", run_numba), ("NumPy (CPU, 1 thread)", run_numpy)]
    results = {"card": json.loads(out.stdout)["card"], "sizes": {}}
    for n in sizes:
        t0 = start_values(n)
        e = engine[n]
        rows = [{"name": "engine, lazy readback, contracts off", "ms": e["lazy_p50_ms"], "version": "this repo"},
                {"name": "engine, contracts on, every step read back", "ms": e["engine_step_ms"], "version": "this repo"},
                {"name": "handwritten CUDA (ping-pong)", "ms": e["hand_ms"], "version": "bench_stencil", "card_only": True}]
        for name, fn in runners:
            try:
                ms, result, version = fn(t0, a.steps)
            except Exception as ex:  # not installed, or no such device
                rows.append({"name": name, "ms": None, "version": f"not run: {type(ex).__name__}: {str(ex)[:120]}"})
                print(f"n={n} {name}: not run ({ex})")
                continue
            rows.append({"name": name, "ms": ms, "version": version, "result": result})
            print(f"n={n} {name}: {ms:.3f} ms a step ({version})")
        # Agreement: every framework against NumPy, after the same steps.
        base = next((r["result"] for r in rows if r["name"].startswith("NumPy") and "result" in r), None)
        for r in rows:
            if "result" in r:
                r["max_abs_diff_vs_numpy_K"] = float(np.max(np.abs(r.pop("result") - base))) if base is not None else None
            r["gbs_8b"] = 8.0 * n ** 3 / (r["ms"] * 1e6) if r["ms"] else None
        results["sizes"][str(n)] = {"rows": rows, "copy_gbs": e["copy_gbs"], "engine_max_abs_diff_vs_hand_K": e["max_abs_diff_K"]}

    md = [f"# The same stencil in the engine and in other tools\n\nCard: {results['card']}. 7-point explicit heat on N³, float32, "
          f"{a.steps} steps; median wall time of a step with the device synchronised after it (launch overheads count). "
          "Effective bandwidth at the minimum 8 bytes a cell.\n\n"]
    for n, block in results["sizes"].items():
        md.append(f"## {n}³ (copy kernel: {block['copy_gbs']:.0f} GB/s)\n\n| implementation | ms a step | GB/s at 8 B/cell | max abs diff vs NumPy | version |\n|---|---|---|---|---|\n")
        for r in sorted(block["rows"], key=lambda r: r["ms"] if r["ms"] else 1e9):
            ms = f"{r['ms']:.3f}" if r["ms"] else "-"
            gbs = f"{r['gbs_8b']:.0f}" if r.get("gbs_8b") else "-"
            diff = r.get("max_abs_diff_vs_numpy_K")
            md.append(f"| {r['name']} | {ms} | {gbs} | {'-' if diff is None else f'{diff:.1e} K'} | {r['version']} |\n")
        md.append(f"\nThe engine against the handwritten kernel after the same steps: {block['engine_max_abs_diff_vs_hand_K']:.1e} K. "
                  "The handwritten row is card time (CUDA events); the others are wall time.\n\n")
    save("frameworks", results, "".join(md))


if __name__ == "__main__":
    os.chdir(BUILD)
    main()

"""Shared helpers of the benchmarks: where things are, running lawc, reading and writing states.

Every benchmark runs from the build directory (where lawc and the assets are) and writes its results to
bench/results/<name>.json and a short bench/results/<name>.md, which the README's report quotes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(HERE, "results")
EXE = ".exe" if os.name == "nt" else ""
CONFIG_DIRS = ("Release", "")

sys.path.insert(0, os.path.join(ROOT, "tools", "lawjax"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def binary(name: str) -> str:
    for sub in CONFIG_DIRS:
        path = os.path.join(BUILD, sub, name + EXE)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not built (cmake --build build --config Release)")


def lawc(*args: str, quiet: bool = True) -> str:
    """Runs lawc in the build directory; returns its stdout."""
    out = subprocess.run([binary("lawc"), *args], cwd=BUILD, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"lawc {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    if not quiet:
        print(out.stdout)
    return out.stdout


def lawc_json(*args: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.json")
        lawc(*args, "--json", path)
        with open(path) as f:
            return json.load(f)


def write_scene(path: str, grid, laws: Iterable[str], lines: Iterable[str] = ()) -> str:
    nx, ny, nz, h = grid
    with open(path, "w") as f:
        f.write(f"grid {nx} {ny} {nz} {h}\n")
        for law in laws:
            f.write(f"law {law}\n")
        for line in lines:
            f.write(line + "\n")
    return path


def write_state(directory: str, fields: Dict[str, np.ndarray], h: float) -> str:
    """The format lawc --dump writes and --init reads: DIR/meta.json, DIR/FIELD.bin (float32, x fastest)."""
    os.makedirs(directory, exist_ok=True)
    shape = next(iter(fields.values())).shape  # (nz, ny, nx)
    for name, arr in fields.items():
        np.asarray(arr, np.float32).reshape(shape).tofile(os.path.join(directory, name + ".bin"))
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump({"nx": shape[2], "ny": shape[1], "nz": shape[0], "h": h, "fields": list(fields)}, f)
    return directory


def read_state(directory: str):
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    shape = (meta["nz"], meta["ny"], meta["nx"])
    fields = {n: np.fromfile(os.path.join(directory, n + ".bin"), np.float32).reshape(shape) for n in meta["fields"]}
    return fields, shape, meta["h"]


def save(name: str, data: dict, markdown: Optional[str] = None) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    if markdown is not None:
        with open(os.path.join(RESULTS, name + ".md"), "w", encoding="utf-8") as f:
            f.write(markdown)
    print(f"wrote bench/results/{name}.json" + (" and .md" if markdown is not None else ""))


def mean_std(values) -> str:
    v = np.asarray(values, np.float64)
    return f"{v.mean():.3g} ± {v.std(ddof=1) if v.size > 1 else 0.0:.2g}"


def cuda_available() -> bool:
    """Whether lawc can run a law on CUDA here (a card, its driver and NVRTC): one step of a one-cell lattice."""
    law = os.path.join(HERE, "laws", "diffusion.law")
    try:
        out = subprocess.run([binary("lawc"), law, "--grid", "4", "1", "1", "1", "--steps", "1", "--backend", "cuda"], cwd=BUILD,
                             capture_output=True, text=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False

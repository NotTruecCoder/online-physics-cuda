"""JAX against the engine: the same law, the same start, N steps, field by field.

    cd build && python ../tools/lawjax/parity.py --scene assets/scenes/fire_test.scene --steps 5 --dt 0.25

lawc lowers the law to IR JSON and runs the CPU reference, dumping the first and the last state; lawjax runs the same
steps from the first state in JAX; the two are compared.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawjax.ir import load_law, read_state  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--lawc", default=os.path.join("Release", "lawc.exe"))
    p.add_argument("--learned", action="store_true")
    p.add_argument("--tolerance", type=float, default=1e-4)
    a = p.parse_args()
    work = tempfile.mkdtemp(prefix="lawjax_parity_")
    ir = os.path.join(work, "law.json")
    args = [a.lawc, "--scene", a.scene, "--ir-json", ir, "--steps", str(a.steps), "--dt", str(a.dt), "--dump", work]
    if a.learned:
        args.append("--learned")
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    law = load_law(ir)
    state, shape, h = read_state(os.path.join(work, "step_0"))
    want, _, _ = read_state(os.path.join(work, f"step_{a.steps}"))
    for _ in range(a.steps):
        state = law.step(state, a.dt, h, learned=a.learned)
    worst = 0.0
    for name in law.fields:
        got = np.asarray(state[name])
        ref = np.asarray(want[name])
        diff = float(np.max(np.abs(got - ref)))
        scale = max(1e-6, float(np.max(np.abs(ref))))
        worst = max(worst, diff / scale)
        if diff > 0:
            print(f"  {name:14s} max |jax - cpu| {diff:.3e}  ({diff / scale:.3e} of its largest)")
    print(f"{len(law.fields)} fields, {len(law.terms)} terms, {a.steps} steps: worst relative difference {worst:.3e}")
    print("parity ok" if worst <= a.tolerance else "PARITY BROKEN")
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())

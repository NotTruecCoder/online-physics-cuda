"""Networks of the law: the .net text format (src/field/nn/Net.hpp) and their forward pass in jnp.

The arithmetic is the engine's: y = net((x - in_shift) * in_scale) * out_scale, hidden layers through the activation,
the last one linear; W_l is (out x in), row-major in the file, then b_l.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

ACTS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "silu": lambda x: x * jax.nn.sigmoid(x),
    "linear": lambda x: x,
}


@dataclass
class Net:
    name: str
    sizes: List[int]
    act: str = "tanh"
    in_shift: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    in_scale: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    out_scale: float = 1.0
    weights: List[tuple] = field(default_factory=list)  # [(W, b), ...] as numpy

    def params(self):
        """The trainable part, as a pytree."""
        return [(jnp.asarray(w), jnp.asarray(b)) for w, b in self.weights]

    def with_params(self, params):
        return Net(self.name, list(self.sizes), self.act, self.in_shift, self.in_scale, self.out_scale,
                   [(np.asarray(w), np.asarray(b)) for w, b in params])


def init_net(name: str, sizes: List[int], act: str = "tanh", seed: int = 0) -> Net:
    """Glorot-uniform, the last layer zero: an untrained net outputs exactly 0 (as the engine's init_net)."""
    rng = np.random.default_rng(seed)
    weights = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        limit = np.sqrt(6.0 / (a + b))
        w = np.zeros((b, a), np.float32) if last else rng.uniform(-limit, limit, (b, a)).astype(np.float32)
        weights.append((w, np.zeros(b, np.float32)))
    return Net(name, list(sizes), act, np.zeros(sizes[0], np.float32), np.ones(sizes[0], np.float32), 1.0, weights)


def forward(net: Net, params, x):
    """x: (..., in) -> (..., out)."""
    act = ACTS[net.act]
    h = (x - jnp.asarray(net.in_shift)) * jnp.asarray(net.in_scale)
    for i, (w, b) in enumerate(params):
        h = h @ w.T + b
        if i + 1 < len(params):
            h = act(h)
    return h * net.out_scale


def read_net(path: str, name: str = "") -> Net:
    words = open(path).read().split()
    i = 0
    sizes, shift, scale, flat = [], [], [], []
    act, out_scale, section = "tanh", 1.0, ""
    while i < len(words):
        w = words[i]
        i += 1
        if w == "net":
            name = words[i]
            i += 1
        elif w in ("sizes", "in_shift", "in_scale", "weights"):
            section = w
        elif w == "act":
            act = words[i]
            i += 1
        elif w == "out_scale":
            out_scale = float(words[i])
            i += 1
        else:
            {"sizes": sizes, "in_shift": shift, "in_scale": scale, "weights": flat}[section].append(float(w))
    sizes = [int(s) for s in sizes]
    weights, at = [], 0
    for a, b in zip(sizes[:-1], sizes[1:]):
        wm = np.asarray(flat[at:at + a * b], np.float32).reshape(b, a)
        at += a * b
        bv = np.asarray(flat[at:at + b], np.float32)
        at += b
        weights.append((wm, bv))
    return Net(name, sizes, act, np.asarray(shift, np.float32), np.asarray(scale, np.float32), out_scale, weights)


def write_net(net: Net, path: str) -> None:
    flat = []
    for w, b in net.weights:
        flat.extend(np.asarray(w, np.float32).reshape(-1).tolist())
        flat.extend(np.asarray(b, np.float32).reshape(-1).tolist())
    with open(path, "w") as out:
        out.write(f"net {net.name}\nsizes {' '.join(str(s) for s in net.sizes)}\nact {net.act}\n")
        out.write("in_shift " + " ".join(f"{v:.9g}" for v in np.asarray(net.in_shift).tolist()) + "\n")
        out.write("in_scale " + " ".join(f"{v:.9g}" for v in np.asarray(net.in_scale).tolist()) + "\n")
        out.write(f"out_scale {net.out_scale:.9g}\nweights\n")
        for k in range(0, len(flat), 8):
            out.write(" ".join(f"{v:.9g}" for v in flat[k:k + 8]) + "\n")

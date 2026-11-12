"""The law's IR (lawc --ir-json) as a JAX program.

Every instruction of the C++ IR has its jnp twin here with the engine's semantics: stencils clamp at the edge (the edge
cell repeats), diffuse is the flux form with the harmonic mean of k, adv is upwind, fall/spread/topple move matter by
antisymmetric fluxes, flux() and ldiffuse() are the learned face fluxes (antisymmetric by construction, flux() limited).
A step is the engine's: terms in order (Lie splitting), each with its substeps, every equation of a term reading the same
pre-state, then the derives.

Arrays are (nz, ny, nx): the engine's linear order, x fastest, so np.fromfile(...).reshape(nz, ny, nx) is a field.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .nets import Net, forward, init_net, read_net

AXIS = {0: 2, 1: 1, 2: 0}  # the engine's x, y, z -> array axes


def sh(a, axis, d):
    """a at index + d along an engine axis, clamped at the edge."""
    ax = AXIS[axis]
    n = a.shape[ax]
    idx = jnp.clip(jnp.arange(n) + d, 0, n - 1)
    return jnp.take(a, idx, axis=ax)


def index_along(shape, axis):
    ax = AXIS[axis]
    n = shape[ax]
    view = [1, 1, 1]
    view[ax] = n
    return jnp.arange(n).reshape(view), n


def is_cell(v):
    return hasattr(v, "ndim") and v.ndim == 3


def as_cell(v, shape):
    return v if is_cell(v) else jnp.full(shape, v, jnp.float32)


def divergence(faces, shape, h):
    """faces[a]: the flux from each cell through its +a face (0 where there is no face). In minus out, over h."""
    out = jnp.zeros(shape, jnp.float32)
    for a in range(3):
        f = faces[a]
        idx, n = index_along(shape, a)
        incoming = jnp.where(idx > 0, sh(f, a, -1), 0.0)
        outgoing = jnp.where(idx + 1 < n, f, 0.0)
        out = out + incoming - outgoing
    return out / h


def limit_face(r, fi, fj, lo, hi, dt, h):
    if dt <= 0:
        return r
    per = 0.999 * h / (6.0 * dt)
    pos = r
    neg = r
    if lo > -1e29:
        pos = jnp.minimum(pos, jnp.maximum(0.0, (fi - lo) * per))
        neg = jnp.maximum(neg, -jnp.maximum(0.0, (fj - lo) * per))
    if hi < 1e29:
        pos = jnp.minimum(pos, jnp.maximum(0.0, (hi - fj) * per))
        neg = jnp.maximum(neg, -jnp.maximum(0.0, (hi - fi) * per))
    return jnp.where(r > 0, pos, jnp.where(r < 0, neg, r))


def edt_axis(f, axis):
    """Squared distance along one axis: d[q] = min_r f[r] + (q - r)^2 (exact)."""
    ax = AXIS[axis]
    g = jnp.moveaxis(f, ax, -1)
    n = g.shape[-1]
    q = jnp.arange(n, dtype=jnp.float32)
    cost = (q[:, None] - q[None, :]) ** 2  # (q, r)
    out = jax.lax.map(lambda row: jnp.min(row[..., None, :] + cost, axis=-1), g)
    return jnp.moveaxis(out, -1, ax)


class Law:
    def __init__(self, ir: dict, base_dir: str = "."):
        self.ir = ir
        self.fields = [f["name"] for f in ir["fields"]]
        self.decl = ir["fields"]
        self.param_names = [p["name"] for p in ir["params"]]
        self.param_values = {p["name"]: p["value"] for p in ir["params"]}
        self.insts = ir["insts"]
        self.terms = ir["terms"]
        self.learned_terms = ir["learned"]
        self.derives = ir["derives"]
        self.nets: Dict[str, Net] = {}
        for n in ir["nets"]:
            path = n["file"] if os.path.isabs(n["file"]) else os.path.join(base_dir, n["file"])
            self.nets[n["name"]] = read_net(path, n["name"]) if os.path.exists(path) else init_net(n["name"], n["sizes"], n["act"])
        self.net_order = [n["name"] for n in ir["nets"]]

    # ---- state -----------------------------------------------------------------------------------------------------

    def initial(self, shape):
        return {f["name"]: jnp.full(shape, f["init"], jnp.float32) for f in self.decl}

    def net_params(self):
        return {name: net.params() for name, net in self.nets.items()}

    # ---- one value -------------------------------------------------------------------------------------------------

    def _eval(self, i, ctx):
        memo = ctx["memo"]
        if i in memo:
            return memo[i]
        inst = self.insts[i]
        op = inst["op"]
        args = inst["args"]
        a = lambda k: self._eval(args[k], ctx)  # noqa: E731
        shape = ctx["shape"]
        h = ctx["h"]
        state = ctx["state"]
        if op == "const":
            v = jnp.float32(inst["number"])
        elif op == "param":
            v = jnp.float32(ctx["params"][self.param_names[inst["index"]]])
        elif op == "h":
            v = jnp.float32(h)
        elif op == "dt":
            v = jnp.float32(ctx["dt"])
        elif op == "tridiag":
            # lo u[i-1] + di u[i] + up u[i+1] = rhs along the axis, the edge cell repeating past the ends (its missing
            # neighbour's coefficient on the diagonal), as the engine solves it. Differentiable (JAX's own rules).
            ax = AXIS[inst["index"]]
            lo, di, up, rhs = (jnp.moveaxis(as_cell(a(k), shape), ax, -1) for k in range(4))
            n = lo.shape[-1]
            first = (jnp.arange(n) == 0).astype(jnp.float32)
            last = (jnp.arange(n) == n - 1).astype(jnp.float32)
            u = jax.lax.linalg.tridiagonal_solve(lo * (1 - first), di + lo * first + up * last, up * (1 - last), rhs[..., None])[..., 0]
            v = jnp.moveaxis(u, -1, ax)
        elif op in ("shift_t", "gx_t", "gy_t", "gz_t"):
            # Transposes of shift and gx/gy/gz: each value added back into the cell its stencil read it from.
            def transpose(f, axis, d):
                ax = AXIS[axis]
                n = f.shape[ax]
                g = jnp.moveaxis(f, ax, -1)
                idx = jnp.clip(jnp.arange(n) + d, 0, n - 1)
                return jnp.moveaxis(jnp.zeros_like(g).at[..., idx].add(g), -1, ax)

            f = as_cell(a(0), shape)
            if op == "shift_t":
                v = f
                for ax in range(3):
                    if inst["offset"][ax] != 0:
                        v = transpose(v, ax, inst["offset"][ax])
            else:
                ax = {"gx_t": 0, "gy_t": 1, "gz_t": 2}[op]
                idx, n = index_along(shape, ax)
                span = (jnp.minimum(idx + 1, n - 1) - jnp.maximum(idx - 1, 0)).astype(jnp.float32)
                w = jnp.where(span > 0, f / (h * jnp.maximum(span, 1.0)), 0.0)
                v = transpose(w, ax, 1) - transpose(w, ax, -1)
        elif op == "rim":
            z, y, x = [jnp.arange(n) for n in shape]
            v = ((x[None, None, :] == 0) | (y[None, :, None] == 0) | (z[:, None, None] == 0) | (x[None, None, :] == shape[2] - 1)
                 | (y[None, :, None] == shape[1] - 1) | (z[:, None, None] == shape[0] - 1)).astype(jnp.float32)
        elif op == "local":
            v = jnp.float32(0.0)
        elif op == "field":
            v = state[self.fields[inst["index"]]]
        elif op == "neg":
            v = -a(0)
        elif op == "abs":
            v = jnp.abs(a(0))
        elif op == "sqrt":
            v = jnp.sqrt(jnp.maximum(0.0, a(0)))
        elif op == "exp":
            v = jnp.exp(jnp.minimum(a(0), 80.0))
        elif op == "log":
            v = jnp.log(jnp.maximum(a(0), 1e-30))
        elif op == "step":
            v = jnp.where(a(0) > 0, 1.0, 0.0).astype(jnp.float32)
        elif op == "pos":
            v = jnp.maximum(a(0), 0.0)
        elif op == "add":
            v = a(0) + a(1)
        elif op == "sub":
            v = a(0) - a(1)
        elif op == "mul":
            v = a(0) * a(1)
        elif op == "div":
            num, den = a(0), a(1)
            v = jnp.where(den == 0, 0.0, num / jnp.where(den == 0, 1.0, den))
        elif op == "pow":
            v = jnp.power(a(0), a(1))
        elif op == "min":
            v = jnp.minimum(a(0), a(1))
        elif op == "max":
            v = jnp.maximum(a(0), a(1))
        elif op == "clamp":
            lo = a(1)
            v = jnp.clip(a(0), lo, jnp.maximum(lo, a(2)))
        elif op == "shift":
            v = a(0)
            for ax in range(3):
                if inst["offset"][ax] != 0 and is_cell(v):
                    v = sh(v, ax, inst["offset"][ax])
        elif op == "lap":
            f = a(0)
            v = sum((sh(f, ax, -1) - f) + (sh(f, ax, 1) - f) for ax in range(3)) / (h * h) if is_cell(f) else jnp.float32(0)
        elif op in ("gx", "gy", "gz"):
            ax = {"gx": 0, "gy": 1, "gz": 2}[op]
            f = a(0)
            if not is_cell(f):
                v = jnp.float32(0)
            else:
                idx, n = index_along(shape, ax)
                span = (jnp.minimum(idx + 1, n - 1) - jnp.maximum(idx - 1, 0)).astype(jnp.float32)
                v = jnp.where(span > 0, (sh(f, ax, 1) - sh(f, ax, -1)) / (h * jnp.maximum(span, 1.0)), 0.0)
        elif op == "diffuse":
            k = as_cell(a(0), shape)
            f = a(1)
            faces = []
            for ax in range(3):
                kb, fb = sh(k, ax, 1), sh(f, ax, 1)
                s = k + kb
                kf = jnp.where(s > 0, 2 * k * kb / jnp.where(s > 0, s, 1.0), 0.0)
                faces.append(-(kf * (fb - f)))  # the flux out through +ax is k grad pointing down: -(k (f_b - f))
            v = divergence(faces, shape, h) / h
        elif op == "adv":
            f = a(0)
            speeds = [as_cell(a(1), shape), as_cell(a(2), shape), as_cell(a(3), shape)]
            total = 0.0
            for ax in range(3):
                idx, n = index_along(shape, ax)
                u = speeds[ax]
                up = jnp.where(u > 0, jnp.where(idx > 0, u * (f - sh(f, ax, -1)) / h, 0.0),
                               jnp.where(idx + 1 < n, u * (sh(f, ax, 1) - f) / h, 0.0))
                total = total + up
            v = -total
        elif op == "distance":
            f = as_cell(a(0), shape)
            inside = f >= 0.5
            far = float(sum(shape) * h)
            big = 1e30

            def squared(seed):
                d = jnp.where(seed, 0.0, big)
                for ax in range(3):
                    if shape[AXIS[ax]] > 1:
                        d = edt_axis(d, ax)
                return d

            to_in = squared(inside)
            to_out = squared(~inside)
            z, y, x = [jnp.arange(n, dtype=jnp.float32) for n in shape]
            border = jnp.full(shape, 1e30, jnp.float32)
            for n, c, view in ((shape[2], x, (1, 1, -1)), (shape[1], y, (1, -1, 1)), (shape[0], z, (-1, 1, 1))):
                if n > 1:
                    border = jnp.minimum(border, jnp.minimum(c + 1, n - c).reshape(view))
            out_cells = jnp.minimum(jnp.where(to_out < 1e29, jnp.sqrt(to_out), far / h), border)
            v = jax.lax.stop_gradient(jnp.where(inside, -(out_cells - 0.5) * h, jnp.where(to_in < 1e29, (jnp.sqrt(to_in) - 0.5) * h, far)))
        elif op == "poisson":
            src = as_cell(a(0), shape)
            key = f"__poisson_{i}"
            phi = state.get(key, jnp.zeros(shape, jnp.float32))
            axes = sum(1 for n in shape if n > 1)
            for _ in range(inst["iterations"]):
                total = jnp.zeros(shape, jnp.float32)
                for ax in range(3):
                    if shape[AXIS[ax]] > 1:
                        idx, n = index_along(shape, ax)
                        total = total + jnp.where(idx > 0, sh(phi, ax, -1), 0.0) + jnp.where(idx + 1 < n, sh(phi, ax, 1), 0.0)
                phi = (total - h * h * src) / (2.0 * max(axes, 1))
            ctx["carry"][key] = phi
            v = phi
        elif op in ("maxof", "minof", "sumof"):
            f = a(0)
            if not is_cell(f):
                v = f * float(np.prod(shape)) if op == "sumof" else f
            else:
                v = {"maxof": jnp.max, "minof": jnp.min, "sumof": jnp.sum}[op](f)
        elif op == "fall":
            f, b, speed = a(0), as_cell(a(1), shape), as_cell(a(2), shape)
            idx, n = index_along(shape, 2)
            room = jnp.clip(1.0 - sh(b, 2, -1) - sh(f, 2, -1), 0.0, 1.0)
            flux = jnp.where(idx > 0, jnp.maximum(0.0, speed) / h * jnp.maximum(0.0, f) * room, 0.0)
            v = -flux + jnp.where(idx + 1 < n, sh(flux, 2, 1), 0.0)
        elif op == "spread":
            f, b, k = a(0), as_cell(a(1), shape), as_cell(a(2), shape)
            out = jnp.zeros(shape, jnp.float32)
            for ax in (1, 0):  # the engine adds the -y face, then -x, then takes +x, then +y
                idx, n = index_along(shape, ax)
                fc, bc, kc = sh(f, ax, 1), sh(b, ax, 1), sh(k, ax, 1)
                into_c = f > fc
                room = jnp.clip(1.0 - jnp.where(into_c, bc, b) - jnp.where(into_c, fc, f), 0.0, 1.0)
                kf = 0.5 * (jnp.maximum(0.0, k) + jnp.maximum(0.0, kc))
                face = jnp.where(idx + 1 < n, kf / (h * h) * (f - fc) * room, 0.0)
                out = out + jnp.where(idx > 0, sh(face, ax, -1), 0.0) - face
            v = out
        elif op == "topple":
            f, b, rate = a(0), as_cell(a(1), shape), as_cell(a(2), shape)
            room = jnp.clip(1.0 - b - f, 0.0, 1.0)
            zi, nz = index_along(shape, 2)
            rests = (f > 0) & (zi > 0) & (sh(room, 2, -1) <= 0.01)
            out = jnp.zeros(shape, jnp.float32)
            sends = []
            for ax, d in ((0, -1), (0, 1), (1, -1), (1, 1)):
                idx, n = index_along(shape, ax)
                ok = (idx + d >= 0) & (idx + d < n)
                rj = sh(room, ax, d)
                rjd = sh(sh(room, 2, -1), ax, d)
                s = jnp.where(rests & ok, 0.25 * jnp.maximum(0.0, rate) / h * f * jnp.minimum(rj, rjd), 0.0)
                sends.append((ax, d, s))
                out = out - s
            for ax, d, s in sends:
                # what slid along d from the cell above-and-behind lands here
                idx, n = index_along(shape, ax)
                src_ok = (idx - d >= 0) & (idx - d < n) & (zi + 1 < nz)
                out = out + jnp.where(src_ok, sh(sh(s, 2, 1), ax, -d), 0.0)
            v = out
        elif op == "mlp":
            net = self.nets[self.net_order[inst["index"]]]
            xs = jnp.stack([as_cell(a(k), shape) for k in range(len(args))], axis=-1)
            v = forward(net, ctx["nets"][net.name], xs)[..., 0]
        elif op == "flux":
            net = self.nets[self.net_order[inst["index"]]]
            m = len(args) - 2
            lo = self.insts[args[m]]["number"]
            hi = self.insts[args[m + 1]]["number"]
            vals = [as_cell(a(k), shape) for k in range(m)]
            faces = []
            for ax in range(3):
                idx, n = index_along(shape, ax)
                nb = [sh(q, ax, 1) for q in vals]
                gij = forward(net, ctx["nets"][net.name], jnp.stack(vals + nb, -1))[..., 0]
                gji = forward(net, ctx["nets"][net.name], jnp.stack(nb + vals, -1))[..., 0]
                r = limit_face(0.5 * (gij - gji), vals[0], nb[0], lo, hi, ctx["dt"], h)
                faces.append(jnp.where(idx + 1 < n, r, 0.0))
            v = divergence(faces, shape, h)
        elif op == "ldiffuse":
            net = self.nets[self.net_order[inst["index"]]]
            k = as_cell(a(0), shape)
            f = a(1)
            faces = []
            base = jnp.asarray([0.0, -1.0, 1.0, 0.0], jnp.float32)
            for ax in range(3):
                idx, n = index_along(shape, ax)
                w = jnp.stack([sh(f, ax, -1), f, sh(f, ax, 1), sh(f, ax, 2)], -1)
                kb = sh(k, ax, 1)
                s = k + kb
                kf = jnp.where(s > 0, 2 * k * kb / jnp.where(s > 0, s, 1.0), 0.0)

                def raw(window):
                    centre = 0.5 * (window[..., 1] + window[..., 2])
                    spread = jnp.maximum(1e-6, jnp.max(jnp.abs(window - centre[..., None]), -1))
                    x = jnp.concatenate([(window - centre[..., None]) / spread[..., None],
                                         (kf * ctx["dt"] / (h * h))[..., None]], -1)
                    y = forward(net, ctx["nets"][net.name], x)
                    y = y - jnp.mean(y, -1, keepdims=True)
                    return -kf / h * jnp.sum((base + y) * window, -1)

                face = 0.5 * (raw(w) - raw(w[..., ::-1]))
                faces.append(jnp.where(idx + 1 < n, face, 0.0))
            v = divergence(faces, shape, h)
        else:
            raise ValueError(f"op {op} has no JAX twin")
        memo[i] = v
        return v

    # ---- a step ----------------------------------------------------------------------------------------------------

    def _context(self, state, shape, h, params, nets, dt):
        return {"memo": {}, "state": state, "shape": shape, "h": h, "params": params, "nets": nets, "dt": dt, "carry": {}}

    def max_dt(self, program, state, h, params=None, nets=None):
        if program["max_dt"] < 0:
            return math.inf
        shape = next(iter(state.values())).shape
        ctx = self._context(state, shape, h, params or self.param_values, nets or self.net_params(), 0.0)
        return float(self._eval(program["max_dt"], ctx))

    def substeps(self, state, dt, h, learned=False, params=None):
        """How many substeps each term takes now (eager: max_dt may read the fields)."""
        out = []
        for t, term in enumerate(self.terms):
            program = self.learned_terms[t] if learned and self.learned_terms[t]["equations"] else term
            limit = self.max_dt(program, state, h, params)
            out.append(max(1, int(math.ceil(dt / limit))) if math.isfinite(limit) and limit > 0 and dt > limit else 1)
        return out

    def step(self, state, dt, h, counts=None, params=None, nets=None, learned=False, only=None):
        """One step. `counts`: substeps per term (fixed: the step is then jit-able and differentiable)."""
        params = params or self.param_values
        nets = nets if nets is not None else self.net_params()
        if counts is None:
            counts = self.substeps(state, dt, h, learned, params)
        shape = next(v for k, v in state.items() if not k.startswith("__")).shape
        state = dict(state)
        for t, term in enumerate(self.terms):
            if only is not None and term["name"] not in only:
                continue
            program = self.learned_terms[t] if learned and self.learned_terms[t]["equations"] else term
            sub = dt / counts[t]
            for _ in range(counts[t]):
                ctx = self._context(state, shape, h, params, nets, sub)
                values = [self._eval(v, ctx) for _, v in program["equations"]]
                state.update(ctx["carry"])
                for (target, _), value in zip(program["equations"], values):
                    name = self.fields[target]
                    state[name] = state[name] + (value * sub).astype(jnp.float32)
            for target in program["writes"]:
                d = self.decl[target]
                if d["clamp"]:
                    state[d["name"]] = jnp.clip(state[d["name"]], d["lo"], d["hi"])
        for program in self.derives:
            ctx = self._context(state, shape, h, params, nets, 0.0)
            target, value = program["equations"][0]
            v = self._eval(value, ctx)
            state.update(ctx["carry"])
            name = self.fields[target]
            state[name] = as_cell((state[name] if program["accumulate"] else 0.0) + v, shape).astype(jnp.float32)
        return state


def load_law(path: str, base_dir: Optional[str] = None) -> Law:
    with open(path) as f:
        ir = json.load(f)
    return Law(ir, base_dir or os.getcwd())


def read_state(directory: str):
    """A state lawc --dump wrote: (fields dict, shape, h)."""
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    shape = (meta["nz"], meta["ny"], meta["nx"])
    state = {name: jnp.asarray(np.fromfile(os.path.join(directory, name + ".bin"), np.float32).reshape(shape)) for name in meta["fields"]}
    return state, shape, meta["h"]

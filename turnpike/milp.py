from math import inf
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import _core
from ._validate import integer, pack, scalar


class Fit(NamedTuple):
    status: str
    x: Optional[np.ndarray]
    fun: Optional[float]
    gap: Optional[float]
    bound: Optional[float]
    message: str


class _M:
    def __init__(self):
        self.c = []
        self.lo = []
        self.hi = []
        self.z = []
        self.r = []
        self.j = []
        self.a = []
        self.bl = []
        self.bu = []

    def var(self, lo=0.0, hi=inf, c=0.0, z=0):
        j = len(self.c)
        self.c.append(c)
        self.lo.append(lo)
        self.hi.append(hi)
        self.z.append(z)
        return ({j: 1.0}, 0.0)

    def add(self, e, lo=-inf, hi=inf):
        j = len(self.bl)
        for k, a in e[0].items():
            if a:
                self.r.append(j)
                self.j.append(k)
                self.a.append(a)
        self.bl.append(lo - e[1])
        self.bu.append(hi - e[1])

    def solve(self, options):
        A = coo_matrix(
            (self.a, (self.r, self.j)), shape=(len(self.bl), len(self.c))
        ).tocsc()
        A.indices = A.indices.astype(np.int32, copy=False)
        A.indptr = A.indptr.astype(np.int32, copy=False)
        return milp(
            np.asarray(self.c),
            integrality=np.asarray(self.z),
            bounds=Bounds(self.lo, self.hi),
            constraints=LinearConstraint(A, self.bl, self.bu),
            options=options,
        )


def _sum(*e):
    a = {}
    b = 0.0
    for x, s in e:
        b += s
        for j, v in x.items():
            a[j] = a.get(j, 0.0) + v
    return a, b


def _mul(a, e):
    return ({j: a * x for j, x in e[0].items()}, a * e[1])


def _const(a):
    return {}, float(a)


def _cmp(M, a, b, u):
    lo = M.var(0, u)
    hi = M.var(0, u)
    p = M.var(0, 1, z=1)
    M.add(_sum(lo, hi, _mul(-1, a), _mul(-1, b)), 0, 0)
    M.add(_sum(lo, _mul(-1, a)), hi=0)
    M.add(_sum(lo, _mul(-1, b)), hi=0)
    M.add(_sum(a, _mul(-u, p), _mul(-1, lo)), hi=0)
    M.add(_sum(b, _mul(u, p), _mul(-1, lo)), hi=u)
    M.add(_sum(a, _mul(-1, hi)), hi=0)
    M.add(_sum(b, _mul(-1, hi)), hi=0)
    M.add(_sum(hi, _mul(-1, a), _mul(u, p)), hi=u)
    M.add(_sum(hi, _mul(-1, b), _mul(-u, p)), hi=0)
    return lo, hi


def network(n):
    """Return the padded bitonic network size and comparator triples."""
    return _core.net(integer(n, "n", 1))


def _sort(M, w, u):
    n = len(w)
    N, C = network(len(w))
    w = list(w) + [_const(u)] * (N - len(w))
    for i, j, up in C:
        lo, hi = _cmp(M, w[i], w[j], u)
        w[i], w[j] = (lo, hi) if up else (hi, lo)
    return w[:n]


def best_fit(D, I, n, u=None, loss="l1", options=None):
    """Solve the bounded sorting-network MILP for independent cells."""
    n = integer(n, "n", 2)
    d, o, g = pack(D, I, n)
    if u is None:
        raise ValueError("u is required: it is the proved coordinate-span bound")
    u = scalar(u, "u", 0, strict=True)
    if loss not in ("l1", "linf"):
        raise ValueError("loss must be 'l1' or 'linf'")
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be a dictionary")

    M = _M()
    x = [M.var(0, 0)] + [M.var(0, u) for _ in range(n - 1)]
    for i in range(n - 1):
        M.add(_sum(x[i], _mul(-1, x[i + 1])), hi=0)

    P = _core.pairs(n)
    t = M.var(0, inf, c=1) if loss == "linf" else None
    for c in range(len(o) - 1):
        e = np.flatnonzero(g == c + 1)
        w = [_sum(x[P[k, 1]], _mul(-1, x[P[k, 0]])) for k in e]
        y = _sort(M, w, u)
        dc = np.sort(d[o[c] : o[c + 1]])
        for yi, di in zip(y, dc):
            r = M.var(0, inf, c=1) if loss == "l1" else t
            M.add(_sum(yi, _mul(-1, r)), hi=di)
            M.add(_sum(_mul(-1, yi), _mul(-1, r)), hi=-di)

    opt = {"mip_rel_gap": 0.0}
    if options is not None:
        opt.update(options)
    res = M.solve(opt)
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(
        res.status, "error"
    )
    xx = None if res.x is None else np.asarray(res.x[:n])
    fun = None if res.fun is None else float(res.fun)
    gap = getattr(res, "mip_gap", None)
    bound = getattr(res, "mip_dual_bound", None)
    gap = None if gap is None else float(gap)
    bound = None if bound is None else float(bound)
    return Fit(status, xx, fun, gap, bound, str(res.message))

from functools import wraps
from inspect import signature
from math import isqrt

import numpy as np


def vectors(*names, dtype=np.float64):
    def deco(f):
        sig = signature(f)

        @wraps(f)
        def g(*args, **kwargs):
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            for name in names:
                x = np.asarray(b.arguments[name])
                if x.dtype.kind not in "iuf":
                    raise TypeError("%s must be real" % name)
                if x.ndim != 1:
                    raise ValueError("%s must be one-dimensional" % name)
                x = np.ascontiguousarray(x) if dtype is None else np.ascontiguousarray(x, dtype=dtype)
                if not np.all(np.isfinite(x)):
                    raise ValueError("%s must be finite" % name)
                b.arguments[name] = x
            return f(*b.args, **b.kwargs)

        return g

    return deco


def integer(x, name, lo=None):
    if isinstance(x, (bool, np.bool_)) or int(x) != x:
        raise TypeError("%s must be an integer" % name)
    x = int(x)
    if lo is not None and x < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return x


def scalar(x, name, lo=None, strict=False):
    if not np.isscalar(x) or not np.isreal(x) or not np.isfinite(x):
        raise TypeError("%s must be a finite real scalar" % name)
    x = float(x)
    if lo is not None and (x <= lo if strict else x < lo):
        op = "greater than" if strict else "at least"
        raise ValueError("%s must be %s %g" % (name, op, lo))
    return x


def tri_n(m):
    m = integer(m, "number of distances", 1)
    n = (1 + isqrt(1 + 8 * m)) // 2
    if n * (n - 1) // 2 != m:
        raise ValueError("the number of distances must be triangular")
    return n


def _indices(e, n):
    e = np.asarray(e)
    m = n * (n - 1) // 2
    if e.ndim == 1:
        if e.dtype.kind not in "iu":
            raise TypeError("interval indices must be integers")
        a = np.ascontiguousarray(e, dtype=np.intc)
    elif e.ndim == 2 and e.shape[1] == 2:
        if e.dtype.kind not in "iu":
            raise TypeError("interval endpoints must be integers")
        i = e[:, 0].astype(np.int64, copy=False)
        j = e[:, 1].astype(np.int64, copy=False)
        if np.any(i < 0) or np.any(i >= j) or np.any(j >= n):
            raise ValueError("interval endpoints must satisfy 0 <= i < j < n")
        a = np.ascontiguousarray(j * (j - 1) // 2 + i, dtype=np.intc)
    else:
        raise ValueError("intervals must be row indices or pairs")
    if np.any(a < 0) or np.any(a >= m):
        raise ValueError("interval index out of range")
    return a


def pack(D, I, n):
    if len(D) != len(I) or not len(D):
        raise ValueError("D and I must contain the same nonzero number of cells")
    m = n * (n - 1) // 2
    g = np.zeros(m, dtype=np.intc)
    d = []
    o = [0]
    for c, (dc, ic) in enumerate(zip(D, I), 1):
        x = np.asarray(dc)
        if x.dtype.kind not in "iuf":
            raise TypeError("cell distances must be real")
        if x.ndim != 1 or not len(x):
            raise ValueError("each distance cell must be a nonempty vector")
        x = np.ascontiguousarray(np.sort(x), dtype=np.float64)
        if not np.all(np.isfinite(x)) or np.any(x < 0):
            raise ValueError("cell distances must be finite and nonnegative")
        e = _indices(ic, n)
        if len(x) != len(e):
            raise ValueError("each cell needs one interval per distance")
        if len(np.unique(e)) != len(e) or np.any(g[e]):
            raise ValueError("observed interval cells must be disjoint")
        g[e] = c
        d.append(x)
        o.append(o[-1] + len(x))
    d = np.ascontiguousarray(np.concatenate(d), dtype=np.float64)
    if not np.any(d > 0):
        raise ValueError("at least one observed distance must be positive")
    return d, np.asarray(o, dtype=np.intc), g

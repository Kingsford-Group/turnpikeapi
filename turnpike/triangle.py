from bisect import bisect_left, bisect_right
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import _core
from ._validate import integer, scalar, tri_n, vectors


class Relations(NamedTuple):
    y: np.ndarray
    mu: np.ndarray
    S: np.ndarray


class Model(NamedTuple):
    A: object
    b: np.ndarray
    ub: np.ndarray
    np: int
    nt: int


class Certificate(NamedTuple):
    status: str
    P: Optional[np.ndarray]
    x: Optional[np.ndarray]
    integral: bool
    guaranteed: bool
    rel: Relations
    message: str


class Calibration(NamedTuple):
    r: Optional[float]
    cert: Certificate


@vectors("d", dtype=None)
def hist(d):
    """Return decreasing distinct distances and their multiplicities."""
    tri_n(len(d))
    if np.any(d <= 0):
        raise ValueError("exact distances must be positive")
    y, mu = np.unique(d, return_counts=True)
    return np.ascontiguousarray(y[::-1]), np.ascontiguousarray(mu[::-1], dtype=np.intc)


def parts(y, mu, tau=0):
    """Enumerate ordered, multiplicity-aware two-partitions."""
    y = np.asarray(y)
    mu = np.asarray(mu)
    if y.dtype.kind not in "iuf" or mu.dtype.kind not in "iu":
        raise TypeError("y and mu must be numeric vectors")
    if y.ndim != 1 or mu.ndim != 1 or len(y) != len(mu) or not len(y):
        raise ValueError("y and mu must be same-length nonempty vectors")
    mu = np.ascontiguousarray(mu, dtype=np.intc)
    if not np.all(np.isfinite(y)) or np.any(y <= 0) or np.any(mu <= 0):
        raise ValueError("y and mu must be positive and finite")
    if np.any(y[:-1] <= y[1:]):
        raise ValueError("y must be strictly decreasing")
    tau = scalar(tau, "tau", 0)
    if tau == 0:
        if np.any(y != np.rint(y)) or np.any(y >= 2 ** 63):
            raise ValueError("exact relations require scaled 64-bit integers")
        return _core.parts_i(np.ascontiguousarray(y, dtype=np.int64), mu)
    return _core.parts(np.ascontiguousarray(y, dtype=np.float64), mu, tau)


@vectors("d", dtype=None)
def two_partitions(d, tau=0):
    """Return the histogram and its ordered two-partitions."""
    y, mu = hist(d)
    return Relations(y, mu, parts(y, mu, tau))


@vectors("y", dtype=None)
def gap(y):
    """Return the least nonzero two-sum residual."""
    if not len(y) or np.any(y <= 0):
        raise ValueError("y must be a positive vector")
    y = np.ascontiguousarray(np.unique(y)[::-1])
    if np.all(y == np.rint(y)) and np.all(y < 2 ** 63):
        v = [int(x) for x in y]
        s = sorted(a + b for i, a in enumerate(v) for b in v[i:])
        g = None
        for x in v:
            i = bisect_left(s, x)
            j = bisect_right(s, x, i)
            if i:
                g = x - s[i - 1] if g is None else min(g, x - s[i - 1])
            if j < len(s):
                g = s[j] - x if g is None else min(g, s[j] - x)
        return np.inf if g is None else g
    return float(_core.gap(np.ascontiguousarray(y, dtype=np.float64)))


def rank_mask(n, mu):
    """Return the admissible interval-label mask from rank bounds."""
    n = integer(n, "n", 2)
    mu = np.asarray(mu)
    if mu.ndim != 1 or mu.dtype.kind not in "iu" or np.any(mu <= 0):
        raise ValueError("mu must be a positive integer vector")
    mu = np.ascontiguousarray(mu, dtype=np.intc)
    if np.sum(mu) != n * (n - 1) // 2:
        raise ValueError("mu must sum to n choose 2")
    return _core.rank_mask(n, mu)


def model(n, mu, S, basis="spine", prune=True):
    """Build the solver-independent sparse triangle-equality model."""
    n = integer(n, "n", 2)
    mu = np.asarray(mu)
    S = np.asarray(S)
    if mu.ndim != 1 or mu.dtype.kind not in "iu" or np.any(mu <= 0):
        raise ValueError("mu must be a positive integer vector")
    if S.ndim != 2 or S.shape[1] != 3 or S.dtype.kind not in "iu":
        raise ValueError("S must be an integer array with three columns")
    mu = np.ascontiguousarray(mu, dtype=np.intc)
    S = np.ascontiguousarray(S, dtype=np.intc)
    if np.sum(mu) != n * (n - 1) // 2:
        raise ValueError("mu must sum to n choose 2")
    if len(S) and (np.any(S < 0) or np.any(S >= len(mu))):
        raise ValueError("relation label out of range")
    if basis not in ("spine", "full"):
        raise ValueError("basis must be 'spine' or 'full'")
    if not isinstance(prune, (bool, np.bool_)):
        raise TypeError("prune must be boolean")
    r, c, a, b, ub, npv, nt = _core.tri_model(
        n, mu, S, basis == "spine", bool(prune)
    )
    A = coo_matrix((a, (r, c)), shape=(len(b), len(ub))).tocsc()
    A.indices = A.indices.astype(np.int32, copy=False)
    A.indptr = A.indptr.astype(np.int32, copy=False)
    return Model(A, b, ub.astype(float), int(npv), int(nt))


def _solve(rel, lp, basis, prune, options, guaranteed):
    n = tri_n(int(np.sum(rel.mu)))
    if basis is None:
        basis = "full" if lp else "spine"
    M = model(n, rel.mu, rel.S, basis, prune)
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be a dictionary")
    res = milp(
        np.zeros(len(M.ub)),
        integrality=np.zeros(len(M.ub), dtype=int)
        if lp
        else np.ones(len(M.ub), dtype=int),
        bounds=Bounds(np.zeros(len(M.ub)), M.ub),
        constraints=LinearConstraint(M.A, M.b, M.b),
        options={} if options is None else dict(options),
    )
    if res.status == 2:
        return Certificate("infeasible", None, None, False, False, rel, str(res.message))
    if res.status != 0 or res.x is None:
        status = "limit" if res.status == 1 else "unknown"
        return Certificate(status, None, None, False, False, rel, str(res.message))
    P = np.asarray(res.x[: M.np]).reshape(-1, len(rel.y))
    integral = bool(np.all(np.abs(P - np.rint(P)) <= 1e-7))
    valid = guaranteed and integral and verify(np.repeat(rel.y, rel.mu), P)
    if guaranteed and ((not lp and not valid) or (lp and integral and not valid)):
        return Certificate("numerical", P, None, integral, False, rel, str(res.message))
    if guaranteed and valid:
        d = rel.y[np.argmax(P, axis=1)]
        x = np.zeros(n, dtype=rel.y.dtype)
        for j in range(1, n):
            x[j] = d[j * (j - 1) // 2]
    elif guaranteed:
        d = P @ rel.y
        x = np.zeros(n)
        for j in range(1, n):
            x[j] = d[j * (j - 1) // 2]
    else:
        d = P @ rel.y
        x = _core.qt(np.ascontiguousarray(d), n) / n
        x -= x[0]
    status = "feasible" if not lp or integral else "fractional"
    cert = guaranteed and valid
    return Certificate(status, P, x, integral, cert, rel, str(res.message))


@vectors("d", dtype=None)
def triangle(d, lp=False, basis=None, prune=True, options=None):
    """Solve the exact scaled-integer triangle LP or ILP."""
    if not isinstance(lp, (bool, np.bool_)):
        raise TypeError("lp must be boolean")
    rel = two_partitions(d)
    return _solve(rel, bool(lp), basis, prune, options, True)


@vectors("d")
def rounded_relations(d, r, R):
    """Round observations and recover diagnostic relations at radius r."""
    tri_n(len(d))
    if np.any(d <= 0):
        raise ValueError("observed distances must be positive")
    r = scalar(r, "r", 0)
    R = scalar(R, "R", 0)
    if R:
        lim = (2 ** 63 - 0.5) * R
        if np.isfinite(lim) and np.any(d >= lim):
            raise OverflowError("rounded grid indices exceed signed 64-bit range")
        a = d / R + 0.5
        k = np.floor(a).astype(np.int64)
        if np.any(k <= 0):
            raise ValueError("rounding collapsed a positive distance to zero")
        u, mu = np.unique(k, return_counts=True)
        y = np.ascontiguousarray((u[::-1] * R).astype(float))
        mu = np.ascontiguousarray(mu[::-1], dtype=np.intc)
    else:
        y, mu = hist(d)
    S = _core.parts_r(
        np.ascontiguousarray(y, dtype=np.float64), mu, r, R
    )
    return Relations(y, mu, S)


@vectors("d")
def critical_radii(d, R=0):
    """Return the radii at which recovered relation sets change."""
    R = scalar(R, "R", 0)
    rel = rounded_relations(d, 0, R)
    return _core.critical(rel.y, rel.mu, R)


@vectors("d")
def robust(d, r, R=0, lp=False, basis=None, prune=True, options=None):
    """Solve the diagnostic triangle model from noisy recovered relations."""
    if not isinstance(lp, (bool, np.bool_)):
        raise TypeError("lp must be boolean")
    rel = rounded_relations(d, r, R)
    return _solve(rel, bool(lp), basis, prune, options, False)


@vectors("d")
def calibrate(d, R=0, lp=False, basis=None, prune=True, options=None):
    """Find the least critical radius with a feasible recovered-relation model."""
    if not isinstance(lp, (bool, np.bool_)):
        raise TypeError("lp must be boolean")
    R = scalar(R, "R", 0)
    psi = np.unique(np.r_[0.0, critical_radii(d, R)])
    lo, hi = 0, len(psi)
    best = None
    last = None
    while lo < hi:
        j = (lo + hi) // 2
        z = robust(d, float(psi[j]), R, lp, basis, prune, options)
        last = z
        if z.status in ("feasible", "fractional"):
            best = (j, z)
            hi = j
        elif z.status == "infeasible":
            lo = j + 1
        else:
            raise RuntimeError("solver did not classify the calibration model")
    if best is None:
        if lo < len(psi):
            z = robust(d, float(psi[lo]), R, lp, basis, prune, options)
            if z.status in ("feasible", "fractional"):
                best = (lo, z)
            else:
                last = z
        return Calibration(None, last)
    return Calibration(float(psi[best[0]]), best[1])


@vectors("d", dtype=None)
def verify(d, P, tol=1e-7):
    """Verify an integral interval-label assignment against exact distances."""
    y, mu = hist(d)
    if np.any(y != np.rint(y)) or np.any(y >= 2 ** 63):
        return False
    n = tri_n(len(d))
    P = np.asarray(P)
    if P.shape != (len(d), len(y)) or P.dtype.kind not in "iuf":
        return False
    tol = scalar(tol, "tol", 0)
    if not np.all(np.isfinite(P)) or np.any(np.abs(P - np.rint(P)) > tol):
        return False
    P = np.rint(P).astype(np.int64)
    if np.any((P != 0) & (P != 1)):
        return False
    if not np.array_equal(P.sum(1), np.ones(len(d), dtype=np.int64)):
        return False
    if not np.array_equal(P.sum(0), mu):
        return False
    a = y[np.argmax(P, axis=1)]
    for k in range(2, n):
        for j in range(1, k):
            for i in range(j):
                ij = j * (j - 1) // 2 + i
                jk = k * (k - 1) // 2 + j
                ik = k * (k - 1) // 2 + i
                if int(a[ij]) + int(a[jk]) != int(a[ik]):
                    return False
    return True

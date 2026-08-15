import numpy as np

from . import _core
from ._validate import integer, pack, scalar, vectors
from .core import anchor, normalize, unit


def _check(d, z):
    n = len(z)
    if n < 2 or len(d) != n * (n - 1) // 2:
        raise ValueError("d must contain one distance per point pair")
    if np.any(d < 0) or not np.any(d > 0):
        raise ValueError("d must be nonnegative and nonzero")
    if not np.any(z != z[0]):
        raise ValueError("the initializer must have nonzero centered norm")


@vectors("d", "z")
def raw(d, z):
    """Return the raw MM back-projection for the assignment induced by z."""
    _check(d, z)
    return _core.mmu(np.sort(d), np.sort(z))


@vectors("d", "z")
def step(d, z):
    """Return one normalized basic-MM step."""
    return unit(raw(d, z))


@vectors("d", "z")
def fit(d, z):
    """Return the centered least-squares fit for the assignment induced by z."""
    _check(d, z)
    return _core.mmu(np.sort(d), np.sort(z)) / len(z)


@vectors("d", "z")
def mm(d, z, tol=1e-10, max_iter=1000):
    """Iterate basic MM and return its normalized representative."""
    _check(d, z)
    tol = scalar(tol, "tol", 0)
    max_iter = integer(max_iter, "max_iter", 0)
    return _core.mm(np.sort(d), np.sort(z), tol, max_iter)


@vectors("d", "z")
def solve(d, z, tol=1e-10, max_iter=1000):
    """Iterate MM and return anchored physical coordinates."""
    _check(d, z)
    tol = scalar(tol, "tol", 0)
    max_iter = integer(max_iter, "max_iter", 0)
    d = np.sort(d)
    _, s = _core.mm_state(d, np.sort(z), tol, max_iter)
    return anchor(_core.mmu(d, s) / len(z))


def _bstep(D, I, z, impute):
    s = np.sort(z)
    z = normalize(s)
    d, o, g = pack(D, I, len(s))
    u = _core.bmmu(d, o, g, s, z, impute)
    return unit(u)


@vectors("z")
def bstep(D, I, z):
    """Return one observed-correlation coordinate-ascent step.

    Omitted intervals contribute zero. This ascends the observed correlation
    sum_c <Q_c z, P_c D_c>; when rows are missing that is not observed least
    squares, since sum_c ||Q_c z||^2 then varies with z. Use `impute_step` for
    the profiled observed loss.
    """
    return _bstep(D, I, z, False)


@vectors("z")
def impute_step(D, I, z):
    """Return one profiled-observed-loss imputation-MM step.

    Missing rows are filled with their current predictions and enter the
    surrogate only; they are optimization-transfer terms, not observations.
    The step exactly minimizes a tight global majorizer of the profiled
    observed half squared loss, so that loss is nonincreasing.
    """
    return _bstep(D, I, z, True)


def _bmm(D, I, z, tol, max_iter, impute):
    tol = scalar(tol, "tol", 0)
    max_iter = integer(max_iter, "max_iter", 0)
    s = np.sort(z)
    if not np.any(s != s[0]):
        raise ValueError("the initializer must have nonzero centered norm")
    d, o, g = pack(D, I, len(s))
    z = _core.bmm(d, o, g, s, tol, max_iter, impute)
    if not np.all(np.isfinite(z)):
        raise ValueError("the projected block update is zero")
    return z


@vectors("z")
def bmm(D, I, z, tol=1e-10, max_iter=1000):
    """Iterate observed-correlation coordinate ascent.

    Unlisted intervals contribute zero. With deterministic ties the exact
    uncapped map reaches a fixed point finitely; this floating-point routine
    stops on `tol` or `max_iter`. A fixed point need not be optimal.
    """
    return _bmm(D, I, z, tol, max_iter, False)


@vectors("z")
def impute(D, I, z, tol=1e-10, max_iter=1000):
    """Iterate profiled-observed-loss imputation MM.

    Missing rows enter the surrogate only. Each step minimizes a tight global
    majorizer, so the profiled observed half squared loss is nonincreasing.
    The surrogate depends continuously on the current iterate, so a finite
    assignment space does not imply finite termination here; the guarantee is
    monotone bounded loss values, not global optimality.
    """
    return _bmm(D, I, z, tol, max_iter, True)

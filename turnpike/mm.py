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
    """Return one observed-only projected block-MM step."""
    return _bstep(D, I, z, False)


@vectors("z")
def impute_step(D, I, z):
    """Return one diagnostic block step with missing intervals imputed."""
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
    """Iterate observed-only block MM; unlisted intervals contribute zero."""
    return _bmm(D, I, z, tol, max_iter, False)


@vectors("z")
def impute(D, I, z, tol=1e-10, max_iter=1000):
    """Iterate the historical null-imputing block update."""
    return _bmm(D, I, z, tol, max_iter, True)

import numpy as np

from . import _core
from ._validate import integer, tri_n, vectors


def pairs(n):
    """Return the j-major interval endpoints."""
    return _core.pairs(integer(n, "n", 2))


@vectors("z")
def q(z):
    """Return all forward differences Qz."""
    if len(z) < 2:
        raise ValueError("z must contain at least two points")
    return _core.q(z)


@vectors("d")
def qt(d, n=None):
    """Return the incidence back-projection Q.T @ d."""
    n = tri_n(len(d)) if n is None else integer(n, "n", 2)
    if len(d) != n * (n - 1) // 2:
        raise ValueError("d has the wrong length for n")
    return _core.qt(d, n)


@vectors("z")
def pav(z):
    """Project onto the nondecreasing cone by unweighted PAV."""
    if not len(z):
        raise ValueError("z must be nonempty")
    return _core.iso(z)


@vectors("z")
def unit(z):
    """Return the Euclidean unit representative."""
    r = np.linalg.norm(z)
    if r == 0:
        raise ValueError("cannot normalize the zero vector")
    return z / r


@vectors("z")
def normalize(z):
    """Sort, center, and unit-normalize a point vector."""
    if len(z) < 2:
        raise ValueError("z must contain at least two points")
    z = np.sort(z) - np.mean(z)
    return unit(z)


@vectors("x")
def anchor(x):
    """Translate coordinates so that x[0] is zero."""
    if not len(x):
        raise ValueError("x must be nonempty")
    return x - x[0]

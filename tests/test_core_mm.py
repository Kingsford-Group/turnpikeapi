import numpy as np
import pytest

import turnpike as tp
from turnpike import _core

from .reference import Q0, order0, pairs0, pav0, raw0


def test_compiled_backend_is_exercised():
    assert _core.__file__.endswith((".so", ".pyd"))
    assert _core.version == "0.1.0"


def test_pair_order_q_and_qt():
    rng = np.random.default_rng(1)
    for n in range(2, 9):
        Q = Q0(n)
        z = rng.normal(size=n)
        d = rng.normal(size=len(Q))
        assert np.array_equal(tp.pairs(n), pairs0(n))
        assert np.allclose(tp.q(z), Q @ z)
        assert np.allclose(tp.qt(d, n), Q.T @ d)


def test_streaming_raw_matches_dense_oracle():
    rng = np.random.default_rng(2)
    for n in range(2, 9):
        for _ in range(30):
            z = np.sort(rng.integers(-3, 7, size=n)).astype(float)
            if np.ptp(z) == 0:
                z[-1] += 1
            d = rng.integers(0, 9, size=n * (n - 1) // 2).astype(float)
            if not np.any(d):
                d[0] = 1
            assert np.array_equal(tp.raw(d, z), raw0(d, z))


def test_canonical_tie_survives_public_boundary_and_iterations():
    z = np.array([0.0, 1.0, 2.0, 4.0])
    assert np.array_equal(tp.raw(np.arange(1, 7), z), [-11, -6, 3, 14])
    d = np.array([1, 1, 1, 2, 6, 11], dtype=float)
    u = np.array([-14, -6, 2, 18], dtype=float)
    assert np.allclose(tp.mm(d, z, tol=0, max_iter=2), u / np.linalg.norm(u))
    assert np.allclose(tp.solve(d, z, tol=0, max_iter=2), (u - u[0]) / len(z))


def test_fit_and_step_are_distinct_rays():
    d = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    z = np.array([0, 1, 2, 4], dtype=float)
    u = raw0(d, z)
    assert np.allclose(tp.step(d, z), u / np.linalg.norm(u))
    assert np.allclose(tp.fit(d, z), u / len(z))
    assert np.array_equal(order0(u), order0(u / len(z)))


def test_mm_trace_and_objective():
    rng = np.random.default_rng(3)
    for n in range(2, 8):
        d = np.sort(rng.uniform(0.1, 5, n * (n - 1) // 2))
        s = np.sort(rng.normal(size=n))
        s0 = s.copy()
        z = tp.normalize(s)
        f = []
        for _ in range(8):
            f.append(np.dot(np.sort(tp.q(z)), d))
            s = raw0(d, s)
            z = s / np.linalg.norm(s)
        assert np.all(np.diff(f) >= -1e-10)
        assert np.allclose(tp.mm(d, s0, tol=0, max_iter=8), z)
        assert np.allclose(tp.mm(d, s0, max_iter=0), tp.normalize(s0))


def test_scaling_and_pav():
    rng = np.random.default_rng(4)
    for n in range(2, 8):
        z = rng.normal(size=n)
        assert np.allclose(tp.pav(z), pav0(z))
    d = np.arange(1, 7, dtype=float)
    z = [0, 1, 2, 4]
    assert np.allclose(tp.step(7 * d, z), tp.step(d, z))
    assert np.allclose(tp.fit(7 * d, z), 7 * tp.fit(d, z))


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda: tp.step([1, 2], [0, 1, 2]), ValueError),
        (lambda: tp.step([-1], [0, 1]), ValueError),
        (lambda: tp.normalize([1, 1]), ValueError),
        (lambda: tp.mm([1], [2, 2], max_iter=0), ValueError),
        (lambda: tp.q([[0, 1]]), ValueError),
        (lambda: tp.mm([1], [0, 1], max_iter=-1), ValueError),
    ],
)
def test_validation(call, error):
    with pytest.raises(error):
        call()

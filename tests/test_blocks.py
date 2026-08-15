import numpy as np
import pytest

import turnpike as tp

from .reference import block0, block_trace0, order0


def test_block_step_matches_dense_oracle():
    rng = np.random.default_rng(5)
    for n in range(3, 8):
        m = n * (n - 1) // 2
        e = rng.permutation(m)
        I = [e[: m // 3], e[m // 3 : 2 * m // 3]]
        D = [rng.integers(1, 8, len(I[0])), rng.integers(1, 8, len(I[1]))]
        z = np.sort(rng.integers(0, 8, n)).astype(float)
        if np.ptp(z) == 0:
            z[-1] += 1
        u = block0(D, I, z)
        assert np.allclose(tp.bstep(D, I, z), u / np.linalg.norm(u))


def test_one_block_reduces_to_mm_step():
    z = np.array([0, 1, 2, 4], dtype=float)
    d = np.array([1, 1, 2, 3, 5, 8], dtype=float)
    assert np.allclose(tp.bstep([d], [np.arange(6)], z), tp.step(d, z))


def test_projection_counterexample():
    z = np.array([0, 1, 2, 8], dtype=float)
    I = [np.array([0, 1, 4, 5]), np.array([2, 3])]
    D = [np.array([1, 1, 7, 7]), np.array([6, 8])]
    u = block0(D, I, z)
    assert np.array_equal(u, [-11, -11, 0, 22])
    assert np.allclose(tp.bstep(D, I, z), u / np.linalg.norm(u))


def test_imputation_has_separate_contract():
    z = np.array([0, 1, 3, 7], dtype=float)
    D = [[1, 2]]
    I = [[0, 4]]
    u = block0(D, I, tp.normalize(z), impute=True)
    assert np.allclose(tp.impute_step(D, I, z), u / np.linalg.norm(u))
    assert not np.allclose(tp.impute_step(D, I, z), tp.bstep(D, I, z))


def test_multi_iteration_tie_state_is_raw():
    z = np.array([-14, -6, 2, 18], dtype=float)
    D = [np.array([1.0, 9.0])]
    I = [np.array([5, 1])]
    assert not np.array_equal(order0(z), order0(tp.normalize(z)))
    for impute, f in ((False, tp.bmm), (True, tp.impute)):
        want = block_trace0(D, I, z, 5, impute)
        assert np.allclose(f(D, I, z, tol=0, max_iter=5), want)


def test_fixed_assignments_are_singleton_cells():
    z = [0, 1, 3]
    D = [[3], [1], [2]]
    I = [[1], [0], [2]]
    u = block0(D, I, np.asarray(z, dtype=float))
    assert np.allclose(tp.bstep(D, I, z), u / np.linalg.norm(u))


def test_block_validation():
    with pytest.raises(ValueError):
        tp.bstep([[1], [2]], [[0], [0]], [0, 1, 2])
    with pytest.raises(ValueError):
        tp.bstep([[1, 2]], [[0]], [0, 1, 2])
    with pytest.raises(ValueError):
        tp.bstep([[1]], [[9]], [0, 1, 2])

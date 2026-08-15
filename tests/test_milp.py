from itertools import product

import numpy as np
import pytest

import turnpike as tp
from turnpike.milp import _M, _cmp, _const

from .reference import best_fit0, network0


def test_network_zero_one_principle():
    for n in range(1, 13):
        N, C = tp.network(n)
        for x in product((0, 1), repeat=n):
            y = network0(x, N, C)
            assert np.array_equal(y[:n], np.sort(x))


def test_nonpower_network_on_reals():
    rng = np.random.default_rng(6)
    for n in (3, 5, 6, 7, 9):
        N, C = tp.network(n)
        for _ in range(20):
            x = rng.random(n)
            assert np.allclose(network0(x, N, C)[:n], np.sort(x))


def test_tied_comparator_is_integral_and_exact():
    M = _M()
    _cmp(M, _const(0.5), _const(0.5), 1)
    z = M.solve({"mip_rel_gap": 0.0})
    assert M.z == [0, 0, 1]
    assert np.allclose(z.x[:2], [0.5, 0.5])


@pytest.mark.parametrize("loss", ["l1", "linf"])
def test_best_fit_matches_naive_permutation_lp(loss):
    D = [np.array([1.0, 2.2]), np.array([1.1])]
    I = [np.array([0, 2]), np.array([1])]
    ref, _ = best_fit0(D, I, 3, 4, loss)
    fit = tp.best_fit(D, I, 3, u=4, loss=loss)
    assert fit.status == "optimal"
    assert fit.gap is not None and fit.gap <= 1e-10
    assert fit.bound is not None and np.isclose(fit.bound, ref, atol=1e-9)
    assert np.isclose(fit.fun, ref)


def test_partitioned_nonpower_exact_fit():
    x = np.array([0, 1, 4, 7], dtype=float)
    d = tp.q(x)
    I = [np.array([0, 2, 5]), np.array([1, 3]), np.array([4])]
    D = [d[e] for e in I]
    fit = tp.best_fit(D, I, 4, u=7)
    assert fit.status == "optimal"
    assert np.isclose(fit.fun, 0)
    assert np.allclose(np.sort(tp.q(fit.x)[I[0]]), np.sort(D[0]))


def test_bound_is_required_and_material():
    D = [[1], [1]]
    I = [[0], [2]]
    with pytest.raises(ValueError):
        tp.best_fit(D, I, 3)
    tight = tp.best_fit(D, I, 3, u=1)
    valid = tp.best_fit(D, I, 3, u=2)
    assert tight.fun > valid.fun
    assert np.isclose(valid.fun, 0)


def test_invalid_loss_and_bound():
    with pytest.raises(ValueError):
        tp.best_fit([[1]], [[0]], 2, u=1, loss="l2")
    with pytest.raises(TypeError):
        tp.best_fit([[1]], [[0]], 2, u=np.inf)


def test_time_limit_is_not_an_optimality_certificate():
    fit = tp.best_fit([[1, 2, 3]], [[0, 1, 2]], 3, u=3, options={"time_limit": 0})
    assert fit.status == "limit"
    assert fit.x is None and fit.fun is None and fit.gap is None and fit.bound is None

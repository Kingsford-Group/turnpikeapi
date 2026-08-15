from itertools import combinations_with_replacement
from types import SimpleNamespace
import importlib

import numpy as np
import pytest

import turnpike as tp

from .reference import feasible0, parts0


def test_parts_matches_cubic_oracle():
    rng = np.random.default_rng(7)
    for p in range(1, 9):
        y = np.sort(rng.choice(np.arange(1, 40), p, replace=False))[::-1]
        mu = rng.integers(1, 4, p, dtype=np.int32)
        for tau in (0, 0.5, 2):
            assert np.array_equal(tp.parts(y, mu, tau), parts0(y, mu, tau))


def test_self_pair_needs_multiplicity():
    assert len(tp.parts([2, 1], [1, 1])) == 0
    assert np.array_equal(tp.parts([2, 1], [1, 2]), [[1, 1, 0]])


def test_exact_path_requires_scaled_integers():
    with pytest.raises(ValueError):
        tp.triangle([0.1, 0.2, 0.3])
    assert tp.triangle([1, 2, 3]).status == "feasible"


def test_large_int64_reconstruction_stays_exact():
    d = np.array([2 ** 53 + 1, 2 ** 53 + 3, 2], dtype=np.int64)
    z = tp.triangle(d)
    assert z.guaranteed and z.integral and z.x.dtype.kind in "iu"
    got = sorted(int(z.x[j]) - int(z.x[i]) for j in range(1, 3) for i in range(j))
    assert got == sorted(int(x) for x in d)
    assert tp.verify(d, z.P)


def test_verify_rejects_signed_nonbinary_assignment():
    P = np.array([[-2, 0, 3], [0, 2, -1], [3, -1, -1]], dtype=float)
    assert not tp.verify([1, 2, 3], P)


def test_exact_gap_above_float_integer_limit():
    y = np.array([2 ** 53 + 3, 2 ** 53 + 2, 2 ** 53 + 1, 2], dtype=np.int64)
    assert tp.gap(y) == 1


@pytest.mark.parametrize(
    "d, status",
    [
        ([1, 2, 3], "feasible"),
        ([2, 2, 2, 4, 4, 6], "feasible"),
        ([1, 1, 3], "infeasible"),
        ([1, 2, 2], "infeasible"),
    ],
)
def test_known_certificates(d, status):
    z = tp.triangle(d)
    assert z.status == status
    if status == "feasible":
        assert z.guaranteed
        assert z.integral
        assert tp.verify(d, z.P)
    else:
        assert not z.guaranteed


def test_small_histograms_against_permutation_oracle():
    for n, alphabet in ((3, range(1, 6)), (4, range(1, 5))):
        m = n * (n - 1) // 2
        for d in combinations_with_replacement(alphabet, m):
            want = feasible0(d) is not None
            a = tp.triangle(d, basis="spine", prune=True)
            b = tp.triangle(d, basis="full", prune=False)
            assert (a.status == "feasible") == want
            assert (b.status == "feasible") == want


def test_sparse_model_shape_and_empty_relation_rows():
    M = tp.model(3, [1, 1, 1], np.empty((0, 3), dtype=int), basis="full")
    assert M.A.shape == (16, 9)
    assert M.A.getnnz(axis=1)[6] == 0 and M.b[6] == 1
    z = tp.triangle([1, 1, 3], basis="full")
    assert z.status == "infeasible"


def test_lp_contract():
    z = tp.triangle([1, 2, 3], lp=True)
    assert z.status == "feasible"
    assert z.integral and z.guaranteed
    q = tp.triangle([1, 1, 3], lp=True)
    assert q.status == "infeasible" and not q.guaranteed


def test_ilp_guarantee_requires_exact_post_verification(monkeypatch):
    mod = importlib.import_module("turnpike.triangle")
    monkeypatch.setattr(
        mod,
        "milp",
        lambda c, **kwargs: SimpleNamespace(status=0, x=np.zeros(len(c)), message="mock"),
    )
    z = mod._solve(tp.two_partitions([1, 2, 3]), False, "spine", True, None, True)
    assert z.status == "numerical" and not z.guaranteed


def test_robust_is_diagnostic_and_critical_events_are_closed():
    d = np.array([10.1, 19.9, 30.4])
    R = 1.0
    psi = tp.critical_radii(d, R)
    count = []
    for r in psi:
        at = len(tp.rounded_relations(d, r, R).S)
        below = len(tp.rounded_relations(d, np.nextafter(r, -np.inf), R).S) if r else 0
        assert at > below if r else at >= below
        count.append(at)
    assert np.all(np.diff(count) >= 0)
    z = tp.robust(d, 0, R)
    assert not z.guaranteed
    with pytest.raises(TypeError):
        tp.robust(d, 0, R, lp="false")
    with pytest.raises(OverflowError):
        tp.rounded_relations(d, 0, np.finfo(float).tiny)


def test_ulp_critical_endpoint_enters_strictly():
    from turnpike import _core

    y = np.array([32.12914108549334, 0.8561706558140311, 0.0011030410478530348])
    mu = np.ones(3, dtype=np.intc)
    R = 15.17195032851071
    for r in _core.critical(y, mu, R):
        if r:
            at = len(_core.parts_r(y, mu, r, R))
            below = len(_core.parts_r(y, mu, np.nextafter(r, -np.inf), R))
            assert at > below


def test_calibration_finds_first_feasible_event():
    d = np.array([1.0, 1.0, 3.0])
    z = tp.calibrate(d)
    assert np.isclose(z.r, 1 / 3)
    assert z.cert.status == "feasible" and not z.cert.guaranteed
    psi = np.unique(np.r_[0.0, tp.critical_radii(d)])
    first = next(r for r in psi if tp.robust(d, r).status == "feasible")
    assert z.r == first


def test_rank_mask_keeps_true_assignments():
    for x in ([0, 1, 3, 7], [0, 2, 4, 6], [0, 2, 5, 9]):
        d = tp.q(x).astype(int)
        y, mu = tp.hist(d)
        A = tp.rank_mask(4, mu)
        for e, a in enumerate(d):
            r = int(np.flatnonzero(y == a)[0])
            assert A[e, r]

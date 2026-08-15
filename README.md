# turnpike

A Python package, with C++17 kernels, for reconstructing one-dimensional point
sets from **unordered pairwise distances** — the Turnpike problem and its
circular counterpart, the Beltway problem — including the case where the
distance measurements are uncertain.

## The problem

Given a set of `n` points on a line, you can read off all `m = n(n-1)/2`
pairwise distances between them. The Turnpike problem is the inverse: you are
handed that multiset of distances *with no indication of which pair produced
which distance*, and asked to recover the original points. The Beltway problem
is the same question with the points arranged on a circle.

This arises in practice whenever an instrument reports distances but does not
associate them with the entities that produced them — partial digestion in
genomic mapping, molecular structure determination, tandem mass spectrometry,
and DNA-based error-correcting codes among them.

Exact Turnpike is tractable in practice, but **both problems become strongly
NP-hard once the measurements carry any uncertainty**, which is the regime every
real instrument operates in. This package implements three approaches to that
uncertain regime, trading off scale against the strength of the guarantee:

| Approach | Scales to | Guarantee |
|---|---|---|
| Majorization–minimization (MM) | ~10⁵ points | local/fixed-point; no global claim |
| Sorting-network MILP | moderate `n` | global best fit, solver-certified |
| Triangle-equality LP/ILP | moderate `n` | exact integer certificate of realizability |

## Provenance

This is the consolidated implementation behind three papers from the
[Kingsford group](https://www.cs.cmu.edu/~ckingsf/) at Carnegie Mellon
University's Ray and Stephanie Lane Computational Biology Department, and behind
the PhD dissertation of C. S. Elder:

1. **C. S. Elder, M. Hoang, M. Ferdosi, C. Kingsford.** *A Scalable Optimization
   Algorithm for Solving the Beltway and Turnpike Problems with Uncertain
   Measurements.* RECOMB. — the MM solver.
2. **The extended journal version** of the above, in the *Journal of
   Computational Biology* — partitioned MM and the sorting-network MILP models.
3. **C. S. Elder, G. Marçais, C. Kingsford.** *Turnpike with Uncertain
   Measurements: Triangle-Equality Integer Programming with a Deterministic
   Recovery Guarantee.* WABI. Full version:
   [arXiv:2603.18283](https://arxiv.org/abs/2603.18283) — the triangle
   certificates.

The papers are the mathematical authority; this package is the reference
implementation. Where the two diverge, one such case is documented explicitly
below under *partitioned MM*.

## Install

```bash
python3 -m pip install .
python3 -m pytest -q          # optional: run the test suite
```

Requires a C++17 compiler, Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.9, and
PyBind11 ≥ 2.10 (pulled in automatically at build time). The tests exercise the
compiled extension against deliberately naive pure-Python oracles.

The mathematical kernels are C++17 bound with PyBind11; validation, solver
calls, and result semantics stay in Python. The private `_core` module assumes
valid contiguous buffers — **the public functions are the checked API**, and are
what you should call.

## Conventions

For `n` points there are `m = n(n-1)/2` intervals. Every module uses the same
j-major ordering:

```python
>>> import turnpike as tp
>>> tp.pairs(4)
array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
```

`q(x)` computes all `x[j] - x[i]` in this order; `qt(d, n)` computes `Q.T @ d`.
Rank ties are resolved by predicted value, then shorter interval, then smaller
left endpoint.

## Majorization–minimization

The scalable solver. Each iteration runs in `O(m log m)` time and `O(√m)`
working space, which is what lets it reach instances of ~100,000 points.

```python
import numpy as np
import turnpike as tp

d  = np.array([1, 2, 3, 4, 5, 6])   # unordered pairwise distances
z0 = np.array([0, 1, 2, 4])         # initial guess

z = tp.mm(d, z0)       # centered, sorted, unit MM representative
x = tp.solve(d, z0)    # anchored physical least-squares coordinates
```

Smaller primitives keep the mathematically distinct outputs explicit:

- `raw(d, z)` → `u = Q.T @ P(z) @ d`
- `step(d, z)` → `u / ||u||`
- `fit(d, z)` → the centered regression `u / n`
- `solve(d, z)` → iterates MM, regresses the final assignment, anchors at zero

**MM is a deterministic local/fixed-point method, not a global reconstruction
guarantee.** `max_iter` and the normalized-iterate tolerance are always
retained. Internally the iteration ranks an unscaled raw representative and
keeps a separate unit vector for stopping and output; this prevents
floating-point normalization from splitting exact interval ties.

## Partitioned MM

For distances observed in groups rather than as one undifferentiated multiset.
Each observed cell is a multiset `D[c]` assigned to intervals `I[c]`, specified
by j-major row indices or `(i, j)` pairs. Cells are disjoint; omitted intervals
are treated as missing.

```python
D = [np.array([1, 1, 7, 7]), np.array([6, 8])]
I = [np.array([0, 1, 4, 5]), np.array([2, 3])]

z = tp.bmm(D, I, [0, 1, 2, 8])
```

`bmm` implements products of independent symmetric cells; a fixed assignment is
a singleton cell. General permutation families need their own assignment oracle
and are not silently treated as cells.

The observed-only update is `unit(pav(Q_obs.T @ P @ D))`. The PAV projection is
necessary — independently co-sorted cells can produce a nonmonotone gradient.

> **Known divergence from the journal paper.** This corrects a seam in that
> presentation, whose displayed objective omits the null block although its
> pseudocode imputes it. `impute` and `impute_step` retain the original
> missing-data update under a separate, diagnostic contract; they do **not**
> inherit the observed-objective monotonicity claim.

## Sorting-network MILP

Exact global best fit for moderate `n`.

```python
fit = tp.best_fit(D, I, n=4, u=10, loss="l1")
```

`best_fit` builds one exact bitonic sorting network per cell and supports `l1`
and `linf` residuals. The finite span bound `u` is **required**: it is
simultaneously the coordinate-domain bound, the comparator big-M, and the
padding sentinel, and cannot be inferred safely from noisy or partial
observations.

`Fit` reports `status`, `x`, `fun`, `gap`, `bound`, and the solver message. The
default relative MIP gap is zero. "Optimal" remains a floating-point solver
certificate, and any user-supplied nonzero gap stays visible in the result.
These are moderate-size global best-fit models, **not** the scalable solver.
Squared `l2` would be an MIQP and Euclidean `l2` a conic model, so neither is
offered here mislabeled as an MILP.

## Triangle certificates

Exact-arithmetic feasibility certificates.

```python
cert = tp.triangle([1, 2, 3])

cert.status      # "feasible"
cert.P           # interval-to-distinct-value assignment
cert.x           # x[0] = 0 spine reconstruction
cert.guaranteed  # True
```

The backend constructs a zero-objective feasibility model with `P` interval
labels and `T` ordered two-partitions. Exact relations use scaled signed 64-bit
integers with overflow-safe unsigned 64-bit sums. **Decimal or rational data
must be scaled to integers first**; no hidden tolerance is introduced into an
exact certificate.

`model`, `parts`, `rank_mask`, and `verify` expose the modular components
without requiring a solver-specific C++ layer.

### What each outcome does and does not establish

This distinction is deliberate and load-bearing:

| Outcome | Meaning |
|---|---|
| ILP | exact spine basis and rank pruning |
| LP | full triangle system |
| integral LP `P` | realizability certificate |
| fractional LP `P` | relaxation diagnostic — **not** a nonrealizability certificate |
| infeasible LP/ILP | solver-reported nonrealizability for the supplied relations; no independently checkable infeasibility proof artifact is returned |

`guaranteed=True` is reserved for an integral assignment that `verify` re-checks
in exact integer arithmetic. Solver-reported infeasibility and fractional
relaxations are deliberately **not** given that label.

### Bounded-error observations

```python
rel  = tp.rounded_relations(d_obs, r=0.02, R=0.01)
crit = tp.critical_radii(d_obs, R=0.01)
diag = tp.robust(d_obs, r=0.02, R=0.01)
cal  = tp.calibrate(d_obs, R=0.01)
```

`R` is the rounding-grid spacing, so its rounding error is at most `R/2`;
half-grid ties round upward for positive distances. Relations enter when
`max(0, |y[r]+y[s]-y[t]|/3 - R/2) <= r`, including the critical endpoint.
`calibrate` binary-searches these nested events for the first feasible
diagnostic model. Robust coordinates are a regression, not an exact spine claim.

`robust(...).guaranteed` is deliberately `False`. A theorem-level guarantee also
requires externally known strict separation conditions, and independently
perturbed duplicate copies can split across rounded bins. Treat it as a
relation-recovery and feasibility diagnostic unless those additional assumptions
are established outside this API.

## Citing

If you use this software, please cite the paper corresponding to the method you
used — the MM solver, the partitioned/MILP extensions, or the triangle
certificates — from the *Provenance* section above.

## Contact

Issues and questions: the Kingsford group, Carnegie Mellon University.

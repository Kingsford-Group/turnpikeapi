# Turnpike

`turnpike` is the compact implementation of the three algorithmic contributions in
Chapters 2--4 of the dissertation:

1. streaming majorization--minimization (MM);
2. partitioned MM and exact sorting-network MILP models;
3. triangle-equality LP/ILP certificates and noisy two-partition recovery.

The mathematical kernels are C++17 and are bound with PyBind11. NumPy/SciPy,
validation, solver calls, and result semantics stay in Python. The private `_core`
module assumes valid contiguous buffers; the public functions are the checked API.

## Install and test

```bash
python3 -m pip install .
python3 -m pytest -q
```

A C++17 compiler, Python 3.9 or newer, NumPy, SciPy, and PyBind11 are required.
The tests execute the compiled extension and compare it with deliberately naive
Python oracles.

## Shared conventions

For `n` points there are `m = n(n-1)/2` intervals. Every module uses the same
j-major order:

```python
>>> turnpike.pairs(4)
array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
```

`q(x)` computes all `x[j] - x[i]` in this order and `qt(d, n)` computes
`Q.T @ d`. Rank ties are resolved by predicted value, shorter interval, then
smaller left endpoint.

## Chapter 2: basic MM

```python
import numpy as np
import turnpike as tp

d = np.array([1, 2, 3, 4, 5, 6])
z0 = np.array([0, 1, 2, 4])

z = tp.mm(d, z0)       # centered, sorted, unit MM representative
x = tp.solve(d, z0)    # anchored physical least-squares coordinates
```

The small primitives keep the three mathematically distinct outputs explicit:

- `raw(d, z)` returns `u = Q.T @ P(z) @ d`;
- `step(d, z)` returns `u / ||u||`;
- `fit(d, z)` returns the centered regression `u / n`;
- `solve(d, z)` iterates MM, regresses the final assignment, and anchors at zero.

MM is a deterministic local/fixed-point method, not a global reconstruction
guarantee. `max_iter` and the normalized iterate tolerance are always retained.
Internally, iteration ranks an unscaled raw representative and keeps a separate
unit vector for stopping/output. This prevents floating-point normalization from
splitting exact interval ties.

## Chapter 3: partitioned MM

Each observed cell is a multiset `D[c]` assigned to the intervals `I[c]`. An
interval cell may be specified with j-major row indices or `(i, j)` pairs. Cells
are disjoint; omitted intervals are missing.

```python
D = [np.array([1, 1, 7, 7]), np.array([6, 8])]
I = [np.array([0, 1, 4, 5]), np.array([2, 3])]

z = tp.bmm(D, I, [0, 1, 2, 8])
```

`bmm` implements products of independent symmetric cells. A fixed assignment is
represented by a singleton cell. General permutation families require their own
assignment oracle and are not silently treated as cells.

The observed-only update is

```text
unit(pav(Q_obs.T @ P @ D)).
```

The PAV projection is necessary: independently co-sorted cells can produce a
nonmonotone gradient. This corrects a seam in the Chapter 3 presentation, whose
displayed objective omits the null block although its pseudocode imputes it.
`impute` and `impute_step` retain that historical missing-data update under a
separate, diagnostic contract; they do not inherit the observed-objective
monotonicity claim.

## Chapter 3: sorting-network MILP

```python
fit = tp.best_fit(D, I, n=4, u=10, loss="l1")
```

`best_fit` builds one exact bitonic sorting network per cell and supports `l1`
and `linf` residuals. The finite span bound `u` is required: it is simultaneously
the coordinate-domain bound, comparator big-M, and padding sentinel. It cannot be
inferred safely from noisy or partial observations.

`Fit` reports `status`, `x`, `fun`, `gap`, `bound`, and the solver message. The
default relative MIP gap is zero. “Optimal” remains a floating-point solver
certificate, and a user-supplied nonzero gap is visible in the result. These are
moderate-size global best-fit models, not the scalable solver. Squared `l2` is an
MIQP and Euclidean `l2` is a conic model, so neither is mislabeled as an MILP here.

## Chapter 4: triangle certificates

```python
cert = tp.triangle([1, 2, 3])

cert.status      # "feasible"
cert.P           # interval-to-distinct-value assignment
cert.x           # x[0] = 0 spine reconstruction
cert.guaranteed  # True
```

The backend constructs the zero-objective feasibility model with `P` interval
labels and `T` ordered two-partitions. Exact relations use scaled signed 64-bit
integers and overflow-safe unsigned 64-bit sums. Decimal/rational data must first
be scaled to integers;
no hidden tolerance is introduced into an exact certificate.

Defaults follow the mathematical distinction in the dissertation:

- ILP: exact spine basis and rank pruning;
- LP: full triangle system;
- integral LP `P`: realizability certificate;
- fractional LP `P`: relaxation diagnostic, not a nonrealizability certificate;
- infeasible LP/ILP: solver-reported nonrealizability for the supplied relations;
  no independently checkable infeasibility proof artifact is returned.

`model`, `parts`, `rank_mask`, and `verify` expose the modular components without
requiring a solver-specific C++ layer.

`guaranteed=True` is reserved for an integral assignment that `verify` checks
again with exact integer arithmetic. Solver-reported infeasibility and fractional
relaxations are not given that independently checkable label.

For bounded-error observations:

```python
rel = tp.rounded_relations(d_obs, r=0.02, R=0.01)
crit = tp.critical_radii(d_obs, R=0.01)
diag = tp.robust(d_obs, r=0.02, R=0.01)
cal = tp.calibrate(d_obs, R=0.01)
```

`R` is the rounding-grid spacing, so its rounding error is at most `R/2`.
Half-grid ties round upward for positive distances. Relations enter when
`max(0, |y[r]+y[s]-y[t]|/3 - R/2) <= r`, including their critical endpoint.
`calibrate` binary-searches these nested events for the first feasible diagnostic
model. Robust coordinates are a regression, not an exact spine claim.

`robust(...).guaranteed` is deliberately `False`. A theorem-level guarantee
also needs externally known strict separation conditions, and independently
perturbed duplicate copies can split across rounded bins. The function is a
relation-recovery and feasibility diagnostic unless those additional assumptions
are established outside the API.

## Source and verification boundary

The live dissertation TeX is the mathematical authority:

- `chapters/ch2/`: basic MM and streaming tie order;
- `chapters/ch3/`: structured partitions and sorting-network MILP;
- `chapters/ch4/`: triangle feasibility, LP/ILP, rank pruning, and noise.

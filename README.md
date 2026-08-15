# turnpike

A Python package, with C++17 kernels, for reconstructing one-dimensional point
sets from **unordered pairwise distances** — the Turnpike problem — including
uncertain, structured, and incomplete observations.

## The problem

Given a set of `n` points on a line, you can read off all `m = n(n-1)/2`
pairwise distances between them. The Turnpike problem is the inverse: you are
handed that multiset of distances *with no indication of which pair produced
which distance*, and asked to recover the original points. The related Beltway
problem places the points on a circle.

The same loss of correspondence arises in partial digestion for genomic mapping,
molecular structure determination, tandem mass spectrometry, and DNA-based
error-correcting codes.

Exact Turnpike has practical reconstruction algorithms, although its worst-case
complexity remains open. Bounded-error Turnpike and related noisy variants are
strongly NP-hard, and real observations are also commonly rounded, incomplete,
or duplicated. The package therefore combines large-scale fixed-point fitting,
moderate-size global fitting, and exact or diagnostic certification:

| Approach | Regime | Guarantee |
|---|---|---|
| Streaming coordinate ascent / MM | ~10⁵ points | fixed-point method; no local or global optimality claim |
| Sorting-network MILP | moderate `n` | solver-reported global best fit within the supplied span bound |
| Triangle-equality LP/ILP | moderate `n` | exact realizability certificate only when integral and post-verified |

**Beltway** is handled through the missing-distance machinery rather than a
separate code path: in the real-arithmetic model, Beltway on `n` points reduces
to Turnpike with missing distances on `2n + 1` points, and the partitioned
routines below are exactly what that reduction requires. The reduction itself is
laid out in the MM work.

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

The papers give the original formulations; this package is their reference
implementation. One distinction is easy to miss in the journal presentation and
is made explicit below: the observed-only and missing-row-imputation updates
optimize **different objectives**.

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

## Full-data coordinate ascent and MM

The scalable full-data solver uses the j-major difference matrix $Q$ and the
centered, sorted unit search space

$$
\mathcal Z_n=\{z\in\mathbb R^n:\mathbf 1^\top z=0,\ \lVert z\rVert_2=1,
z_1\le\cdots\le z_n\}.
$$

For a current $z$, let $P(z)$ denote the deterministic co-sorting of the input
distances $d$ with the predicted differences $Qz$. One update is

$$
u=Q^\top P(z)d,\qquad z^+=u/\lVert u\rVert_2.
$$

This is exact two-block coordinate ascent on

$$
\max_{z\in\mathcal Z_n,\;P\in\mathcal S_m}\langle Qz,Pd\rangle.
$$

It is also MM on the profiled objective
$\Phi(z)=\max_P\langle Qz,Pd\rangle$: the active branch
$G_t(z)=\langle Qz,P_t d\rangle$ is a global minorizer of $\Phi$ that touches it
at the current iterate. Because $\lVert Qz\rVert_2^2=n$ on $\mathcal Z_n$,
maximizing this correlation is equivalent to minimizing the complete-data half
squared loss. No PAV projection is needed in the full-data update: canonical
co-sorting makes $u$ nondecreasing.

After sorting `d` once, the streaming kernel takes `O(m log n)` time per
iteration and `O(n) = O(√m)` auxiliary words beyond the stored distances and
outputs. This is the structure behind the ~100,000-point scale.

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
- `mm(d, z)` → iterates the normalized assignment–coordinate update
- `solve(d, z)` → iterates MM, regresses the final assignment, anchors at zero

In exact arithmetic with deterministic ties, the uncapped map has a finite
assignment image and reaches a fixed point after finitely many iterations. The
floating-point APIs retain `tol` and `max_iter`. A fixed point need not be a
local or global optimum, so no reconstruction-optimality guarantee is claimed.
Internally the iteration ranks an unscaled raw representative and keeps a
separate unit vector for stopping and output; this prevents normalization from
splitting exact interval ties.

## Partitioned updates: correlation ascent and imputation MM

For distances observed in groups rather than as one undifferentiated multiset,
each observed cell is a multiset `D[c]` assigned to intervals `I[c]`, specified
by j-major row indices or `(i, j)` pairs. Cells are disjoint; omitted intervals
are treated as missing.

```python
D = [np.array([1, 1, 7]), np.array([6, 8])]
I = [np.array([0, 1, 4]), np.array([2, 3])]  # interval row 5 is missing

z_corr = tp.bmm(D, I, [0, 1, 2, 8])
z_loss = tp.impute(D, I, [0, 1, 2, 8])
```

The implementation supports products of independent exchangeable cells. A fixed
assignment is a singleton cell; a general permutation family needs its own exact
assignment oracle and is not silently treated as a cell. The two public API
pairs share the same cellwise assignment and unweighted coordinate-space PAV
machinery, but they optimize different objectives when rows are missing:

- `bstep(D, I, z)` / `bmm(D, I, z, tol=..., max_iter=...)` perform
  observed-correlation projected ascent; omitted rows contribute zero.
- `impute_step(D, I, z)` / `impute(D, I, z, tol=..., max_iter=...)` perform
  imputation MM for profiled observed half squared loss; omitted rows contribute
  their current predictions to the surrogate only.

For cell $c$, let $Q_c$ contain its interval rows and let $P_{c,t}$ be the
co-sorted assignment at the current normalized iterate $z_t$. Define

$$
g_t=\sum_c Q_c^\top P_{c,t}D_c,
\qquad
\Phi_{\mathrm{corr}}(z,P)=\sum_c\langle Q_cz,P_cD_c\rangle.
$$

The observed-only step is

$$
z_{t+1}=\mathrm{unit}(\mathrm{pav}(g_t)).
$$

Thus `bstep` is one exact projected coordinate-ascent step for
$\Phi_{\mathrm{corr}}$, and `bmm` iterates it. PAV is necessary because
independently co-sorted cells can make $g_t$ nonmonotone. With deterministic
ties, the exact uncapped `bmm` map reaches a fixed point finitely; the
floating-point routine uses `tol` and `max_iter`. If rows are missing, this
correlation objective is not observed least squares because
$\sum_c\lVert Q_cz\rVert_2^2$ varies with $z$.

For the imputation mode, let $Q_0$ contain the missing rows,
$L_0=Q_0^\top Q_0$, and

$$
L_{\mathrm{obs}}(z,P)=\frac12\sum_c\lVert Q_cz-P_cD_c\rVert_2^2,
\qquad
\rho(z)=\min_P L_{\mathrm{obs}}(z,P).
$$

The implementation fills each missing row with its current prediction $Q_0z_t$
and returns

$$
z_{t+1}=\mathrm{unit}\!\left(
  \mathrm{pav}(g_t+L_0z_t)
\right).
$$

This update exactly minimizes the tight global majorizer

$$
G_t(z)=L_{\mathrm{obs}}(z,P_t)
       +\frac12\lVert Q_0(z-z_t)\rVert_2^2.
$$

Indeed, $\rho(z)\le L_{\mathrm{obs}}(z,P_t)\le G_t(z)$ for every $z$, while
$G_t(z_t)=\rho(z_t)$. Consequently, in exact arithmetic,

$$
\rho(z_{t+1})\le G_t(z_{t+1})\le G_t(z_t)=\rho(z_t),
$$

with strict decrease when the normalized iterate changes. Missing predictions
are optimization-transfer terms, not observations. Because the surrogate
depends continuously on $z_t$, finite assignment space does not imply finite
termination for `impute`; the guarantee is monotone, bounded profiled-loss
values, not global optimality or convergence of the iterates.

> **Clarification relative to the journal presentation.** Its displayed
> observed-only objective omits the null block, while its pseudocode fills that
> block with current predictions. The API exposes both mathematically distinct
> updates. `bstep`/`bmm` are correlation ascent; `impute_step`/`impute` implement
> the pseudocode and are genuine MM for $\rho$. They therefore guarantee
> observed-loss descent, not observed-correlation ascent.

If no intervals are missing, $Q_0$ is empty and the two modes coincide:
`impute_step` equals `bstep`, and `impute` equals `bmm`, up to floating-point
roundoff. If one exchangeable cell contains every row, PAV is inactive and these
calls reduce to `step` and `mm`.

A three-point example shows why the two objectives cannot be conflated. Using
the API's zero-based endpoints, observe singleton distance 1 on `(0, 1)` and
distance 2 on `(0, 2)`, omit `(1, 2)`, and take
$z_t=(-5,1,4)/\sqrt{42}$. One `impute_step` lowers observed correlation from
approximately `3.703280` to `3.690960` while lowering profiled observed half
squared loss from approximately `0.189577` to `0.188267` — exactly the behavior
promised by the imputation-MM contract, and impossible for a correlation-ascent
step.

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

| Mode or returned result | Meaning |
|---|---|
| `triangle(..., lp=False)` | ILP with spine basis and rank pruning by default |
| `triangle(..., lp=True)` | LP with the full triangle system by default |
| integral, exactly post-verified `P` | realizability certificate (`guaranteed=True`) |
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

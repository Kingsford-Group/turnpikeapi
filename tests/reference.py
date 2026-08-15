from itertools import permutations, product

import numpy as np
from scipy.optimize import linprog


def pairs0(n):
    return np.array([(i, j) for j in range(1, n) for i in range(j)], dtype=int)


def Q0(n):
    p = pairs0(n)
    Q = np.zeros((len(p), n))
    for k, (i, j) in enumerate(p):
        Q[k, i] = -1
        Q[k, j] = 1
    return Q


def order0(z):
    p = pairs0(len(z))
    d = z[p[:, 1]] - z[p[:, 0]]
    return np.lexsort((p[:, 0], p[:, 1] - p[:, 0], d))


def raw0(d, z):
    Q = Q0(len(z))
    a = order0(z)
    y = np.empty(len(d), dtype=float)
    y[a] = np.sort(d)
    return Q.T @ y


def pav0(z):
    n = len(z)
    best = None
    val = np.inf
    for mask in range(1 << (n - 1)):
        cuts = [0] + [i + 1 for i in range(n - 1) if mask >> i & 1] + [n]
        x = np.empty(n)
        a = []
        for i, j in zip(cuts[:-1], cuts[1:]):
            a.append(np.mean(z[i:j]))
            x[i:j] = a[-1]
        if np.all(np.diff(a) >= 0):
            e = np.sum((x - z) ** 2)
            if e < val:
                best, val = x, e
    return best


def block0(D, I, z, impute=False, null_z=None):
    Q = Q0(len(z))
    p = pairs0(len(z))
    qz = Q @ z
    y = Q @ (z if null_z is None else null_z) if impute else np.zeros(len(Q))
    for d, e in zip(D, I):
        e = np.asarray(e)
        if e.ndim == 2:
            e = e[:, 1] * (e[:, 1] - 1) // 2 + e[:, 0]
        a = np.lexsort((p[e, 0], p[e, 1] - p[e, 0], qz[e]))
        y[e[a]] = np.sort(d)
    return pav0(Q.T @ y)


def block_trace0(D, I, z, it, impute=False):
    s = np.sort(np.asarray(z, dtype=float))
    v = s - np.mean(s)
    v /= np.linalg.norm(v)
    for _ in range(it):
        s = block0(D, I, s, impute, null_z=v)
        v = s / np.linalg.norm(s)
    return v


def parts0(y, mu, tau=0):
    S = []
    for r in range(len(y)):
        for s in range(len(y)):
            if r == s and mu[r] < 2:
                continue
            for t in range(len(y)):
                if abs(y[r] + y[s] - y[t]) <= tau:
                    S.append((r, s, t))
    return np.asarray(S, dtype=int).reshape(-1, 3)


def triples0(n):
    return [(i, j, k) for k in range(2, n) for j in range(1, k) for i in range(j)]


def feasible0(d):
    m = len(d)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    p = pairs0(n)
    row = {(i, j): k for k, (i, j) in enumerate(p)}
    for a in set(permutations(tuple(d))):
        if all(
            a[row[i, j]] + a[row[j, k]] == a[row[i, k]]
            for i, j, k in triples0(n)
        ):
            return np.asarray(a)
    return None


def network0(x, N, C, u=1):
    x = list(x) + [u] * (N - len(x))
    for i, j, up in C:
        lo, hi = sorted((x[i], x[j]))
        x[i], x[j] = (lo, hi) if up else (hi, lo)
    return np.asarray(x)


def best_fit0(D, I, n, u, loss="l1"):
    p = pairs0(n)
    I = [np.asarray(e, dtype=int) for e in I]
    A = [set(permutations(tuple(d))) for d in D]
    best = (np.inf, None)
    for P in product(*A):
        e = np.concatenate(I)
        a = np.asarray([v for pc in P for v in pc], dtype=float)
        k = len(e)
        if loss == "l1":
            c = np.r_[np.zeros(n), np.ones(k)]
            nr = n + k
        else:
            c = np.r_[np.zeros(n), 1.0]
            nr = n + 1
        A_ub, b_ub = [], []
        for i in range(n - 1):
            row = np.zeros(nr)
            row[i], row[i + 1] = 1, -1
            A_ub.append(row)
            b_ub.append(0)
        for h, (q, d) in enumerate(zip(e, a)):
            i, j = p[q]
            v = np.zeros(nr)
            v[i], v[j] = -1, 1
            r = n + h if loss == "l1" else n
            v[r] = -1
            A_ub.append(v)
            b_ub.append(d)
            v = -v
            v[r] = -1
            A_ub.append(v)
            b_ub.append(-d)
        bounds = [(0, 0)] + [(0, u)] * (n - 1) + [(0, None)] * (nr - n)
        z = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if z.success and z.fun < best[0] - 1e-9:
            best = (z.fun, z.x[:n])
    return best

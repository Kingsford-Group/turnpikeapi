#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<T> parts_i(const std::vector<std::int64_t>& y, const I& mu) {
    std::vector<T> S;
    const int p = static_cast<int>(y.size());
    for (int t = 0; t < p; ++t) {
        int r = 0;
        int s = p - 1;
        while (r <= s) {
            const std::uint64_t x = static_cast<std::uint64_t>(y[r]) + y[s];
            const std::uint64_t z = y[t];
            if (x > z) {
                ++r;
            } else if (x < z) {
                --s;
            } else {
                if (r != s || mu[r] >= 2) {
                    S.push_back({r, s, t});
                    if (r != s) S.push_back({s, r, t});
                }
                ++r;
                --s;
            }
        }
    }
    std::sort(S.begin(), S.end());
    return S;
}

std::vector<T> parts(const V& y, const I& mu, double tau) {
    std::vector<T> S;
    const int p = static_cast<int>(y.size());
    if (tau == 0.0) {
        for (int t = 0; t < p; ++t) {
            int r = 0;
            int s = p - 1;
            while (r <= s) {
                const double x = y[r] + y[s];
                if (x > y[t]) {
                    ++r;
                } else if (x < y[t]) {
                    --s;
                } else {
                    if (r != s || mu[r] >= 2) {
                        S.push_back({r, s, t});
                        if (r != s) S.push_back({s, r, t});
                    }
                    ++r;
                    --s;
                }
            }
        }
        std::sort(S.begin(), S.end());
        return S;
    }
    for (int r = 0; r < p; ++r) {
        for (int s = r; s < p; ++s) {
            if (r == s && mu[r] < 2) continue;
            const double x = y[r] + y[s];
            auto it = std::lower_bound(y.begin(), y.end(), x + tau, std::greater<double>());
            for (; it != y.end() && *it >= x - tau; ++it) {
                const int t = static_cast<int>(it - y.begin());
                S.push_back({r, s, t});
                if (r != s) S.push_back({s, r, t});
            }
        }
    }
    std::sort(S.begin(), S.end());
    return S;
}

std::vector<T> parts_r(const V& y, const I& mu, double r0, double R) {
    std::vector<T> S;
    const int p = static_cast<int>(y.size());
    for (int r = 0; r < p; ++r) {
        for (int s = 0; s < p; ++s) {
            if (r == s && mu[r] < 2) continue;
            for (int t = 0; t < p; ++t) {
                const double x = std::abs(y[r] + y[s] - y[t]);
                const double q = std::max(0.0, x / 3.0 - R / 2.0);
                if (q <= r0) S.push_back({r, s, t});
            }
        }
    }
    return S;
}

double gap(const V& y) {
    V s;
    const int p = static_cast<int>(y.size());
    s.reserve(p * (p + 1) / 2);
    for (int i = 0; i < p; ++i)
        for (int j = i; j < p; ++j) s.push_back(y[i] + y[j]);
    std::sort(s.begin(), s.end());
    double g = std::numeric_limits<double>::infinity();
    for (const double x : y) {
        auto a = std::lower_bound(s.begin(), s.end(), x);
        auto b = std::upper_bound(a, s.end(), x);
        if (a != s.begin()) g = std::min(g, x - *std::prev(a));
        if (b != s.end()) g = std::min(g, *b - x);
    }
    return g;
}

V critical(const V& y, const I& mu, double R) {
    V v;
    const int p = static_cast<int>(y.size());
    v.reserve(static_cast<long>(p) * p * p);
    for (int r = 0; r < p; ++r) {
        for (int s = 0; s < p; ++s) {
            if (r == s && mu[r] < 2) continue;
            for (int t = 0; t < p; ++t)
                v.push_back(std::max(0.0, std::abs(y[r] + y[s] - y[t]) / 3.0 - R / 2.0));
        }
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

std::vector<std::uint8_t> rank_mask(int n, const I& mu) {
    const int m = static_cast<int>(c2(n));
    const int p = static_cast<int>(mu.size());
    I M(p + 1, 0);
    for (int r = 0; r < p; ++r) M[r + 1] = M[r] + mu[r];
    std::vector<std::uint8_t> a(static_cast<long>(m) * p, 1);
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            const int lo = (i + 1) * (n - j);
            const int h = j - i + 1;
            const int hi = m - h * (h - 1) / 2 + 1;
            for (int r = 0; r < p; ++r)
                a[ix(i, j) * p + r] = M[r + 1] >= lo && M[r] + 1 <= hi;
        }
    }
    return a;
}

namespace {

void add(TM& M, long r, long c, double a) {
    M.r.push_back(r);
    M.c.push_back(c);
    M.a.push_back(a);
}

}  // namespace

TM tri_model(int n, const I& mu, const std::vector<T>& S, bool basis, bool prune) {
    const long m = c2(n);
    const long p = mu.size();
    const auto H = triples(n, basis);
    const long h = H.size();
    const long q = S.size();
    TM M;
    M.np = m * p;
    M.nt = h * q;
    M.ub.assign(M.np + M.nt, 1);
    M.b.reserve(m + p + h + 3 * h * p);
    const long nnz = 2 * m * p + 3 * h * p + 4 * h * q;
    M.r.reserve(nnz);
    M.c.reserve(nnz);
    M.a.reserve(nnz);
    std::vector<std::uint8_t> A;
    if (prune) A = rank_mask(n, mu);
    std::vector<I> B(3 * p);
    for (long s = 0; s < q; ++s)
        for (int w = 0; w < 3; ++w) B[w * p + S[s][w]].push_back(s);
    if (prune)
        std::copy(A.begin(), A.end(), M.ub.begin());

    long r = 0;
    for (long e = 0; e < m; ++e, ++r) {
        for (long a = 0; a < p; ++a) add(M, r, e * p + a, 1.0);
        M.b.push_back(1.0);
    }
    for (long a = 0; a < p; ++a, ++r) {
        for (long e = 0; e < m; ++e) add(M, r, e * p + a, 1.0);
        M.b.push_back(mu[a]);
    }
    for (long u = 0; u < h; ++u) {
        const long z = M.np + u * q;
        for (long a = 0; a < q; ++a) {
            add(M, r, z + a, 1.0);
            if (prune) {
                const auto [i, j, k] = H[u];
                const auto [x, y, w] = S[a];
                M.ub[z + a] = A[ix(i, j) * p + x] && A[ix(j, k) * p + y] && A[ix(i, k) * p + w];
            }
        }
        ++r;
        M.b.push_back(1.0);
    }
    for (long u = 0; u < h; ++u) {
        const auto [i, j, k] = H[u];
        const long e[3] = {ix(i, j), ix(j, k), ix(i, k)};
        const long z = M.np + u * q;
        for (int w = 0; w < 3; ++w) {
            for (long a = 0; a < p; ++a, ++r) {
                add(M, r, e[w] * p + a, 1.0);
                for (const int s : B[w * p + a]) add(M, r, z + s, -1.0);
                M.b.push_back(0.0);
            }
        }
    }
    return M;
}

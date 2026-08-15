#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

long c2(long n) { return n * (n - 1) / 2; }

long ix(int i, int j) { return static_cast<long>(j) * (j - 1) / 2 + i; }

std::vector<std::array<int, 2>> pairs(int n) {
    std::vector<std::array<int, 2>> p;
    p.reserve(c2(n));
    for (int j = 1; j < n; ++j)
        for (int i = 0; i < j; ++i) p.push_back({i, j});
    return p;
}

std::vector<T> triples(int n, bool basis) {
    std::vector<T> h;
    if (basis) {
        h.reserve(c2(n - 1));
        for (int k = 2; k < n; ++k)
            for (int j = 1; j < k; ++j) h.push_back({0, j, k});
    } else {
        h.reserve(n * (n - 1) * (n - 2) / 6);
        for (int k = 2; k < n; ++k)
            for (int j = 1; j < k; ++j)
                for (int i = 0; i < j; ++i) h.push_back({i, j, k});
    }
    return h;
}

V q(const V& z) {
    V d(c2(z.size()));
    for (int j = 1; j < static_cast<int>(z.size()); ++j)
        for (int i = 0; i < j; ++i) d[ix(i, j)] = z[j] - z[i];
    return d;
}

V qt(const V& d, int n) {
    V z(n, 0.0);
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            const double x = d[ix(i, j)];
            z[i] -= x;
            z[j] += x;
        }
    }
    return z;
}

V iso(const V& z) {
    const int n = static_cast<int>(z.size());
    V s(n), w(n), v(n);
    I a(n), b(n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        s[k] = z[i];
        w[k] = 1.0;
        a[k] = b[k] = i;
        while (k && s[k - 1] / w[k - 1] > s[k] / w[k]) {
            s[k - 1] += s[k];
            w[k - 1] += w[k];
            b[k - 1] = b[k];
            --k;
        }
        ++k;
    }
    for (int t = 0; t < k; ++t)
        std::fill(v.begin() + a[t], v.begin() + b[t] + 1, s[t] / w[t]);
    return v;
}

namespace {

struct E {
    double x;
    int i;
    int j;
};

struct C {
    bool operator()(const E& a, const E& b) const {
        if (a.x != b.x) return a.x > b.x;
        const int la = a.j - a.i;
        const int lb = b.j - b.i;
        if (la != lb) return la > lb;
        return a.i > b.i;
    }
};

template <class F>
V stream(const V& z, F pick) {
    const int n = static_cast<int>(z.size());
    std::priority_queue<E, std::vector<E>, C> h;
    V u(n, 0.0);
    for (int i = 0; i + 1 < n; ++i) h.push({z[i + 1] - z[i], i, i + 1});
    for (long t = 0; t < c2(n); ++t) {
        const E e = h.top();
        h.pop();
        const double d = pick(t, e);
        u[e.i] -= d;
        u[e.j] += d;
        if (e.j + 1 < n) h.push({z[e.j + 1] - z[e.i], e.i, e.j + 1});
    }
    return u;
}

void unit(V& z) {
    const double r = std::sqrt(std::inner_product(z.begin(), z.end(), z.begin(), 0.0));
    for (double& x : z) x /= r;
}

void normal(V& z) {
    const double a = std::accumulate(z.begin(), z.end(), 0.0) / z.size();
    for (double& x : z) x -= a;
    unit(z);
}

double dist(const V& x, const V& y) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - y[i];
        s += d * d;
    }
    return std::sqrt(s);
}

}  // namespace

V mmu(const double* d0, const V& z) {
    return stream(z, [&](long t, const E&) { return d0[t]; });
}

std::pair<V, V> mm_state(const double* d, V s, double eps, int it) {
    V z = s;
    normal(z);
    for (int t = 0; t < it; ++t) {
        V u = stream(s, [&](long k, const E&) { return d[k]; });
        V v = u;
        unit(v);
        const double e = dist(v, z);
        s.swap(u);
        z.swap(v);
        if (e < eps) break;
    }
    return {z, s};
}

V mm(const double* d, V s, double eps, int it) {
    return mm_state(d, std::move(s), eps, it).first;
}

V bmmu(const double* d, const I& o, const I& g, const V& s, const V& z, bool impute) {
    I c(o.size() - 1, 0);
    V u = stream(s, [&](long, const E& e) {
        const int p = g[ix(e.i, e.j)];
        if (!p) return impute ? z[e.j] - z[e.i] : 0.0;
        return d[o[p - 1] + c[p - 1]++];
    });
    return iso(u);
}

V bmm(const double* d, const I& o, const I& g, V s, double eps, int it, bool impute) {
    V z = s;
    normal(z);
    for (int t = 0; t < it; ++t) {
        I c(o.size() - 1, 0);
        V u = stream(s, [&](long, const E& e) {
            const int p = g[ix(e.i, e.j)];
            if (!p) return impute ? z[e.j] - z[e.i] : 0.0;
            return d[o[p - 1] + c[p - 1]++];
        });
        u = iso(u);
        V v = u;
        unit(v);
        const double e = dist(v, z);
        s.swap(u);
        z.swap(v);
        if (e < eps) break;
    }
    return z;
}

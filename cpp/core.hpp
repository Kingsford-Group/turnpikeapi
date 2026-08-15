#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

using V = std::vector<double>;
using I = std::vector<int>;
using T = std::array<int, 3>;

struct TM {
    std::vector<long> r;
    std::vector<long> c;
    V a;
    V b;
    std::vector<std::uint8_t> ub;
    long np;
    long nt;
};

long c2(long n);
long ix(int i, int j);
std::vector<std::array<int, 2>> pairs(int n);
std::vector<T> triples(int n, bool basis);

V q(const V& z);
V qt(const V& d, int n);
V iso(const V& z);
V mmu(const double* d, const V& z);
std::pair<V, V> mm_state(const double* d, V z, double eps, int it);
V mm(const double* d, V z, double eps, int it);
V bmmu(const double* d, const I& o, const I& g, const V& s, const V& z, bool impute);
V bmm(const double* d, const I& o, const I& g, V z, double eps, int it, bool impute);

std::pair<int, std::vector<T>> net(int n);
std::vector<T> parts(const V& y, const I& mu, double tau);
std::vector<T> parts_i(const std::vector<std::int64_t>& y, const I& mu);
std::vector<T> parts_r(const V& y, const I& mu, double r, double R);
double gap(const V& y);
V critical(const V& y, const I& mu, double R);
std::vector<std::uint8_t> rank_mask(int n, const I& mu);
TM tri_model(int n, const I& mu, const std::vector<T>& S, bool basis, bool prune);

#include "core.hpp"

#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <class X>
std::vector<X> vec(const py::array_t<X, py::array::c_style | py::array::forcecast>& a) {
    return {a.data(), a.data() + a.size()};
}

template <class X>
py::array_t<X> arr(const std::vector<X>& a) {
    py::array_t<X> x(a.size());
    std::copy(a.begin(), a.end(), x.mutable_data());
    return x;
}

py::array_t<int> tri(const std::vector<T>& a) {
    py::array_t<int> x(py::array::ShapeContainer{
        static_cast<py::ssize_t>(a.size()), static_cast<py::ssize_t>(3)});
    int* p = x.mutable_data();
    for (const auto& t : a)
        for (int j = 0; j < 3; ++j) *p++ = t[j];
    return x;
}

PYBIND11_MODULE(_core, m) {
    m.def("pairs", [](int n) {
        const auto a = pairs(n);
        py::array_t<int> x(py::array::ShapeContainer{
            static_cast<py::ssize_t>(a.size()), static_cast<py::ssize_t>(2)});
        int* p = x.mutable_data();
        for (const auto& e : a) { *p++ = e[0]; *p++ = e[1]; }
        return x;
    });
    m.def("triples", [](int n, bool b) { return tri(triples(n, b)); });
    m.def("q", [](py::array_t<double, py::array::c_style | py::array::forcecast> z) { return arr(q(vec(z))); });
    m.def("qt", [](py::array_t<double, py::array::c_style | py::array::forcecast> d, int n) { return arr(qt(vec(d), n)); });
    m.def("iso", [](py::array_t<double, py::array::c_style | py::array::forcecast> z) { return arr(iso(vec(z))); });
    m.def("mmu", [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
                     py::array_t<double, py::array::c_style | py::array::forcecast> z) { return arr(mmu(d.data(), vec(z))); });
    m.def("mm", [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
                    py::array_t<double, py::array::c_style | py::array::forcecast> z,
                    double e, int it) { return arr(mm(d.data(), vec(z), e, it)); });
    m.def("mm_state", [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
                          py::array_t<double, py::array::c_style | py::array::forcecast> z,
                          double e, int it) {
        auto [v, s] = mm_state(d.data(), vec(z), e, it);
        return py::make_tuple(arr(v), arr(s));
    });
    m.def("bmmu", [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
                      py::array_t<int, py::array::c_style | py::array::forcecast> o,
                      py::array_t<int, py::array::c_style | py::array::forcecast> g,
                      py::array_t<double, py::array::c_style | py::array::forcecast> s,
                      py::array_t<double, py::array::c_style | py::array::forcecast> z,
                      bool h) { return arr(bmmu(d.data(), vec(o), vec(g), vec(s), vec(z), h)); });
    m.def("bmm", [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
                     py::array_t<int, py::array::c_style | py::array::forcecast> o,
                     py::array_t<int, py::array::c_style | py::array::forcecast> g,
                     py::array_t<double, py::array::c_style | py::array::forcecast> z,
                     double e, int it, bool h) { return arr(bmm(d.data(), vec(o), vec(g), vec(z), e, it, h)); });
    m.def("net", [](int n) {
        auto [N, C] = net(n);
        return py::make_tuple(N, tri(C));
    });
    m.def("parts", [](py::array_t<double, py::array::c_style | py::array::forcecast> y,
                       py::array_t<int, py::array::c_style | py::array::forcecast> u,
                       double e) { return tri(parts(vec(y), vec(u), e)); });
    m.def("parts_i", [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> y,
                         py::array_t<int, py::array::c_style | py::array::forcecast> u) {
        return tri(parts_i(vec(y), vec(u)));
    });
    m.def("parts_r", [](py::array_t<double, py::array::c_style | py::array::forcecast> y,
                         py::array_t<int, py::array::c_style | py::array::forcecast> u,
                         double r, double R) { return tri(parts_r(vec(y), vec(u), r, R)); });
    m.def("gap", [](py::array_t<double, py::array::c_style | py::array::forcecast> y) { return gap(vec(y)); });
    m.def("critical", [](py::array_t<double, py::array::c_style | py::array::forcecast> y,
                          py::array_t<int, py::array::c_style | py::array::forcecast> u,
                          double R) { return arr(critical(vec(y), vec(u), R)); });
    m.def("rank_mask", [](int n, py::array_t<int, py::array::c_style | py::array::forcecast> u) {
        const auto a = rank_mask(n, vec(u));
        py::array_t<bool> x(py::array::ShapeContainer{
            static_cast<py::ssize_t>(c2(n)), static_cast<py::ssize_t>(u.size())});
        bool* p = x.mutable_data();
        for (auto v : a) *p++ = v;
        return x;
    });
    m.def("tri_model", [](int n,
                           py::array_t<int, py::array::c_style | py::array::forcecast> u,
                           py::array_t<int, py::array::c_style | py::array::forcecast> s,
                           bool b, bool p) {
        std::vector<T> S(s.shape(0));
        for (py::ssize_t i = 0; i < s.shape(0); ++i)
            S[i] = {s.data(i, 0)[0], s.data(i, 1)[0], s.data(i, 2)[0]};
        const TM A = tri_model(n, vec(u), S, b, p);
        return py::make_tuple(arr(A.r), arr(A.c), arr(A.a), arr(A.b), arr(A.ub), A.np, A.nt);
    });
    m.attr("version") = "0.1.0";
}

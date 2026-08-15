#include "core.hpp"

#include <algorithm>

std::pair<int, std::vector<T>> net(int n) {
    int N = 1;
    while (N < n) N <<= 1;
    std::vector<T> C;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j; j >>= 1) {
            for (int i = 0; i < N; ++i) {
                const int q = i ^ j;
                if (q > i) C.push_back({i, q, (i & k) == 0});
            }
        }
    }
    return {N, C};
}


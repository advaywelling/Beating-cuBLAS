#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>

// used once per kernel to confirm correctness before bothering w timing
inline void check_error(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int M, int K, int N) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int idx = 0; idx < K; idx++)
        sum += a[i * K + idx] * b[idx * N + j];
      float diff = std::fabs(c[i * N + j] - sum);
      float tol  = 1e-2f * std::fabs(sum);
      assert(diff <= tol + 1e-3f);
    }
}

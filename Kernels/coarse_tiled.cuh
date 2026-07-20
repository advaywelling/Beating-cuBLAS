#pragma once
#include <cuda_runtime.h> 

constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

__global__ void coarse_tiled(float *a, float *b, float *c, int M, int K, int N) {
  __shared__ float share_a[BM * BK];
  __shared__ float share_b[BK * BN];

  const int tid = threadIdx.x;
  const int cRow = blockIdx.y; // tile index in C
  const int cCol = blockIdx.x;

  int aRow = tid / BK;
  int aCol = tid % BK;
  int bRow = tid / BN;
  int bCol = tid % BN;
  int threadRow = tid / BN; // each thread covers 8 rows
  int threadCol = tid % BN; // each thread covers 1 col

  float threadResults[TM] = {0.0f};

  // iterate over common edge in steps of depth BK
  for (int bkIdx{}; bkIdx < K; bkIdx += BK) {
    share_a[aRow * BK + aCol] = a[(cRow * BM + aRow) * K + bkIdx + aCol];
    share_b[bRow * BN + bCol] = b[((bkIdx + bRow) * N + (cCol * BN + bCol))];

    __syncthreads();

    for (int k{}; k < BK; k++) {
      float bVal = share_b[k * BN + threadCol];
      // actual coarsening
      for (int i{}; i < TM; i++) {
        threadResults[i] += share_a[(threadRow * TM + i) * BK + k] * bVal;
      }
    }

    __syncthreads();
  }
  for (int i{}; i < TM; i++) {
    c[(cRow * BM + threadRow * TM + i) * N + (cCol * BN + threadCol)] = threadResults[i];
  }
}

#pragma once
#include <cuda_runtime.h> 

constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

__global__ void coarse_tiled(float *a, float *b, float *c, int M, int K, int N) {
  __shared__ float share_a[BM * BK];
  __shared__ float share_b[BK * BN];

  const int tid = threadIdx.x;
  const int cRow = blockIdx.y; // tile index in C
  const int cCol = blockIdx.x;

  int aRow = tid / BK; // locations in tile
  int aCol = tid % BK;
  int bRow = tid / BN;
  int bCol = tid % BN;
  int threadRow = tid / BN; // each thread covers 8 rows
  int threadCol = tid % BN; // each thread covers 1 col

  float threadResults[TM] = {0.0f};

  // iterate over common edge in steps of depth BK
  for (int bkIdx{}; bkIdx < K; bkIdx += BK) {
    const int aRow_global = cRow * BM + aRow; // where aRow sits in A
    const int aCol_global = bkIdx + aCol; // where aCol sits in A
    share_a[aRow * BK + aCol] = (aRow_global < M && aCol_global < K) ? a[aRow_global * K + aCol_global] : 0.0f;

    const int bRow_global = bkIdx + bRow; // where bRow sits in B
    const int bCol_global = cCol * BN + bCol; // wehre bCol sits in B
    share_b[bRow * BN + bCol] = (bRow_global < K && bCol_global < N) ? b[bRow_global * N + bCol_global] : 0.0f;

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
  const int cCol_global = cCol * BN + threadCol;
  for (int i = 0; i < TM; i++) {
    const int cRow_global = cRow * BM + threadRow * TM + i;
    if (cRow_global < M && cCol_global < N) c[cRow_global * N + cCol_global] = threadResults[i];
  }
}

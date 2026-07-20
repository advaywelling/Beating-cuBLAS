#pragma once
#include <cuda_runtime.h>

#define TILE 16

__global__ void tiled(float *a, float *b, float *c, int M, int K, int N) {
  __shared__ float share_a[TILE][TILE];
  __shared__ float share_b[TILE][TILE];

  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  float sum = 0.0f;
  for(int i{}; i < (K + TILE - 1)/ TILE; i++) {
    share_a[ty][tx] = (row < M && (i * TILE + tx) < K) ? a[(row * K) + (i * TILE + tx)] : 0.0f;
    share_b[ty][tx] = ((i * TILE + ty) < K && col < N) ? b[((i * TILE + ty) * N + col)] : 0.0f;

    __syncthreads();

    for(int j{}; j < TILE; j++) {
      sum += share_a[ty][j] * share_b[j][tx];
    }

    __syncthreads();

  }
  if (row < M && col < N) {
    c[row * N + col] = sum;
  }
}

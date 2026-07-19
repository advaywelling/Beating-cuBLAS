#pragme once
#include <cuda_runtime.h>

__global__ void naive(float *a, float *b, float *c, int M, int K, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0;
    for (int i{}; i < K; i++) {
      sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
  }
}

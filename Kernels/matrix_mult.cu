%%writefile opt_mm.cu
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_mult(float *a, float *b, float *c, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < K) {
    float sum = 0;
    for (int i{}; i < N; i++) {
      sum += a[row * N + i] * b[i * K + col];
    }
    c[row * K + col] = sum;
  }
}

__global__ void tiled_matrix_mult(float *a, float *b, float *c, int M, int N, int K) {
  __shared__ float share_A[TILE][TILE];
  __shared__ float share_B [TILE][TILE];

  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  float sum = 0;

  // load shared mem
  for (int i{}; i < N / TILE; i++) {
    share_A[ty][tx] = a[row*N + (i * TILE + tx)];
    share_B[ty][tx] = b[(i * TILE + ty) * K + col];

    __syncthreads();

    for(int j{}; j < TILE; j++) {
      sum += share_A[ty][j] * share_B[j][tx];
    }
    __syncthreads();
  }
  c[row * K + col] = sum;
}

void check_error(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, int M, int N, int K) {
    for(int i{}; i < M; i++) {
        for(int j{}; j < K; j++) {
            float sum = 0;
            for(int idx{}; idx < N; idx++) {
                sum += a[i * N + idx] * b[idx * K + j];
            }
            assert(c[i * K + j] == sum);
        }
    }
}

int main() {

  constexpr int M = 1 << 10;
  constexpr int N = 1 << 10;
  constexpr int K = 1 << 10;
  constexpr size_t size_A = sizeof(float) * M * N;
  constexpr size_t size_B = sizeof(float) * N * K;
  constexpr size_t size_C = sizeof(float) * M * K;

  std::vector<float> a;
  a.reserve(M * N);
  std::vector<float> b;
  b.reserve(N * K);
  std::vector<float> c;
  c.resize(M * K);

  for (int i{}; i < M * N; i++) {
    a.push_back(rand() % 100);
  }
  for (int i{}; i < N * K; i++) {
    b.push_back(rand() % 100);
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_A);
  cudaMalloc(&d_b, size_B);
  cudaMalloc(&d_c, size_C);

  cudaMemcpy(d_a, a.data(), size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_B, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,(M + dimBlock.y - 1) / dimBlock.y);

  tiled_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(c.data(), d_c, size_C, cudaMemcpyDeviceToHost);

  check_error(a, b, c, M, N, K);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}

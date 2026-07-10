%%writefile opt_mm.cu
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
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
  __shared__ float share_a[TILE][TILE];
  __shared__ float share_b[TILE][TILE];

  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  float sum = 0.0f;
  for(int i{}; i < (N + TILE - 1)/ TILE; i++) {
    share_a[ty][tx] = (row < M && (i * TILE + tx) < N) ? a[(row * N) + (i * TILE + tx)] : 0.0f;
    share_b[ty][tx] = ((i * TILE + ty) < N && col < K) ? b[((i * TILE + ty) * K + col)] : 0.0f;

    __syncthreads();

    for(int j{}; j < TILE; j++) {
      sum += share_a[ty][j] * share_b[j][tx];
    }
    __syncthreads();
  }
  if (row < M && col < K) {
    c[row * K + col] = sum;
  }
}

void check_error(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, int M, int N, int K) {
    for (int i{}; i < M; i++) {
        for (int j{}; j < K; j++) {
            float sum = 0;
            for (int idx{}; idx < N; idx++) {
                sum += a[i * N + idx] * b[idx * K + j];
            }
            float diff = std::fabs(c[i * K + j] - sum);
            float tol = 1e-2f * std::fabs(sum);   // relative tolerance
            assert(diff <= tol + 1e-3f);          // + small absolute floor
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
  c.resize(M * K); // fills with 0s

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

  dim3 dimBlock(TILE, TILE);
  dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,(M + dimBlock.y - 1) / dimBlock.y);

  tiled_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(c.data(), d_c, size_C, cudaMemcpyDeviceToHost);

  check_error(a, b, c, M, N, K);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}

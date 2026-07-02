%%writefile matrixmult.cu
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>

__global__ void matrix_mult (int *a, int *b, int *c, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < K) {
    int sum = 0;
    for (int i{}; i < N; i++) {
      sum += a[row * N + i] * b[i * K + col];
    }
    c[row * K + col] = sum
  }
}

void check_error(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, int M, int N, int K) {
    for(int i{}; i < M; i++) {
        for(int j{}; j < K; j++) {
            int sum = 0;
            for(int idx{}; idx < N; idx++) {
                sum += a[i * N + idx] * b[idx * K + j];
            }
            assert(c[i * K + j] == sum);
        }
    }
}

int main() {

  constexpr int M = 1 << 10;
  constexpr int N = 1 << 8;
  constexpr int K = 1 << 10;
  constexpr size_t size_A = sizeof(int) * M * N;
  constexpr size_t size_B = sizeof(int) * N * K;
  constexpr size_t size_C = sizeof(int) * M * K;

  std::vector<int> a;
  a.reserve(M * N);
  std::vector<int> b;
  b.reserve(N * K);
  std::vector<int> c;
  c.resize(M * K);

  for (int i{}; i < M * N; i++) {
    a.push_back(rand() % 100);
  }
  for (int i{}; i < N * K; i++) {
    b.push_back(rand() % 100);
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_A);
  cudaMalloc(&d_b, size_B);
  cudaMalloc(&d_c, size_C);

  cudaMemcpy(d_a, a.data(), size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_B, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,(M + dimBlock.y - 1) / dimBlock.y);

  matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(c.data(), d_c, size_C, cudaMemcpyDeviceToHost);

  check_error(a, b, c, M, N, K);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}

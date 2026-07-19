%%writefile opt_mm.cu
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A = M x K 
// B = K x N 
// C = M x N

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s @ %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

#define TILE 16
const int BM = 64, BN = 64, BK = 8, TM = 8;

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

void check_error(std::vector<float>&a, std::vector<float>&b, std::vector<float>&c,
                 int M, int K, int N) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {          // N, not K
      float sum = 0;
      for (int idx = 0; idx < K; idx++)    // K, not N
        sum += a[i*K + idx] * b[idx*N + j];
      float diff = std::fabs(c[i*N + j] - sum);
      float tol = 1e-2f * std::fabs(sum);
      assert(diff <= tol + 1e-3f);
    }
}

int main() {

  constexpr int M = 1 << 9;
  constexpr int K = 1 << 8;
  constexpr int N = 1 << 10;
  constexpr size_t size_A = sizeof(float) * M * K;
  constexpr size_t size_B = sizeof(float) * K * N;
  constexpr size_t size_C = sizeof(float) * M * N;

  std::vector<float> a;
  a.reserve(M * K);
  std::vector<float> b;
  b.reserve(K * N);
  std::vector<float> c;
  c.resize(M * N);

  for (int i{}; i < M * K; i++) {
    a.push_back(rand() % 100);
  }
  for (int i{}; i < K * N; i++) {
    b.push_back(rand() % 100);
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_A);
  cudaMalloc(&d_b, size_B);
  cudaMalloc(&d_c, size_C);

  cudaMemcpy(d_a, a.data(), size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_B, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE, TILE);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,(M + dimBlock.y - 1) / dimBlock.y);

  dim3 block(512);
  dim3 grid(N / BN, M / BM);
  coarse_tiled<<<grid, block>>>(d_a, d_b, d_c, M, K, N);

  //tiled<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, K, N);

  cudaMemcpy(c.data(), d_c, size_C, cudaMemcpyDeviceToHost);

  check_error(a, b, c, M, K, N);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}

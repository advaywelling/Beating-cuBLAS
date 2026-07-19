#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// abort loudly on any failed CUDA call; prints the line so you know where
#define CUDA_CHECK(x) do {                                  \
  cudaError_t err_ = (x);                                   \
  if (err_ != cudaSuccess) {                                \
    printf("CUDA error: %s @ %s:%d\n",                      \
           cudaGetErrorString(err_), __FILE__, __LINE__);   \
    exit(1);                                                \
  }                                                         \
} while (0)

// time a launch: warmup, then `iters` runs, return average ms.
// takes any callable (lambda) so it works for kernels and cuBLAS alike.
template <class Launch>
float bench(Launch launch, int warmup = 5, int iters = 50) {
  for (int i = 0; i < warmup; i++) launch();
  CUDA_CHECK(cudaGetLastError());        // catches a bad launch config now
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// matmul does 2*M*N*K flops (one multiply + one add per inner term)
inline double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (ms * 1e6);
}

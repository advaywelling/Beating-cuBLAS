#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do {                                  \
  cudaError_t err_ = (x);                                   \
  if (err_ != cudaSuccess) {                                \
    printf("CUDA error: %s @ %s:%d\n",                      \
           cudaGetErrorString(err_), __FILE__, __LINE__);   \
    exit(1);                                                \
  }                                                         \
} while (0)

template <class Launch>
float bench(Launch launch, int warmup = 5, int iters = 50) {
  for (int i = 0; i < warmup; i++) launch();
  CUDA_CHECK(cudaGetLastError());     
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

// matmul does 2*M*N*K flops
inline double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (ms * 1e6);
}

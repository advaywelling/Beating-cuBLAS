#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>

#include "common.cuh"
#include "check.cuh"
#include "Kernels/naive.cuh"
#include "Kernels/tiled.cuh"
#include "Kernels/coarse_tiled.cuh"

int main() {
  const int M = 1024, K = 1024, N = 1024;
  const size_t sA = sizeof(float)*M*K, sB = sizeof(float)*K*N, sC = sizeof(float)*M*N;

  cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s | SMs=%d | %.2f GHz | cc %d.%d\n\n", p.name, p.multiProcessorCount, p.clockRate/1e6, p.major, p.minor);

  // host data 
  std::vector<float> hA(M*K), hB(K*N), hC(M*N);
  for (auto& x : hA) x = (rand() % 100) / 100.0f;
  for (auto& x : hB) x = (rand() % 100) / 100.0f;

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sA));
  CUDA_CHECK(cudaMalloc(&dB, sB));
  CUDA_CHECK(cudaMalloc(&dC, sC));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice));

  // launch config
  dim3 blk16(TILE, TILE), grd16((N+TILE-1)/TILE, (M+TILE-1)/TILE);
  dim3 blkC(512),         grdC(N/BN, M/BM);

  auto run_naive  = [&]{ naive       <<<grd16, blk16>>>(dA,dB,dC,M,K,N); };
  auto run_tiled  = [&]{ tiled       <<<grd16, blk16>>>(dA,dB,dC,M,K,N); };
  auto run_coarse = [&]{ coarse_tiled<<<grdC,  blkC >>>(dA,dB,dC,M,K,N); };

  cublasHandle_t h; cublasCreate(&h);
  const float alpha = 1.f, beta = 0.f;
  // row-major C=A*B == col-major C^T=B^T*A^T; pass B,A swapped, dims swapped.
  auto run_cublas = [&]{ cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N); };

  // check for correctness first
  auto verify = [&](const char* name, auto launch){
    launch(); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sC, cudaMemcpyDeviceToHost));
    check_error(hA, hB, hC, M, K, N);
    printf("  %-13s PASS\n", name);
  };
  printf("correctness:\n");
  verify("naive",  run_naive);
  verify("tiled",  run_tiled);
  verify("coarse", run_coarse);

  // am i better than cublas yet
  float t_cub = bench(run_cublas);
  float t_nai = bench(run_naive);
  float t_til = bench(run_tiled);
  float t_coa = bench(run_coarse);
  double gref = gflops(M, N, K, t_cub);

  printf("\n%-13s %9s %11s %10s\n", "kernel", "ms", "GFLOP/s", "% cuBLAS");
  auto row = [&](const char* n, float ms){
    double g = gflops(M, N, K, ms);
    printf("%-13s %9.3f %11.1f %9.1f%%\n", n, ms, g, 100.0*g/gref);
  };
  row("cuBLAS", t_cub);
  row("naive",  t_nai);
  row("tiled",  t_til);
  row("coarse", t_coa);

  cublasDestroy(h);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}

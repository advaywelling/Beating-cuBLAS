%%writefile vectoradd.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

__global__ void add(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void check_error(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  for (size_t i{}; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  constexpr int N = 1 << 16;
  constexpr size_t bytes = sizeof(int) * N;

  std::vector<int> a;
  a.reserve(N);
  std::vector<int> b;
  b.reserve(N);
  std::vector<int> c;
  c.resize(N);

  for (int i{}; i < N; i++) {
    a.push_back(rand() % 100);
    b.push_back(rand() % 100);
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  check_error(a, b, c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  std::cout << "Final value: " << c[N-1] << '\n';
}

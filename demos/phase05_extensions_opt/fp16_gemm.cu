#include <cuda_fp16.h>

#include "cuda_check.cuh"

__global__ void gemm_half_naive(const __half* __restrict__ A, const __half* __restrict__ B,
                                __half* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    const float a = __half2float(A[row * K + k]);
    const float b = __half2float(B[k * N + col]);
    acc += a * b;
  }
  C[row * N + col] = __float2half(acc);
}

void launch_fp16_gemm(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_half_naive<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}

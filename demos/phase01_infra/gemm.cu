#include "cuda_check.cuh"

__global__ void gemm_naive_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

__global__ void gemm_tiled_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int TILE = 16;
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE + ty;
  int col = bx * TILE + tx;
  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + tx;
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
    int b_row = t * TILE + ty;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_naive_ker<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_tiled_ker<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}

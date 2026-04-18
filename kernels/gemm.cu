// kernels/gemm.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "gemm.hpp"
#include "cuda_utils.hpp"

// ============================================================================
// Naive GEMM: C = alpha * A @ B + beta * C
// Each thread computes one element of C
// ============================================================================
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int N, int K,
    float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// Tiling GEMM: Uses shared memory for A and B tiles
// Each block computes a TILE_SIZE x TILE_SIZE block of C
// ============================================================================
#define TILE_SIZE_GEMM 16

__global__ void gemm_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) {
    __shared__ float As[TILE_SIZE_GEMM][TILE_SIZE_GEMM];
    __shared__ float Bs[TILE_SIZE_GEMM][TILE_SIZE_GEMM];

    int row = blockIdx.y * TILE_SIZE_GEMM + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_GEMM + threadIdx.x;
    float c_val = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE_GEMM - 1) / TILE_SIZE_GEMM; ++tile) {
        // Load A tile
        if (row < M && (tile * TILE_SIZE_GEMM + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE_GEMM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (col < N && (tile * TILE_SIZE_GEMM + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE_GEMM + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE_GEMM; ++k) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * c_val + beta * C[row * N + col];
    }
}

void gemm_cuda(const Tensor& A, const Tensor& B, Tensor& C,
               bool transpose_A, bool transpose_B,
               float alpha, float beta) {
    int M = transpose_A ? A.shape_[1] : A.shape_[0];
    int K = transpose_A ? A.shape_[0] : A.shape_[1];
    int N = transpose_B ? B.shape_[0] : B.shape_[1];

    // Use tiling kernel by default
    dim3 block(TILE_SIZE_GEMM, TILE_SIZE_GEMM);
    dim3 grid((N + TILE_SIZE_GEMM - 1) / TILE_SIZE_GEMM,
               (M + TILE_SIZE_GEMM - 1) / TILE_SIZE_GEMM);

    gemm_tiling_kernel<<<grid, block>>>(
        A.data_cuda(), B.data_cuda(), C.data_cuda(),
        M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void gemm_cpu(const Tensor& A, const Tensor& B, Tensor& C,
              bool transpose_A, bool transpose_B,
              float alpha, float beta) {
    int M = transpose_A ? A.shape_[1] : A.shape_[0];
    int K = transpose_A ? A.shape_[0] : A.shape_[1];
    int N = transpose_B ? B.shape_[0] : B.shape_[1];

    const float* a = A.data_cpu();
    const float* b = B.data_cpu();
    float* c = C.data_cpu();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int a_idx = transpose_A ? k * M + i : i * K + k;
                int b_idx = transpose_B ? j * K + k : k * N + j;
                sum += a[a_idx] * b[b_idx];
            }
            int c_idx = i * N + j;
            c[c_idx] = alpha * sum + beta * c[c_idx];
        }
    }
}
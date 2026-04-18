// kernels/layernorm.cu
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include "tensor.hpp"
#include <cmath>

// ============================================================================
// LayerNorm Kernel
// Normalizes over last dim, learns gamma (scale) and beta (bias)
// Each thread block handles one row
// ============================================================================

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    float* output,
    const float* __restrict__ gamma,  // scale params [d]
    const float* __restrict__ beta,   // bias params [d]
    int rows, int d,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;

    // Step 1: Compute mean
    float mean = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        mean += input[row * d + j];
    }
    // Warp-level reduction for mean
    float warp_mean = mean;
    for (int offset = 16; offset >= 1; offset /= 2) {
        warp_mean += __shfl_down_sync(0xffffffff, warp_mean, offset);
    }
    if (tid == 0) mean = warp_mean;
    __syncthreads();
    mean /= d;

    // Step 2: Compute variance
    float var = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        float diff = input[row * d + j] - mean;
        var += diff * diff;
    }
    // Warp-level reduction for var
    float warp_var = var;
    for (int offset = 16; offset >= 1; offset /= 2) {
        warp_var += __shfl_down_sync(0xffffffff, warp_var, offset);
    }
    if (tid == 0) var = warp_var;
    __syncthreads();
    var /= d;
    float inv_std = rsqrtf(var + eps);

    // Step 3: Normalize and apply gamma/beta
    for (int j = tid; j < d; j += blockDim.x) {
        float x_norm = (input[row * d + j] - mean) * inv_std;
        float y = gamma[j] * x_norm + beta[j];
        output[row * d + j] = y;
    }
}

// Host wrapper
void layernorm_cuda(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                    Tensor& output, float eps = 1e-5f) {
    int d = input.shape_.back();
    int rows = input.numel() / d;

    int block_size = 256;
    int num_blocks = rows;

    layernorm_kernel<<<num_blocks, block_size>>>(
        input.data_cuda(),
        output.data_cuda(),
        gamma.data_cuda(),
        beta.data_cuda(),
        rows, d, eps);
    CUDA_CHECK(cudaGetLastError());
}
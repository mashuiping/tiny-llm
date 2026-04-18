// kernels/softmax.cu
#include <cuda_runtime.h>
#include <cmath>
#include "cuda_utils.hpp"
#include "tensor.hpp"

// ============================================================================
// Row-wise Softmax Kernel
// One thread block per row, threads within block compute max and sum
// using warp-level primitives (__shfl_down_sync)
// ============================================================================

__global__ void softmax_row_kernel(
    const float* __restrict__ input,
    float* output,
    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;

    // Step 1: Find row max ( warp reduction )
    float max_val = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, input[row * cols + j]);
    }

    // Warp-level max reduction
    // Note: full warp shuffle requires all threads in warp to participate
    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    // Handle remaining threads within warp
    if (tid < 32) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 16));
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 8));
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 4));
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 2));
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 1));
    }

    // Step 2: Compute exp and sum
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float exp_val = expf(input[row * cols + j] - max_val);
        output[row * cols + j] = exp_val;
        sum += exp_val;
    }

    // Warp-level sum reduction
    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid < 32) {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    // Step 3: Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int j = tid; j < cols; j += blockDim.x) {
        output[row * cols + j] *= inv_sum;
    }
}

// Host wrapper
void softmax_cuda(const Tensor& input, Tensor& output, int dim = -1) {
    int rows = 1, cols = input.numel();
    if (dim == -1 || dim == (int)input.shape_.size() - 1) {
        // Last dim: softmax over each row
        cols = input.shape_.back();
        rows = input.numel() / cols;
    }

    int block_size = 256;
    int num_blocks = rows;

    softmax_row_kernel<<<num_blocks, block_size>>>(
        input.data_cuda(),
        output.data_cuda(),
        rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

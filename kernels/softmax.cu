// kernels/softmax.cu
#include <cuda_runtime.h>
#include <cmath>
#include "cuda_utils.hpp"
#include "tensor.hpp"

// ============================================================================
// Row-wise Softmax Kernel
// One thread block per row, threads within block compute max and sum
// using a two-level reduction: block-level via shared memory, then
// warp-level via __shfl_down_sync
// ============================================================================

__global__ void softmax_row_kernel(
    const float* __restrict__ input,
    float* output,
    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Step 1: Find row max using block reduction
    float thread_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        thread_max = fmaxf(thread_max, input[row * cols + j]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Block-level max reduction (reduce from blockDim.x down to 32)
    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Warp-level reduction (only tid < 32 participate)
    float max_val = sdata[tid];
    if (tid < 32) {
        for (int offset = 16; offset >= 1; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];

    // Step 2: Compute exp and sum
    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float val = expf(input[row * cols + j] - max_val);
        output[row * cols + j] = val;
        thread_sum += val;
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level sum reduction
    float sum = sdata[tid];
    if (tid < 32) {
        for (int offset = 16; offset >= 1; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) sdata[0] = sum;
    }
    __syncthreads();
    sum = sdata[0];

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

    softmax_row_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_cuda(),
        output.data_cuda(),
        rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

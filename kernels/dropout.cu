// kernels/dropout.cu
#include <cuda_runtime.h>
#include <cstdlib>
#include "cuda_utils.hpp"
#include "tensor.hpp"

// ============================================================================
// Dropout Forward Kernel
// During training: mask each element with probability p, scale by 1/(1-p)
// During inference: identity (output = input)
// ============================================================================

__global__ void dropout_forward_kernel(
    const float* __restrict__ input,
    float* output,
    float* mask,  // Store mask for backward
    int n,
    float dropout_prob,
    bool training) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (training) {
        // Generate random mask
        float r = (float)rand() / (float)RAND_MAX;
        if (r < dropout_prob) {
            mask[idx] = 0.0f;
            output[idx] = 0.0f;
        } else {
            mask[idx] = 1.0f / (1.0f - dropout_prob);  // Scale factor
            output[idx] = input[idx] * mask[idx];
        }
    } else {
        mask[idx] = 1.0f;
        output[idx] = input[idx];
    }
}

__global__ void dropout_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ mask,
    float* grad_input,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // grad_input = grad_output * mask (mask already has scale factor)
    grad_input[idx] = grad_output[idx] * mask[idx];
}

void dropout_forward_cuda(const Tensor& input, const Tensor& mask, Tensor& output,
                         float dropout_prob, bool training) {
    int n = input.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    dropout_forward_kernel<<<num_blocks, block_size>>>(
        input.data_cuda(),
        output.data_cuda(),
        mask.data_cuda(),
        n, dropout_prob, training);
    CUDA_CHECK(cudaGetLastError());
}

void dropout_backward_cuda(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input) {
    int n = grad_output.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    dropout_backward_kernel<<<num_blocks, block_size>>>(
        grad_output.data_cuda(),
        mask.data_cuda(),
        grad_input.data_cuda(),
        n);
    CUDA_CHECK(cudaGetLastError());
}
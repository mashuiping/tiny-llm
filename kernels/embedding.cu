// kernels/embedding.cu
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include "tensor.hpp"

// ============================================================================
// Embedding Lookup Kernel
// input: [batch, seq_len] of token IDs (integers)
// weight: [vocab_size, d_model] embedding table
// output: [batch, seq_len, d_model]
// ============================================================================

__global__ void embedding_lookup_kernel(
    const int* __restrict__ input_ids,  // [batch, seq_len], int32
    const float* __restrict__ weight,  // [vocab_size, d_model]
    float* __restrict__ output,         // [batch, seq_len, d_model]
    int batch, int seq_len, int vocab_size, int d_model) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len) return;

    int token_id = input_ids[batch_idx * seq_len + seq_idx];

    // Bounds check for token_id
    if (token_id < 0 || token_id >= vocab_size) return;

    int embedding_offset = token_id * d_model;
    int output_offset = (batch_idx * seq_len + seq_idx) * d_model;

    for (int dim = threadIdx.x; dim < d_model; dim += blockDim.x) {
        output[output_offset + dim] = weight[embedding_offset + dim];
    }
}

void embedding_lookup_cuda(const Tensor& input_ids, const Tensor& weight, Tensor& output,
                         int batch, int seq_len, int vocab_size, int d_model) {
    // input_ids should be on GPU as int32
    dim3 block_size(min(256, d_model));
    dim3 grid_size(seq_len, batch);

    embedding_lookup_kernel<<<grid_size, block_size>>>(
        static_cast<const int*>(input_ids.raw_data()),
        weight.data_cuda(),
        output.data_cuda(),
        batch, seq_len, vocab_size, d_model);
    CUDA_CHECK(cudaGetLastError());
}
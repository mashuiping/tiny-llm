// kernels/attention.cu
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include "tensor.hpp"
#include "gemm.hpp"
#include <cmath>

// ============================================================================
// Attention Forward Kernel
// Q: [batch, seq_len, d_model] -> reshaped to [batch, heads, seq_len, d_k]
// K: [batch, seq_len, d_model] -> reshaped to [batch, heads, seq_len, d_k]
// V: [batch, seq_len, d_model] -> reshaped to [batch, heads, seq_len, d_v]
// Output: [batch, heads, seq_len, d_v]
// ============================================================================

// Host wrapper for attention
void attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                    Tensor& output,
                    int batch, int seq_len, int d_model, int n_heads) {
    int d_k = d_model / n_heads;
    int d_v = d_model / n_heads;

    // Step 1: Q @ K^T / sqrt(d_k)
    // Q: [batch, heads, seq, d_k], K: [batch, heads, seq, d_k]
    // scores: [batch, heads, seq, seq]
    Tensor scores = Tensor::zeros({batch, n_heads, seq_len, seq_len}, Device::CUDA, false);

    // For each head, compute Q @ K^T
    // Simplified: treat as batched matmul across heads
    // In practice would use batched GEMM, but for MVP we compute head by head

    // Allocate temporary tensors for the computation
    Tensor Q_reshaped = Tensor::zeros({batch * n_heads, seq_len, d_k}, Device::CUDA, false);
    Tensor K_reshaped = Tensor::zeros({batch * n_heads, seq_len, d_k}, Device::CUDA, false);
    Tensor scores_reshaped = Tensor::zeros({batch * n_heads, seq_len, seq_len}, Device::CUDA, false);

    // Copy and reshape Q (in practice would be a kernel, using CPU copy for MVP)
    {
        Tensor Q_cpu = Q.cpu();
        Tensor Q_reshaped_cpu = Tensor::zeros({batch * n_heads, seq_len, d_k}, Device::CPU, false);
        const float* q_src = Q_cpu.data_cpu();
        float* q_dst = Q_reshaped_cpu.data_cpu();
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < d_k; ++j) {
                        int src_idx = b * seq_len * d_model + i * d_model + h * d_k + j;
                        int dst_idx = (b * n_heads + h) * seq_len * d_k + i * d_k + j;
                        q_dst[dst_idx] = q_src[src_idx];
                    }
        Q_reshaped = Q_reshaped_cpu.cuda();
    }

    // Copy and reshape K
    {
        Tensor K_cpu = K.cpu();
        Tensor K_reshaped_cpu = Tensor::zeros({batch * n_heads, seq_len, d_k}, Device::CPU, false);
        const float* k_src = K_cpu.data_cpu();
        float* k_dst = K_reshaped_cpu.data_cpu();
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < d_k; ++j) {
                        int src_idx = b * seq_len * d_model + i * d_model + h * d_k + j;
                        int dst_idx = (b * n_heads + h) * seq_len * d_k + i * d_k + j;
                        k_dst[dst_idx] = k_src[src_idx];
                    }
        K_reshaped = K_reshaped_cpu.cuda();
    }

    // scores = Q @ K^T / sqrt(d_k)
    {
        // K^T: [batch*n_heads, d_k, seq_len]
        Tensor K_t = K_reshaped.transpose(1, 2); // [batch*n_heads, d_k, seq_len]
        gemm_cuda(Q_reshaped, K_t, scores_reshaped, false, false, 1.0f / sqrtf((float)d_k), 0.0f);
    }

    // Step 2: Softmax (would call softmax kernel here in practice)
    // For MVP: use CPU softmax and copy back
    {
        Tensor scores_cpu = scores_reshaped.cpu();
        float* s = scores_cpu.data_cpu();
        // Row-wise softmax
        for (int b = 0; b < batch * n_heads; ++b) {
            for (int i = 0; i < seq_len; ++i) {
                float max_val = -INFINITY;
                for (int j = 0; j < seq_len; ++j) {
                    max_val = fmaxf(max_val, s[b * seq_len * seq_len + i * seq_len + j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    s[b * seq_len * seq_len + i * seq_len + j] = expf(s[b * seq_len * seq_len + i * seq_len + j] - max_val);
                    sum += s[b * seq_len * seq_len + i * seq_len + j];
                }
                for (int j = 0; j < seq_len; ++j) {
                    s[b * seq_len * seq_len + i * seq_len + j] /= (sum + 1e-8f);
                }
            }
        }
        scores = scores_cpu.cuda();
    }

    // Step 3: attention @ V
    {
        Tensor V_cpu = V.cpu();
        Tensor V_reshaped_cpu = Tensor::zeros({batch * n_heads, seq_len, d_v}, Device::CPU, false);
        const float* v_src = V_cpu.data_cpu();
        float* v_dst = V_reshaped_cpu.data_cpu();
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < d_v; ++j) {
                        int src_idx = b * seq_len * d_model + i * d_model + h * d_v + j;
                        int dst_idx = (b * n_heads + h) * seq_len * d_v + i * d_v + j;
                        v_dst[dst_idx] = v_src[src_idx];
                    }
        Tensor V_reshaped = V_reshaped_cpu.cuda();

        Tensor output_reshaped = Tensor::zeros({batch * n_heads, seq_len, d_v}, Device::CUDA, false);
        gemm_cuda(scores, V_reshaped, output_reshaped, false, false, 1.0f, 0.0f);

        // Reshape output back to [batch, seq_len, d_model]
        Tensor output_cpu = Tensor::zeros({batch, seq_len, d_model}, Device::CPU, false);
        float* out_dst = output_cpu.data_cpu();
        const float* out_src = output_reshaped.cpu().data_cpu();
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < d_v; ++j) {
                        int src_idx = (b * n_heads + h) * seq_len * d_v + i * d_v + j;
                        int dst_idx = b * seq_len * d_model + i * d_model + h * d_v + j;
                        out_dst[dst_idx] = out_src[src_idx];
                    }
        output = output_cpu.cuda();
    }
}
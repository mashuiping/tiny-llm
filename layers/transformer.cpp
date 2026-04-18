// layers/transformer.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include "layers.hpp"
#include "layers/layernorm.hpp"
#include "layers/embedding.hpp"
#include "attention.hpp"

// Multi-Head Self-Attention
class MultiHeadAttention {
    int d_model_;
    int n_heads_;
    int d_k_;
    Linear wq_, wk_, wv_, wo_;
public:
    MultiHeadAttention(int d_model, int n_heads);
    Tensor forward(const Tensor& x);
};

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads)
    : d_model_(d_model), n_heads_(n_heads), d_k_(d_model / n_heads),
      wq_(d_model, d_model, false),
      wk_(d_model, d_model, false),
      wv_(d_model, d_model, false),
      wo_(d_model, d_model, false) {
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    // x: [batch, seq_len, d_model]
    int batch = x.shape_[0];
    int seq_len = x.shape_[1];
    int d_model = x.shape_[2];

    // Compute Q, K, V
    Tensor Q = wq_.forward(x);
    Tensor K = wk_.forward(x);
    Tensor V = wv_.forward(x);

    // Reshape for multi-head: [batch, seq, d_model] -> [batch, heads, seq, d_k]
    Tensor Q_reshaped = Q.view({batch, seq_len, n_heads_, d_k_}).transpose(1, 2);
    Tensor K_reshaped = K.view({batch, seq_len, n_heads_, d_k_}).transpose(1, 2);
    Tensor V_reshaped = V.view({batch, seq_len, n_heads_, d_k_}).transpose(1, 2);

    // Attention (simplified - would call attention_cuda in production)
    // For MVP: use CPU attention with GPU copies
    Tensor attn_output = attention_cuda(Q_reshaped, K_reshaped, V_reshaped,
                                       batch, seq_len, d_model, n_heads_);

    // Reshape back: [batch, heads, seq, d_k] -> [batch, seq, d_model]
    attn_output = attn_output.transpose(1, 2).view({batch, seq_len, d_model});

    // Output projection
    attn_output = wo_.forward(attn_output);

    return attn_output;
}

// Transformer Encoder Block
class TransformerBlock {
    MultiHeadAttention attention;
    LayerNorm ln1, ln2;
    Linear ffn_w1, ffn_w2;
public:
    TransformerBlock(int d_model, int n_heads, int d_ff);
    Tensor forward(const Tensor& x);
};

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff)
    : attention(d_model, n_heads),
      ln1(d_model),
      ln2(d_model),
      ffn_w1(d_model, d_ff),
      ffn_w2(d_ff, d_model) {
}

Tensor TransformerBlock::forward(const Tensor& x) {
    // Self-attention with residual
    Tensor attn_out = attention.forward(x);
    x = x + attn_out;
    x = ln1.forward(x);

    // FFN with residual
    Tensor ffn_out = ffn_w2.forward(ffn_w1.forward(x).relu());
    x = x + ffn_out;
    x = ln2.forward(x);

    return x;
}

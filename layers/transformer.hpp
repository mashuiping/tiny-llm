// layers/transformer.hpp
#pragma once
#include "tensor.hpp"
#include "layers.hpp"
#include "layers/layernorm.hpp"

class MultiHeadAttention {
    int d_model_;
    int n_heads_;
    int d_k_;
    Linear wq_, wk_, wv_, wo_;
public:
    MultiHeadAttention(int d_model, int n_heads);
    Tensor forward(const Tensor& x);
};

class TransformerBlock {
    MultiHeadAttention attention;
    LayerNorm ln1, ln2;
    Linear ffn_w1, ffn_w2;
public:
    TransformerBlock(int d_model, int n_heads, int d_ff);
    Tensor forward(const Tensor& x);
};

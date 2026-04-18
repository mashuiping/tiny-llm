// include/layers.hpp
#pragma once
#include "tensor.hpp"

class Linear {
    int in_features_;
    int out_features_;
    bool bias_;
    Tensor weight_;
    Tensor bias_;
public:
    Linear(int in_features, int out_features, bool bias = true);
    Tensor forward(const Tensor& input);
};

class LayerNorm {
    int d_model_;
    Tensor gamma_;
    Tensor beta_;
public:
    explicit LayerNorm(int d_model);
    Tensor forward(const Tensor& input);
};

class Embedding {
    int vocab_size_;
    int d_model_;
    Tensor weight_;
public:
    Embedding(int vocab_size, int d_model);
    Tensor forward(const Tensor& input_ids);
};
// layers/layernorm.hpp
#pragma once
#include "tensor.hpp"

class LayerNorm {
    int d_model_;
    Tensor gamma_;  // scale
    Tensor beta_;   // bias
public:
    explicit LayerNorm(int d_model);
    Tensor forward(const Tensor& input);
};
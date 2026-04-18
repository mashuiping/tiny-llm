// layers/embedding.hpp
#pragma once
#include "tensor.hpp"

class Embedding {
    int vocab_size_;
    int d_model_;
    Tensor weight_;
public:
    Embedding(int vocab_size, int d_model);
    Tensor forward(const Tensor& input_ids);
};
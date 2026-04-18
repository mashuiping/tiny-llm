// layers/embedding.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include "embedding.hpp"

Embedding::Embedding(int vocab_size, int d_model)
    : vocab_size_(vocab_size), d_model_(d_model) {
    // Xavier initialization
    float std = sqrt(2.0f / (vocab_size + d_model));
    weight_ = Tensor::randn({vocab_size, d_model}, Device::CUDA, true) * std;
}

Tensor Embedding::forward(const Tensor& input_ids) {
    // input_ids: [batch, seq_len] int token IDs
    // weight: [vocab_size, d_model]
    // output: [batch, seq_len, d_model]
    int batch = input_ids.shape_[0];
    int seq_len = input_ids.shape_[1];

    Tensor output = Tensor::zeros({batch, seq_len, d_model_}, Device::CUDA, false);

    embedding_lookup_cuda(input_ids, weight_, output, batch, seq_len, vocab_size_, d_model_);
    return output;
}
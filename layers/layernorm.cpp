// layers/layernorm.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include "layernorm.hpp"

LayerNorm::LayerNorm(int d_model) : d_model_(d_model) {
    gamma_ = Tensor::randn({d_model}, Device::CUDA, true) * 0.02f;  // gamma = 1 by default
    beta_ = Tensor::zeros({d_model}, Device::CUDA, true);
}

Tensor LayerNorm::forward(const Tensor& input) {
    int d = input.shape_.back();
    int rows = input.numel() / d;
    Tensor output = Tensor::zeros(input.shape_, input.device_, false);

    layernorm_cuda(input, gamma_, beta_, output);
    return output;
}
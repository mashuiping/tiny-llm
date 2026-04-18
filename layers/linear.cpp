// layers/linear.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include "layers.hpp"

// Xavier/Glorot initialization for weights
static float xavier_init(int fan_in, int fan_out) {
    return sqrt(2.0f / (fan_in + fan_out));
}

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), bias_(bias) {
    float scale = xavier_init(in_features, out_features);
    weight_ = Tensor::randn({out_features, in_features}, Device::CUDA, true) * scale;
    if (bias_) {
        bias_ = Tensor::zeros({out_features}, Device::CUDA, true);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input: [..., in_features]
    // weight: [out_features, in_features]
    // output: [..., out_features]

    int batch_dim = input.numel() / in_features_;
    std::vector<int> output_shape = input.shape_;
    output_shape.back() = out_features_;

    Tensor output = Tensor::zeros(output_shape, Device::CUDA, false);

    // Reshape input to [batch, in_features]
    Tensor input_flat = input.view({batch_dim, in_features_});
    Tensor output_flat = output.view({batch_dim, out_features_});

    // y = x @ w^T + b
    gemm_cuda(input_flat, weight_, output_flat, false, false, 1.0f, 0.0f);

    if (bias_) {
        // Add bias: broadcast [1, out_features] to [batch, out_features]
        Tensor bias_view = bias_.view({1, out_features_});
        Tensor bias_broadcast = Tensor::zeros({batch_dim, out_features_}, Device::CUDA, false);
        // Fill bias broadcast with bias values
        const float* b = bias_.data_cuda();
        float* bb = bias_broadcast.data_cuda();
        // This is simplified - would ideally be a kernel
        for (int i = 0; i < batch_dim; ++i) {
            for (int j = 0; j < out_features_; ++j) {
                bb[i * out_features_ + j] = b[j];
            }
        }
        output_flat = output_flat + bias_broadcast;
    }

    return output.view(output_shape);
}
// kernels/attention.hpp
#pragma once
#include "tensor.hpp"

Tensor attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                     int batch, int seq_len, int d_model, int n_heads);

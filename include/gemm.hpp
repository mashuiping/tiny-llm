#pragma once
#include "tensor.hpp"

void gemm_cuda(const Tensor& A, const Tensor& B, Tensor& C,
               bool transpose_A = false, bool transpose_B = false,
               float alpha = 1.0f, float beta = 0.0f);

void gemm_cpu(const Tensor& A, const Tensor& B, Tensor& C,
              bool transpose_A = false, bool transpose_B = false,
              float alpha = 1.0f, float beta = 0.0f);
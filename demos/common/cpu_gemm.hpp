#pragma once
#include <cstddef>
void cpu_gemm_f32(std::size_t M, std::size_t N, std::size_t K,
                  const float* A, const float* B, float* C);

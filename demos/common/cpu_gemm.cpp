#include "cpu_gemm.hpp"

void cpu_gemm_f32(std::size_t M, std::size_t N, std::size_t K,
                  const float* A, const float* B, float* C) {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.f;
      for (std::size_t k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_fp16.h>

#include "cpu_gemm.hpp"
#include "cuda_check.cuh"
#include "numeric.hpp"

void launch_fp16_gemm(const __half* A, const __half* B, __half* C, int M, int N, int K);

int main() {
  // FP16 naive GEMM vs FP32 CPU reference on float-converted inputs. Fast-math tolerant threshold.
  constexpr float kTol = 5e-3f;
  constexpr int M = 64, N = 64, K = 64;

  std::vector<float> Af(static_cast<std::size_t>(M * K));
  std::vector<float> Bf(static_cast<std::size_t>(K * N));
  for (int i = 0; i < M * K; ++i) Af[static_cast<std::size_t>(i)] = 0.01f * std::sin(float(i));
  for (int i = 0; i < K * N; ++i) Bf[static_cast<std::size_t>(i)] = 0.01f * std::cos(float(i));

  std::vector<__half> Ah(Af.size()), Bh(Bf.size());
  for (std::size_t i = 0; i < Af.size(); ++i) Ah[i] = __float2half(Af[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bh[i] = __float2half(Bf[i]);

  std::vector<float> cref(static_cast<std::size_t>(M * N));
  cpu_gemm_f32(static_cast<std::size_t>(M), static_cast<std::size_t>(N), static_cast<std::size_t>(K), Af.data(),
               Bf.data(), cref.data());

  __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CK(cudaMalloc(&dA, Ah.size() * sizeof(__half)));
  CUDA_CK(cudaMalloc(&dB, Bh.size() * sizeof(__half)));
  CUDA_CK(cudaMalloc(&dC, cref.size() * sizeof(__half)));
  CUDA_CK(cudaMemcpy(dA, Ah.data(), Ah.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dB, Bh.data(), Bh.size() * sizeof(__half), cudaMemcpyHostToDevice));

  launch_fp16_gemm(dA, dB, dC, M, N, K);

  std::vector<__half> Ch(static_cast<std::size_t>(M * N));
  CUDA_CK(cudaMemcpy(Ch.data(), dC, Ch.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  std::vector<float> cgpu(Ch.size());
  for (std::size_t i = 0; i < Ch.size(); ++i) cgpu[i] = __half2float(Ch[i]);

  const float err = max_abs_diff_f32(cref.data(), cgpu.data(), cref.size());
  std::printf("FP16 GEMM vs FP32 CPU ref max_abs_err=%e\n", err);

  CUDA_CK(cudaFree(dA));
  CUDA_CK(cudaFree(dB));
  CUDA_CK(cudaFree(dC));

  if (err > kTol) return 1;
  return 0;
}

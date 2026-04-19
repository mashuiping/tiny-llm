#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_gemm.hpp"
#include "cuda_check.cuh"
#include "numeric.hpp"
#include "timer.hpp"

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
  // Fast-math can loosen FP32 ULPs; use 1e-5 vs CPU ref (documented).
  constexpr float kTol = 1e-5f;
  constexpr int M = 256, N = 256, K = 256;

  std::vector<float> hA(static_cast<std::size_t>(M * K));
  std::vector<float> hB(static_cast<std::size_t>(K * N));
  std::vector<float> hRef(static_cast<std::size_t>(M * N));
  std::vector<float> hNaive(static_cast<std::size_t>(M * N));
  std::vector<float> hTile(static_cast<std::size_t>(M * N));

  for (int i = 0; i < M * K; ++i) hA[static_cast<std::size_t>(i)] = std::sin(float(i)) * 1e-3f;
  for (int i = 0; i < K * N; ++i) hB[static_cast<std::size_t>(i)] = std::cos(float(i)) * 1e-3f;
  cpu_gemm_f32(static_cast<std::size_t>(M), static_cast<std::size_t>(N), static_cast<std::size_t>(K),
               hA.data(), hB.data(), hRef.data());

  float *dA = nullptr, *dB = nullptr, *dCn = nullptr, *dCt = nullptr;
  CUDA_CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCn, hNaive.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCt, hTile.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  WallTimer t1;
  launch_gemm_naive(dA, dB, dCn, M, N, K);
  double ms_naive = t1.ms();
  CUDA_CK(cudaMemcpy(hNaive.data(), dCn, hNaive.size() * sizeof(float), cudaMemcpyDeviceToHost));

  WallTimer t2;
  launch_gemm_tiled(dA, dB, dCt, M, N, K);
  double ms_tile = t2.ms();
  CUDA_CK(cudaMemcpy(hTile.data(), dCt, hTile.size() * sizeof(float), cudaMemcpyDeviceToHost));

  float err_naive = max_abs_diff_f32(hRef.data(), hNaive.data(), hRef.size());
  float err_tile = max_abs_diff_f32(hRef.data(), hTile.data(), hRef.size());
  std::printf("GEMM naive vs CPU max_abs_err=%e time=%.3f ms\n", err_naive, ms_naive);
  std::printf("GEMM tiled vs CPU max_abs_err=%e time=%.3f ms\n", err_tile, ms_tile);

  CUDA_CK(cudaFree(dA));
  CUDA_CK(cudaFree(dB));
  CUDA_CK(cudaFree(dCn));
  CUDA_CK(cudaFree(dCt));

  if (err_naive > kTol || err_tile > kTol) return 1;
  return 0;
}

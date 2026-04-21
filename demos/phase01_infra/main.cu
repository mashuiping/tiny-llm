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
  // 容忍度略宽：fast-math 或 FMA 重排可能改变 FP32 累加顺序，与严格 CPU 累加有微小差异。
  constexpr float kTol = 1e-5f;
  constexpr int M = 256, N = 256, K = 256;

  // h*：host 侧（CPU 内存）的矩阵缓冲区；d*：device 侧（GPU 显存）的矩阵指针。
  std::vector<float> hA(static_cast<std::size_t>(M * K));
  std::vector<float> hB(static_cast<std::size_t>(K * N));
  std::vector<float> hRef(static_cast<std::size_t>(M * N));   // CPU 参考结果（黄金标准）
  std::vector<float> hNaive(static_cast<std::size_t>(M * N)); // naive kernel 的 GPU 结果
  std::vector<float> hTile(static_cast<std::size_t>(M * N));  // tiled kernel 的 GPU 结果

  // 用小幅三角函数生成测试数据：避免全零或全常数导致某些索引 bug 被掩盖。
  for (int i = 0; i < M * K; ++i) hA[static_cast<std::size_t>(i)] = std::sin(float(i)) * 1e-3f;
  for (int i = 0; i < K * N; ++i) hB[static_cast<std::size_t>(i)] = std::cos(float(i)) * 1e-3f;

  // cpu_gemm_f32：在 CPU 上用朴素三重循环计算参考结果 C = A × B，作为「真值」锚点。
  // 后续 GPU kernel 的结果会分别与 hRef 做 max_abs_diff，验证正确性而非追求速度。
  cpu_gemm_f32(static_cast<std::size_t>(M), static_cast<std::size_t>(N), static_cast<std::size_t>(K),
               hA.data(), hB.data(), hRef.data());

  // -------------------------------------------------------------------------
  // GPU 内存分配（cudaMalloc）：在设备（GPU 显存）上分配存放矩阵的空间。
  // cudaMalloc 的第一个参数是指向指针的指针，分配成功后 *ptr 指向设备内存。
  // 注意：设备内存和 host 内存是独立的，ptr 不能直接解引用来访问 host 数据。
  // -------------------------------------------------------------------------
  float *dA = nullptr, *dB = nullptr, *dCn = nullptr, *dCt = nullptr;
  CUDA_CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCn, hNaive.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCt, hTile.size() * sizeof(float)));

  // -------------------------------------------------------------------------
  // 数据上传（cudaMemcpy H2D）：把 CPU 内存中的输入矩阵拷贝到 GPU 显存。
  // cudaMemcpy(dst, src, size, direction) 的第四个参数指定拷贝方向：
  //   - cudaMemcpyHostToDevice：CPU → GPU（上传输入）
  //   - cudaMemcpyDeviceToHost：GPU → CPU（下载输出）
  //   - cudaMemcpyDeviceToDevice：GPU → GPU（通常在多 GPU 或 stream 间搬数据）
  // -------------------------------------------------------------------------
  CUDA_CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  // -------------------------------------------------------------------------
  // 第一次 kernel launch：naive GEMM
  // t1.ms() 在 cudaMemcpy 之后才调用，确保计时只包含 kernel 执行时间，不混入拷贝时间。
  // -------------------------------------------------------------------------
  WallTimer t1;
  launch_gemm_naive(dA, dB, dCn, M, N, K);
  double ms_naive = t1.ms();
  // cudaMemcpy D2H：把 GPU 计算结果从显存拷回 CPU 内存。
  CUDA_CK(cudaMemcpy(hNaive.data(), dCn, hNaive.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // -------------------------------------------------------------------------
  // 第二次 kernel launch：tiled GEMM（预期比 naive 快）
  // -------------------------------------------------------------------------
  WallTimer t2;
  launch_gemm_tiled(dA, dB, dCt, M, N, K);
  double ms_tile = t2.ms();
  CUDA_CK(cudaMemcpy(hTile.data(), dCt, hTile.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // -------------------------------------------------------------------------
  // 正确性校验：用 max_abs_diff_f32 对比 GPU 结果与 CPU 参考的全局最大误差。
  // -------------------------------------------------------------------------
  float err_naive = max_abs_diff_f32(hRef.data(), hNaive.data(), hRef.size());
  float err_tile = max_abs_diff_f32(hRef.data(), hTile.data(), hRef.size());
  std::printf("GEMM naive vs CPU max_abs_err=%e time=%.3f ms\n", err_naive, ms_naive);
  std::printf("GEMM tiled vs CPU max_abs_err=%e time=%.3f ms\n", err_tile, ms_tile);

  // -------------------------------------------------------------------------
  // 释放 GPU 显存：不再需要时调用 cudaFree，避免内存泄漏。
  // -------------------------------------------------------------------------
  CUDA_CK(cudaFree(dA));
  CUDA_CK(cudaFree(dB));
  CUDA_CK(cudaFree(dCn));
  CUDA_CK(cudaFree(dCt));

  // 误差超过容忍度返回非零退出码，供 CI 脚本检测。
  if (err_naive > kTol || err_tile > kTol) return 1;
  return 0;
}

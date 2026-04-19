#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

__global__ void layernorm_fwd_ker(const float* __restrict__ in, float* __restrict__ out, int rows,
                                  int cols, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* x = in + r * cols;
  float* y = out + r * cols;
  float mu = 0.f;
  for (int i = 0; i < cols; ++i) mu += x[i];
  mu /= static_cast<float>(cols);
  float var = 0.f;
  for (int i = 0; i < cols; ++i) {
    float d = x[i] - mu;
    var += d * d;
  }
  var /= static_cast<float>(cols);
  const float inv_std = rsqrtf(var + eps);
  for (int i = 0; i < cols; ++i) y[i] = (x[i] - mu) * inv_std;
}

static void layernorm_fwd_cpu(const float* in, float* out, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* x = in + r * cols;
    float* y = out + r * cols;
    float mu = 0.f;
    for (int i = 0; i < cols; ++i) mu += x[i];
    mu /= static_cast<float>(cols);
    float var = 0.f;
    for (int i = 0; i < cols; ++i) {
      float d = x[i] - mu;
      var += d * d;
    }
    var /= static_cast<float>(cols);
    const float inv_std = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < cols; ++i) y[i] = (x[i] - mu) * inv_std;
  }
}

static float ln_loss_sum_sq(const std::vector<float>& x, int rows, int cols) {
  std::vector<float> y(static_cast<std::size_t>(rows * cols));
  layernorm_fwd_cpu(x.data(), y.data(), rows, cols, 1e-5f);
  float L = 0.f;
  for (float v : y) L += v * v;
  return L;
}

static void ln_grad_analytic(const float* x, int rows, int cols, float eps, float* grad) {
  std::vector<float> y(static_cast<std::size_t>(rows * cols));
  layernorm_fwd_cpu(x, y.data(), rows, cols, eps);

  std::vector<float> dy(y.size(), 0.f);
  for (std::size_t i = 0; i < y.size(); ++i) dy[i] = 2.f * y[i];

  for (int r = 0; r < rows; ++r) {
    const float* yr = y.data() + r * cols;
    const float* xr = x + r * cols;
    float* gr = grad + r * cols;
    const float* dyr = dy.data() + r * cols;

    float mean_dy = 0.f;
    for (int i = 0; i < cols; ++i) mean_dy += dyr[i];
    mean_dy /= static_cast<float>(cols);

    float mean_y_dy = 0.f;
    for (int i = 0; i < cols; ++i) mean_y_dy += yr[i] * dyr[i];
    mean_y_dy /= static_cast<float>(cols);

    float mu = 0.f;
    for (int i = 0; i < cols; ++i) mu += xr[i];
    mu /= static_cast<float>(cols);
    float var = 0.f;
    for (int i = 0; i < cols; ++i) {
      float d = xr[i] - mu;
      var += d * d;
    }
    var /= static_cast<float>(cols);
    const float sigma = std::sqrt(var + eps);

    for (int i = 0; i < cols; ++i) {
      gr[i] = (dyr[i] - mean_dy - yr[i] * mean_y_dy) / sigma;
    }
  }
}

int main() {
  const int rows = 8, cols = 4;
  const float eps = 1e-5f;
  std::vector<float> x(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * std::sin(float(i)) + 0.01f * float(i % 7);

  std::vector<float> y_cpu(x.size()), y_gpu(x.size());
  layernorm_fwd_cpu(x.data(), y_cpu.data(), rows, cols, eps);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CK(cudaMalloc(&d_in, x.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&d_out, y_gpu.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(d_in, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  layernorm_fwd_ker<<<rows, 1>>>(d_in, d_out, rows, cols, eps);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(y_gpu.data(), d_out, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float ferr = max_abs_diff_f32(y_cpu.data(), y_gpu.data(), y_cpu.size());
  std::printf("LAYERNORM CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  auto forward_lambda = [&](const std::vector<float>& x0) { return ln_loss_sum_sq(x0, rows, cols); };
  std::vector<float> g(x.size());
  ln_grad_analytic(x.data(), rows, cols, eps, g.data());
  const bool ok = vector_gradcheck(x, forward_lambda, g);
  std::printf("LAYERNORM gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(d_in));
  CUDA_CK(cudaFree(d_out));
  return ok ? 0 : 1;
}

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

__global__ void softmax_rows_ker(const float* __restrict__ in, float* __restrict__ out, int rows,
                                 int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* x = in + r * cols;
  float* y = out + r * cols;
  float m = x[0];
  for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) {
    float v = expf(x[c] - m);
    y[c] = v;
    s += v;
  }
  float inv = 1.f / fmaxf(s, 1e-12f);
  for (int c = 0; c < cols; ++c) y[c] *= inv;
}

static void softmax_rows_cpu(const float* in, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* x = in + r * cols;
    float* y = out + r * cols;
    float m = x[0];
    for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) {
      float v = expf(x[c] - m);
      y[c] = v;
      s += v;
    }
    float inv = 1.f / fmaxf(s, 1e-12f);
    for (int c = 0; c < cols; ++c) y[c] *= inv;
  }
}

static float ce_loss(const std::vector<float>& logits, const std::vector<int>& labels, int rows,
                      int cols) {
  float L = 0.f;
  for (int r = 0; r < rows; ++r) {
    const float* x = logits.data() + r * cols;
    int y = labels[static_cast<std::size_t>(r)];
    float m = x[0];
    for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) s += expf(x[c] - m);
    float logz = logf(fmaxf(s, 1e-12f));
    L += -(x[y] - m - logz);
  }
  return L;
}

static void ce_grad(const float* logits, const int* labels, int rows, int cols, float* grad) {
  std::vector<float> prob(static_cast<std::size_t>(rows * cols));
  softmax_rows_cpu(logits, prob.data(), rows, cols);
  for (int r = 0; r < rows; ++r) {
    const float* p = prob.data() + r * cols;
    float* g = grad + r * cols;
    int y = labels[r];
    for (int c = 0; c < cols; ++c) g[c] = p[c] - (c == y ? 1.f : 0.f);
  }
}

int main() {
  const int rows = 4, cols = 8;
  std::vector<float> h_in(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < h_in.size(); ++i) h_in[i] = 0.02f * std::sin(float(i));
  std::vector<float> h_cpu(h_in.size()), h_gpu(h_in.size());

  softmax_rows_cpu(h_in.data(), h_cpu.data(), rows, cols);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&d_out, h_gpu.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  const int threads = 128;
  const int blocks = (rows + threads - 1) / threads;
  softmax_rows_ker<<<blocks, threads>>>(d_in, d_out, rows, cols);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(h_gpu.data(), d_out, h_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float ferr = max_abs_diff_f32(h_cpu.data(), h_gpu.data(), h_cpu.size());
  std::printf("SOFTMAX CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  std::vector<int> labels = {0, 1, 2, 3};
  auto forward_lambda = [&](const std::vector<float>& x) { return ce_loss(x, labels, rows, cols); };

  std::vector<float> g(static_cast<std::size_t>(rows * cols));
  ce_grad(h_in.data(), labels.data(), rows, cols, g.data());

  const bool ok = vector_gradcheck(h_in, forward_lambda, g);
  std::printf("SOFTMAX gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(d_in));
  CUDA_CK(cudaFree(d_out));
  return ok ? 0 : 1;
}

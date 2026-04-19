#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

static void sdpa_forward_cpu(const float Q[2], const float K[2], const float V[2], float O[2]) {
  float s[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) s[i][j] = Q[i] * K[j];

  float P[2][2];
  for (int i = 0; i < 2; ++i) {
    float m = s[i][0];
    for (int j = 1; j < 2; ++j) m = fmaxf(m, s[i][j]);
    float sum = 0.f;
    for (int j = 0; j < 2; ++j) {
      P[i][j] = expf(s[i][j] - m);
      sum += P[i][j];
    }
    const float inv = 1.f / fmaxf(sum, 1e-8f);
    for (int j = 0; j < 2; ++j) P[i][j] *= inv;
  }

  for (int i = 0; i < 2; ++i) {
    float acc = 0.f;
    for (int j = 0; j < 2; ++j) acc += P[i][j] * V[j];
    O[i] = acc;
  }
}

__global__ void sdpa_tiny_ker(const float* Q, const float* K, const float* V, float* O) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float q0 = Q[0], q1 = Q[1], k0 = K[0], k1 = K[1], v0 = V[0], v1 = V[1];
  const float s00 = q0 * k0, s01 = q0 * k1, s10 = q1 * k0, s11 = q1 * k1;
  auto softmax2 = [](float a, float b, float& p0, float& p1) {
    const float m = fmaxf(a, b);
    const float e0 = expf(a - m), e1 = expf(b - m);
    const float s = e0 + e1;
    p0 = e0 / s;
    p1 = e1 / s;
  };
  float p00, p01, p10, p11;
  softmax2(s00, s01, p00, p01);
  softmax2(s10, s11, p10, p11);
  O[0] = p00 * v0 + p01 * v1;
  O[1] = p10 * v0 + p11 * v1;
}

static void sdpa_backward_cpu(const float Q[2], const float K[2], const float V[2], const float dO[2],
                                float dQ[2], float dK[2], float dV[2]) {
  float s[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) s[i][j] = Q[i] * K[j];

  float P[2][2];
  for (int i = 0; i < 2; ++i) {
    float m = s[i][0];
    for (int j = 1; j < 2; ++j) m = fmaxf(m, s[i][j]);
    float sum = 0.f;
    for (int j = 0; j < 2; ++j) {
      P[i][j] = expf(s[i][j] - m);
      sum += P[i][j];
    }
    const float inv = 1.f / fmaxf(sum, 1e-8f);
    for (int j = 0; j < 2; ++j) P[i][j] *= inv;
  }

  float dP[2][2] = {};
  for (int j = 0; j < 2; ++j) {
    float acc = 0.f;
    for (int i = 0; i < 2; ++i) acc += dO[i] * P[i][j];
    dV[j] = acc;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) dP[i][j] = dO[i] * V[j];

  float dS[2][2] = {};
  for (int i = 0; i < 2; ++i) {
    float pg = 0.f;
    for (int j = 0; j < 2; ++j) pg += P[i][j] * dP[i][j];
    for (int k = 0; k < 2; ++k) dS[i][k] = P[i][k] * (dP[i][k] - pg);
  }

  for (int i = 0; i < 2; ++i) {
    dQ[i] = 0.f;
    for (int j = 0; j < 2; ++j) dQ[i] += dS[i][j] * K[j];
  }
  for (int j = 0; j < 2; ++j) {
    dK[j] = 0.f;
    for (int i = 0; i < 2; ++i) dK[j] += dS[i][j] * Q[i];
  }
}

static float sdpa_loss(const std::vector<float>& p) {
  float Q[2] = {p[0], p[1]};
  float K[2] = {p[2], p[3]};
  float V[2] = {p[4], p[5]};
  float O[2];
  sdpa_forward_cpu(Q, K, V, O);
  return O[0] * O[0] + O[1] * O[1];
}

static void sdpa_analytic_grad(const std::vector<float>& p, std::vector<float>& g) {
  float Q[2] = {p[0], p[1]};
  float K[2] = {p[2], p[3]};
  float V[2] = {p[4], p[5]};
  float O[2];
  sdpa_forward_cpu(Q, K, V, O);
  const float dO[2] = {2.f * O[0], 2.f * O[1]};
  g.resize(6);
  sdpa_backward_cpu(Q, K, V, dO, g.data(), g.data() + 2, g.data() + 4);
}

int main() {
  std::vector<float> h(6);
  h[0] = 0.3f;
  h[1] = -0.2f;
  h[2] = 0.15f;
  h[3] = 0.4f;
  h[4] = -0.1f;
  h[5] = 0.25f;

  float Q[2] = {h[0], h[1]};
  float K[2] = {h[2], h[3]};
  float V[2] = {h[4], h[5]};
  float O_cpu[2], O_gpu[2];
  sdpa_forward_cpu(Q, K, V, O_cpu);

  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  CUDA_CK(cudaMalloc(&dQ, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dK, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dV, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dO, 2 * sizeof(float)));
  CUDA_CK(cudaMemcpy(dQ, Q, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dK, K, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dV, V, 2 * sizeof(float), cudaMemcpyHostToDevice));
  sdpa_tiny_ker<<<1, 1>>>(dQ, dK, dV, dO);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(O_gpu, dO, 2 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ocpu = {O_cpu[0], O_cpu[1]};
  std::vector<float> ogpu = {O_gpu[0], O_gpu[1]};
  const float ferr = max_abs_diff_f32(ocpu.data(), ogpu.data(), 2);
  std::printf("ATTENTION CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  auto forward_lambda = [&](const std::vector<float>& p) { return sdpa_loss(p); };
  std::vector<float> g;
  sdpa_analytic_grad(h, g);
  const bool ok = vector_gradcheck(h, forward_lambda, g);
  std::printf("ATTENTION gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(dQ));
  CUDA_CK(cudaFree(dK));
  CUDA_CK(cudaFree(dV));
  CUDA_CK(cudaFree(dO));
  return ok ? 0 : 1;
}

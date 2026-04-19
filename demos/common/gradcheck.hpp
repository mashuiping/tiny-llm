#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

inline bool vector_gradcheck(
    const std::vector<float>& x0,
    const std::function<float(const std::vector<float>&)>& forward,
    const std::vector<float>& analytic_grad,
    float eps = 1e-3f,
    float tol = 5e-2f) {
  for (std::size_t i = 0; i < x0.size(); ++i) {
    std::vector<float> xp = x0, xm = x0;
    xp[i] += eps;
    xm[i] -= eps;
    float numeric = (forward(xp) - forward(xm)) / (2.f * eps);
    float ag = analytic_grad[i];
    float rel = std::fabs(numeric - ag) / (std::fabs(ag) + 1e-6f);
    if (rel > tol && std::fabs(numeric - ag) > 1e-2f) {
      std::printf("grad mismatch idx=%zu numeric=%f analytic=%f\n", i, numeric, ag);
      return false;
    }
  }
  return true;
}

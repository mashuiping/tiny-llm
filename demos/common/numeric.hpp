#pragma once
#include <cmath>
#include <cstddef>

inline float max_abs_diff_f32(const float* a, const float* b, std::size_t n) {
  float m = 0.f;
  for (std::size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

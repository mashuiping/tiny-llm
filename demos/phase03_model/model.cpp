#include "model.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

namespace {

float xavier_scale(std::size_t fan_in, std::size_t fan_out) {
  return std::sqrt(2.f / static_cast<float>(fan_in + fan_out));
}

void matvec(const std::vector<float>& W, const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float acc = 0.f;
    const float* wr = W.data() + r * cols;
    for (int c = 0; c < cols; ++c) acc += wr[c] * x[c];
    y[r] = acc;
  }
}

std::uint32_t pcg32(std::uint32_t& state) {
  std::uint32_t x = state;
  state = x * 747796405u + 2891336453u;
  std::uint32_t word = ((x >> ((x >> 28u) + 4u)) ^ x) * 2778037370u;
  return (word >> 22u) ^ word;
}

float randf01(std::uint32_t& s) { return (pcg32(s) & 0xFFFFFFu) / 16777216.f; }

}  // namespace

TinyPoetryModel::TinyPoetryModel() {
  emb.assign(static_cast<std::size_t>(vocab * d_model), 0.f);
  w1.assign(static_cast<std::size_t>(d_model * d_model), 0.f);
  b1.assign(static_cast<std::size_t>(d_model), 0.f);
  w2.assign(static_cast<std::size_t>(d_model), 0.f);
  b2.assign(1, 0.f);

  std::uint32_t s = 12345u;
  const float se = xavier_scale(static_cast<std::size_t>(vocab), static_cast<std::size_t>(d_model));
  for (float& v : emb) v = (randf01(s) * 2.f - 1.f) * se;

  const float s1 = xavier_scale(static_cast<std::size_t>(d_model), static_cast<std::size_t>(d_model));
  for (float& v : w1) v = (randf01(s) * 2.f - 1.f) * s1;
  for (float& v : b1) v = (randf01(s) * 2.f - 1.f) * 0.01f;

  const float s2 = xavier_scale(static_cast<std::size_t>(d_model), 1u);
  for (float& v : w2) v = (randf01(s) * 2.f - 1.f) * s2;
  b2[0] = 0.f;
}

float TinyPoetryModel::forward_mse(const std::vector<int>& token_ids, std::vector<float>* dbg_out) {
  std::vector<float> pooled(static_cast<std::size_t>(d_model), 0.f);
  for (int t = 0; t < seq; ++t) {
    const int id = token_ids[static_cast<std::size_t>(t)];
    const float* row = emb.data() + static_cast<std::size_t>(id) * d_model;
    for (int d = 0; d < d_model; ++d) pooled[static_cast<std::size_t>(d)] += row[d];
  }
  const float inv = 1.f / static_cast<float>(seq);
  for (float& v : pooled) v *= inv;

  std::vector<float> h1(static_cast<std::size_t>(d_model));
  matvec(w1, pooled.data(), h1.data(), d_model, d_model);
  for (int d = 0; d < d_model; ++d) {
    float v = h1[static_cast<std::size_t>(d)] + b1[static_cast<std::size_t>(d)];
    h1[static_cast<std::size_t>(d)] = v > 0.f ? v : 0.f;  // ReLU
  }

  float y = b2[0];
  for (int d = 0; d < d_model; ++d) y += w2[static_cast<std::size_t>(d)] * h1[static_cast<std::size_t>(d)];

  std::uint32_t s = 999u;
  float target = (randf01(s) * 2.f - 1.f) * 0.5f;
  const float loss = (y - target) * (y - target);

  if (dbg_out) *dbg_out = h1;
  return loss;
}

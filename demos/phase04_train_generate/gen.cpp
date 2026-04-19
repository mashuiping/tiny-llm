#include "gen.hpp"

#include <cstdint>

namespace {

std::uint32_t pcg32(std::uint32_t& state) {
  std::uint32_t x = state;
  state = x * 747796405u + 2891336453u;
  std::uint32_t word = ((x >> ((x >> 28u) + 4u)) ^ x) * 2778037370u;
  return (word >> 22u) ^ word;
}

float rand01(std::uint32_t& s) { return (pcg32(s) & 0xffffffu) / 16777216.f; }

int sample_multinomial(const std::vector<float>& prob, std::uint32_t& s) {
  float u = rand01(s);
  float c = 0.f;
  for (std::size_t i = 0; i < prob.size(); ++i) {
    c += prob[i];
    if (u <= c) return static_cast<int>(i);
  }
  return static_cast<int>(prob.size() - 1);
}

}  // namespace

std::string greedy_generate(const CharTokenizer& tok, TinyLM& lm, const std::vector<int>& prompt, int new_tokens) {
  std::vector<int> cur = prompt;
  for (int n = 0; n < new_tokens; ++n) {
    std::vector<float> logits = lm.logits_for(cur.back());
    softmax_mut(logits);
    int best = 0;
    for (int v = 1; v < lm.V; ++v) {
      if (logits[static_cast<std::size_t>(v)] > logits[static_cast<std::size_t>(best)]) best = v;
    }
    cur.push_back(best);
  }
  return tok.decode(cur);
}

std::string temperature_generate(const CharTokenizer& tok, TinyLM& lm, const std::vector<int>& prompt, int new_tokens,
                                 float temp, std::uint32_t seed) {
  std::vector<int> cur = prompt;
  for (int n = 0; n < new_tokens; ++n) {
    std::vector<float> logits = lm.logits_for(cur.back());
    for (float& v : logits) v /= temp;
    softmax_mut(logits);
    const int pick = sample_multinomial(logits, seed);
    cur.push_back(pick);
  }
  return tok.decode(cur);
}

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

void softmax_mut(std::vector<float>& logits);

struct TinyLM {
  int V = 0;
  int D = 0;
  std::vector<float> emb;
  std::vector<float> wo;
  std::vector<float> bo;

  TinyLM(int v, int d, std::uint32_t seed);

  float step_batch(const std::vector<std::vector<int>>& x, const std::vector<std::vector<int>>& y, float lr,
                   float wd, bool adam, std::vector<float>& m_emb, std::vector<float>& v_emb,
                   std::vector<float>& m_wo, std::vector<float>& v_wo, int t_step);

  // Returns logits for a single token id (length V).
  std::vector<float> logits_for(int tok_id) const;
};

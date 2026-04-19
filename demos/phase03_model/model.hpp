#pragma once

#include <cstddef>
#include <vector>

// Demo-only tiny block: embedding + two linear layers + SiLU-ish (ReLU) for scalar target MSE.
struct TinyPoetryModel {
  int vocab = 64;
  int d_model = 32;
  int seq = 16;

  std::vector<float> emb;   // [vocab * d_model]
  std::vector<float> w1;    // [d_model * d_model]
  std::vector<float> b1;    // [d_model]
  std::vector<float> w2;    // [d_model]
  std::vector<float> b2;    // [1]

  TinyPoetryModel();

  // Returns MSE loss against internally generated random targets (deterministic seed per call site).
  float forward_mse(const std::vector<int>& token_ids, std::vector<float>* dbg_out = nullptr);
};

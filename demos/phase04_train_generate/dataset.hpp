#pragma once

#include <string>
#include <vector>

#include "tokenizer.hpp"

struct WindowDataset {
  const CharTokenizer* tok = nullptr;
  std::vector<int> data;
  int T = 32;

  WindowDataset(const CharTokenizer& t, const std::string& text, int window);
  bool next_batch(int batch, int step, std::vector<std::vector<int>>& x,
                  std::vector<std::vector<int>>& y) const;
};

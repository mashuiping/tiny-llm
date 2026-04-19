#pragma once

#include <string>
#include <vector>

#include "dataset.hpp"

struct TrainConfig {
  int d_model = 24;
  int steps = 50;
  float lr_sgd = 0.5f;
  float lr_adam = 0.05f;
};

void run_sgd_demo(const CharTokenizer& tok, const WindowDataset& ds, const TrainConfig& cfg, std::string* log);
void run_adamw_demo(const CharTokenizer& tok, const WindowDataset& ds, const TrainConfig& cfg, std::string* log);

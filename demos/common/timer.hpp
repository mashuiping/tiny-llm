#pragma once
#include <chrono>

struct WallTimer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0{clock::now()};
  double ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  }
};

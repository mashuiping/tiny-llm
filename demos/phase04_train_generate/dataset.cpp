#include "dataset.hpp"

WindowDataset::WindowDataset(const CharTokenizer& t, const std::string& text, int window) : tok(&t), T(window) {
  data = t.encode(text);
}

bool WindowDataset::next_batch(int batch, int step, std::vector<std::vector<int>>& x,
                               std::vector<std::vector<int>>& y) const {
  if (static_cast<int>(data.size()) < T + 1) return false;
  const int max_start = static_cast<int>(data.size()) - T - 1;
  if (max_start <= 0) return false;
  x.assign(static_cast<std::size_t>(batch), std::vector<int>(static_cast<std::size_t>(T), 0));
  y.assign(static_cast<std::size_t>(batch), std::vector<int>(static_cast<std::size_t>(T), 0));
  for (int b = 0; b < batch; ++b) {
    const int start = (step * batch + b) % (max_start + 1);
    for (int t = 0; t < T; ++t) {
      x[static_cast<std::size_t>(b)][static_cast<std::size_t>(t)] = data[static_cast<std::size_t>(start + t)];
      y[static_cast<std::size_t>(b)][static_cast<std::size_t>(t)] =
          data[static_cast<std::size_t>(start + t + 1)];
    }
  }
  return true;
}

#include <cstdio>
#include <numeric>
#include <vector>

#include "model.hpp"

int main() {
  TinyPoetryModel m;
  std::vector<int> ids(static_cast<std::size_t>(m.seq), 0);
  for (int i = 0; i < m.seq; ++i) ids[static_cast<std::size_t>(i)] = (i * 7 + 3) % m.vocab;

  const float loss = m.forward_mse(ids);
  const double mean = std::accumulate(m.emb.begin(), m.emb.end(), 0.0) / static_cast<double>(m.emb.size());
  double var = 0.0;
  for (float v : m.emb) {
    const double d = static_cast<double>(v) - mean;
    var += d * d;
  }
  var /= static_cast<double>(m.emb.size());

  std::printf("loss=%f\n", static_cast<double>(loss));
  std::printf("W_emb mean=%f var=%f\n", mean, var);

  if (!(loss == loss) || loss > 1e6f) return 1;
  return 0;
}

#include "train.hpp"

#include <cstdio>
#include <string>

#include "micro_lm.hpp"

namespace {

std::string train_loop(const CharTokenizer& tok, const WindowDataset& ds, const TrainConfig& cfg, bool adam) {
  const int V = tok.vocab_size();
  std::vector<float> m_e, v_e, m_w, v_w;
  TinyLM lm(V, cfg.d_model, adam ? 11u : 7u);
  if (adam) {
    m_e.assign(lm.emb.size(), 0.f);
    v_e.assign(lm.emb.size(), 0.f);
    m_w.assign(lm.wo.size(), 0.f);
    v_w.assign(lm.wo.size(), 0.f);
  }
  std::string out;
  const int batch = 4;
  for (int s = 0; s < cfg.steps; ++s) {
    std::vector<std::vector<int>> x, y;
    if (!ds.next_batch(batch, s, x, y)) break;
    const float lr = adam ? cfg.lr_adam : cfg.lr_sgd;
    const float loss = lm.step_batch(x, y, lr, 1e-4f, adam, m_e, v_e, m_w, v_w, s);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "step=%d loss=%.4f (%s)\n", s, static_cast<double>(loss),
                  adam ? "AdamW-ish" : "SGD");
    out += buf;
  }
  return out;
}

}  // namespace

void run_sgd_demo(const CharTokenizer& tok, const WindowDataset& ds, const TrainConfig& cfg, std::string* log) {
  const std::string s = train_loop(tok, ds, cfg, false);
  if (log) *log += s;
}

void run_adamw_demo(const CharTokenizer& tok, const WindowDataset& ds, const TrainConfig& cfg, std::string* log) {
  const std::string s = train_loop(tok, ds, cfg, true);
  if (log) *log += s;
}

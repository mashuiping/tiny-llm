#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "dataset.hpp"
#include "gen.hpp"
#include "micro_lm.hpp"
#include "tokenizer.hpp"
#include "train.hpp"

static std::string read_all(const char* path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main() {
  const std::string text = read_all("demos/phase04_train_generate/fixture.txt");
  if (text.empty()) {
    std::fprintf(stderr, "fixture missing (run from repo root)\n");
    return 1;
  }

  CharTokenizer tok(text);
  WindowDataset ds(tok, text, 16);
  TrainConfig cfg;
  cfg.steps = 50;
  cfg.d_model = 24;

  std::string log;
  run_sgd_demo(tok, ds, cfg, &log);
  run_adamw_demo(tok, ds, cfg, &log);
  std::fputs(log.c_str(), stdout);

  TinyLM lm(tok.vocab_size(), cfg.d_model, 42u);
  std::vector<float> m_e, v_e, m_w, v_w;
  m_e.assign(lm.emb.size(), 0.f);
  v_e.assign(lm.emb.size(), 0.f);
  m_w.assign(lm.wo.size(), 0.f);
  v_w.assign(lm.wo.size(), 0.f);
  const int batch = 4;
  for (int s = 0; s < 30; ++s) {
    std::vector<std::vector<int>> x, y;
    if (!ds.next_batch(batch, s, x, y)) break;
    lm.step_batch(x, y, 0.05f, 1e-4f, true, m_e, v_e, m_w, v_w, s);
  }

  std::vector<int> prompt = tok.encode("春");
  if (prompt.empty()) prompt.push_back(1);

  const std::string g1 = greedy_generate(tok, lm, prompt, 20);
  const std::string g2 = temperature_generate(tok, lm, prompt, 20, 0.9f, 2026u);
  std::printf("greedy: %s\n", g1.c_str());
  std::printf("temp:   %s\n", g2.c_str());
  return 0;
}

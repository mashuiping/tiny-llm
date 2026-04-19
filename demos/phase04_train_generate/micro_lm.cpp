#include "micro_lm.hpp"

void softmax_mut(std::vector<float>& logits) {
  float m = logits[0];
  for (float v : logits) m = std::fmax(m, v);
  float s = 0.f;
  for (float& v : logits) {
    v = std::exp(v - m);
    s += v;
  }
  const float inv = 1.f / std::fmax(s, 1e-8f);
  for (float& v : logits) v *= inv;
}

TinyLM::TinyLM(int v, int d, std::uint32_t seed) : V(v), D(d) {
  emb.assign(static_cast<std::size_t>(V * D), 0.f);
  wo.assign(static_cast<std::size_t>(D * V), 0.f);
  bo.assign(static_cast<std::size_t>(V), 0.f);
  for (std::size_t i = 0; i < emb.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    emb[i] = ((seed & 0xffff) / 65536.f - 0.5f) * 0.2f;
  }
  for (std::size_t i = 0; i < wo.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    wo[i] = ((seed & 0xffff) / 65536.f - 0.5f) * 0.2f;
  }
}

std::vector<float> TinyLM::logits_for(int tok_id) const {
  std::vector<float> h(static_cast<std::size_t>(D));
  for (int d = 0; d < D; ++d) h[static_cast<std::size_t>(d)] = emb[static_cast<std::size_t>(tok_id * D + d)];
  std::vector<float> logits(static_cast<std::size_t>(V), 0.f);
  for (int v = 0; v < V; ++v) {
    float acc = bo[static_cast<std::size_t>(v)];
    for (int d = 0; d < D; ++d) acc += h[static_cast<std::size_t>(d)] * wo[static_cast<std::size_t>(d * V + v)];
    logits[static_cast<std::size_t>(v)] = acc;
  }
  return logits;
}

float TinyLM::step_batch(const std::vector<std::vector<int>>& x, const std::vector<std::vector<int>>& y, float lr,
                         float wd, bool adam, std::vector<float>& m_emb, std::vector<float>& v_emb,
                         std::vector<float>& m_wo, std::vector<float>& v_wo, int t_step) {
  const int B = static_cast<int>(x.size());
  const int T = static_cast<int>(x[0].size());
  std::vector<float> g_emb(emb.size(), 0.f);
  std::vector<float> g_wo(wo.size(), 0.f);
  std::vector<float> g_bo(bo.size(), 0.f);
  float loss_acc = 0.f;

  for (int b = 0; b < B; ++b) {
    for (int ti = 0; ti < T; ++ti) {
      const int tok = x[static_cast<std::size_t>(b)][static_cast<std::size_t>(ti)];
      const int tgt = y[static_cast<std::size_t>(b)][static_cast<std::size_t>(ti)];
      std::vector<float> h(static_cast<std::size_t>(D));
      for (int d = 0; d < D; ++d) h[static_cast<std::size_t>(d)] = emb[static_cast<std::size_t>(tok * D + d)];

      std::vector<float> logits = logits_for(tok);
      softmax_mut(logits);
      const float py = logits[static_cast<std::size_t>(tgt)];
      loss_acc += -std::log(std::fmax(py, 1e-8f));

      std::vector<float> dlogit(static_cast<std::size_t>(V), 0.f);
      for (int v = 0; v < V; ++v) dlogit[static_cast<std::size_t>(v)] = logits[static_cast<std::size_t>(v)] - (v == tgt ? 1.f : 0.f);

      for (int d = 0; d < D; ++d) {
        float dh = 0.f;
        for (int v = 0; v < V; ++v) dh += wo[static_cast<std::size_t>(d * V + v)] * dlogit[static_cast<std::size_t>(v)];
        g_emb[static_cast<std::size_t>(tok * D + d)] += dh;
      }
      for (int d = 0; d < D; ++d) {
        for (int v = 0; v < V; ++v) {
          g_wo[static_cast<std::size_t>(d * V + v)] += h[static_cast<std::size_t>(d)] * dlogit[static_cast<std::size_t>(v)];
        }
      }
      for (int v = 0; v < V; ++v) g_bo[static_cast<std::size_t>(v)] += dlogit[static_cast<std::size_t>(v)];
    }
  }
  const float inv_n = 1.f / static_cast<float>(B * T);
  loss_acc *= inv_n;

  auto apply = [&](std::vector<float>& W, const std::vector<float>& G, std::vector<float>& M, std::vector<float>& Vv,
                   float lr_local) {
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for (std::size_t i = 0; i < W.size(); ++i) {
      float g = G[i] * inv_n + wd * W[i];
      if (!adam) {
        W[i] -= lr_local * g;
      } else {
        M[i] = beta1 * M[i] + (1.f - beta1) * g;
        Vv[i] = beta2 * Vv[i] + (1.f - beta2) * g * g;
        const float mhat = M[i] / (1.f - std::pow(beta1, static_cast<float>(t_step + 1)));
        const float vhat = Vv[i] / (1.f - std::pow(beta2, static_cast<float>(t_step + 1)));
        W[i] -= lr_local * mhat / (std::sqrt(vhat) + eps);
      }
    }
  };

  apply(emb, g_emb, m_emb, v_emb, lr);
  apply(wo, g_wo, m_wo, v_wo, lr);
  for (std::size_t i = 0; i < bo.size(); ++i) {
    float g = g_bo[i] * inv_n;
    bo[i] -= lr * g;
  }
  return loss_acc;
}

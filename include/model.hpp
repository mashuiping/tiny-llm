#pragma once
#include <cstdint>

struct ModelConfig {
    static constexpr int vocab_size = 8192;
    static constexpr int d_model = 256;
    static constexpr int n_heads = 4;
    static constexpr int n_layers = 4;
    static constexpr int d_ff = 1024;
    static constexpr int seq_len = 128;
    static constexpr float dropout_prob = 0.1f;
    static constexpr int max_tokens = 100;
};

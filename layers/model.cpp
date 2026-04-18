// layers/model.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include "model.hpp"
#include "transformer.hpp"

// PoetryModel: Embedding + N * TransformerBlock + LayerNorm + LM Head
class PoetryModel {
    Embedding embedding;
    std::vector<TransformerBlock> blocks;
    LayerNorm final_norm;
    Linear lm_head;
public:
    PoetryModel();
    Tensor forward(const Tensor& input_ids);  // input_ids: [batch, seq_len]
    Tensor generate(const Tensor& input_ids, int max_new_tokens);
};

PoetryModel::PoetryModel()
    : embedding(ModelConfig::vocab_size, ModelConfig::d_model),
      final_norm(ModelConfig::d_model),
      lm_head(ModelConfig::d_model, ModelConfig::vocab_size, false) {
    for (int i = 0; i < ModelConfig::n_layers; ++i) {
        blocks.emplace_back(ModelConfig::d_model, ModelConfig::n_heads, ModelConfig::d_ff);
    }
}

Tensor PoetryModel::forward(const Tensor& input_ids) {
    // input_ids: [batch, seq_len] token IDs
    // Embedding
    Tensor x = embedding.forward(input_ids);

    // Transformer blocks
    for (auto& block : blocks) {
        x = block.forward(x);
    }

    // Final LayerNorm
    x = final_norm.forward(x);

    // LM head (no bias, tied with embedding)
    x = lm_head.forward(x);

    return x;
}

Tensor PoetryModel::generate(const Tensor& input_ids, int max_new_tokens) {
    // TODO: implement proper autoregressive generation
    // This requires a dynamic computation graph that can grow sequence length
    assert(false && "generate() not fully implemented - TODO");
    return forward(input_ids);
}

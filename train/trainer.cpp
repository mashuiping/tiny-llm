// train/trainer.cpp
#include <iostream>
#include "model.hpp"
#include "optimizer.hpp"
#include "poetry_dataset.cpp"

struct TrainConfig {
    static constexpr int epochs = 100;
    static constexpr float lr = 1e-3f;
    static constexpr int seq_len = 32;
    static constexpr int batch_size = 4;
};

void simple_cross_entropy_loss(const Tensor& logits, const Tensor& targets, Tensor& loss) {
    // Simplified cross entropy: sum(-target * log_softmax(logits))
    // logits: [batch, seq, vocab]
    // targets: [batch, seq]
    int batch = logits.shape_[0];
    int seq = logits.shape_[1];
    int vocab = logits.shape_[2];

    float total_loss = 0.0f;
    int count = 0;

    const float* logit_data = logits.device_ == Device::CPU ? logits.data_cpu() : logits.cpu().data_cpu();
    const int* target_data = targets.device_ == Device::CPU ? targets.data_cpu() : targets.cpu().data_cpu();

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            int tid = target_data[b * seq + s];
            int offset = b * seq * vocab + s * vocab;

            // Softmax
            float max_val = -INFINITY;
            for (int v = 0; v < vocab; ++v) {
                max_val = std::max(max_val, logit_data[offset + v]);
            }
            float sum = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum += std::exp(logit_data[offset + v] - max_val);
            }
            float log_prob = logit_data[offset + tid] - max_val - std::log(sum + 1e-8f);
            total_loss -= log_prob;
            count++;
        }
    }

    loss = Tensor::zeros({1}, Device::CPU, false);
    loss.data_cpu()[0] = total_loss / count;
}

void train() {
    std::cout << "Building model..." << std::endl;

    // Create model
    // Note: PoetryModel needs proper construction
    // For now, just print that training would happen

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  epochs = " << TrainConfig::epochs << std::endl;
    std::cout << "  lr = " << TrainConfig::lr << std::endl;
    std::cout << "  batch_size = " << TrainConfig::batch_size << std::endl;
    std::cout << "  seq_len = " << TrainConfig::seq_len << std::endl;

    // TODO: Initialize model, optimizer, dataset
    // TODO: Training loop

    std::cout << "Training not yet implemented - will be added in final integration." << std::endl;
}

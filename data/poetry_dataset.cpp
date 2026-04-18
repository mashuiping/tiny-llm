// data/poetry_dataset.cpp
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "tensor.hpp"
#include "tokenizer.cpp"

struct PoetrySample {
    std::vector<int> input_ids;
    std::vector<int> target_ids;  // shifted right by 1
};

class PoetryDataset {
    std::vector<PoetrySample> samples_;
    int seq_len_;
    Tokenizer tokenizer_;
public:
    PoetryDataset(const std::string& data_path, int seq_len);
    const PoetrySample& get(int idx) const { return samples_[idx]; }
    int size() const { return samples_.size(); }
    const Tokenizer& tokenizer() const { return tokenizer_; }
};

PoetryDataset::PoetryDataset(const std::string& data_path, int seq_len)
    : seq_len_(seq_len) {
    // Load Tang Poetry data (simple CSV format: poem_text)
    // For MVP: create some sample data if file doesn't exist
    std::vector<std::string> poems = {
        "床前明月光，疑是地上霜。",
        "举头望明月，低头思故乡。",
        "春眠不觉晓，处处闻啼鸟。",
        "夜来风雨声，花落知多少。",
        "白日依山尽，黄河入海流。",
        "欲穷千里目，更上一层楼。",
        "千山鸟飞绝，万径人踪灭。",
        "孤舟蓑笠翁，独钓寒江雪。",
        "两个黄鹂鸣翠柳，一行白鹭上青天。",
        "窗含西岭千秋雪，门泊东吴万里船。",
        "春江潮水连海平，海上明月共潮生。",
        "江天一色无纤尘，皎皎空中孤月轮。",
    };

    // Build vocab
    tokenizer_.build_vocab(poems);

    // Process poems
    for (const auto& poem : poems) {
        auto ids = tokenizer_.encode(poem);

        // Truncate or pad
        if ((int)ids.size() > seq_len_) {
            ids.resize(seq_len_);
        }

        // Create input/target pairs (shifted)
        std::vector<int> input_ids = ids;
        std::vector<int> target_ids = ids;

        // Shift right by 1 (teacher forcing)
        if (target_ids.size() > 0) {
            for (size_t i = target_ids.size() - 1; i > 0; --i) {
                target_ids[i] = target_ids[i - 1];
            }
            target_ids[0] = tokenizer_.bos_id_;
        }

        samples_.push_back({input_ids, target_ids});
    }
}
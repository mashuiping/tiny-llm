// data/tokenizer.hpp
#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include "tensor.hpp"

class Tokenizer {
public:
    int pad_id_ = 0;
    int unk_id_ = 1;
    int bos_id_ = 2;
    int eos_id_ = 3;

    Tokenizer();
    void build_vocab(const std::vector<std::string>& texts);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    int vocab_size() const;
};
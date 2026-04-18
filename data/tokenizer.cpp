// data/tokenizer.cpp
#include <unordered_map>
#include <vector>
#include <string>
#include "tensor.hpp"

class Tokenizer {
    std::unordered_map<char, int> char_to_id_;
    std::unordered_map<int, char> id_to_char_;
    int vocab_size_;
public:
    Tokenizer() : vocab_size_(0) {}

    void build_vocab(const std::vector<std::string>& texts) {
        // Count all characters
        for (const auto& text : texts) {
            for (char c : text) {
                if (char_to_id_.find(c) == char_to_id_.end()) {
                    int id = vocab_size_;
                    char_to_id_[c] = id;
                    id_to_char_[id] = c;
                    vocab_size_++;
                }
            }
        }
        // Add special tokens
        // <PAD> = 0, <UNK> = 1, <BOS> = 2, <EOS> = 3
        // Shift all existing ids by 4
        if (vocab_size_ > 0) {
            std::unordered_map<char, int> new_char_to_id;
            std::unordered_map<int, char> new_id_to_char;
            for (auto& [c, id] : char_to_id_) {
                new_char_to_id[c] = id + 4;
                new_id_to_char[id + 4] = c;
            }
            char_to_id_ = std::move(new_char_to_id);
            id_to_char_ = std::move(new_id_to_char);
        }
        vocab_size_ += 4;

        // Special token IDs
        pad_id_ = 0;
        unk_id_ = 1;
        bos_id_ = 2;
        eos_id_ = 3;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        ids.push_back(bos_id_);
        for (char c : text) {
            auto it = char_to_id_.find(c);
            if (it != char_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(unk_id_);
            }
        }
        ids.push_back(eos_id_);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string text;
        for (int id : ids) {
            if (id == eos_id_ || id == bos_id_) break;
            if (id == unk_id_) {
                text += '<';
                text += 'UNK';
                text += '>';
            } else {
                auto it = id_to_char_.find(id);
                if (it != id_to_char_.end()) {
                    text += it->second;
                }
            }
        }
        return text;
    }

    int vocab_size() const { return vocab_size_; }
};
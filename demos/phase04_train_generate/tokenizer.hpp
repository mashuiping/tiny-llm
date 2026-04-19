#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct CharTokenizer {
  std::unordered_map<char, int> stoi;
  std::vector<char> itos;

  explicit CharTokenizer(const std::string& corpus_text);
  std::vector<int> encode(const std::string& s) const;
  std::string decode(const std::vector<int>& ids) const;
  int vocab_size() const { return static_cast<int>(itos.size()); }
};

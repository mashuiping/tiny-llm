#include "tokenizer.hpp"

#include <unordered_set>

CharTokenizer::CharTokenizer(const std::string& corpus_text) {
  std::unordered_set<char> uniq;
  for (unsigned char ch : corpus_text) {
    if (ch == '\r') continue;
    uniq.insert(static_cast<char>(ch));
  }
  itos.push_back('<pad>');
  stoi['<pad>'] = 0;
  for (char c : uniq) {
    if (c == '\n' || c == '\t') continue;
    if (stoi.count(c)) continue;
    stoi[c] = static_cast<int>(itos.size());
    itos.push_back(c);
  }
}

std::vector<int> CharTokenizer::encode(const std::string& s) const {
  std::vector<int> out;
  for (unsigned char ch : s) {
    auto it = stoi.find(static_cast<char>(ch));
    if (it != stoi.end()) out.push_back(it->second);
  }
  return out;
}

std::string CharTokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(itos.size())) continue;
    s.push_back(itos[static_cast<std::size_t>(id)]);
  }
  return s;
}

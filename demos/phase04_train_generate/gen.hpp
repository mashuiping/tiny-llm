#pragma once

#include <string>
#include <vector>

#include "micro_lm.hpp"
#include "tokenizer.hpp"

std::string greedy_generate(const CharTokenizer& tok, TinyLM& lm, const std::vector<int>& prompt, int new_tokens);
std::string temperature_generate(const CharTokenizer& tok, TinyLM& lm, const std::vector<int>& prompt, int new_tokens,
                                 float temp, std::uint32_t seed);

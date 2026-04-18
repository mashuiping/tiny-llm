// generate/generator.cpp
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "model.hpp"
#include "tokenizer.hpp"

std::string generate_text(PoetryModel& model, const Tokenizer& tokenizer,
                         const std::string& prompt, int max_new_tokens = 50) {
    // Encode prompt
    auto ids = tokenizer.encode(prompt);

    // For MVP: just return the prompt decoded
    // Full autoregressive generation requires handling dynamic sequence length
    // which is complex without a proper autograd engine that supports graph building
    return prompt + " [generation requires full autograd integration]";
}

void run_generation(const std::string& model_path, const std::string& prompt) {
    // Load tokenizer (simplified)
    std::vector<std::string> sample_poems = {
        "床前明月光，疑是地上霜。",
        "举头望明月，低头思故乡。",
    };

    Tokenizer tokenizer;
    tokenizer.build_vocab(sample_poems);

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generating..." << std::endl;

    // TODO: Load model weights from model_path
    // PoetryModel model;
    // model.load(model_path);

    // For MVP: just return the prompt
    std::string generated = prompt + " [generation requires full model integration]";
    std::cout << "Generated: " << generated << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <prompt>" << std::endl;
        std::cout << "Example: " << argv[0] << " \"春眠不觉晓\"" << std::endl;
        return 1;
    }

    std::string prompt = argv[1];
    run_generation("", prompt);
    return 0;
}

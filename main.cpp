#include <iostream>
#include "model.hpp"

int main(int argc, char** argv) {
    std::cout << "TinyLLM - 从0构建大模型" << std::endl;
    std::cout << "Model config:" << std::endl;
    std::cout << "  d_model = " << ModelConfig::d_model << std::endl;
    std::cout << "  n_heads = " << ModelConfig::n_heads << std::endl;
    std::cout << "  n_layers = " << ModelConfig::n_layers << std::endl;
    std::cout << "  vocab_size = " << ModelConfig::vocab_size << std::endl;
    return 0;
}

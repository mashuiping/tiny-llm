#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

class Context {
public:
    std::unordered_map<std::string, Tensor> data;
    void save_for_backward(const std::string& key, const Tensor& t) {
        data[key] = t;
    }
    Tensor get(const std::string& key) {
        return data[key];
    }
};

class Function {
public:
    virtual ~Function() = default;
    virtual Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) = 0;
};

// Forward declarations of concrete functions
class AddFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class MulFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class MatMulFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class ReLUFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class SoftmaxFunction : public Function {
    int dim_;
public:
    explicit SoftmaxFunction(int dim = -1) : dim_(dim) {}
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class SumFunction : public Function {
    int dim_;
public:
    explicit SumFunction(int dim = -1) : dim_(dim) {}
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class LogFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class SigmoidFunction : public Function {
public:
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

// Utility function to call a function and attach to graph
Tensor call_function(std::shared_ptr<Function> func, std::vector<Tensor> inputs);
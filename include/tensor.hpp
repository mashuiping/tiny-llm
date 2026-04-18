#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cassert>
#include <cuda_runtime.h>

enum class Device { CPU, CUDA };

class Function;
class Context;

class Tensor {
public:
    std::vector<int> shape_;
    std::vector<int> strides_;
    void* data_ = nullptr;
    Device device_ = Device::CPU;
    bool requires_grad_ = false;
    bool is_leaf_ = true;
    std::shared_ptr<Function> creator_;
    std::shared_ptr<Context> creator_ctx_;
    Tensor grad_;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU, bool requires_grad = false);
    Tensor(std::vector<int> shape, const std::vector<float>& data, Device device = Device::CPU, bool requires_grad = false);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    static Tensor zeros(std::vector<int> shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor randn(std::vector<int> shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor from_cpu(const std::vector<int>& shape, const float* data, bool requires_grad = false);

    float* data_cpu() const;
    float* data_cuda() const;
    void* raw_data() const { return data_; }

    int numel() const;
    Tensor view(std::vector<int> shape);
    Tensor transpose(int dim0, int dim1);
    Tensor cpu() const;
    Tensor cuda() const;

    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor matmul(const Tensor& other);
    Tensor softmax(int dim = -1);
    Tensor relu();
    Tensor sigmoid();
    Tensor log_();
    Tensor sum(int dim = -1);

    void backward();

    void set_creator(std::shared_ptr<Function> func) {
        if (!is_leaf_) creator_ = std::move(func);
    }
};

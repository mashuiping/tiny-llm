#include "tensor.hpp"
#include <cstring>
#include <cstdlib>

// Stub implementations - will be filled in Task 2

Tensor::Tensor(std::vector<int> shape, Device device, bool requires_grad)
    : shape_(std::move(shape)), device_(device), requires_grad_(requires_grad), is_leaf_(true) {
    int n = 1;
    for (int s : shape_) n *= s;
    data_ = malloc(n * sizeof(float));
}

Tensor::Tensor(std::vector<int> shape, const std::vector<float>& data, Device device, bool requires_grad)
    : shape_(std::move(shape)), device_(device), requires_grad_(requires_grad), is_leaf_(true) {
    int n = 1;
    for (int s : shape_) n *= s;
    data_ = malloc(n * sizeof(float));
    memcpy(data_, data.data(), n * sizeof(float));
}

Tensor::~Tensor() {
    if (data_) free(data_);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      data_(other.data_),
      device_(other.device_),
      requires_grad_(other.requires_grad_),
      is_leaf_(other.is_leaf_),
      creator_(std::move(other.creator_)),
      creator_ctx_(std::move(other.creator_ctx_)),
      grad_(std::move(other.grad_)) {
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data_) free(data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        data_ = other.data_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        is_leaf_ = other.is_leaf_;
        creator_ = std::move(other.creator_);
        creator_ctx_ = std::move(other.creator_ctx_);
        grad_ = std::move(other.grad_);
        other.data_ = nullptr;
    }
    return *this;
}

Tensor Tensor::zeros(std::vector<int> shape, Device device, bool requires_grad) {
    return Tensor(shape, device, requires_grad);
}

Tensor Tensor::randn(std::vector<int> shape, Device device, bool requires_grad) {
    return Tensor(shape, device, requires_grad);
}

Tensor Tensor::from_cpu(const std::vector<int>& shape, const float* data, bool requires_grad) {
    std::vector<float> vec(data, data + 1); // stub
    return Tensor(shape, Device::CPU, requires_grad);
}

float* Tensor::data_cpu() const { return static_cast<float*>(data_); }
float* Tensor::data_cuda() const { return static_cast<float*>(data_); }

int Tensor::numel() const {
    int n = 1;
    for (int s : shape_) n *= s;
    return n;
}

Tensor Tensor::view(std::vector<int> shape) { return Tensor(shape, device_, requires_grad_); }
Tensor Tensor::transpose(int dim0, int dim1) { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::cpu() const { return Tensor(shape_, Device::CPU, requires_grad_); }
Tensor Tensor::cuda() const { return Tensor(shape_, Device::CUDA, requires_grad_); }

Tensor Tensor::operator+(const Tensor& other) { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::operator*(const Tensor& other) { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::matmul(const Tensor& other) { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::softmax(int dim) { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::relu() { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::sigmoid() { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::log_() { return Tensor(shape_, device_, requires_grad_); }
Tensor Tensor::sum(int dim) { return Tensor({}, device_, requires_grad_); }

void Tensor::backward() {
    // stub - will be implemented in Task 4
}

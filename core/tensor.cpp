#include "tensor.hpp"
#include "cuda_utils.hpp"
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <cmath>

// Forward declare Context from autograd.hpp (will be filled in Task 4)
class Context;

Tensor::Tensor(std::vector<int> shape, Device device, bool requires_grad)
    : shape_(std::move(shape)), device_(device), requires_grad_(requires_grad), is_leaf_(true) {
    int n = numel();
    if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMalloc(&data_, n * sizeof(float)));
    } else {
        data_ = malloc(n * sizeof(float));
    }
    strides_.resize(shape_.size());
    if (shape_.size() > 1) {
        strides_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
}

Tensor::Tensor(std::vector<int> shape, const std::vector<float>& data, Device device, bool requires_grad)
    : Tensor(std::move(shape), device, requires_grad) {
    int n = numel();
    if (device_ == Device::CPU) {
        std::memcpy(data_, data.data(), n * sizeof(float));
    } else {
        float* tmp = new float[n];
        std::memcpy(tmp, data.data(), n * sizeof(float));
        CUDA_CHECK(cudaMemcpy(data_, tmp, n * sizeof(float), cudaMemcpyHostToDevice));
        delete[] tmp;
    }
}

Tensor::~Tensor() {
    if (data_ != nullptr) {
        if (device_ == Device::CUDA) {
            cudaFree(data_);
        } else {
            free(data_);
        }
        data_ = nullptr;
    }
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
        if (data_ != nullptr) {
            if (device_ == Device::CUDA) cudaFree(data_);
            else free(data_);
        }
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
    Tensor t(std::move(shape), device, requires_grad);
    int n = t.numel();
    if (device == Device::CPU) {
        std::fill_n(static_cast<float*>(t.data_), n, 0.0f);
    } else {
        CUDA_CHECK(cudaMemset(t.data_, 0, n * sizeof(float)));
    }
    return t;
}

Tensor Tensor::randn(std::vector<int> shape, Device device, bool requires_grad) {
    Tensor t(std::move(shape), device, requires_grad);
    int n = t.numel();
    std::vector<float> tmp(n);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) tmp[i] = dist(gen);
    if (device == Device::CPU) {
        std::memcpy(t.data_, tmp.data(), n * sizeof(float));
    } else {
        CUDA_CHECK(cudaMemcpy(t.data_, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    return t;
}

Tensor Tensor::from_cpu(const std::vector<int>& shape, const float* data, bool requires_grad) {
    Tensor t(shape, Device::CPU, requires_grad);
    int n = t.numel();
    std::memcpy(t.data_, data, n * sizeof(float));
    return t;
}

float* Tensor::data_cpu() const {
    assert(device_ == Device::CPU);
    return static_cast<float*>(data_);
}

float* Tensor::data_cuda() const {
    assert(device_ == Device::CUDA);
    return static_cast<float*>(data_);
}

int Tensor::numel() const {
    int n = 1;
    for (int d : shape_) n *= d;
    return n;
}

// NOTE: view() and transpose() create non-owning tensors that share the original's data.
// The original tensor must outlive any view/transpose returned. This is acceptable for MVP phase.
Tensor Tensor::view(std::vector<int> shape) {
    Tensor out = *this;
    out.shape_ = std::move(shape);
    out.strides_.resize(out.shape_.size());
    out.strides_[out.shape_.size() - 1] = 1;
    for (int i = out.shape_.size() - 2; i >= 0; --i) {
        out.strides_[i] = out.strides_[i + 1] * out.shape_[i + 1];
    }
    return out;
}

Tensor Tensor::cpu() const {
    if (device_ == Device::CPU) return *this;
    Tensor out(shape_, Device::CPU, requires_grad_);
    CUDA_CHECK(cudaMemcpy(out.data_, data_, numel() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

Tensor Tensor::cuda() const {
    if (device_ == Device::CUDA) return *this;
    Tensor out(shape_, Device::CUDA, requires_grad_);
    CUDA_CHECK(cudaMemcpy(out.data_, data_, numel() * sizeof(float), cudaMemcpyHostToDevice));
    return out;
}

Tensor Tensor::transpose(int dim0, int dim1) {
    assert(dim0 >= 0 && dim0 < (int)shape_.size() && dim1 >= 0 && dim1 < (int)shape_.size());
    Tensor out = *this;
    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

// Stub implementations for operators (will be filled in Task 4)
Tensor Tensor::operator+(const Tensor& other) { assert(false && "not implemented"); }
Tensor Tensor::operator*(const Tensor& other) { assert(false && "not implemented"); }
Tensor Tensor::matmul(const Tensor& other) { assert(false && "not implemented"); }
Tensor Tensor::softmax(int dim) { assert(false && "not implemented"); }
Tensor Tensor::relu() { assert(false && "not implemented"); }
Tensor Tensor::sigmoid() { assert(false && "not implemented"); }
Tensor Tensor::log_() { assert(false && "not implemented"); }
Tensor Tensor::sum(int dim) { assert(false && "not implemented"); }
void Tensor::backward() { assert(false && "not implemented"); }

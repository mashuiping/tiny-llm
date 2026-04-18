// include/optimizer.hpp
#pragma once
#include "tensor.hpp"
#include <vector>
#include <unordered_map>

class Optimizer {
public:
    virtual void step() = 0;
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
    std::vector<Tensor*> params_;
    float lr_;
public:
    SGD(float lr = 1e-3f) : lr_(lr) {}
    void add_param(Tensor* p) { params_.push_back(p); }
    void step() override;
};

class AdamW : public Optimizer {
    std::vector<Tensor*> params_;
    float lr_;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weight_decay_ = 0.01f;
    int t_ = 0;
    std::unordered_map<Tensor*, Tensor> m_;
    std::unordered_map<Tensor*, Tensor> v_;
public:
    AdamW(float lr = 1e-3f) : lr_(lr) {}
    void add_param(Tensor* p) { params_.push_back(p); }
    void step() override;
};

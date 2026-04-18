// core/optimizer.cpp
#include "optimizer.hpp"
#include "cuda_utils.hpp"
#include <cmath>

void SGD::step() {
    for (auto* p : params_) {
        if (!p->requires_grad_) continue;
        const float* grad = p->device_ == Device::CPU ? p->grad_.data_cpu() : p->grad_.data_cuda();
        float* data = p->device_ == Device::CPU ? p->data_cpu() : p->data_cuda();
        int n = p->numel();

        for (int i = 0; i < n; ++i) {
            data[i] -= lr_ * grad[i];
        }
    }
}

void AdamW::step() {
    t_++;
    float lr_t = lr_ * std::sqrt(1.0f - std::pow(beta2_, t_)) / (1.0f - std::pow(beta1_, t_));

    for (auto* p : params_) {
        if (!p->requires_grad_ || p->grad_.numel() == 0) continue;

        int n = p->numel();
        Tensor grad = p->grad_;

        // Get or create m and v
        if (m_.find(p) == m_.end()) {
            m_[*p] = Tensor::zeros(p->shape_, p->device_, false);
            v_[*p] = Tensor::zeros(p->shape_, p->device_, false);
        }

        Tensor& m = m_[*p];
        Tensor& v = v_[*p];

        const float* g = grad.device_ == Device::CPU ? grad.data_cpu() : grad.data_cuda();
        float* m_data = m.device_ == Device::CPU ? m.data_cpu() : m.data_cuda();
        float* v_data = v.device_ == Device::CPU ? v.data_cpu() : v.data_cuda();
        float* p_data = p->device_ == Device::CPU ? p->data_cpu() : p->data_cuda();

        for (int i = 0; i < n; ++i) {
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g
            m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g[i];
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g[i] * g[i];
            // p_t = p_{t-1} - lr_t * m_t / (sqrt(v_t) + eps) - lr_t * weight_decay * p_{t-1}
            p_data[i] = p_data[i] - lr_t * m_data[i] / (std::sqrt(v_data[i]) + eps_) - lr_t * weight_decay_ * p_data[i];
        }
    }
}

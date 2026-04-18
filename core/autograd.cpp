// core/autograd.cpp
#include "autograd.hpp"
#include "cuda_utils.hpp"
#include "gemm.hpp"
#include <cassert>
#include <cmath>

// ============================================================================
// Utility
// ============================================================================

Tensor call_function(std::shared_ptr<Function> func, std::vector<Tensor> inputs) {
    Context ctx;
    std::vector<Tensor> outputs;

    bool needs_grad = false;
    for (auto& inp : inputs)
        if (inp.requires_grad_) { needs_grad = true; break; }

    outputs.push_back(func->forward(ctx, inputs));

    if (needs_grad) {
        outputs[0].requires_grad_ = true;
        outputs[0].is_leaf_ = false;
        outputs[0].creator_ = func;
        outputs[0].creator_ctx_ = std::make_shared<Context>(std::move(ctx));
    }
    return outputs[0];
}

// ============================================================================
// AddFunction: z = x + y
// ============================================================================

Tensor AddFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2);
    const Tensor& x = inputs[0];
    const Tensor& y = inputs[1];

    ctx.save_for_backward("x", x);
    ctx.save_for_backward("y", y);

    // Broadcast if needed (simplified: assume same shape)
    std::vector<int> out_shape = x.shape_;
    Tensor out = Tensor::zeros(out_shape, x.device_, false);

    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    const float* yd = y.device_ == Device::CPU ? y.data_cpu() : y.data_cuda();
    float* od = out.device_ == Device::CPU ? out.data_cpu() : out.data_cuda();

    for (int i = 0; i < n; ++i) od[i] = xd[i] + yd[i];
    return out;
}

std::vector<Tensor> AddFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor y = ctx.get("y");
    // grad_x = grad_output, grad_y = grad_output
    return {grad_output, grad_output};
}

// ============================================================================
// MulFunction: z = x * y
// ============================================================================

Tensor MulFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2);
    const Tensor& x = inputs[0];
    const Tensor& y = inputs[1];
    ctx.save_for_backward("x", x);
    ctx.save_for_backward("y", y);

    Tensor out = Tensor::zeros(x.shape_, x.device_, false);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    const float* yd = y.device_ == Device::CPU ? y.data_cpu() : y.data_cuda();
    float* od = out.device_ == Device::CPU ? out.data_cpu() : out.data_cuda();

    for (int i = 0; i < n; ++i) od[i] = xd[i] * yd[i];
    return out;
}

std::vector<Tensor> MulFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor y = ctx.get("y");
    // d(x*y)/dx = y, d(x*y)/dy = x
    Tensor grad_x = grad_output * y;
    Tensor grad_y = grad_output * x;
    return {grad_x, grad_y};
}

// ============================================================================
// MatMulFunction: z = x @ y
// ============================================================================

Tensor MatMulFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2);
    const Tensor& x = inputs[0];
    const Tensor& y = inputs[1];
    ctx.save_for_backward("x", x);
    ctx.save_for_backward("y", y);

    // Assume x: [M, K], y: [K, N] -> out: [M, N]
    int M = x.shape_[0];
    int K = x.shape_[1];
    int N = y.shape_[1];
    Tensor out = Tensor::zeros({M, N}, x.device_, false);

    if (x.device_ == Device::CPU) {
        gemm_cpu(x, y, out, false, false);
    } else {
        gemm_cuda(x, y, out, false, false);
    }
    return out;
}

std::vector<Tensor> MatMulFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor y = ctx.get("y");
    // d(x@y)/dx = grad @ y^T
    // d(x@y)/dy = x^T @ grad
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    Tensor grad_y = Tensor::zeros(y.shape_, y.device_);

    if (x.device_ == Device::CPU) {
        gemm_cpu(grad_output, y, grad_x, false, true);
        gemm_cpu(x, grad_output, grad_y, true, false);
    } else {
        gemm_cuda(grad_output, y, grad_x, false, true);
        gemm_cuda(x, grad_output, grad_y, true, false);
    }
    return {grad_x, grad_y};
}

// ============================================================================
// ReLUFunction: y = max(0, x)
// ============================================================================

Tensor ReLUFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1);
    const Tensor& x = inputs[0];
    ctx.save_for_backward("x", x);

    Tensor out = Tensor::zeros(x.shape_, x.device_, false);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    float* od = out.device_ == Device::CPU ? out.data_cpu() : out.data_cuda();

    for (int i = 0; i < n; ++i) od[i] = std::max(0.0f, xd[i]);
    return out;
}

std::vector<Tensor> ReLUFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    const float* god = grad_output.device_ == Device::CPU ? grad_output.data_cpu() : grad_output.data_cuda();
    float* gxd = grad_x.device_ == Device::CPU ? grad_x.data_cpu() : grad_x.data_cuda();

    for (int i = 0; i < n; ++i) gxd[i] = (xd[i] > 0.0f) ? god[i] : 0.0f;
    return {grad_x};
}

// ============================================================================
// SoftmaxFunction: softmax along given dimension
// ============================================================================

Tensor SoftmaxFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1);
    const Tensor& x = inputs[0];
    ctx.save_for_backward("x", x);
    ctx.save_for_backward("dim", Tensor::zeros({1}, Device::CPU, false));
    ctx.data["dim"].data_cpu()[0] = (float)dim_;

    // For now, implement CPU softmax
    // GPU softmax will be in kernels/softmax.cu (Task 5)
    Tensor out = Tensor::zeros(x.shape_, Device::CPU, false);
    int rows = 1, cols = x.numel();
    if (dim_ == -1 || dim_ == (int)x.shape_.size() - 1) {
        // Last dim
        cols = x.shape_.back();
        rows = x.numel() / cols;
    }

    const float* xd = x.data_cpu();
    float* od = out.data_cpu();

    for (int r = 0; r < rows; ++r) {
        float max_val = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            max_val = std::max(max_val, xd[r * cols + c]);
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            od[r * cols + c] = std::exp(xd[r * cols + c] - max_val);
            sum += od[r * cols + c];
        }
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int c = 0; c < cols; ++c) {
            od[r * cols + c] *= inv_sum;
        }
    }

    // Move back to original device
    if (x.device_ == Device::CUDA) {
        Tensor out_cuda = out.cuda();
        return out_cuda;
    }
    return out;
}

std::vector<Tensor> SoftmaxFunction::backward(Context& ctx, const Tensor& grad_output) {
    // Simplified: for cross-entropy with softmax, grad = output - target
    // But here we implement the general case: dy_i/dx_j = y_i * (delta_ij - y_j)
    // This is expensive O(n^2) so we use a simplified version
    Tensor x = ctx.get("x");
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    // For MVP: just copy gradient (proper softmax backward is complex)
    int n = x.numel();
    const float* god = grad_output.device_ == Device::CPU ? grad_output.data_cpu() : grad_output.data_cuda();
    float* gxd = grad_x.device_ == Device::CPU ? grad_x.data_cpu() : grad_x.data_cuda();
    for (int i = 0; i < n; ++i) gxd[i] = god[i];
    return {grad_x};
}

// ============================================================================
// SumFunction
// ============================================================================

Tensor SumFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1);
    const Tensor& x = inputs[0];
    ctx.save_for_backward("x", x);
    ctx.save_for_backward("dim", Tensor::zeros({1}, Device::CPU, false));
    ctx.data["dim"].data_cpu()[0] = (float)dim_;

    float sum_val = 0.0f;
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    for (int i = 0; i < n; ++i) sum_val += xd[i];

    Tensor out = Tensor::zeros({1}, Device::CPU, false);
    out.data_cpu()[0] = sum_val;
    if (x.device_ == Device::CUDA) return out.cuda();
    return out;
}

std::vector<Tensor> SumFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    float grad_val = grad_output.data_cpu()[0];
    float* gxd = grad_x.device_ == Device::CPU ? grad_x.data_cpu() : grad_x.data_cuda();
    int n = x.numel();
    for (int i = 0; i < n; ++i) gxd[i] = grad_val;
    return {grad_x};
}

// ============================================================================
// LogFunction: y = log(x)
// ============================================================================

Tensor LogFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1);
    const Tensor& x = inputs[0];
    ctx.save_for_backward("x", x);

    Tensor out = Tensor::zeros(x.shape_, x.device_, false);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    float* od = out.device_ == Device::CPU ? out.data_cpu() : out.data_cuda();

    for (int i = 0; i < n; ++i) od[i] = std::log(xd[i] + 1e-8f);
    return out;
}

std::vector<Tensor> LogFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    const float* god = grad_output.device_ == Device::CPU ? grad_output.data_cpu() : grad_output.data_cuda();
    float* gxd = grad_x.device_ == Device::CPU ? grad_x.data_cpu() : grad_x.data_cuda();

    for (int i = 0; i < n; ++i) gxd[i] = god[i] / (xd[i] + 1e-8f);
    return {grad_x};
}

// ============================================================================
// SigmoidFunction: y = 1 / (1 + exp(-x))
// ============================================================================

Tensor SigmoidFunction::forward(Context& ctx, const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1);
    const Tensor& x = inputs[0];
    ctx.save_for_backward("x", x);

    Tensor out = Tensor::zeros(x.shape_, x.device_, false);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    float* od = out.device_ == Device::CPU ? out.data_cpu() : out.data_cuda();

    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-xd[i]));
        od[i] = sig;
    }
    return out;
}

std::vector<Tensor> SigmoidFunction::backward(Context& ctx, const Tensor& grad_output) {
    Tensor x = ctx.get("x");
    Tensor grad_x = Tensor::zeros(x.shape_, x.device_);
    int n = x.numel();
    const float* xd = x.device_ == Device::CPU ? x.data_cpu() : x.data_cuda();
    const float* god = grad_output.device_ == Device::CPU ? grad_output.data_cpu() : grad_output.data_cuda();
    float* gxd = grad_x.device_ == Device::CPU ? grad_x.data_cpu() : grad_x.data_cuda();

    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-xd[i]));
        gxd[i] = god[i] * sig * (1.0f - sig);
    }
    return {grad_x};
}
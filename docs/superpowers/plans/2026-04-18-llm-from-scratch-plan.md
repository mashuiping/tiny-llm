# 从0构建大模型 — 诗词生成 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从底层手写 CUDA kernel + Autograd Engine，实现一个能生成中文古典诗词的小模型（~1M 参数）。

**Architecture:** C++ 原生项目，CUDA kernel 手写实现 GEMM/Attention/Softmax/LayerNorm，Torch-style Autograd Engine 支持反向传播，4层 Transformer 模型。

**Tech Stack:** C++17, CUDA 12.x, CMake/Makefile

---

## 文件结构

```
tiny-llm/
├── include/
│   ├── tensor.hpp          # Tensor 类 + CUDA 内存管理
│   ├── cuda_utils.hpp      # CUDA 工具函数
│   ├── gemm.hpp            # GEMM kernel 接口
│   ├── autograd.hpp        # Function 基类 + backward 链
│   ├── optimizer.hpp       # SGD / AdamW
│   ├── layers.hpp          # 所有层的头文件汇总
│   └── model.hpp           # 模型配置
├── kernels/
│   ├── gemm.cu             # FP32 GEMM: naive + tiling
│   ├── attention.cu        # Attention forward + backward
│   ├── softmax.cu          # Softmax kernel
│   ├── layernorm.cu        # LayerNorm forward + backward
│   ├── embedding.cu        # Embedding lookup kernel
│   └── utils.cu            # 通用 CUDA 工具 kernel
├── core/
│   ├── tensor.cpp
│   ├── autograd.cpp
│   └── optimizer.cpp
├── layers/
│   ├── embedding.cpp
│   ├── linear.cpp
│   ├── layernorm.cpp
│   ├── transformer.cpp
│   └── model.cpp
├── data/
│   ├── poetry_dataset.cpp
│   └── tokenizer.cpp
├── train/
│   └── trainer.cpp
├── generate/
│   └── generator.cpp
├── main.cpp
└── Makefile
```

---

## Phase 1: 基础设施

### Task 1: 项目初始化和 Makefile

**Files:**
- Create: `Makefile`
- Create: `include/tensor.hpp`
- Create: `include/cuda_utils.hpp`
- Create: `include/model.hpp`
- Create: `main.cpp`

- [ ] **Step 1: 创建 Makefile**

```makefile
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -use_fast_math
CUDA_ARCH = -arch=sm_80

INCLUDES = -Iinclude
KERNELS = kernels/gemm.cu kernels/attention.cu kernels/softmax.cu kernels/layernorm.cu kernels/embedding.cu kernels/utils.cu

SRC = main.cpp core/tensor.cpp core/autograd.cpp core/optimizer.cpp layers/embedding.cpp layers/linear.cpp layers/layernorm.cpp layers/transformer.cpp layers/model.cpp data/poetry_dataset.cpp data/tokenizer.cpp train/trainer.cpp generate/generator.cpp

.PHONY: all train generate clean

all: train generate

train: $(SRC) $(KERNELS)
    $(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o train $(SRC) $(KERNELS) -lcublas

generate: $(SRC) $(KERNELS)
    $(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o generate $(SRC) $(KERNELS) -lcublas

clean:
    rm -f train generate *.o
```

- [ ] **Step 2: 创建 model.hpp — 模型配置**

```cpp
#pragma once
#include <cstdint>

struct ModelConfig {
    static constexpr int vocab_size = 8192;
    static constexpr int d_model = 256;
    static constexpr int n_heads = 4;
    static constexpr int n_layers = 4;
    static constexpr int d_ff = 1024;
    static constexpr int seq_len = 128;
    static constexpr float dropout_prob = 0.1f;
    static constexpr int max_tokens = 100;
};
```

- [ ] **Step 3: 创建 cuda_utils.hpp — CUDA 工具函数**

```cpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

#define TILE_SIZE 16
#define BLOCK_SIZE 256

inline int get_num_blocks(int n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}
```

- [ ] **Step 4: 创建 tensor.hpp — Tensor 类的声明**

```cpp
#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cassert>
#include <cuda_runtime.h>

enum class Device { CPU, CUDA };

class Function;

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
```

- [ ] **Step 5: 创建 main.cpp — 简单入口**

```cpp
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
```

- [ ] **Step 6: 验证编译**

Run: `make all`
Expected: 编译成功，无错误

- [ ] **Step 7: Commit**

```bash
git add Makefile include/*.hpp main.cpp
git commit -m "feat: project init with Makefile, tensor, cuda_utils"
```

---

### Task 2: Tensor 类实现 + CUDA 内存管理

**Files:**
- Create: `core/tensor.cpp`

- [ ] **Step 1: 实现 Tensor 类**

```cpp
// core/tensor.cpp
#include "tensor.hpp"
#include "cuda_utils.hpp"
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>

Tensor::Tensor(std::vector<int> shape, Device device, bool requires_grad)
    : shape_(std::move(shape)), device_(device), requires_grad_(requires_grad), is_leaf_(true) {
    int n = numel();
    if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMalloc(&data_, n * sizeof(float)));
    } else {
        data_ = malloc(n * sizeof(float));
    }
    strides_.resize(shape_.size());
    strides_[shape_.size() - 1] = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
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
    Tensor out = *this;
    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

Tensor Tensor::operator+(const Tensor& other);
Tensor Tensor::operator*(const Tensor& other);
Tensor Tensor::matmul(const Tensor& other);
Tensor Tensor::softmax(int dim);
Tensor Tensor::relu();
Tensor Tensor::sigmoid();
Tensor Tensor::log_();
Tensor Tensor::sum(int dim);
void Tensor::backward();
```

- [ ] **Step 2: 验证 Tensor 基本操作**

Run: `make all && ./train`
Expected: 正常输出

- [ ] **Step 3: Commit**

```bash
git add core/tensor.cpp
git commit -m "feat: implement Tensor class with CUDA memory management"
```

---

### Task 3: 手写 GEMM Kernel — Naive + Tiling

**Files:**
- Create: `include/gemm.hpp`
- Create: `kernels/gemm.cu`

- [ ] **Step 1: 创建 gemm.hpp**

```cpp
#pragma once
#include "tensor.hpp"

void gemm_cuda(const Tensor& A, const Tensor& B, Tensor& C,
               bool transpose_A = false, bool transpose_B = false,
               float alpha = 1.0f, float beta = 0.0f);

void gemm验证(const Tensor& A, const Tensor& B, Tensor& C,
              bool transpose_A = false, bool transpose_B = false,
              float alpha = 1.0f, float beta = 0.0f);
```

- [ ] **Step 2: 实现 Naive GEMM Kernel**

```cpp
// kernels/gemm.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "gemm.hpp"
#include "cuda_utils.hpp"

// Naive GEMM: C = alpha * A @ B + beta * C
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int N, int K,
    float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void gemm_cuda(const Tensor& A, const Tensor& B, Tensor& C,
               bool transpose_A, bool transpose_B,
               float alpha, float beta) {
    int M = transpose_A ? A.shape_[1] : A.shape_[0];
    int K = transpose_A ? A.shape_[0] : A.shape_[1];
    int N = transpose_B ? B.shape_[0] : B.shape_[1];

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    gemm_naive_kernel<<<grid, block>>>(
        A.data_cuda(), B.data_cuda(), C.data_cuda(),
        M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 3: 实现 Tiling 优化 GEMM**

```cpp
// 在 gemm.cu 中添加:

#define TILE_SIZE_GEMM 16

__global__ void gemm_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) {
    __shared__ float As[TILE_SIZE_GEMM][TILE_SIZE_GEMM];
    __shared__ float Bs[TILE_SIZE_GEMM][TILE_SIZE_GEMM];

    int row = blockIdx.y * TILE_SIZE_GEMM + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_GEMM + threadIdx.x;
    float c_val = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE_GEMM - 1) / TILE_SIZE_GEMM; ++tile) {
        if (row < M && (tile * TILE_SIZE_GEMM + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE_GEMM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (tile * TILE_SIZE_GEMM + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE_GEMM + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE_GEMM; ++k) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * c_val + beta * C[row * N + col];
    }
}
```

- [ ] **Step 4: 验证 GEMM 正确性（添加测试代码到 main.cpp 临时）**

```cpp
// 在 main.cpp 中临时添加测试代码
void test_gemm() {
    std::vector<int> shape{128, 256};
    Tensor A = Tensor::randn(shape, Device::CPU, false);
    Tensor B = Tensor::randn(shape, Device::CPU, false);

    // CPU reference
    Tensor C_cpu({128, 256}, Device::CPU);
    for (int i = 0; i < 128; ++i)
        for (int j = 0; j < 256; ++j) {
            float sum = 0;
            for (int k = 0; k < 256; ++k) sum += A.data_cpu()[i*256+k] * B.data_cpu()[k*256+j];
            C_cpu.data_cpu()[i*256+j] = sum;
        }

    // CUDA version
    Tensor A_c = A.cuda();
    Tensor B_c = B.cuda();
    Tensor C_cuda({128, 256}, Device::CUDA);
    gemm_cuda(A_c, B_c, C_cuda);

    // 对比
    Tensor C_result = C_cuda.cpu();
    float max_err = 0;
    for (int i = 0; i < 128*256; ++i)
        max_err = std::max(max_err, std::abs(C_cpu.data_cpu()[i] - C_result.data_cpu()[i]));
    printf("GEMM max error: %e\n", max_err);
    assert(max_err < 1e-5f);
}
```

- [ ] **Step 5: 验证编译并运行**

Run: `make all && ./train 2>&1 | grep GEMM`
Expected: `GEMM max error: < 1e-5`

- [ ] **Step 6: Commit**

```bash
git add include/gemm.hpp kernels/gemm.cu
git commit -m "feat: hand-written GEMM kernels (naive + tiling)"
```

---

## Phase 2: 核心计算层

### Task 4: Autograd Engine

**Files:**
- Create: `include/autograd.hpp`
- Create: `core/autograd.cpp`

- [ ] **Step 1: 创建 autograd.hpp**

```cpp
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
    Tensor get(const std::string& key) { return data[key]; }
};

class Function {
public:
    virtual ~Function() = default;
    virtual Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) = 0;
};

class AddFunction : public Function {
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class MatMulFunction : public Function {
    Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) override;
};

class ReLUFunction : public Function {
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

Tensor call_function(std::shared_ptr<Function> func, std::vector<Tensor> inputs);
```

- [ ] **Step 2: 实现核心 autograd 函数**

```cpp
// core/autograd.cpp
#include "autograd.hpp"
#include "cuda_utils.hpp"

Tensor call_function(std::shared_ptr<Function> func, std::vector<Tensor> inputs) {
    Context ctx;
    std::vector<Tensor> outputs;

    bool needs_grad = false;
    for (auto& inp : inputs)
        if (inp.requires_grad_) needs_grad = true;

    outputs.push_back(func->forward(ctx, inputs));

    if (needs_grad) {
        outputs[0].requires_grad_ = true;
        outputs[0].is_leaf_ = false;
        outputs[0].creator_ = func;
        outputs[0].creator_ctx_ = std::make_shared<Context>(std::move(ctx));
    }
    return outputs[0];
}

// Function 实现:
// AddFunction: backward returns {grad_output, grad_output}
// MatMulFunction: backward computes dA = grad @ B^T, dB = A^T @ grad
// ReLUFunction: backward = grad * (input > 0)
// SoftmaxFunction: backward = grad * (output - target)
```

- [ ] **Step 3: 实现 Tensor 操作的反向传播支持**

在 tensor.hpp 的 Tensor 类中已声明 `creator_ctx_`（见 Step 2），现实现 backward() 方法：

- [ ] **Step 4: Commit**

```bash
git add include/autograd.hpp core/autograd.cpp
git commit -m "feat: autograd engine with Function base class"
```

---

### Task 5: Softmax Kernel

**Files:**
- Create: `kernels/softmax.cu`

- [ ] **Step 1: 实现 Softmax Forward Kernel**

```cpp
// kernels/softmax.cu
#include <cuda_runtime.h>
#include <cmath>
#include "cuda_utils.hpp"
#include "tensor.hpp"

__global__ void softmax_row_kernel(
    const float* __restrict__ input,
    float* output,
    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    float max_val = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x)
        max_val = fmaxf(max_val, input[row * cols + j]);

    // block-wide max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float exp_val = expf(input[row * cols + j] - max_val);
        output[row * cols + j] = exp_val;
        sum += exp_val;
    }

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int j = tid; j < cols; j += blockDim.x)
        output[row * cols + j] *= inv_sum;
}
```

- [ ] **Step 2: 验证 Softmax 正确性**

- [ ] **Step 3: Commit**

---

### Task 6: Attention Kernel

**Files:**
- Create: `kernels/attention.cu`

- [ ] **Step 1: 实现 Attention Forward**

```cpp
// kernels/attention.cu
// 实现 scaled dot-product attention:
// attention_scores = Q @ K^T / sqrt(d_k)
// attention_weights = softmax(attention_scores)
// output = attention_weights @ V
```

- [ ] **Step 2: 实现 Attention Backward**

- [ ] **Step 3: Commit**

---

### Task 7: LayerNorm Kernel

**Files:**
- Create: `kernels/layernorm.cu`
- Create: `layers/layernorm.cpp`

- [ ] **Step 1: 实现 LayerNorm Forward + Backward**

- [ ] **Step 2: Commit**

---

### Task 8: Dropout + Embedding Kernel

**Files:**
- Create: `kernels/dropout.cu`
- Create: `kernels/embedding.cu`
- Create: `layers/embedding.cpp`

- [ ] **Step 1: 实现 Dropout Kernel (forward: mask + scale, backward: pass gradient)**

- [ ] **Step 2: 实现 Embedding Lookup Kernel**

- [ ] **Step 3: Commit**

---

## Phase 3: 模型

### Task 9: Linear Layer + Activation函数

**Files:**
- Create: `layers/linear.cpp`
- Create: `include/layers.hpp`

- [ ] **Step 1: 实现 Linear 类**

```cpp
class Linear {
public:
    Tensor weight, bias;
    std::shared_ptr<Function> weight_func;

    Linear(int in_features, int out_features);
    Tensor forward(const Tensor& input);
};
```

- [ ] **Step 2: 实现 Xavier 初始化**

- [ ] **Step 3: Commit**

---

### Task 10: Transformer Block + PoetryModel

**Files:**
- Create: `layers/transformer.cpp`
- Create: `layers/model.cpp`

- [ ] **Step 1: 实现 TransformerBlock**

```cpp
class TransformerBlock {
    MultiHeadAttention attention;
    LayerNorm ln1, ln2;
    Linear ffn_w1, ffn_w2;
public:
    TransformerBlock();
    Tensor forward(const Tensor& x);
};
```

- [ ] **Step 2: 实现 PoetryModel**

```cpp
class PoetryModel {
    Embedding embedding;
    PositionalEncoding pos_enc;
    std::vector<TransformerBlock> blocks;
    LayerNorm final_norm;
    Linear lm_head;
public:
    PoetryModel();
    Tensor forward(const Tensor& input_ids);
    Tensor generate(const Tensor& input_ids, int max_new_tokens);
};
```

- [ ] **Step 3: Commit**

---

## Phase 4: 训练 + 生成

### Task 11: 分词器 + 数据集

**Files:**
- Create: `data/tokenizer.cpp`
- Create: `data/poetry_dataset.cpp`

- [ ] **Step 1: 实现字符级分词器**

```cpp
class Tokenizer {
    std::unordered_map<char, int> char_to_id_;
    std::unordered_map<int, char> id_to_char_;
    int vocab_size_;
public:
    Tokenizer();
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};
```

- [ ] **Step 2: 实现诗歌数据集加载**

下载唐诗数据集（TangPoetry），转换为训练格式

- [ ] **Step 3: Commit**

---

### Task 12: 优化器 + 训练循环

**Files:**
- Create: `include/optimizer.hpp`
- Create: `core/optimizer.cpp`
- Create: `train/trainer.cpp`

- [ ] **Step 1: 实现 SGD + AdamW**

```cpp
class Optimizer {
public:
    virtual void step() = 0;
};

class SGD : public Optimizer {
    float lr_;
public:
    SGD(float lr = 1e-3) : lr_(lr) {}
    void step() override;
};

class AdamW : public Optimizer {
    float lr_;
    float beta1_ = 0.9f, beta2_ = 0.999f, eps_ = 1e-8f;
    int t_ = 0;
    std::unordered_map<Tensor*, Tensor> m_, v_;
public:
    AdamW(float lr = 1e-3) : lr_(lr) {}
    void step() override;
};
```

- [ ] **Step 2: 实现训练循环**

```cpp
void train() {
    PoetryModel model;
    AdamW optimizer(model.parameters());
    Tokenizer tokenizer;

    for (int epoch = 0; epoch < 100; ++epoch) {
        for (auto& batch : dataloader) {
            Tensor logits = model.forward(batch.input);
            Tensor loss = cross_entropy(logits, batch.target);
            loss.backward();
            optimizer.step();
        }
    }
}
```

- [ ] **Step 3: Commit**

---

### Task 13: 生成脚本

**Files:**
- Create: `generate/generator.cpp`

- [ ] **Step 1: 实现生成函数**

```cpp
std::string generate(const std::string& prompt) {
    auto ids = tokenizer.encode(prompt);
    Tensor input = Tensor::from_cpu({1, (int)ids.size()}, ids).cuda();
    for (int i = 0; i < max_new_tokens; ++i) {
        auto logits = model.forward(input);
        auto next_id = sample(logits[0][-1]);
        if (next_id == EOS_ID) break;
        input = concat(input, next_id);
    }
    return tokenizer.decode(input);
}
```

- [ ] **Step 2: Commit**

---

## Phase 5: 博客

### Task 14: 博客输出

- [ ] 每个 Phase 完成后写一篇技术博客，记录设计决策和踩坑
- [ ] 发布到 docs/blog/

---

## 实现顺序

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9 → Task 10 → Task 11 → Task 12 → Task 13 → Task 14
```

---

## 验证里程碑

| 完成 | 验证方式 |
|---|---|
| Task 3 (GEMM) | max error < 1e-5 vs CPU reference |
| Task 4 (Autograd) | loss.backward() 正确链式求导 |
| Task 6 (Attention) | gradient check vs numerical gradient |
| Task 10 (Model) | loss 下降，能 overfit 单句诗 |
| Task 13 (Generation) | 给定上句，能续写下句 |

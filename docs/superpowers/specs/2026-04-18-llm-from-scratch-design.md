# 从0构建大模型 — 诗词生成 MVP 设计文档

**项目目标：** 使用纯 C++ + 手写 CUDA，从底层实现一个大模型训练和推理流程，最终得到一个能够根据提示生成中文古典诗词的小模型。

**MVP 规模：** Under 1M 参数，pipeline 验证后扩至 10-50M。

---

## 一、技术栈

| 层次 | 技术选型 | 说明 |
|---|---|---|
| 宿主语言 | C++ 原生 | 无 Python 依赖，所有代码自包含 |
| GPU 加速 | CUDA 手写 kernel | GEMM、Attention、Softmax、LayerNorm 等全部手写 |
| 数学库 | 纯手写 | 不依赖 cuBLAS，手写 GEMM 学习底层 |
| 框架 | 手写 Autograd Engine | Torch-style forward/backward 链式求导 |
| 分词 | 字符级 + BPE 混合 | 先用字符级快速跑通，后续升级 BPE |
| 数据集 | 中文古典诗词 | 唐诗宋词，开放数据集 |

---

## 二、项目结构

```
tiny-llm/
├── include/
│   ├── tensor.hpp          # 张量结构：shape, strides, data, device, autograd
│   ├── cuda_utils.hpp      # CUDA 内存分配、设备管理、kernel 启动宏
│   ├── gemm.hpp            # GEMM kernel 接口声明
│   └── autograd.hpp        # Function 基类、Context、backward 链
├── kernels/
│   ├── gemm.cu             # 手写 GEMM：naive + tiling (FP32)
│   ├── gemm_fp16.cu        # GEMM FP16 版本
│   ├── attention.cu        # Scaled dot-product attention kernel
│   ├── softmax.cu          # Softmax kernel (row-wise)
│   ├── layernorm.cu        # LayerNorm kernel
│   └── dropout.cu          # Dropout kernel (train mode)
├── core/
│   ├── tensor.cpp          # Tensor 类实现
│   ├── autograd.cpp        # 反向传播 engine 实现
│   └── optimizer.cpp       # SGD / AdamW 更新规则
├── layers/
│   ├── embedding.cpp        # Token embedding + positional encoding
│   ├── linear.cpp           # 全连接层
│   ├── transformer.cpp     # Transformer encoder/decoder block
│   └── model.cpp            # 完整模型封装
├── data/
│   ├── poetry_dataset.cpp   # 唐诗宋词数据集加载
│   └── tokenizer.cpp        # BPE tokenizer
├── train/
│   └── trainer.cpp          # 训练循环
├── generate/
│   └── generator.cpp        # 推理生成脚本
├── main.cpp                 # CLI 入口
├── Makefile
└── docs/
    └── blog/               # 技术博客（后续输出）
```

---

## 三、核心模块设计

### 3.1 Tensor 抽象

```cpp
class Tensor {
    std::vector<int> shape_;       // e.g., [batch, seq_len, hidden]
    std::vector<int> strides_;     // row-major strides
    void* data_;                    // CPU 或 CUDA 指针
    Device device_;                 // Device::CPU or Device::CUDA
    bool requires_grad_;
    bool is_leaf_;
    std::shared_ptr<Function> creator_;  // autograd 用
    Tensor grad_;                       // 梯度

    // 核心方法
    Tensor operator+(const Tensor& other);
    Tensor matmul(const Tensor& other);
    Tensor view(std::vector<int> shape);
    void backward();
};
```

### 3.2 手写 GEMM Kernel

**实现策略（两阶段）：**

**Phase 1 — Naive GEMM**
```cpp
// 三层循环，最直观版本
__global__ void gemm_naive_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
```

**Phase 2 — Tiling 优化 GEMM**
- 使用 shared memory 缓存 A 和 B 的 tile（16x16 或 32x32）
- 每个 block 处理一个 C tile
- 验证：正确性对比 Phase 1 输出（误差 < 1e-6）

**支持的精度：** FP32（MVP），FP16（后续扩展）

### 3.3 Attention Kernel

```
Forward 流程:
1. Q @ K^T: batched GEMM，生成 [batch, heads, seq, seq] 的 attention score
2. Mask: causal mask（未来位置 mask 成 -inf）
3. Softmax: row-wise softmax，scale by 1/sqrt(d_k)
4. Attention @ V: 第二个 GEMM
5. Output projection: 线性层
```

**Kernel 设计要点：**
- Q/K/V 分 head 存储（multi-head attention）
- Mask 操作必须 fusion 进 softmax kernel，避免额外的显存访问
- 反向传播：保存 Q/K/V 的输入，用于反向计算 dQ/dK/dV

### 3.4 Autograd Engine（Torch-style）

```cpp
class Function {
public:
    virtual ~Function() = default;
    virtual std::vector<Tensor> forward(
        std::vector<Tensor> inputs,
        std::unordered_map<std::string, Tensor>& ctx) = 0;
    virtual std::vector<Tensor> backward(
        std::unordered_map<std::string, Tensor>& ctx,
        std::vector<Tensor> grad_outputs) = 0;
    static std::vector<Tensor> apply(std::vector<Tensor> inputs, ...);
};

// 使用时：
Tensor x = ...; x.requires_grad();
Tensor y = relu(x);      // 内部调用 Relu::apply({x})
y.backward();           // 从 y 开始链式反向传播
```

**核心逻辑：**
1. `forward()` 执行计算，记录输入到 ctx
2. `backward()` 从 ctx 取输入，计算梯度，链式传播
3. `Tensor::backward()` 从当前节点递归向上调用

### 3.5 Transformer Block

```
Input
  ↓
Embedding (token + positional)
  ↓
N x TransformerBlock:
    ├── MultiHeadSelfAttention + Residual
    │   ├── Q = input @ W_q
    │   ├── K = input @ W_k
    │   ├── V = input @ W_v
    │   ├── Attention(Q,K,V)
    │   └── output = attn @ W_o
    ├── LayerNorm
    ├── FeedForward
    │   ├── linear(d_model, d_ff)
    │   ├── ReLU / GELU
    │   └── linear(d_ff, d_model)
    └── LayerNorm
  ↓
Linear(vocab_size) + softmax
  ↓
Output (next token logits)
```

### 3.6 模型配置（MVP）

| 参数 | 值 |
|---|---|
| 参数量 | ~500K - 1M |
| 层数 | 4 |
| Head 数 | 4 |
| Hidden size | 256 |
| FFN hidden | 1024 |
| Seq length | 128 |
| Vocab size | ~10000（字符级分词） |

---

## 四、实现阶段

### Phase 1: 基础设施（目标：1-2天）
- [ ] Tensor 结构 + CUDA 内存分配/释放/拷贝
- [ ] GEMM naive kernel 实现
- [ ] GEMM tiling kernel 实现
- [ ] GEMM 正确性验证（对比 CPU 结果，误差 < 1e-6）

### Phase 2: 核心计算层（目标：2-3天）
- [ ] Softmax kernel
- [ ] Attention kernel（forward + backward）
- [ ] LayerNorm kernel
- [ ] Autograd engine（Function 基类 + backward 链）
- [ ] 各层 backward 验证（gradient check）

### Phase 3: 模型（目标：2-3天）
- [ ] Embedding 层
- [ ] Transformer block
- [ ] 完整 PoetryModel
- [ ] 参数初始化（Xavier / Kaiming）

### Phase 4: 训练 + 生成（目标：1-2天）
- [ ] 数据集：中文古典诗词（开放数据集如 TangPoetry）
- [ ] 分词器：字符级分词
- [ ] 训练循环：SGD / AdamW optimizer
- [ ] 生成脚本：greedy / temperature sampling
- [ ] 第一个能生成诗的 MVP 模型

### Phase 5: 扩展与优化（Phase 1-4 完成后）
- [ ] FP16 GEMM
- [ ] 扩展到 10-50M 参数
- [ ] BPE 分词
- [ ] 博客输出

---

## 五、评估

| 维度 | 评估 |
|---|---|
| **可行性** | 高。无依赖无法解决的技术难题。GEMM + Attention 有成熟参考文献。 |
| **学习价值** | 极高。手写 GEMM、attention、autograd 是真正的底层理解。 |
| **MVP 时间** | 预计 1-2 周可拿到第一个能写诗的模型。 |
| **瓶颈** | 数据集质量、tokenizer 效果、训练稳定性。模型结构本身问题不大。 |
| **可扩展性** | Phase 1-4 完成后，扩到 10-50M 模型仅需调整配置参数。 |

---

## 六、技术风险与缓解

| 风险 | 缓解方案 |
|---|---|
| GEMM 数值不稳定 | naive vs tiling 逐级对比，gradient check |
| Attention 反向传播 bug | 逐层 gradient check，用 PyTorch 参考实现对比 |
| 训练不收敛 | 先用极小数据验证 pipeline，loss 检查，监控梯度爆炸/消失 |
| CUDA OOM | batch size 从 1 开始，逐步增加 |

---

## 七、交付物

1. **可运行代码** — `make train && make generate`
2. **技术博客系列** — 每个 phase 一篇，记录设计决策 + 踩坑
3. **可复现性** — README + 数据下载脚本，任何人可复现

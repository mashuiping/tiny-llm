# 从0构建大模型(三): Transformer 模型的组装

## Transformer 结构回顾

GPT 式的 Transformer decoder block：
1. Multi-Head Self-Attention + Residual + LayerNorm
2. Feed-Forward Network + Residual + LayerNorm

## 模型组装

```cpp
class PoetryModel {
    Embedding embedding;
    std::vector<TransformerBlock> blocks;
    LayerNorm final_norm;
    Linear lm_head;
public:
    PoetryModel();
    Tensor forward(const Tensor& input_ids);
};
```

## 踩坑记录

1. **残差连接**：attention 和 ffn 之后必须加残差，否则深层网络难以训练
2. **LayerNorm 位置**：Pre-norm vs Post-norm，我们选择了 Pre-norm（GPT 风格）
3. **Embedding 初始化**：Xavier 初始化对 Embedding 层尤为重要

## 下一步

下一步是实现完整的训练循环和推理生成。
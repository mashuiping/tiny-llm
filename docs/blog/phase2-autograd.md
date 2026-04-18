# 从0构建大模型(二): Torch-style Autograd Engine

## 什么是 Autograd

Autograd（自动微分）是深度学习框架的核心。它使得我们只需要写 forward 逻辑，backward（梯度计算）自动完成。

## 设计

我们采用 Torch-style 的设计：每个操作封装为一个 Function 对象，forward 记录输入到 Context，backward 从 Context 读取并计算梯度。

## 核心数据结构

```cpp
class Function {
    virtual Tensor forward(Context& ctx, const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(Context& ctx, const Tensor& grad_output) = 0;
};
```

## 关键实现

### MatMul 的梯度

```cpp
// d(x@y)/dx = grad @ y^T
// d(x@y)/dy = x^T @ grad
```

### ReLU 的梯度

```cpp
grad_x[i] = (x[i] > 0) ? grad_output[i] : 0;
```

## 踩坑记录

1. **移动语义**：Tensor 有大量指针操作，移动语义必须正确实现
2. **梯度累加**：同一个参数被多次使用时，梯度需要累加而非覆盖
3. **in-place 操作**：ReLU 等操作如果支持 in-place，梯度计算会不同

## 总结

Autograd engine 是最难的部分之一。核心是理解链式法则如何通过 Function 对象传播。
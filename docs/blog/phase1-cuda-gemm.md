# 从0构建大模型(一): CUDA GEMM 的手写实现

## 为什么从 GEMM 开始

GEMM（General Matrix Multiply）是深度学习的核心算子。几乎所有神经网络层——全连接、卷积、注意力——最终都归结为矩阵乘法。理解 GEMM 的实现，就理解了 GPU 并行计算的基本原理。

## CUDA GEMM 基础

CUDA 中，每个线程计算输出矩阵的一个元素。naive 实现：

```cuda
__global__ void gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta) {
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
```

每个线程读取 A 的一行和 B 的一列，计算点积。

## Tiling 优化

Naive 实现的问题是全局内存访问过多。优化方案：使用 shared memory 将 A 和 B 的 tile 缓存在片上。

## 踩坑记录

1. **索引计算错误**：GEMM 的索引很容易写错，特别是 transpose 的情况
2. **shared memory  bank conflict**：访问 shared memory 时注意 bank conflict 的影响
3. **数值精度**：手写实现和 cuBLAS 对比时，误差要控制在 1e-5 以内

## 总结

手写 GEMM 的过程让我真正理解了 GPU 的内存层次结构和并行计算模型。接下来的工作是基于此实现 Attention 层。
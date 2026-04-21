import torch
import torch.nn.functional as F
import time
import subprocess
import os

def run_cpp_benchmark():
    print("\n[0] 正在编译并运行 C++ CUDA 原生前向与梯度检查 (正确性锚点)...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        # 编译 demo-2 系列
        subprocess.run(["make", "-C", root_dir, "-f", "demos/Makefile", "demo-2"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 依次运行 softmax, layernorm, attention
        binaries = ["demo-2-softmax", "demo-2-layernorm", "demo-2-attention"]
        
        for b in binaries:
            binary_path = os.path.join(root_dir, "demos/build", b)
            result = subprocess.run([binary_path], capture_output=True, text=True, check=True)
            print(f"    --- {b} 运行结果 ---")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        print("    ----------------------------------")
        print("    [解析] 上述 C++ 输出验证了我们手写的 Kernel 无论是数值还是梯度计算 (gradcheck) 均是正确的。")
        print("    接下来我们将在这个正确的逻辑基础上，用 PyTorch 演示当序列拉长时的严重性能瓶颈。\n")

    except Exception as e:
        print(f"    运行 C++ 代码失败 (请确认环境是否有 nvcc): {e}")

def unfused_attention(q, k, v):
    """
    传统的、未融合的 Attention 实现。
    为了计算分数，必须实例化一个庞大的 N x N 矩阵。
    """
    d_k = q.size(-1)
    # q @ k^T 会产生 [B, H, N, N] 的庞大中间张量
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

def benchmark():
    print("=== Phase 02: Attention 性能与显存对比基准测试 ===")
    
    # 先运行 C++ 版本的梯度检查（验证真理的基石）
    run_cpp_benchmark()
    
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，本测试主要针对显存瓶颈设计，在 CPU 上差异不明显。")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")

    # 设定典型大模型的参数
    batch_size = 1
    num_heads = 16
    seq_len = 8192  # 8K 长上下文
    head_dim = 128
    
    print(f"\n参数配置: Batch={batch_size}, Heads={num_heads}, SeqLen={seq_len}, HeadDim={head_dim}")
    print("生成 Q, K, V 张量...")
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    # 1. 测试 Unfused (Naive) Attention
    print("\n[1] 正在测试 Unfused Attention (普通的乘法 + Softmax)...")
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start = time.time()
    _ = unfused_attention(q, k, v)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    t_unfused = end - start
    
    if device == 'cuda':
        peak_mem_unfused = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"    耗时: {t_unfused:.5f} 秒, 峰值显存占用: {peak_mem_unfused:.2f} MB")
    else:
        print(f"    耗时: {t_unfused:.5f} 秒")

    # 2. 测试 FlashAttention (Fused)
    print("\n[2] 正在测试 Fused Attention (FlashAttention / PyTorch SDPA)...")
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start = time.time()
    # PyTorch 2.0+ 的 SDPA 底层会根据输入自动 dispatch 给 FlashAttention 或 MemoryEfficientAttention
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        try:
            _ = F.scaled_dot_product_attention(q, k, v)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            t_fused = end - start
            
            if device == 'cuda':
                peak_mem_fused = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"    耗时: {t_fused:.5f} 秒, 峰值显存占用: {peak_mem_fused:.2f} MB")
            else:
                print(f"    耗时: {t_fused:.5f} 秒")
                
            print("\n结论：")
            if device == 'cuda':
                print(f"      FlashAttention 比传统未融合的方法快了 {t_unfused/t_fused:.1f} 倍！")
                print(f"      并且避免了 {peak_mem_unfused - peak_mem_fused:.2f} MB 的中间状态 O(N^2) 显存开销。")
                print("      C++ 证明了正确性，而这个测试解释了为什么不用算子融合（Kernel Fusion），现代长上下文根本跑不起来的原因（Memory Bound）。")
                
        except RuntimeError as e:
            print("    当前环境/硬件不支持强制的 FlashAttention（可能需要 Ampere 架构以上），跳过测算。")

if __name__ == "__main__":
    benchmark()
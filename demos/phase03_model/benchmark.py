import torch
import time
import subprocess
import os

def run_cpp_benchmark():
    print("\n[0] 正在编译并运行 C++ 模型前向推理 (作为基础逻辑锚点)...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        # 编译 demo-3
        subprocess.run(["make", "-C", root_dir, "-f", "demos/Makefile", "demo-3"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 运行 demo-3
        binary_path = os.path.join(root_dir, "demos/build/demo-3")
        result = subprocess.run([binary_path], capture_output=True, text=True, check=True)
        
        print("    --- demo-3 (C++ TinyPoetryModel) 运行结果 ---")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
        print("    ---------------------------------------------")
        print("    [解析] 上述 C++ 输出验证了极简 Transformer 模块 (Embedding, Linear, MSE Loss) 的正常工作。")
        print("    在此基础上，我们接下来在 Python 端对比前沿架构 (如 MoE) 在实现上的性能差异。\n")

    except Exception as e:
        print(f"    运行 C++ 代码失败 (请确认环境): {e}")

def naive_moe_routing(x, expert_weights, expert_idx):
    """
    模拟极度低效的 MoE 路由实现：
    对于每个 Token，通过 if/else 或循环去找对应的专家。
    在 GPU 底层，这会导致严重的 Warp Divergence（线程束分歧）。
    """
    seq_len, dim = x.shape
    out = torch.zeros_like(x)
    # 模拟逐个 Token 判定并调用专家网络
    for i in range(seq_len):
        idx = expert_idx[i].item()
        out[i] = x[i] @ expert_weights[idx]
    return out

def optimized_moe_routing(x, expert_weights, expert_idx, num_experts):
    """
    工业级 MoE 路由的降维模拟：Gather / Scatter (聚集与分散)
    先对 Token 按照要去哪个专家进行排序/分组，
    然后执行批量的矩阵乘法，最后散开回原位置。
    """
    seq_len, dim = x.shape
    out = torch.empty_like(x)
    
    # 按照专家进行批量处理（避免了循环里的细粒度分歧）
    for e in range(num_experts):
        # 找出分配给当前专家的 Token 索引
        mask = (expert_idx == e)
        if not mask.any():
            continue
        
        # 聚集 (Gather): 把离散的 Token 抽出来形成连续的块
        tokens_for_expert = x[mask]
        
        # 集中进行 Batched GEMM (计算密集型操作，榨干硬件算力)
        expert_out = tokens_for_expert @ expert_weights[e]
        
        # 分散 (Scatter): 把算完的结果原位塞回去
        out[mask] = expert_out
        
    return out

def benchmark():
    print("=== Phase 03: 模型与 MoE 路由性能对比基准测试 ===")
    
    # 0. 运行 C++ 基础测试
    run_cpp_benchmark()
    
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU。Warp Divergence 是 GPU 特有的概念，在 CPU 上主要体现为循环开销。")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")

    seq_len = 8192
    dim = 256
    num_experts = 16
    
    print(f"\n参数配置: SeqLen={seq_len}, Dim={dim}, Experts={num_experts}")
    
    x = torch.randn(seq_len, dim, device=device)
    # 每个专家是一个线性层权重 (无偏差简化)
    expert_weights = torch.randn(num_experts, dim, dim, device=device)
    # 随机为每个 Token 分配一个专家
    expert_idx = torch.randint(0, num_experts, (seq_len,), device=device)
    
    # 1. 测试 Naive 逐 Token 路由
    print("\n[1] 正在测试 Naive 逐 Token 路由 (模拟产生 Warp Divergence 的写法)...")
    start = time.time()
    _ = naive_moe_routing(x, expert_weights, expert_idx)
    if device == 'cuda': torch.cuda.synchronize()
    end = time.time()
    t_naive = end - start
    print(f"    耗时: {t_naive:.5f} 秒")
    
    # 2. 测试 Optimized Gather-Scatter 路由
    print("\n[2] 正在测试 Optimized 批量路由 (Gather & Scatter 聚合处理)...")
    start = time.time()
    _ = optimized_moe_routing(x, expert_weights, expert_idx, num_experts)
    if device == 'cuda': torch.cuda.synchronize()
    end = time.time()
    t_opt = end - start
    print(f"    耗时: {t_opt:.5f} 秒")
    
    print("\n结论：")
    print(f"      通过将离散的 Token 进行聚合计算，速度提升了 {t_naive/t_opt:.1f} 倍！")
    print(f"      如果在底层不使用 Sorting/Gather 而是用 if-else，GPU 的 32 个线程将无法同步，导致算力彻底退化为串行。")
    print(f"      现代的 DeepSeekMoE、Triton Kernel，正是在这种“内存整理换算力”的思想上做到了极致。")

if __name__ == "__main__":
    benchmark()
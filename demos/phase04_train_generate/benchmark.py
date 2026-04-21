import torch
import time
import subprocess
import os

def run_cpp_benchmark():
    print("\n[0] 正在编译并运行 C++ 微型语言模型 (字符级 Tokenizer, 训练与生成全流程)...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        # 编译 demo-4
        subprocess.run(["make", "-C", root_dir, "-f", "demos/Makefile", "demo-4"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 运行 demo-4
        binary_path = os.path.join(root_dir, "demos/build/demo-4")
        result = subprocess.run([binary_path], capture_output=True, text=True, check=True)
        
        print("    --- demo-4 (C++ MicroLM 训练与生成) 运行日志 ---")
        lines = result.stdout.strip().split('\n')
        # 只打印前15行和后10行，防止输出过长
        for line in lines[:15]:
            print(f"    {line}")
        print("    ... (省略部分训练日志) ...")
        for line in lines[-10:]:
            print(f"    {line}")
        print("    ------------------------------------------------")
        print("    [解析] 上述 C++ 输出证明了我们手写的端到端模型 (SGD/AdamW 优化器、自回归生成) 是跑通的。")
        print("    在此基础上，我们接下来在 Python 端对比前沿推理系统 (如 vLLM) 解决大并发速度瓶颈的核心思想。\n")

    except Exception as e:
        print(f"    运行 C++ 代码失败 (请确认环境): {e}")

def naive_autoregressive_generation(seq_len, dim):
    """
    极度低效的生成：
    每次生成一个新 Token 时，把历史上所有的 Token 重新算一遍 Attention。
    时间复杂度: O(N^3)
    """
    # 假设目前已经生成了 i 个 token，正在算第 i+1 个
    q = torch.randn(1, 1, seq_len, dim)
    k = torch.randn(1, 1, seq_len, dim)
    v = torch.randn(1, 1, seq_len, dim)
    
    # 重新计算整个序列的分数
    scores = (q @ k.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out

def kv_cache_generation(seq_len, dim):
    """
    工业界标准的生成：
    使用 KV Cache，只需要算当前这 1 个 Token 与历史所有 KV 的 Attention。
    时间复杂度: O(N^2)
    """
    # 当前这 1 个新生成的 Token 的 Q
    q = torch.randn(1, 1, 1, dim)
    
    # 历史缓存的 K 和 V (长度为 seq_len)
    k_cache = torch.randn(1, 1, seq_len, dim)
    v_cache = torch.randn(1, 1, seq_len, dim)
    
    # 只计算 1 x N 的点积
    scores = (q @ k_cache.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v_cache
    return out

def benchmark():
    print("=== Phase 04: 文本生成 (解码) 性能对比基准测试 ===")
    
    # 0. 运行 C++ 基础测试
    run_cpp_benchmark()
    
    # 因为只是测算量，用 CPU 跑就能明显看出数量级差距
    device = 'cpu'
    dim = 128
    
    print("\n[对比测试] 假设我们正在长上下文中逐字生成回复...")
    
    # 1. 较短上下文
    seq_short = 512
    print(f"\n[测试 1] 当前上下文长度: {seq_short} Tokens")
    
    start = time.time()
    for _ in range(100):
        naive_autoregressive_generation(seq_short, dim)
    t_naive_short = time.time() - start
    
    start = time.time()
    for _ in range(100):
        kv_cache_generation(seq_short, dim)
    t_kv_short = time.time() - start
    
    print(f"    全量重算 Attention 耗时: {t_naive_short:.4f} 秒")
    print(f"    使用 KV Cache 增量解码: {t_kv_short:.4f} 秒")
    print(f"    -> KV Cache 快了 {t_naive_short/t_kv_short:.1f} 倍")

    # 2. 较长上下文 (长文本时代)
    seq_long = 8192
    print(f"\n[测试 2] 当前上下文长度: {seq_long} Tokens")
    
    start = time.time()
    for _ in range(10):
        naive_autoregressive_generation(seq_long, dim)
    t_naive_long = time.time() - start
    
    start = time.time()
    for _ in range(10):
        kv_cache_generation(seq_long, dim)
    t_kv_long = time.time() - start
    
    print(f"    全量重算 Attention 耗时: {t_naive_long:.4f} 秒")
    print(f"    使用 KV Cache 增量解码: {t_kv_long:.4f} 秒")
    print(f"    -> KV Cache 快了 {t_naive_long/t_kv_long:.1f} 倍")
    
    print("\n结论：")
    print("      如果不保存 KV Cache，生成越长的句子，每一步都越来越慢，耗时呈现平方级爆炸。")
    print("      这就是为什么推理引擎（如 vLLM）把所有精力都花在如何高效管理海量的 KV Cache 内存上，")
    print("      甚至借用操作系统的“分页机制”（PagedAttention）来消除内存碎片。")

if __name__ == "__main__":
    benchmark()
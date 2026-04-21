import torch
import time
import subprocess
import re
import os

def naive_python_gemm(A, B, M, N, K):
    """
    用纯 Python 模拟极度未经优化的 Naive 矩阵乘法
    (仅用极小的尺寸演示，否则会执行数小时)
    """
    C = torch.zeros((M, N), dtype=torch.float32)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C

def run_cpp_benchmark():
    print("\n[0] 正在编译并运行 C++ CUDA 原生实现 (作为性能锚点)...")
    # 确保我们在 demos/phase01_infra 目录下执行，或者在根目录执行
    # 这里通过往上两级调用 Makefile
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        # 编译 demo-1
        subprocess.run(["make", "-C", root_dir, "-f", "demos/Makefile", "demo-1"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 运行 demo-1
        demo1_path = os.path.join(root_dir, "demos/build/demo-1")
        result = subprocess.run([demo1_path], capture_output=True, text=True, check=True)
        
        print("    --- C++ CUDA 运行结果 ---")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
        print("    -------------------------")
        
        # 提取时间 (demo-1 写死 M=N=K=256)
        cpp_size = 256
        flops = 2 * (cpp_size ** 3)
        
        for line in result.stdout.split('\n'):
            if "GEMM naive" in line:
                m = re.search(r"time=([\d\.]+)\s*ms", line)
                if m:
                    ms_naive = float(m.group(1))
                    tflops_naive_cu = (flops / (ms_naive / 1000.0)) / 1e12
                    print(f"    [解析] C++ Naive CUDA 算力: {tflops_naive_cu:.6f} TFLOPS (规模 {cpp_size}x{cpp_size})")
            elif "GEMM tiled" in line:
                m = re.search(r"time=([\d\.]+)\s*ms", line)
                if m:
                    ms_tiled = float(m.group(1))
                    tflops_tiled_cu = (flops / (ms_tiled / 1000.0)) / 1e12
                    print(f"    [解析] C++ Tiled CUDA 算力: {tflops_tiled_cu:.6f} TFLOPS (规模 {cpp_size}x{cpp_size})")

    except Exception as e:
        print(f"    运行 C++ 代码失败 (请确认环境是否有 nvcc): {e}")

def benchmark():
    print("=== Phase 01: GEMM 性能对比基准测试 ===")
    
    # 0. 运行 C++ Benchmark
    run_cpp_benchmark()

    # 1. 纯 Python 循环测试（使用极小矩阵，因为太慢了）
    small_size = 128
    A_small = torch.randn(small_size, small_size)
    B_small = torch.randn(small_size, small_size)
    
    print(f"\n[1] 正在测试纯 Python Naive GEMM ({small_size}x{small_size})...")
    start = time.time()
    _ = naive_python_gemm(A_small, B_small, small_size, small_size, small_size)
    end = time.time()
    t_naive = end - start
    flops_naive = 2 * (small_size ** 3)
    tflops_naive = (flops_naive / t_naive) / 1e12
    print(f"    耗时: {t_naive:.4f} 秒, 算力: {tflops_naive:.6f} TFLOPS")

    # 2. PyTorch 底层 cuBLAS/CUTLASS 测试 (真实的工业级性能)
    # 使用足够大的矩阵来填满 GPU 流水线
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\n未检测到 GPU，使用 CPU 运行工业级底层测试 (如 MKL/OpenBLAS)")

    large_size = 4096
    A_large = torch.randn(large_size, large_size, device=device)
    B_large = torch.randn(large_size, large_size, device=device)
    
    # 预热 (Warm-up)
    for _ in range(5):
        _ = torch.matmul(A_large, B_large)
    if device == 'cuda':
        torch.cuda.synchronize()

    print(f"\n[2] 正在测试 PyTorch 工业级底层 GEMM ({large_size}x{large_size})...")
    start = time.time()
    iters = 20
    for _ in range(iters):
        _ = torch.matmul(A_large, B_large)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    t_fused = (end - start) / iters
    flops_large = 2 * (large_size ** 3)
    tflops_fused = (flops_large / t_fused) / 1e12
    
    print(f"    单次耗时: {t_fused:.5f} 秒, 算力: {tflops_fused:.2f} TFLOPS")
    
    print(f"\n结论：")
    print(f"  1. 纯 Python 循环极度受限，只有 {tflops_naive:.6f} TFLOPS。")
    print(f"  2. C++ Tiled CUDA 通过 Shared Memory 突破了内存墙，相比 C++ Naive CUDA 有质的飞跃。")
    print(f"  3. 工业级底层库（如 PyTorch 的 torch.matmul 调用的 cuBLAS/CUTLASS）通过汇编级优化，")
    print(f"     能把硬件算力压榨到 {tflops_fused:.2f} TFLOPS。")
    print(f"  因此，虽然我们手写 Tiled CUDA 能跑通，但在生产环境中往往是调用高层接口或使用 Triton/CUTLASS！")

if __name__ == "__main__":
    benchmark()
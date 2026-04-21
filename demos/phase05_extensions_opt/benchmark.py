import torch
import time
import subprocess
import os

def run_cpp_benchmark():
    print("\n[0] 正在编译并运行 C++ FP16 Naive GEMM 实现 (探究数值误差底线)...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        # 编译 demo-5
        subprocess.run(["make", "-C", root_dir, "-f", "demos/Makefile", "demo-5"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 运行 demo-5
        binary_path = os.path.join(root_dir, "demos/build/demo-5")
        result = subprocess.run([binary_path], capture_output=True, text=True, check=True)
        
        print("    --- demo-5 (C++ FP16 GEMM vs FP32 CPU) 运行结果 ---")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
        print("    ---------------------------------------------------")
        print("    [解析] 上述 C++ 输出了我们手写的 __half 矩阵乘法对比全精度 CPU 时的最大误差 (max_abs_err)。")
        print("    这证明了低精度的混合运算在数值上是可控的。接下来我们将用 PyTorch 揭示降低精度带来的硬件算力飞跃。\n")

    except Exception as e:
        print(f"    运行 C++ 代码失败 (请确认环境): {e}")

def benchmark():
    print("=== Phase 05: 混合精度与 Tensor Core 算力对比基准测试 ===")
    
    # 先运行 C++ FP16 误差验证
    run_cpp_benchmark()
    
    if not torch.cuda.is_available():
        print("警告: 只有在拥有 Tensor Core 的 NVIDIA GPU (Volta 架构及以上) 才能感受到差异。当前环境为 CPU，测试将跳过。")
        return

    device = 'cuda'
    print(f"\n[对比测试] 检测到 GPU: {torch.cuda.get_device_name(0)}")

    # 设定大矩阵规模，使得 GPU 能充分热身进入 Compute Bound
    size = 8192
    print(f"正在生成 {size}x{size} 的矩阵...")

    # 1. FP32 测试 (走 CUDA Core 标量乘加)
    A_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
    B_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # 强制禁用 PyTorch 在 Ampere 架构上偷偷把 FP32 转成 TF32 的优化，
    # 这样我们才能测出真正的纯 FP32 标量算力。
    torch.backends.cuda.matmul.allow_tf32 = False

    # 预热
    for _ in range(5):
        _ = torch.matmul(A_fp32, B_fp32)
    torch.cuda.synchronize()

    print("\n[1] 正在测试 FP32 (强制使用普通 CUDA Core 计算)...")
    start = time.time()
    iters = 10
    for _ in range(iters):
        _ = torch.matmul(A_fp32, B_fp32)
    torch.cuda.synchronize()
    t_fp32 = (time.time() - start) / iters

    flops = 2 * (size ** 3)
    tflops_fp32 = (flops / t_fp32) / 1e12
    print(f"    单次耗时: {t_fp32:.4f} 秒, 算力: {tflops_fp32:.2f} TFLOPS")

    # 2. FP16 测试 (将自动唤醒 Tensor Core)
    A_fp16 = A_fp32.to(torch.float16)
    B_fp16 = B_fp32.to(torch.float16)

    # 预热
    for _ in range(5):
        _ = torch.matmul(A_fp16, B_fp16)
    torch.cuda.synchronize()

    print("\n[2] 正在测试 FP16 (底层将触发 mma.sync 汇编指令唤醒 Tensor Core)...")
    start = time.time()
    for _ in range(iters):
        _ = torch.matmul(A_fp16, B_fp16)
    torch.cuda.synchronize()
    t_fp16 = (time.time() - start) / iters

    tflops_fp16 = (flops / t_fp16) / 1e12
    print(f"    单次耗时: {t_fp16:.4f} 秒, 算力: {tflops_fp16:.2f} TFLOPS")

    print("\n结论：")
    print(f"      在当前 GPU 上，FP16 (Tensor Core) 的计算速度是 FP32 (CUDA Core) 的 {tflops_fp16/tflops_fp32:.1f} 倍。")
    print("      除了计算速度的爆炸级提升，FP16 更是把显存容量需求和通信带宽直接砍半。")
    print("      DeepSeek 等顶尖模型在万卡集群上甚至把精度压到了极致的 FP8（理论算力再翻倍），")
    print("      辅以精妙的底层对齐工程和 PTX 汇编微操，这才是当代 AI 降本增效的终极引擎。")

if __name__ == "__main__":
    benchmark()
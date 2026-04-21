#include "cuda_check.cuh"

// naive kernel：每个线程独立算 C[row][col]，没有共享内存，K 次内积循环全部直接从 DRAM 读 A/B。
// 设计目的：先验证索引和边界公式正确、避免隐蔽的同步错误；速度慢但逻辑最透明。
__global__ void gemm_naive_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  // blockIdx / threadIdx：CUDA 运行时自动赋值的线程坐标，无需手动管理。
  // blockDim：每个 block 的维度，此处固定为 (16,16)。
  // row/col：二维线程到二维矩阵元素的直接映射，与 C[row][col] 的书写习惯一致。
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // 非整除网格（当 M/N 不是 16 的倍数时）会在边缘产生越界线程；不做边界判断会导致越界读写。
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// tiled kernel：把 A/B 的小块先从 DRAM 读到 GPU 片上的 __shared__ 缓冲区（等价于 GPU 上的 L1 cache），
// 同一 tile 内所有线程复用这一小块数据，大幅减少全局内存带宽使用——这是高性能 GEMM 的起点。
__global__ void gemm_tiled_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int TILE = 16;
  // __shared__：block 内 256 个线程共享的片上内存，延迟比全局显存低一个数量级。
  // As[row][k] 和 Bs[k][col] 组成一小块矩阵乘法的 A、B 片段。
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  // bx/by、tx/ty：CUDA 运行时传入的 block/thread 坐标，block 内线程用 (tx,ty) 协作读写共享内存。
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  // 当前线程负责的输出位置（row, col），对应 C[row][col]。
  int row = by * TILE + ty;
  int col = bx * TILE + tx;
  float acc = 0.f;

  // 在 K 维上按 tile 推进：ceil(K / TILE) 次循环覆盖完整的内积累加。
  // 最后一轮若 K 不是 TILE 的整数倍，越界位置填 0（不改变数学结果）。
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + tx;
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
    int b_row = t * TILE + ty;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

    // 第一次 __syncthreads()：等整个 block 所有线程把当前 tile 的 A/B 从 DRAM 读到共享内存完成，
    // 再开始计算——否则有人还在写、有人已经开始读，产生数据竞争。
    __syncthreads();

    // 在 shared memory 上做本 tile 的 K 维部分积：不再访问 DRAM，带宽压力转移到片上。
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];

    // 第二次 __syncthreads()：等所有线程用完当前 tile，再覆盖 As/Bs 进入下一轮——否则有人还在读，
    // 有人已经开始覆盖下一轮的数据。
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// dim3：CUDA 三维整数向量类型，描述线程块（block）和网格（grid）的维度。
// block(16,16)：每个 block 是 16×16 = 256 个线程的二维方阵，是 CUDA 中最常用的 tile 大小之一。
// grid((N+15)/16, (M+15)/16)：向上取整，保证覆盖完整输出矩阵；grid 总线程数 = grid.x × grid.y × 256。
void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_naive_ker<<<grid, block>>>(A, B, C, M, N, K);
  // cudaGetLastError()：检查刚才的 kernel launch 是否有非法配置（共享内存不足、参数越界等）。
  // 注意：launch 是异步的——此检查只验证「发射是否合法」，不等待 kernel 跑完。
  CUDA_CK(cudaGetLastError());
  // cudaDeviceSynchronize()：强制 CPU 等待 GPU 全部跑完。
  // 作用：① 保证计时只含 kernel 时间，不混入后续 CPU 操作；② 确保在继续前拿到实际执行结果。
  CUDA_CK(cudaDeviceSynchronize());
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_tiled_ker<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}

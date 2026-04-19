# Phase Demos Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `demos/` tree with a root `demos/Makefile` and five phase-scoped, self-contained CUDA/C++ programs (`demo-1` … `demo-5`) that compile **only** sources under `demos/`, never linking parent `core/`, `kernels/`, `layers/`, etc., matching `docs/superpowers/specs/2026-04-19-phase-demos-sandbox-design.md`.

**Architecture:** One orchestrating `demos/Makefile` declares explicit source lists per target. Shared helpers live in `demos/common/` (CPU reference GEMM, CUDA check macros, small numerics). Each phase directory owns its kernels and `main` entry; binaries emit human-readable metrics and exit non-zero on failure.

**Tech Stack:** `nvcc`, CUDA runtime, C++17, FP32 baseline; Phase 5 adds FP16 storage/compute using CUDA half types with explicit numerical checks.

**Spec:** `docs/superpowers/specs/2026-04-19-phase-demos-sandbox-design.md`

---

## File map (all paths relative to repository root)

| Path | Role |
|------|------|
| `demos/Makefile` | Variables + targets `demo-1` … `demo-5`, `all`, `clean` |
| `demos/build/` | Gitignored output directory for binaries (created by make) |
| `demos/common/cuda_check.cuh` | `CUDA_CK` macro |
| `demos/common/cpu_gemm.cpp` | Row-major FP32 CPU GEMM reference |
| `demos/common/cpu_gemm.hpp` | Declaration |
| `demos/common/numeric.hpp` | `max_abs_diff` for FP32 host buffers |
| `demos/common/timer.hpp` | Simple `std::chrono` wall timer (header-only) |
| `demos/common/gradcheck.hpp` | Central-difference vs analytic vector grad helper |
| `demos/phase01_infra/gemm.cu` | Naive + tiled GEMM kernels + host wrappers |
| `demos/phase01_infra/main.cu` | Alloc/memcpy, launch, compare to CPU |
| `demos/phase02_ops_autograd/sm.cu` | Row softmax CUDA vs CPU + CE gradcheck |
| `demos/phase02_ops_autograd/ln.cu` | LayerNorm CUDA vs CPU + scalar gradcheck |
| `demos/phase02_ops_autograd/attn.cu` | Tiny SDPA forward CUDA vs CPU + full-parameter gradcheck (host analytic backward) |
| `demos/phase03_model/model.cpp` | Tiny embedding + one transformer block + forward MSE on random batch |
| `demos/phase03_model/main.cpp` | Runs forward, prints weight stats (mean/var sanity) |
| `demos/phase04_train_generate/{tokenizer.hpp,tok.cpp,dataset.cpp,train.cpp,gen.cpp,main.cpp}` | Char tokenizer on synthetic or bundled tiny corpus file under `demos/phase04_train_generate/fixture.txt` |
| `demos/phase05_extensions_opt/fp16_gemm.cu` | Naive FP16 GEMM kernel + FP32 accumulation reference |
| `demos/phase05_extensions_opt/main.cu` | Compares vs FP32 CPU reference with relaxed threshold documented in source |

Add `demos/build/` to `.gitignore` in **this** plan via a task that only appends if missing (allowed: `.gitignore` is not in the spec’s “do not modify” list for *source* — spec forbids root `Makefile` and main tree sources; ignoring build artifacts is standard). If you prefer zero repo churn outside `demos/`, omit `.gitignore` edit and document “user ignores `demos/build/` locally”; **default in this plan:** append `demos/build/` to `.gitignore`.

---

### Task 1: Create `demos/Makefile` with explicit sources

**Files:**
- Create: `demos/Makefile`

- [ ] **Step 1: Add `demos/Makefile`**

```makefile
# Demos build — does not include parent core/, kernels/, etc.
NVCC := nvcc
NVCC_FLAGS := -std=c++17 -O3 --use_fast_math
CUDA_ARCH ?= -arch=sm_80
COMMON_INC := -Idemos/common

BUILD_DIR := demos/build
$(shell mkdir -p $(BUILD_DIR))

DEMO1 := $(BUILD_DIR)/demo-1
DEMO1_SRC := demos/phase01_infra/main.cu demos/phase01_infra/gemm.cu demos/common/cpu_gemm.cpp

DEMO2_SM := $(BUILD_DIR)/demo-2-softmax
DEMO2_SM_SRC := demos/phase02_ops_autograd/sm.cu

DEMO2_LN := $(BUILD_DIR)/demo-2-layernorm
DEMO2_LN_SRC := demos/phase02_ops_autograd/ln.cu

DEMO2_ATTN := $(BUILD_DIR)/demo-2-attention
DEMO2_ATTN_SRC := demos/phase02_ops_autograd/attn.cu

DEMO3 := $(BUILD_DIR)/demo-3
DEMO3_SRC := demos/phase03_model/main.cpp demos/phase03_model/model.cpp

DEMO4 := $(BUILD_DIR)/demo-4
DEMO4_SRC := \
	demos/phase04_train_generate/main.cpp \
	demos/phase04_train_generate/tok.cpp \
	demos/phase04_train_generate/dataset.cpp \
	demos/phase04_train_generate/train.cpp \
	demos/phase04_train_generate/gen.cpp

DEMO5 := $(BUILD_DIR)/demo-5
DEMO5_SRC := demos/phase05_extensions_opt/main.cu demos/phase05_extensions_opt/fp16_gemm.cu

.PHONY: all demo-1 demo-2 demo-2-softmax demo-2-layernorm demo-2-attention demo-3 demo-4 demo-5 clean

all: demo-1 demo-2 demo-3 demo-4 demo-5

demo-2: demo-2-softmax demo-2-layernorm demo-2-attention

demo-1: $(DEMO1_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO1) $(DEMO1_SRC)

demo-2-softmax: $(DEMO2_SM_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_SM) $(DEMO2_SM_SRC)

demo-2-layernorm: $(DEMO2_LN_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_LN) $(DEMO2_LN_SRC)

demo-2-attention: $(DEMO2_ATTN_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_ATTN) $(DEMO2_ATTN_SRC)

demo-3: $(DEMO3_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO3) $(DEMO3_SRC)

demo-4: $(DEMO4_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO4) $(DEMO4_SRC)

demo-5: $(DEMO5_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO5) $(DEMO5_SRC)

clean:
	rm -rf $(BUILD_DIR)
```

- [ ] **Step 2: Verify syntax**

Run: `make -n -f demos/Makefile demo-1`  
Expected: prints `nvcc` command without error.

- [ ] **Step 3: Commit**

```bash
git add demos/Makefile
git commit -m "chore(demos): add orchestrating Makefile for phase demos"
```

---

### Task 2: Common CUDA check + numerics

**Files:**
- Create: `demos/common/cuda_check.cuh`
- Create: `demos/common/numeric.hpp`
- Create: `demos/common/timer.hpp`
- Create: `demos/common/gradcheck.hpp`

- [ ] **Step 1: Add headers**

`demos/common/cuda_check.cuh`:

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
    std::exit(1);
  }
}
#define CUDA_CK(x) cuda_check((x), __FILE__, __LINE__)
```

`demos/common/numeric.hpp`:

```cpp
#pragma once
#include <cmath>
#include <cstddef>

inline float max_abs_diff_f32(const float* a, const float* b, std::size_t n) {
  float m = 0.f;
  for (std::size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}
```

`demos/common/timer.hpp`:

```cpp
#pragma once
#include <chrono>

struct WallTimer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0{clock::now()};
  double ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  }
};
```

`demos/common/gradcheck.hpp`:

```cpp
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

inline bool vector_gradcheck(
    const std::vector<float>& x0,
    const std::function<float(const std::vector<float>&)>& forward,
    const std::vector<float>& analytic_grad,
    float eps = 1e-3f,
    float tol = 5e-2f) {
  for (std::size_t i = 0; i < x0.size(); ++i) {
    std::vector<float> xp = x0, xm = x0;
    xp[i] += eps;
    xm[i] -= eps;
    float numeric = (forward(xp) - forward(xm)) / (2.f * eps);
    float ag = analytic_grad[i];
    float rel = std::fabs(numeric - ag) / (std::fabs(ag) + 1e-6f);
    if (rel > tol && std::fabs(numeric - ag) > 1e-2f) {
      std::printf("grad mismatch idx=%zu numeric=%f analytic=%f\n", i, numeric, ag);
      return false;
    }
  }
  return true;
}
```

- [ ] **Step 2: Commit**

```bash
git add demos/common/cuda_check.cuh demos/common/numeric.hpp demos/common/timer.hpp demos/common/gradcheck.hpp
git commit -m "chore(demos): add common CUDA and numerics helpers"
```

---

### Task 3: CPU GEMM reference

**Files:**
- Create: `demos/common/cpu_gemm.hpp`
- Create: `demos/common/cpu_gemm.cpp`

- [ ] **Step 1: Implement row-major GEMM `C[M,N] = A[M,K] * B[K,N]`**

`demos/common/cpu_gemm.hpp`:

```cpp
#pragma once
#include <cstddef>
void cpu_gemm_f32(std::size_t M, std::size_t N, std::size_t K,
                  const float* A, const float* B, float* C);
```

`demos/common/cpu_gemm.cpp`:

```cpp
#include "cpu_gemm.hpp"

void cpu_gemm_f32(std::size_t M, std::size_t N, std::size_t K,
                  const float* A, const float* B, float* C) {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.f;
      for (std::size_t k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add demos/common/cpu_gemm.hpp demos/common/cpu_gemm.cpp
git commit -m "feat(demos): add CPU FP32 GEMM reference"
```

---

### Task 4: Phase 1 — device GEMM + correctness

**Files:**
- Create: `demos/phase01_infra/gemm.cu`
- Create: `demos/phase01_infra/main.cu`

**Includes:** `COMMON_INC := -Idemos/common` (already in `demos/Makefile`). Use `#include "cuda_check.cuh"` etc.

- [ ] **Step 1: Create `demos/phase01_infra/gemm.cu` (complete file)**

```cpp
#include "cuda_check.cuh"

__global__ void gemm_naive_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

__global__ void gemm_tiled_ker(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int TILE = 16;
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE + ty;
  int col = bx * TILE + tx;
  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + tx;
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
    int b_row = t * TILE + ty;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_naive_ker<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_tiled_ker<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
}
```

- [ ] **Step 2: Create `demos/phase01_infra/main.cu` (complete file)**

```cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_gemm.hpp"
#include "cuda_check.cuh"
#include "numeric.hpp"
#include "timer.hpp"

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
  // Fast-math can loosen FP32 ULPs; use 1e-5 vs CPU ref (documented).
  constexpr float kTol = 1e-5f;
  constexpr int M = 256, N = 256, K = 256;

  std::vector<float> hA(static_cast<std::size_t>(M * K));
  std::vector<float> hB(static_cast<std::size_t>(K * N));
  std::vector<float> hRef(static_cast<std::size_t>(M * N));
  std::vector<float> hNaive(static_cast<std::size_t>(M * N));
  std::vector<float> hTile(static_cast<std::size_t>(M * N));

  for (int i = 0; i < M * K; ++i) hA[static_cast<std::size_t>(i)] = std::sin(float(i)) * 1e-3f;
  for (int i = 0; i < K * N; ++i) hB[static_cast<std::size_t>(i)] = std::cos(float(i)) * 1e-3f;
  cpu_gemm_f32(static_cast<std::size_t>(M), static_cast<std::size_t>(N), static_cast<std::size_t>(K),
               hA.data(), hB.data(), hRef.data());

  float *dA = nullptr, *dB = nullptr, *dCn = nullptr, *dCt = nullptr;
  CUDA_CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCn, hNaive.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&dCt, hTile.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  WallTimer t1;
  launch_gemm_naive(dA, dB, dCn, M, N, K);
  double ms_naive = t1.ms();
  CUDA_CK(cudaMemcpy(hNaive.data(), dCn, hNaive.size() * sizeof(float), cudaMemcpyDeviceToHost));

  WallTimer t2;
  launch_gemm_tiled(dA, dB, dCt, M, N, K);
  double ms_tile = t2.ms();
  CUDA_CK(cudaMemcpy(hTile.data(), dCt, hTile.size() * sizeof(float), cudaMemcpyDeviceToHost));

  float err_naive = max_abs_diff_f32(hRef.data(), hNaive.data(), hRef.size());
  float err_tile = max_abs_diff_f32(hRef.data(), hTile.data(), hRef.size());
  std::printf("GEMM naive vs CPU max_abs_err=%e time=%.3f ms\n", err_naive, ms_naive);
  std::printf("GEMM tiled vs CPU max_abs_err=%e time=%.3f ms\n", err_tile, ms_tile);

  CUDA_CK(cudaFree(dA));
  CUDA_CK(cudaFree(dB));
  CUDA_CK(cudaFree(dCn));
  CUDA_CK(cudaFree(dCt));

  if (err_naive > kTol || err_tile > kTol) return 1;
  return 0;
}
```

- [ ] **Step 3: Build and run**

Run:

```bash
make -C . -f demos/Makefile demo-1
./demos/build/demo-1
```

Expected: printed errors `< 1e-5`, timings printed, exit code `0`.

- [ ] **Step 4: Commit**

```bash
git add demos/phase01_infra/gemm.cu demos/phase01_infra/main.cu
git commit -m "feat(demos): phase1 GEMM naive/tile vs CPU reference"
```

---

### Task 5: Phase 2 — CUDA ops + gradcheck (three small binaries)

**Rationale:** One monolithic `demo-2` would be too large to embed twice; the Makefile target `demo-2` builds three executables: `demo-2-softmax`, `demo-2-layernorm`, `demo-2-attention`. Each file is self-contained and includes only `demos/common/*`.

**Files:**
- Create: `demos/phase02_ops_autograd/sm.cu`
- Create: `demos/phase02_ops_autograd/ln.cu`
- Create: `demos/phase02_ops_autograd/attn.cu`

- [ ] **Step 1: Add `demos/phase02_ops_autograd/sm.cu` (complete file)**

```cpp
#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

__global__ void softmax_rows_ker(const float* __restrict__ in, float* __restrict__ out, int rows,
                                 int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* x = in + r * cols;
  float* y = out + r * cols;
  float m = x[0];
  for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) {
    float v = expf(x[c] - m);
    y[c] = v;
    s += v;
  }
  float inv = 1.f / fmaxf(s, 1e-12f);
  for (int c = 0; c < cols; ++c) y[c] *= inv;
}

static void softmax_rows_cpu(const float* in, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* x = in + r * cols;
    float* y = out + r * cols;
    float m = x[0];
    for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) {
      float v = expf(x[c] - m);
      y[c] = v;
      s += v;
    }
    float inv = 1.f / fmaxf(s, 1e-12f);
    for (int c = 0; c < cols; ++c) y[c] *= inv;
  }
}

static float ce_loss(const std::vector<float>& logits, const std::vector<int>& labels, int rows,
                      int cols) {
  float L = 0.f;
  for (int r = 0; r < rows; ++r) {
    const float* x = logits.data() + r * cols;
    int y = labels[static_cast<std::size_t>(r)];
    float m = x[0];
    for (int c = 1; c < cols; ++c) m = fmaxf(m, x[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) s += expf(x[c] - m);
    float logz = logf(fmaxf(s, 1e-12f));
    L += -(x[y] - m - logz);
  }
  return L;
}

static void ce_grad(const float* logits, const int* labels, int rows, int cols, float* grad) {
  std::vector<float> prob(static_cast<std::size_t>(rows * cols));
  softmax_rows_cpu(logits, prob.data(), rows, cols);
  for (int r = 0; r < rows; ++r) {
    const float* p = prob.data() + r * cols;
    float* g = grad + r * cols;
    int y = labels[r];
    for (int c = 0; c < cols; ++c) g[c] = p[c] - (c == y ? 1.f : 0.f);
  }
}

int main() {
  const int rows = 4, cols = 8;
  std::vector<float> h_in(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < h_in.size(); ++i) h_in[i] = 0.02f * std::sin(float(i));
  std::vector<float> h_cpu(h_in.size()), h_gpu(h_in.size());

  softmax_rows_cpu(h_in.data(), h_cpu.data(), rows, cols);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&d_out, h_gpu.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  const int threads = 128;
  const int blocks = (rows + threads - 1) / threads;
  softmax_rows_ker<<<blocks, threads>>>(d_in, d_out, rows, cols);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(h_gpu.data(), d_out, h_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float ferr = max_abs_diff_f32(h_cpu.data(), h_gpu.data(), h_cpu.size());
  std::printf("SOFTMAX CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  std::vector<int> labels = {0, 1, 2, 3};
  auto forward_lambda = [&](const std::vector<float>& x) { return ce_loss(x, labels, rows, cols); };

  std::vector<float> g(static_cast<std::size_t>(rows * cols));
  ce_grad(h_in.data(), labels.data(), rows, cols, g.data());

  const bool ok = vector_gradcheck(h_in, forward_lambda, g);
  std::printf("SOFTMAX gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(d_in));
  CUDA_CK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

- [ ] **Step 2: Add `demos/phase02_ops_autograd/ln.cu` (complete file)**

```cpp
#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

__global__ void layernorm_fwd_ker(const float* __restrict__ in, float* __restrict__ out, int rows,
                                  int cols, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* x = in + r * cols;
  float* y = out + r * cols;
  float mu = 0.f;
  for (int i = 0; i < cols; ++i) mu += x[i];
  mu /= static_cast<float>(cols);
  float var = 0.f;
  for (int i = 0; i < cols; ++i) {
    float d = x[i] - mu;
    var += d * d;
  }
  var /= static_cast<float>(cols);
  const float inv_std = rsqrtf(var + eps);
  for (int i = 0; i < cols; ++i) y[i] = (x[i] - mu) * inv_std;
}

static void layernorm_fwd_cpu(const float* in, float* out, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* x = in + r * cols;
    float* y = out + r * cols;
    float mu = 0.f;
    for (int i = 0; i < cols; ++i) mu += x[i];
    mu /= static_cast<float>(cols);
    float var = 0.f;
    for (int i = 0; i < cols; ++i) {
      float d = x[i] - mu;
      var += d * d;
    }
    var /= static_cast<float>(cols);
    const float inv_std = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < cols; ++i) y[i] = (x[i] - mu) * inv_std;
  }
}

static float ln_loss_sum_sq(const std::vector<float>& x, int rows, int cols) {
  std::vector<float> y(static_cast<std::size_t>(rows * cols));
  layernorm_fwd_cpu(x.data(), y.data(), rows, cols, 1e-5f);
  float L = 0.f;
  for (float v : y) L += v * v;
  return L;
}

static void ln_grad_analytic(const float* x, int rows, int cols, float eps, float* grad) {
  std::vector<float> y(static_cast<std::size_t>(rows * cols));
  layernorm_fwd_cpu(x, y.data(), rows, cols, eps);

  std::vector<float> dy(y.size(), 0.f);
  for (std::size_t i = 0; i < y.size(); ++i) dy[i] = 2.f * y[i];

  for (int r = 0; r < rows; ++r) {
    const float* yr = y.data() + r * cols;
    const float* xr = x + r * cols;
    float* gr = grad + r * cols;
    const float* dyr = dy.data() + r * cols;

    float mean_dy = 0.f;
    for (int i = 0; i < cols; ++i) mean_dy += dyr[i];
    mean_dy /= static_cast<float>(cols);

    float mean_y_dy = 0.f;
    for (int i = 0; i < cols; ++i) mean_y_dy += yr[i] * dyr[i];
    mean_y_dy /= static_cast<float>(cols);

    float mu = 0.f;
    for (int i = 0; i < cols; ++i) mu += xr[i];
    mu /= static_cast<float>(cols);
    float var = 0.f;
    for (int i = 0; i < cols; ++i) {
      float d = xr[i] - mu;
      var += d * d;
    }
    var /= static_cast<float>(cols);
    const float sigma = std::sqrt(var + eps);

    for (int i = 0; i < cols; ++i) {
      gr[i] = (dyr[i] - mean_dy - yr[i] * mean_y_dy) / sigma;
    }
  }
}

int main() {
  const int rows = 8, cols = 4;
  const float eps = 1e-5f;
  std::vector<float> x(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * std::sin(float(i)) + 0.01f * float(i % 7);

  std::vector<float> y_cpu(x.size()), y_gpu(x.size());
  layernorm_fwd_cpu(x.data(), y_cpu.data(), rows, cols, eps);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CK(cudaMalloc(&d_in, x.size() * sizeof(float)));
  CUDA_CK(cudaMalloc(&d_out, y_gpu.size() * sizeof(float)));
  CUDA_CK(cudaMemcpy(d_in, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  layernorm_fwd_ker<<<rows, 1>>>(d_in, d_out, rows, cols, eps);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(y_gpu.data(), d_out, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float ferr = max_abs_diff_f32(y_cpu.data(), y_gpu.data(), y_cpu.size());
  std::printf("LAYERNORM CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  auto forward_lambda = [&](const std::vector<float>& x0) { return ln_loss_sum_sq(x0, rows, cols); };
  std::vector<float> g(x.size());
  ln_grad_analytic(x.data(), rows, cols, eps, g.data());
  const bool ok = vector_gradcheck(x, forward_lambda, g);
  std::printf("LAYERNORM gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(d_in));
  CUDA_CK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

- [ ] **Step 3: Add `demos/phase02_ops_autograd/attn.cu` (complete file)**

```cpp
#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_check.cuh"
#include "gradcheck.hpp"
#include "numeric.hpp"

static void sdpa_forward_cpu(const float Q[2], const float K[2], const float V[2], float O[2]) {
  float s[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) s[i][j] = Q[i] * K[j];

  float P[2][2];
  for (int i = 0; i < 2; ++i) {
    float m = s[i][0];
    for (int j = 1; j < 2; ++j) m = fmaxf(m, s[i][j]);
    float sum = 0.f;
    for (int j = 0; j < 2; ++j) {
      P[i][j] = expf(s[i][j] - m);
      sum += P[i][j];
    }
    const float inv = 1.f / fmaxf(sum, 1e-8f);
    for (int j = 0; j < 2; ++j) P[i][j] *= inv;
  }

  for (int i = 0; i < 2; ++i) {
    float acc = 0.f;
    for (int j = 0; j < 2; ++j) acc += P[i][j] * V[j];
    O[i] = acc;
  }
}

__global__ void sdpa_tiny_ker(const float* Q, const float* K, const float* V, float* O) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float q0 = Q[0], q1 = Q[1], k0 = K[0], k1 = K[1], v0 = V[0], v1 = V[1];
  const float s00 = q0 * k0, s01 = q0 * k1, s10 = q1 * k0, s11 = q1 * k1;
  auto softmax2 = [](float a, float b, float& p0, float& p1) {
    const float m = fmaxf(a, b);
    const float e0 = expf(a - m), e1 = expf(b - m);
    const float s = e0 + e1;
    p0 = e0 / s;
    p1 = e1 / s;
  };
  float p00, p01, p10, p11;
  softmax2(s00, s01, p00, p01);
  softmax2(s10, s11, p10, p11);
  O[0] = p00 * v0 + p01 * v1;
  O[1] = p10 * v0 + p11 * v1;
}

static void sdpa_backward_cpu(const float Q[2], const float K[2], const float V[2], const float dO[2],
                                float dQ[2], float dK[2], float dV[2]) {
  float s[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) s[i][j] = Q[i] * K[j];

  float P[2][2];
  for (int i = 0; i < 2; ++i) {
    float m = s[i][0];
    for (int j = 1; j < 2; ++j) m = fmaxf(m, s[i][j]);
    float sum = 0.f;
    for (int j = 0; j < 2; ++j) {
      P[i][j] = expf(s[i][j] - m);
      sum += P[i][j];
    }
    const float inv = 1.f / fmaxf(sum, 1e-8f);
    for (int j = 0; j < 2; ++j) P[i][j] *= inv;
  }

  float dP[2][2] = {};
  for (int j = 0; j < 2; ++j) {
    float acc = 0.f;
    for (int i = 0; i < 2; ++i) acc += dO[i] * P[i][j];
    dV[j] = acc;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) dP[i][j] = dO[i] * V[j];

  float dS[2][2] = {};
  for (int i = 0; i < 2; ++i) {
    float pg = 0.f;
    for (int j = 0; j < 2; ++j) pg += P[i][j] * dP[i][j];
    for (int k = 0; k < 2; ++k) dS[i][k] = P[i][k] * (dP[i][k] - pg);
  }

  for (int i = 0; i < 2; ++i) {
    dQ[i] = 0.f;
    for (int j = 0; j < 2; ++j) dQ[i] += dS[i][j] * K[j];
  }
  for (int j = 0; j < 2; ++j) {
    dK[j] = 0.f;
    for (int i = 0; i < 2; ++i) dK[j] += dS[i][j] * Q[i];
  }
}

static float sdpa_loss(const std::vector<float>& p) {
  float Q[2] = {p[0], p[1]};
  float K[2] = {p[2], p[3]};
  float V[2] = {p[4], p[5]};
  float O[2];
  sdpa_forward_cpu(Q, K, V, O);
  return O[0] * O[0] + O[1] * O[1];
}

static void sdpa_analytic_grad(const std::vector<float>& p, std::vector<float>& g) {
  float Q[2] = {p[0], p[1]};
  float K[2] = {p[2], p[3]};
  float V[2] = {p[4], p[5]};
  float O[2];
  sdpa_forward_cpu(Q, K, V, O);
  const float dO[2] = {2.f * O[0], 2.f * O[1]};
  g.resize(6);
  sdpa_backward_cpu(Q, K, V, dO, g.data(), g.data() + 2, g.data() + 4);
}

int main() {
  std::vector<float> h(6);
  h[0] = 0.3f;
  h[1] = -0.2f;
  h[2] = 0.15f;
  h[3] = 0.4f;
  h[4] = -0.1f;
  h[5] = 0.25f;

  float Q[2] = {h[0], h[1]};
  float K[2] = {h[2], h[3]};
  float V[2] = {h[4], h[5]};
  float O_cpu[2], O_gpu[2];
  sdpa_forward_cpu(Q, K, V, O_cpu);

  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  CUDA_CK(cudaMalloc(&dQ, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dK, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dV, 2 * sizeof(float)));
  CUDA_CK(cudaMalloc(&dO, 2 * sizeof(float)));
  CUDA_CK(cudaMemcpy(dQ, Q, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dK, K, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dV, V, 2 * sizeof(float), cudaMemcpyHostToDevice));
  sdpa_tiny_ker<<<1, 1>>>(dQ, dK, dV, dO);
  CUDA_CK(cudaGetLastError());
  CUDA_CK(cudaDeviceSynchronize());
  CUDA_CK(cudaMemcpy(O_gpu, dO, 2 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ocpu = {O_cpu[0], O_cpu[1]};
  std::vector<float> ogpu = {O_gpu[0], O_gpu[1]};
  const float ferr = max_abs_diff_f32(ocpu.data(), ogpu.data(), 2);
  std::printf("ATTENTION CUDA vs CPU max_abs_err=%e\n", ferr);
  if (ferr > 1e-4f) return 1;

  auto forward_lambda = [&](const std::vector<float>& p) { return sdpa_loss(p); };
  std::vector<float> g;
  sdpa_analytic_grad(h, g);
  const bool ok = vector_gradcheck(h, forward_lambda, g);
  std::printf("ATTENTION gradcheck: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CK(cudaFree(dQ));
  CUDA_CK(cudaFree(dK));
  CUDA_CK(cudaFree(dV));
  CUDA_CK(cudaFree(dO));
  return ok ? 0 : 1;
}
```

- [ ] **Step 4: Build and run**

```bash
make -f demos/Makefile demo-2-softmax && ./demos/build/demo-2-softmax
make -f demos/Makefile demo-2-layernorm && ./demos/build/demo-2-layernorm
make -f demos/Makefile demo-2-attention && ./demos/build/demo-2-attention
```

Expected: each binary prints a small `max_abs_err` line and a `PASS` gradcheck line; exit code `0`.

- [ ] **Step 5: Commit**

```bash
git add demos/phase02_ops_autograd/sm.cu demos/phase02_ops_autograd/ln.cu demos/phase02_ops_autograd/attn.cu
git commit -m "feat(demos): phase2 softmax, layernorm, attention checks"
```

### Task 6: Phase 3 — tiny model forward

**Files:**
- Create: `demos/phase03_model/model.cpp`
- Create: `demos/phase03_model/main.cpp`

- [ ] **Step 1: Implement a toy `TinyPoetryModel`**

Parameters:

- `V=64`, `d_model=32`, `n_heads=4`, `seq=16`, `layers=1`.

Components (host-side loops acceptable; optional CUDA matmul from Phase 1 reused by linking same `.cu` is **not allowed** unless copied into this directory — **copy** the minimal `launch_gemm_naive` into `demos/phase03_model/gemm_stub.cu` if needed, or use naive triple loops on CPU for simplicity).

Forward: random `token_ids` → embedding lookup → one pre-norm transformer block (self-attn + residual + FF + residual) simplified without dropout.

Print:

- `loss=<finite float>`
- `W_emb mean/var ...` within reasonable bounds.

Assert `std::isfinite(loss)`.

- [ ] **Step 2: Build/run/commit**

```bash
make -f demos/Makefile demo-3 && ./demos/build/demo-3
git add demos/phase03_model
git commit -m "feat(demos): phase3 tiny transformer forward"
```

---

### Task 7: Phase 4 — char tokenizer + train + generate

**Files:**
- Create: `demos/phase04_train_generate/fixture.txt` (10–30 lines Chinese or ASCII placeholder text committed as data)
- Create: `demos/phase04_train_generate/tokenizer.hpp`, `tok.cpp`
- Create: `demos/phase04_train_generate/dataset.cpp`, `dataset.hpp`
- Create: `demos/phase04_train_generate/train.cpp`, `train.hpp`
- Create: `demos/phase04_train_generate/gen.cpp`, `gen.hpp`
- Create: `demos/phase04_train_generate/main.cpp`

- [ ] **Step 1: Tokenizer**

Char-level `encode(std::string)->std::vector<int>`, `decode` inverse; `vocab` size = unique chars + small specials.

- [ ] **Step 2: Dataset**

Sliding windows of length `T=32`, batch `B=8`, random shuffle optional.

- [ ] **Step 3: Train**

Reuse `TinyPoetryModel` pattern from Phase 3 (copy code into this directory to keep self-contained) or a smaller `MicroLM` class inline.

Optimizer: SGD **and** AdamW each run for `50` steps on separate clones of weights (print two loss curves summary lines).

- [ ] **Step 4: Generate**

Greedy decode for `20` tokens; temperature sample for `20` tokens with `T=0.9`.

- [ ] **Step 5: Build/run/commit**

```bash
make -f demos/Makefile demo-4 && ./demos/build/demo-4
git add demos/phase04_train_generate
git commit -m "feat(demos): phase4 micro train and generate"
```

---

### Task 8: Phase 5 — FP16 GEMM demo

**Files:**
- Create: `demos/phase05_extensions_opt/fp16_gemm.cu`
- Create: `demos/phase05_extensions_opt/main.cu`

- [ ] **Step 1: Kernel**

Naive `__half` GEMM with accumulation in `float`, write `__half` results.

- [ ] **Step 2: Reference**

Convert `__half` inputs to float on host, use `cpu_gemm_f32`, compare after downcast with tolerance `5e-3` relative (document fast-math).

- [ ] **Step 3: Build/run/commit**

```bash
make -f demos/Makefile demo-5 && ./demos/build/demo-5
git add demos/phase05_extensions_opt
git commit -m "feat(demos): phase5 fp16 naive gemm check"
```

---

### Task 9: Ignore build artifacts (optional but recommended)

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append**

```
demos/build/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore && git commit -m "chore: ignore demos build output"
```

---

### Task 10: Document how to run (inside existing docs tree)

**Files:**
- Create: `demos/README.md`

- [ ] **Step 1: Add short instructions**

Contents must include:

```markdown
# Phase Demos

Build all: `make -f demos/Makefile all`

Run: `./demos/build/demo-1`, `./demos/build/demo-5`, and Phase 2 suite `./demos/build/demo-2-softmax`, `./demos/build/demo-2-layernorm`, `./demos/build/demo-2-attention` (or `make -f demos/Makefile demo-2`).

Override arch: `make -f demos/Makefile demo-1 CUDA_ARCH=-arch=sm_86`
```

- [ ] **Step 2: Commit**

```bash
git add demos/README.md && git commit -m "docs(demos): add runbook for phase demos"
```

---

## Plan self-review (completed)

1. **Spec coverage:** Hard boundary (no parent TU) enforced by Makefile source lists. Phases 1–5 mapped. CUDA error handling required in each `main`. Exit codes specified. `common/` scope constrained.
2. **Placeholders removed:** Phase 1 lists complete `gemm.cu` / `main.cu`. Phase 2 lists complete `sm.cu`, `ln.cu`, `attn.cu` (CUDA forward vs CPU reference + host analytic gradients for gradcheck). Phases 3–5 remain prose-sized but each task names concrete files and verification commands; implementers expand them using the same “complete file per step” rule before merging.
3. **Consistency:** Includes use `-Idemos/common` with `#include "cuda_check.cuh"` / `#include "gradcheck.hpp"` / `#include "numeric.hpp"` / `#include "cpu_gemm.hpp"` as shown.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-phase-demos-sandbox.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development  
2. **Inline Execution** — execute tasks in this session using executing-plans with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans  

Which approach?

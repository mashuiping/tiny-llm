# Phase 自包含 Demo 沙箱 — 设计说明

**日期：** 2026-04-19  
**状态：** 已评审（对话确认）

---

## 1. 目标

在**不修改**仓库根目录既有源码与根 `Makefile` 的前提下，于 `demos/` 下为各实现阶段提供**可独立编译、独立运行**的小程序，用于：

- 阶段能力验证（正确性、数值稳定、训练是否「能动」）；
- 博客 / 演示的固定「锚点」与复现实验入口。

各 demo 与主工程 **API 不必一致**；以 `demos/` 内实现为准，差异用注释标明即可。

---

## 2. 硬边界

| 项 | 要求 |
|----|------|
| 主工程源码 | **不修改** `core/`、`layers/`、`data/`、`kernels/`、`train/`、`generate/`、`include/`（根）、根 `main.cpp`、根 `Makefile` |
| Demo 代码位置 | 仅 `demos/` 下新增或调整 |
| 链接范围 | Demo 目标 **不得** 编译或链接上述主工程翻译单元；仅允许编译 `demos/**` 内源文件 |
| 可选共享 | 仅允许 `demos/common/` 内脚手架（见下文），**禁止**其演化为第二套完整框架 |

---

## 3. 组织方式（已定案）

采用 **统一 `demos/Makefile` + 分 phase 子目录 + 可选 `demos/common/`**：

- **统一入口**：例如 `make demo-1` … `make demo-5`，避免各子目录重复维护 `NVCC_FLAGS`、`CUDA_ARCH`。
- **`demos/common/`**：仅 CPU 参考 GEMM、计时、简单随机填充、`max_abs_diff` 等；不得依赖主工程头文件或实现。
- **不推荐**：每 phase 各自一份完整 Makefile 且 flags 多处复制（维护成本高）；若个别 phase 需特殊 flags，在根 `demos/Makefile` 用目标级变量覆盖即可。

---

## 4. 建议目录布局

```text
demos/
  Makefile
  common/                    # 仅 demo 使用的小工具
  phase01_infra/
  phase02_ops_autograd/
  phase03_model/
  phase04_train_generate/
  phase05_extensions_opt/
```

目录名可按需微调；语义与下表「阶段映射」一致即可。

---

## 5. 阶段与「可运行」含义

与主路线图对齐；每个 phase 至少一个可执行目标，成功 **exit 0**，失败 **非 0** 并打印可读指标。

| Phase | 目录建议 | 证明内容 | 典型输出 |
|-------|-----------|-----------|----------|
| 1 基础设施 | `phase01_infra/` | CUDA 分配/释放/Host–Device 拷贝；naive / tiling GEMM；与 CPU 参考对比，FP32 `max_abs_err < 1e-6`（规模在注释中说明） | 规模、误差、耗时 |
| 2 核心计算 + autograd | `phase02_ops_autograd/` | Softmax、LayerNorm、Attention 等 forward；迷你 autograd；各 op **梯度检查**（数值梯度 vs 解析梯度，容差在源码常量中写明） | 每 op PASS/FAIL |
| 3 模型 | `phase03_model/` | 迷你 Embedding / Transformer block / 小 PoetryModel；可选 Xavier/Kaiming 统计检验 | loss 有限、统计量在合理区间 |
| 4 训练 + 生成 | `phase04_train_generate/` | 小语料或合成数据；字符级 tokenizer；数步训练（SGD/AdamW 任选其一或两者小目标）；greedy / temperature 采样 | 样本文本、loss 曲线打印 |
| 5 扩展优化 | `phase05_extensions_opt/` | FP16 GEMM、或 10–50M 量级以外的**缩小版**对比实验（仍以 demo 自包含为准） | 误差或吞吐对比 |

Phase 5 与「BPE、博客」等：BPE/博客**不强制**在本沙箱内实现；若加 demo，仍须满足「仅 `demos/`、不链主工程」。

---

## 6. 构建与运行约定

- **编译器**：与主工程一致使用 `nvcc`，C++ 标准建议 `c++17`；`CUDA_ARCH` 等与主 `Makefile` **风格对齐**，变量集中在 `demos/Makefile`。
- **每个目标**：显式列出参与编译的 `demos/**/*.cu`、`.cpp`，避免隐式拉入主目录。
- **依赖**：若需数学库，仅在 demo 范围内声明；**不**要求与根目标 `-lcublas` 等保持一致（主工程手写 GEMM 与 demo 可各自独立）。
- **退出码**：校验失败必须非 0，便于脚本化 CI（将来若在 `demos/` 外加 CI，不在本 spec 强制范围）。

---

## 7. 错误处理与数值策略

- **CUDA**：`cudaMalloc` / `cudaMemcpy` / kernel launch 后检查；失败打印 `cudaGetErrorString` 并 `return 1`。
- **数值比较**：统一通过 `demos/common/` 或各 `main` 顶部的阈值常量；大输入或大算子在注释中说明放宽理由。
- **梯度检查**：相对误差容差写死在各 demo 中；文档化「敏感 op 可单独放宽」。

---

## 8. 与主工程的关系

- **主工程**：完整产品与基准实现。
- **`demos/`**：教学与阶段切片；刻意允许与主工程 **分叉**，避免 demo 被主工程重构牵制。若未来要对齐 API，单独立项，不在本沙箱范围。

---

## 9. 实现顺序建议

按 Phase 1 → 5 递增；每完成一个 `make demo-N` 可在博客对应章节引用该命令与预期输出片段。

---

## 10. 非目标（明确排除）

- 不要求 demo 与主工程二进制行为逐字节一致。
- 不要求在本阶段修改根 `Makefile` 或把 demo 并入主 `all` 目标。
- 不在此 spec 中规定具体矩阵规模、网络宽度（由各 demo 源码常量给出即可）。

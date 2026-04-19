# Phase demos (self-contained)

These programs live under `demos/` and are **not** linked against the main project’s `core/`, `kernels/`, etc.

## 学习资料（每个 phase 一份）

各子目录 `LEARN.md`：说明、图、链接、自检、自测题（附答案）。

- `phase01_infra/LEARN.md` — GEMM / CUDA 内存与分块
- `phase02_ops_autograd/LEARN.md` — Softmax、LayerNorm、Attention、gradcheck
- `phase03_model/LEARN.md` — 小模型与初始化直觉
- `phase04_train_generate/LEARN.md` — tokenizer、训练、生成
- `phase05_extensions_opt/LEARN.md` — FP16 与数值权衡

## Build

From the **repository root**:

```bash
make -f demos/Makefile all
```

Or from **inside `demos/`** (paths in the Makefile are anchored to that file):

```bash
cd demos && make demo-1
```

Override GPU architecture:

```bash
make -f demos/Makefile demo-1 CUDA_ARCH=-arch=sm_86
```

## Run

After a successful build, binaries are written to `demos/build/`:

- `./demos/build/demo-1` — GEMM naive/tiled vs CPU reference
- `./demos/build/demo-2-softmax`, `./demos/build/demo-2-layernorm`, `./demos/build/demo-2-attention` — Phase 2 checks (or `make -f demos/Makefile demo-2` to build all three)
- `./demos/build/demo-3` — tiny model forward sanity
- `./demos/build/demo-4` — micro training + greedy / temperature generation (loads `fixture.txt` from repo root, `demos/`, or `demos/build/` cwd)
- `./demos/build/demo-5` — FP16 naive GEMM vs FP32 CPU reference

## Clean

```bash
make -f demos/Makefile clean
```

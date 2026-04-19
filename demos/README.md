# Phase demos (self-contained)

These programs live under `demos/` and are **not** linked against the main project’s `core/`, `kernels/`, etc.

## Build

From the repository root:

```bash
make -f demos/Makefile all
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
- `./demos/build/demo-4` — micro training + greedy / temperature generation (expects CWD at repo root for the bundled `fixture.txt` path)
- `./demos/build/demo-5` — FP16 naive GEMM vs FP32 CPU reference

## Clean

```bash
make -f demos/Makefile clean
```

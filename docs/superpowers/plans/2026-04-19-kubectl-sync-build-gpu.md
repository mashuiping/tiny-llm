# kubectl 同步到 build-gpu Pod 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将本地仓库子集同步到集群中 `build-gpu` Pod 的 `$REMOTE_ROOT/$REMOTE_DIR`（默认 `/workspace/<basename>`），使用 `git ls-files` + `.syncignore`、`kubectl exec` + `tar` 流，并完成 `k8s/`、`tools/k8s/` 目录规范化。

**Architecture:** 以 `tools/k8s/sync-to-build-gpu.sh` 为唯一入口：解析配置合并顺序（默认 → `k8s.env` → 环境变量 → CLI），在仓库根生成归档字节流，经 `kubectl exec -i … tar -x` 解压；同步前 `kubectl exec … mkdir -p` 创建远端目录。清单 `k8s/build-gpu.pod.yaml` 由根目录迁入，脚本不解析 YAML。

**Tech Stack:** Bash 3.2+、`git`、`kubectl`、`tar`（macOS 自带 BSD tar；若 `--exclude-from` 与 `-T` 组合行为异常，改用文档注明的 `gtar` 或调整调用方式）。

**Spec:** `docs/superpowers/specs/2026-04-19-kubectl-sync-build-gpu-design.md`

---

## 文件映射（落地前总览）

| 文件 | 动作 | 职责 |
|------|------|------|
| `k8s/build-gpu.pod.yaml` | 新建（自根目录 `build-pod.yaml` 复制） | Pod 清单单一来源 |
| `build-pod.yaml`（根目录） | 删除 | 避免重复；由 `k8s/` 替代 |
| `tools/k8s/k8s.env.example` | 新建 | 可提交的变量模板与注释 |
| `tools/k8s/sync-to-build-gpu.sh` | 新建 | 可执行同步脚本 |
| `.syncignore` | 新建 | Pod 同步额外排除（如 `docs/`） |
| `.gitignore` | 已有 `tools/k8s/k8s.env` | 无需改除非路径变更 |
| `Makefile` | 修改 | 增加可选 `sync` 目标调用脚本 |

---

### Task 1: 迁入 Pod 清单到 `k8s/`

**Files:**
- Create: `k8s/build-gpu.pod.yaml`
- Delete: `build-pod.yaml`

- [ ] **Step 1: 复制清单**

将当前根目录 `build-pod.yaml` 全文复制为 `k8s/build-gpu.pod.yaml`（内容与现有一致，含 `metadata.name: build-gpu`、`namespace: tai-production`、容器名 `build`）。

- [ ] **Step 2: 删除旧文件**

删除仓库根的 `build-pod.yaml`，避免两处漂移。

- [ ] **Step 3: 提交**

```bash
git add k8s/build-gpu.pod.yaml
git rm build-pod.yaml 2>/dev/null || git add -u build-pod.yaml
git commit -m "chore(k8s): move build pod manifest under k8s/"
```

---

### Task 2: 添加 `tools/k8s/k8s.env.example`

**Files:**
- Create: `tools/k8s/k8s.env.example`

- [ ] **Step 1: 写入模板**

内容示例（实现时可微调空格，语义保持一致）：

```bash
# Copy to tools/k8s/k8s.env and adjust. Values must match k8s/build-gpu.pod.yaml (name/namespace/container).
# tools/k8s/k8s.env is gitignored.

# Optional: leave empty to use current kubectl context
# KUBE_CONTEXT=

NAMESPACE=tai-production
POD_NAME=build-gpu
CONTAINER_NAME=build

REMOTE_ROOT=/workspace
# Remote subdirectory under REMOTE_ROOT; default in script is repo basename if unset
# REMOTE_DIR=tiny-llm
```

- [ ] **Step 2: 提交**

```bash
git add tools/k8s/k8s.env.example
git commit -m "chore(tools): add k8s.env example for pod sync"
```

---

### Task 3: 添加根目录 `.syncignore`

**Files:**
- Create: `.syncignore`

- [ ] **Step 1: 写入初始规则**

在实现阶段按仓库实际内容填写；首版建议至少包含（与 spec「子集同步」一致）：

```
# Extra excludes for Pod sync (in addition to git ls-files --exclude-standard)
docs/
.git/
```

若 `data/` 下有大文件且不应上传，可追加 `data/` 或更细规则；保持模式简单（目录级）。

- [ ] **Step 2: 提交**

```bash
git add .syncignore
git commit -m "chore: add .syncignore for kubectl sync to build pod"
```

---

### Task 4: 实现 `tools/k8s/sync-to-build-gpu.sh`

**Files:**
- Create: `tools/k8s/sync-to-build-gpu.sh`

**约定：**

- `set -euo pipefail`
- 仓库根：`SRC_ROOT=$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel 2>/dev/null)` —— 更稳：`SCRIPT_DIR` → `REPO_ROOT=$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)`；脚本位于 `tools/k8s/`，向上两级为仓库根。
- 配置合并（每个变量独立）：**CLI > 环境变量 > `tools/k8s/k8s.env` > 内置默认**（与已批准 spec「k8s.env → 环境 → CLI」对齐：实现时用「默认值 → 读文件 → 读环境 → 解析 CLI 覆盖」等价效果）。
- `kubectl` 基础参数数组：`KUBECTL=(kubectl)`，若 `KUBECONFIG` 或 `--kubeconfig` 有值则追加；若 `context` 非空则 `--context`。
- 同步前：`"${KUBECTL[@]}" exec -n "$NAMESPACE" "$POD_NAME" -c "$CONTAINER_NAME" -- mkdir -p "$REMOTE_DEST"`
- 归档：在 `REPO_ROOT` 执行，`git ls-files -co --exclude-standard` 生成路径列表；若存在 `.syncignore`，`tar` 使用 `--exclude-from="$REPO_ROOT/.syncignore"`（路径写仓库根绝对路径或先 `cd "$REPO_ROOT"` 后用相对 `.syncignore`）。
- 流传输：`( cd "$REPO_ROOT" && tar ... ) | "${KUBECTL[@]}" exec -i -n "$NAMESPACE" "$POD_NAME" -c "$CONTAINER_NAME" -- tar -xzf - -C "$REMOTE_DEST"`

**BSD tar（macOS）注意：** 使用 `git ls-files` 换行分隔路径 + `tar -czf - -T <(git ls-files ...)` 或 `git ls-files ... | tar -czf - -T -`（确认本机 `tar -T -` 是否支持从 stdin 读列表；若不支持，改用临时文件列表 `/tmp/sync-files-$$` 并在脚本退出时 `rm -f`）。若 `-T -` 不可靠，**必须**用临时文件列清单（计划允许）。

- [ ] **Step 1: 创建脚本并 `chmod +x`**

将下列脚本一次性写入 `tools/k8s/sync-to-build-gpu.sh`。配置合并顺序：**默认值 → `k8s.env` 逐行解析 → 若用户在调用前已 `export` 同名变量则覆盖文件 → `parse_args` 中 CLI 最后覆盖**。脚本开头用 `SAVED_*` + `[[ "${VAR+x}" == x ]]` 记录「父环境是否传入该变量」，在应用文件后再恢复 `SAVED_*`，以实现「环境 > 文件 > 默认」。

```bash
#!/usr/bin/env bash
# Sync local repo subset to build-gpu Pod. See docs/superpowers/specs/2026-04-19-kubectl-sync-build-gpu-design.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/k8s.env"

# Remember exported env from parent (before defaults), for "env > k8s.env"
HAS_ENV_NAMESPACE=0 HAS_ENV_POD=0 HAS_ENV_CONTAINER=0 HAS_ENV_REMOTE_ROOT=0 HAS_ENV_REMOTE_DIR=0 HAS_ENV_KUBE_CONTEXT=0
[[ "${NAMESPACE+x}" == x ]] && { HAS_ENV_NAMESPACE=1; SAVED_ENV_NAMESPACE="$NAMESPACE"; }
[[ "${POD_NAME+x}" == x ]] && { HAS_ENV_POD=1; SAVED_ENV_POD_NAME="$POD_NAME"; }
[[ "${CONTAINER_NAME+x}" == x ]] && { HAS_ENV_CONTAINER=1; SAVED_ENV_CONTAINER_NAME="$CONTAINER_NAME"; }
[[ "${REMOTE_ROOT+x}" == x ]] && { HAS_ENV_REMOTE_ROOT=1; SAVED_ENV_REMOTE_ROOT="$REMOTE_ROOT"; }
[[ "${REMOTE_DIR+x}" == x ]] && { HAS_ENV_REMOTE_DIR=1; SAVED_ENV_REMOTE_DIR="$REMOTE_DIR"; }
[[ "${KUBE_CONTEXT+x}" == x ]] && { HAS_ENV_KUBE_CONTEXT=1; SAVED_ENV_KUBE_CONTEXT="$KUBE_CONTEXT"; }

NAMESPACE_DEFAULT="tai-production"
POD_NAME_DEFAULT="build-gpu"
CONTAINER_NAME_DEFAULT="build"
REMOTE_ROOT_DEFAULT="/workspace"
REMOTE_DIR_DEFAULT="$(basename "$REPO_ROOT")"

NAMESPACE="$NAMESPACE_DEFAULT"
POD_NAME="$POD_NAME_DEFAULT"
CONTAINER_NAME="$CONTAINER_NAME_DEFAULT"
REMOTE_ROOT="$REMOTE_ROOT_DEFAULT"
REMOTE_DIR="$REMOTE_DIR_DEFAULT"

load_env_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      local name="${BASH_REMATCH[1]}" val="${BASH_REMATCH[2]}"
      val="${val%\"}"; val="${val#\"}"
      val="${val%\'}"; val="${val#\'}"
      case "$name" in
        NAMESPACE) NAMESPACE="$val" ;;
        POD_NAME) POD_NAME="$val" ;;
        CONTAINER_NAME) CONTAINER_NAME="$val" ;;
        REMOTE_ROOT) REMOTE_ROOT="$val" ;;
        REMOTE_DIR) REMOTE_DIR="$val" ;;
        KUBE_CONTEXT) KUBE_CONTEXT="$val" ;;
      esac
    fi
  done <"$f"
}

apply_saved_env_over_file() {
  [[ "$HAS_ENV_NAMESPACE" -eq 1 ]] && NAMESPACE="$SAVED_ENV_NAMESPACE"
  [[ "$HAS_ENV_POD" -eq 1 ]] && POD_NAME="$SAVED_ENV_POD_NAME"
  [[ "$HAS_ENV_CONTAINER" -eq 1 ]] && CONTAINER_NAME="$SAVED_ENV_CONTAINER_NAME"
  [[ "$HAS_ENV_REMOTE_ROOT" -eq 1 ]] && REMOTE_ROOT="$SAVED_ENV_REMOTE_ROOT"
  [[ "$HAS_ENV_REMOTE_DIR" -eq 1 ]] && REMOTE_DIR="$SAVED_ENV_REMOTE_DIR"
  [[ "$HAS_ENV_KUBE_CONTEXT" -eq 1 ]] && KUBE_CONTEXT="$SAVED_ENV_KUBE_CONTEXT"
}

KUBECONFIG_ARG=""
CONTEXT_ARG=""
SRC_ROOT="$REPO_ROOT"
DRY_RUN=0
DELETE_REMOTE_EXTRA=0

usage() {
  cat <<'EOF'
Usage: sync-to-build-gpu.sh [options]

Sync tracked/untracked non-ignored files (git ls-files -co --exclude-standard),
minus patterns in .syncignore (if present), to Pod via tar stream.

Options:
  --kubeconfig PATH   Pass to kubectl
  --context NAME      kubectl context (overrides KUBE_CONTEXT)
  --namespace NS      Kubernetes namespace
  --pod NAME          Pod name
  --container NAME    Container name
  --src PATH          Local repo root (default: two levels above this script)
  --remote-root PATH  Remote parent (default: /workspace)
  --remote-dir NAME   Directory under remote root (default: repo basename)
  --dry-run           Print remote path and file count; no kubectl/tar
  --delete            Not implemented in v1 (exits with error)
  -h, --help          This help

Environment: NAMESPACE, POD_NAME, CONTAINER_NAME, REMOTE_ROOT, REMOTE_DIR, KUBE_CONTEXT
Config file: tools/k8s/k8s.env (optional, same variable names)

Precedence: defaults < k8s.env < environment < CLI
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --kubeconfig) KUBECONFIG_ARG="$2"; shift 2 ;;
      --context) CONTEXT_ARG="$2"; shift 2 ;;
      --namespace) NAMESPACE="$2"; shift 2 ;;
      --pod) POD_NAME="$2"; shift 2 ;;
      --container) CONTAINER_NAME="$2"; shift 2 ;;
      --src) SRC_ROOT="$2"; shift 2 ;;
      --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
      --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      --delete) DELETE_REMOTE_EXTRA=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
  done
}

build_kubectl_base() {
  KUBECTL=(kubectl)
  if [[ -n "${KUBECONFIG_ARG}" ]]; then
    KUBECTL+=(--kubeconfig="$KUBECONFIG_ARG")
  fi
  if [[ -n "${CONTEXT_ARG}" ]]; then
    KUBECTL+=(--context="$CONTEXT_ARG")
  elif [[ -n "${KUBE_CONTEXT:-}" ]]; then
    KUBECTL+=(--context="$KUBE_CONTEXT")
  fi
}

main() {
  load_env_file "$ENV_FILE"
  apply_saved_env_over_file
  parse_args "$@"

  REMOTE_DEST="${REMOTE_ROOT%/}/${REMOTE_DIR}"
  LIST_FILE="$(mktemp)"
  trap 'rm -f "$LIST_FILE"' EXIT

  if [[ "$DELETE_REMOTE_EXTRA" -eq 1 ]]; then
    echo "error: --delete is not implemented in v1; refusing." >&2
    exit 2
  fi

  (cd "$SRC_ROOT" && git rev-parse --show-toplevel >/dev/null)

  (cd "$SRC_ROOT" && git ls-files -co --exclude-standard >"$LIST_FILE")
  if ! [[ -s "$LIST_FILE" ]]; then
    echo "error: no files to sync (git ls-files empty)." >&2
    exit 1
  fi
  FILE_COUNT="$(wc -l <"$LIST_FILE" | tr -d ' ')"

  TAR_EXCLUDE=()
  if [[ -f "$SRC_ROOT/.syncignore" ]]; then
    TAR_EXCLUDE=(--exclude-from="$SRC_ROOT/.syncignore")
  fi

  echo "Remote: ${NAMESPACE}/${POD_NAME} (container ${CONTAINER_NAME}) -> ${REMOTE_DEST}"
  echo "Files (git ls-files -co --exclude-standard): ${FILE_COUNT}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run: skipping kubectl and tar stream"
    exit 0
  fi

  build_kubectl_base

  "${KUBECTL[@]}" exec -n "$NAMESPACE" "$POD_NAME" -c "$CONTAINER_NAME" -- \
    mkdir -p "$REMOTE_DEST"

  # BSD/GNU: --exclude-from before -T; newline-separated paths in LIST_FILE
  (cd "$SRC_ROOT" && tar -czf - "${TAR_EXCLUDE[@]}" -T "$LIST_FILE") | \
    "${KUBECTL[@]}" exec -i -n "$NAMESPACE" "$POD_NAME" -c "$CONTAINER_NAME" -- \
      tar -xzf - -C "$REMOTE_DEST"

  echo "Done."
}

main "$@"
```

- [ ] **Step 2: `tar` 与空列表**

在 macOS 上执行 `tar --version`；若 `tar -czf - -T list` 报错，将 `-T list` 与 `--exclude-from` 顺序对调试验。若 `LIST_FILE` 为空已在 Step 1 中 `exit 1`。

- [ ] **Step 3: `chmod +x`**

```bash
chmod +x tools/k8s/sync-to-build-gpu.sh
```

- [ ] **Step 4: 提交**

```bash
git add tools/k8s/sync-to-build-gpu.sh
git commit -m "feat(tools): add kubectl tar sync script for build-gpu pod"
```

---

### Task 5: Makefile 增加 `sync` 目标

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: 增加目标**

在 `.PHONY` 行加入 `sync`，并追加：

```makefile
.PHONY: all train generate clean sync

sync:
	./tools/k8s/sync-to-build-gpu.sh
```

- [ ] **Step 2: 提交**

```bash
git add Makefile
git commit -m "build: add make sync delegating to kubectl sync script"
```

---

### Task 6: 手工验证（必做）

**Files:** 无（仅命令）

- [ ] **Step 1: 干跑**

```bash
./tools/k8s/sync-to-build-gpu.sh --dry-run
```

预期：打印 `Remote:` 行与文件数，退出码 0，无 `kubectl` 调用。

- [ ] **Step 2: 集群实测（有 kube 访问时）**

复制 `tools/k8s/k8s.env.example` → `tools/k8s/k8s.env`，填正确 context/namespace。确保 Pod Running。执行：

```bash
./tools/k8s/sync-to-build-gpu.sh
```

在 Pod 内：

```bash
kubectl exec -n tai-production build-gpu -c build -- ls -la /workspace
```

预期：存在 `REMOTE_DIR` 目录且含源码树。

- [ ] **Step 3: 远端目录不存在**

删除 Pod 内 `$REMOTE_DEST` 后重跑脚本；预期：因 `mkdir -p` 成功重建。

---

## Spec 对照自检

| Spec 条款 | 对应任务 |
|-----------|----------|
| `k8s/build-gpu.pod.yaml` 迁入 | Task 1 |
| `tools/k8s/` 脚本与 `k8s.env.example` | Task 2、4 |
| `.syncignore` + `git ls-files` | Task 3、4 |
| `mkdir -p` 远端 | Task 4 |
| tar 流 + `kubectl exec` | Task 4 |
| 配置优先级与 CLI 覆盖 | Task 4（`save env → defaults → k8s.env → restore env → parse_args`） |
| `--dry-run` | Task 4 |
| `--delete` 默认关闭；首版可拒绝 | Task 4 |
| Makefile 可选 `sync` | Task 5 |
| 手工验证 | Task 6 |

---

## 执行交接

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-kubectl-sync-build-gpu.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints  

**Which approach?**

（若在本会话直接实现：从 Task 1 顺序执行到 Task 6；Task 4 中配置合并与 `tar` 行为需在 macOS 上实机验证。）

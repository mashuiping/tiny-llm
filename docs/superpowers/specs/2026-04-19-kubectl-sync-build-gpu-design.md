# kubectl 同步本地代码到 build-gpu Pod（`/workspace`）— 设计说明

## 背景与目标

在 GPU 构建 Pod（`build-gpu`）内编译与运行本仓库时，需要一条可靠的、可重复的命令：**将本地工作区子集同步到 Pod 内固定目录**（默认在 `/workspace` 下），并与现有 `build-pod` 清单、团队 kube 习惯对齐。

**目标：**

- 一键脚本（或 Makefile 薄封装）完成同步；默认排除已由 Git 忽略的文件，并可通过 `.syncignore` 额外排除「仍被 Git 跟踪但不应进 Pod」的路径（如大文档树）。
- 配置可本地私有化（`k8s.env`），同时支持环境变量与 CLI 覆盖，便于 CI 与临时换集群。
- 仓库目录结构清晰：`k8s/` 放清单，`tools/k8s/` 放脚本与配置模板。

**非目标（首版不做）：**

- 从 Pod 拉回本地、双向同步。
- 用 `yq`/`kubectl` 从 YAML 自动解析 Pod 名（可列为后续增强）；首版以 **人工保持 `k8s.env` 与清单一致** 为准。
- Pod 内编译命令、远程 shell 封装（仅定义同步）。

## 若远端不存在 `/workspace` 会怎样？

`kubectl cp` 与向 Pod 解压归档时，**目标父目录必须已存在**；若仅写 `pod:/workspace/...` 而 `/workspace` 未创建，通常会失败（例如 “No such file or directory”）。因此流程中必须包含：**`kubectl exec … -- mkdir -p <远端父目录>`**（以及按需创建项目子目录），**不**依赖镜像是否自带 `/workspace`。

## 仓库布局（规范化）

| 路径 | 职责 |
|------|------|
| `k8s/build-gpu.pod.yaml` | 由根目录 `build-pod.yaml` **迁入**；内容与现有一致，仅路径变更。 |
| `tools/k8s/sync-to-build-gpu.sh` | 同步入口脚本（`bash`，`set -euo pipefail`）。 |
| `tools/k8s/k8s.env.example` | 变量模板；提交到 Git。 |
| `tools/k8s/k8s.env` | 本地副本，**不提交**（见 `.gitignore`）。 |
| `.syncignore`（仓库根） | Git 风格忽略规则，**叠加**在 Git 排除规则之上；提交到 Git。 |

可选：根 `Makefile` 增加 `sync` 目标，调用上述脚本（实现阶段决定）。

## 忽略规则（`.gitignore` + `.syncignore`）

1. **Git 侧**：使用 `git ls-files -co --exclude-standard` 生成「已应用 `.gitignore` / Git 排除标准」的文件相对路径列表（含已跟踪与未跟踪、且未被 Git 忽略的文件）。
2. **同步侧**：若存在仓库根 `.syncignore`，在打包或复制阶段通过 `tar --exclude-from=.syncignore` 或等效机制**继续排除**（与 `rsync --exclude-from` 语义同属「路径模式」，细则以脚本实现为准；建议在 spec 中注明：**极复杂的 gitignore 边角与 tar/rsync 不完全等价**，`.syncignore` 宜保持简单目录级规则）。
3. 若不存在 `.syncignore`，仅使用 Git 侧列表。

## 传输方式

**推荐首版**：在本地由上述文件清单生成 **tar 流**，经 `kubectl exec -i … tar -x` 解压到远端目标目录，**单次管道、少落盘**。

**备选**：`rsync` 到临时目录再 `kubectl cp`（更易调试，但占本地磁盘）。

实现阶段需验证：开发机为 **macOS 时 BSD `tar` 与 GNU `tar` 参数差异**；若必要，文档中要求安装 `gnu-tar` 或使用脚本内兼容分支。

## 配置与优先级

`tools/k8s/k8s.env.example` 至少包含（名称可微调，语义固定）：

- `KUBE_CONTEXT`（可选，空则使用当前 kubectl 默认 context）
- `NAMESPACE`（与清单一致，如 `tai-production`）
- `POD_NAME`（如 `build-gpu`）
- `CONTAINER_NAME`（默认 `build`，与 `build-pod.yaml` 中容器名一致）
- `REMOTE_ROOT`（默认 `/workspace`）
- `REMOTE_DIR`（默认仓库目录 basename，或显式字符串）

**优先级（后者覆盖前者）**：`k8s.env` 中的默认值 → 环境变量 → CLI 参数。

脚本应 **显式支持** `--kubeconfig`、`--context`、`--namespace`、`--pod`、`--container` 等常见覆盖（具体名称以实现为准，与 `kubectl` 习惯一致）。

## 同步语义

- **源目录**：默认为 Git 仓库根（`git rev-parse --show-toplevel`）；允许 CLI 指定 `--src`。
- **目标路径**：`$REMOTE_ROOT/$REMOTE_DIR`（例如 `/workspace/tiny-llm`）。同步前 `mkdir -p` 该路径的父级与路径本身（按实现二合一即可）。
- **`--dry-run`**：打印将包含的路径或统计信息，不执行 `kubectl exec`/不传 tar 流。
- **`--delete`（可选）**：是否删除远端存在但本次未发送的文件；**默认关闭**；若开启，建议在文档与脚本中强调风险，且 dry-run 可列出将删除项（实现阶段细化）。

## 与 `build-gpu.pod.yaml` 的关系

- Pod/命名空间/容器名以 **`k8s.env` + 覆盖** 为准。
- `k8s/build-gpu.pod.yaml` 为**单一事实来源的清单**，`k8s.env.example` 内用注释提示「须与清单 metadata 一致」。
- 不在首版自动解析 YAML。

## 验证与测试

- **手工**：`--dry-run` 输出合理；在测试集群上对空目录与非空目录各执行一次；故意删除远端 `/workspace` 下子目录后重跑，确认 `mkdir -p` 恢复成功。
- **静态**：脚本 `shellcheck`（若项目已采用则纳入 CI；未采用则作为可选）。

## 修订记录

- 2026-04-19：初稿（brainstorming 定稿）。

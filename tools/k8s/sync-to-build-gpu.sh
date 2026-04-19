#!/usr/bin/env bash
# Sync local repo subset to build-gpu Pod. See docs/superpowers/specs/2026-04-19-kubectl-sync-build-gpu-design.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/k8s.env"

# Remember exported env from parent (before defaults), for "env > k8s.env"
HAS_ENV_NAMESPACE=0 HAS_ENV_POD=0 HAS_ENV_CONTAINER=0 HAS_ENV_REMOTE_ROOT=0 HAS_ENV_REMOTE_DIR=0 HAS_ENV_KUBE_CONTEXT=0 HAS_ENV_KUBECONFIG=0
if [[ "${NAMESPACE+x}" == x ]]; then HAS_ENV_NAMESPACE=1; SAVED_ENV_NAMESPACE="$NAMESPACE"; fi
if [[ "${POD_NAME+x}" == x ]]; then HAS_ENV_POD=1; SAVED_ENV_POD_NAME="$POD_NAME"; fi
if [[ "${CONTAINER_NAME+x}" == x ]]; then HAS_ENV_CONTAINER=1; SAVED_ENV_CONTAINER_NAME="$CONTAINER_NAME"; fi
if [[ "${REMOTE_ROOT+x}" == x ]]; then HAS_ENV_REMOTE_ROOT=1; SAVED_ENV_REMOTE_ROOT="$REMOTE_ROOT"; fi
if [[ "${REMOTE_DIR+x}" == x ]]; then HAS_ENV_REMOTE_DIR=1; SAVED_ENV_REMOTE_DIR="$REMOTE_DIR"; fi
if [[ "${KUBE_CONTEXT+x}" == x ]]; then HAS_ENV_KUBE_CONTEXT=1; SAVED_ENV_KUBE_CONTEXT="$KUBE_CONTEXT"; fi
if [[ "${KUBECONFIG+x}" == x ]]; then HAS_ENV_KUBECONFIG=1; fi

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
    if [[ "$line" =~ ^[[:space:]]*# ]]; then continue; fi
    if [[ -z "${line// }" ]]; then continue; fi
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
        KUBECONFIG) KUBECONFIG_ARG="$val" ;;
      esac
    fi
  done <"$f"
}

apply_saved_env_over_file() {
  if [[ "$HAS_ENV_NAMESPACE" -eq 1 ]]; then NAMESPACE="$SAVED_ENV_NAMESPACE"; fi
  if [[ "$HAS_ENV_POD" -eq 1 ]]; then POD_NAME="$SAVED_ENV_POD_NAME"; fi
  if [[ "$HAS_ENV_CONTAINER" -eq 1 ]]; then CONTAINER_NAME="$SAVED_ENV_CONTAINER_NAME"; fi
  if [[ "$HAS_ENV_REMOTE_ROOT" -eq 1 ]]; then REMOTE_ROOT="$SAVED_ENV_REMOTE_ROOT"; fi
  if [[ "$HAS_ENV_REMOTE_DIR" -eq 1 ]]; then REMOTE_DIR="$SAVED_ENV_REMOTE_DIR"; fi
  if [[ "$HAS_ENV_KUBE_CONTEXT" -eq 1 ]]; then KUBE_CONTEXT="$SAVED_ENV_KUBE_CONTEXT"; fi
  # Env KUBECONFIG wins over k8s.env file: do not pass --kubeconfig so kubectl uses inherited env
  if [[ "$HAS_ENV_KUBECONFIG" -eq 1 ]]; then KUBECONFIG_ARG=""; fi
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

Environment: NAMESPACE, POD_NAME, CONTAINER_NAME, REMOTE_ROOT, REMOTE_DIR,
  KUBECONFIG (path to kubeconfig file), KUBE_CONTEXT (context name inside that file, not a path)
Config file: tools/k8s/k8s.env (optional; supports KUBECONFIG, KUBE_CONTEXT, and pod vars above)

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

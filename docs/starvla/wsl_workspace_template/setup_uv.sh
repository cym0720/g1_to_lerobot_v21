#!/usr/bin/env bash
# 用 uv 创建 WSL 原生 StarVLA 验证环境；不读取或修改 Windows 侧虚拟环境。
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON_VERSION="3.10"
readonly PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
readonly PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
readonly PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
# CUDA 运行时 wheel 较大，公共索引偶发 TLS 中断时允许 uv 以较长超时重试。
readonly UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
readonly UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_HTTP_TIMEOUT UV_HTTP_RETRIES

cd "${PROJECT_ROOT}"
if [[ ! -d third_party/starvla/.git ]]; then
  echo "缺少 third_party/starvla；请先运行 Windows 仓库中的 bootstrap 脚本。" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  uv venv --python "${PYTHON_VERSION}" .venv
fi
uv pip install --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
  --index-strategy unsafe-best-match \
  -r requirements-cuda124.txt
uv pip install --python "${PYTHON_BIN}" --index-url "${PYPI_INDEX_URL}" -r requirements.txt
uv pip install --python "${PYTHON_BIN}" -e third_party/starvla
"${PYTHON_BIN}" validate_environment.py

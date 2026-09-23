#!/usr/bin/env bash
# 在 WSL 原生文件系统创建独立 StarVLA G1 验证工作区。
# 不移动 Windows 清洗仓库、原始数据或既有虚拟环境；目标目录已存在时直接失败。
set -euo pipefail

readonly SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TARGET_ROOT="${HOME}/Projects_2026/starvla-g1-validation"
readonly TEMPLATE_ROOT="${SOURCE_ROOT}/docs/starvla/wsl_workspace_template"
readonly STARVLA_SOURCE="${SOURCE_ROOT}/external/starVLA"

if [[ -e "${TARGET_ROOT}" ]]; then
  echo "目标工作区已存在：${TARGET_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${STARVLA_SOURCE}/.git" ]]; then
  echo "找不到 StarVLA Git checkout：${STARVLA_SOURCE}" >&2
  exit 2
fi

mkdir -p "${TARGET_ROOT}/scripts" "${TARGET_ROOT}/docs" "${TARGET_ROOT}/third_party"
cp -a "${TEMPLATE_ROOT}/." "${TARGET_ROOT}/"
cp -a "${SOURCE_ROOT}/starvla_act_scripts/." "${TARGET_ROOT}/scripts/"
cp -a "${SOURCE_ROOT}/starvla_guide/." "${TARGET_ROOT}/docs/"

# 从本地 Git checkout 克隆，保留 Windows 工作区中尚未推送的提交；不使用硬链接，
# 避免跨 NTFS/WSL 文件系统共享对象带来的损坏风险。
git clone --no-hardlinks "${STARVLA_SOURCE}" "${TARGET_ROOT}/third_party/starvla"

git -C "${TARGET_ROOT}" init --initial-branch main
git -C "${TARGET_ROOT}" add README.md pyproject.toml requirements.txt setup_uv.sh \
  validate_environment.py .gitignore scripts docs
git -C "${TARGET_ROOT}" commit -m "Create StarVLA G1 validation workspace"

echo "已创建：${TARGET_ROOT}"
echo "下一步：cd ${TARGET_ROOT} && ./setup_uv.sh"

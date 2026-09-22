#!/usr/bin/env bash
# Detached-safe runner for one of the G1 ACT jobs. The caller supplies 04, 05, or 06.
set -euo pipefail

task="${1:?expected task number: 04, 05, or 06}"
case "$task" in 04|05|06) ;; *) echo "Unsupported task: $task" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
data_root="${DATA_ROOT_DIR:-/mnt/d/go work/数据清洗/output/618_all_rgb_test/03_lerobot_g1_3cam_cleaned}"
run_root="${RUN_ROOT_DIR:-/mnt/d/go work/数据清洗/output/starvla_act_g1}"
log_dir="${LOG_DIR:-/mnt/d/go work/数据清洗/output/starvla_act_g1/logs}"
python_bin="${STARVLA_PYTHON:-/home/cym/.venvs/starvla-act/bin/python}"
mkdir -p "$run_root" "$log_dir"
exec >>"$log_dir/g1_act_task_${task}.log" 2>&1

exec env WANDB_MODE=disabled STARVLA_DISABLE_DEEPSPEED=1 "$python_bin" \
  -m accelerate.commands.launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/G1ThreeCam/train_files/train_g1_act.yaml \
  --framework.name ACT \
  --datasets.vla_data.data_root_dir "$data_root" \
  --datasets.vla_data.data_mix "g1_task_$task" \
  --run_root_dir "$run_root" --run_id "g1_act_task_$task"

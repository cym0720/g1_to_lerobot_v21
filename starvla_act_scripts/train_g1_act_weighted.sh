#!/usr/bin/env bash
# Train ACT on task_04 with weighted loss (right-arm focus)
set -euo pipefail

task="04"
repo_dir="/mnt/d/go work/数据清洗/external/starVLA"
data_root="/mnt/d/go work/数据清洗/output/618_all_rgb_test/03_lerobot_g1_3cam_cleaned"
run_root="/mnt/d/go work/数据清洗/output/starvla_act_g1_weighted"
log_dir="${run_root}/logs"
python_bin="/home/cym/.venvs/starvla-act/bin/python"
mkdir -p "$run_root" "$log_dir"

cd "$repo_dir"

WANDB_MODE=disabled STARVLA_DISABLE_DEEPSPEED=1 "$python_bin" \
  -m accelerate.commands.launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/G1ThreeCam/train_files/train_g1_act.yaml \
  --framework.name ACT \
  --datasets.vla_data.data_root_dir "$data_root" \
  --datasets.vla_data.data_mix "g1_task_$task" \
  --run_root_dir "$run_root" --run_id "g1_act_task_${task}_weighted" \
  2>&1 | tee "$log_dir/g1_act_task_${task}_weighted.log"

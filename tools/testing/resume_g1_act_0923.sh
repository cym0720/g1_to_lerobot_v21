#!/usr/bin/env bash
# Resume the interrupted original experiment; preserve its run/checkpoints.
# StarVLA checkpoints contain weights only, so AdamW moments start afresh.
set -euo pipefail
cd /home/cym/Projects_2026/starvla-g1-validation/third_party/starvla
export WANDB_MODE=disabled STARVLA_DISABLE_DEEPSPEED=1
exec /home/cym/Projects_2026/starvla-g1-validation/.venv/bin/python \
  -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/G1ThreeCam/train_files/train_g1_act.yaml \
  --framework.name ACT \
  --datasets.vla_data.data_root_dir /home/cym/datasets/g1/0923_g1_pick_apple \
  --datasets.vla_data.data_mix g1_0923_pink_apple_plate \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.lerobot_version v2.0 \
  --trainer.max_train_steps 15000 --trainer.save_interval 1000 \
  --trainer.logging_frequency 1 --trainer.gradient_accumulation_steps 1 \
  --trainer.gradient_checkpointing true --trainer.is_resume true \
  --run_root_dir /home/cym/artifacts/starvla/runs \
  --run_id g1_act_0923_pick_apple_plate_16step

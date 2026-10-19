#!/bin/bash
set -euo pipefail
ROOT=/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014
source "$ROOT/scripts/env_baselines.sh"
MODE=${1:-formal}
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export STARVLA_DISABLE_DEEPSPEED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
PY=$ROOT/.envs/baselines/bin/python
CFG=$ROOT/baselines/experiment/1008_only_10k/dp_1015_30k.yaml
ENTRY=$ROOT/baselines/experiment/1008_only_10k/train_single_gpu.py
cd "$ROOT/baselines"
if [ "$MODE" = "smoke" ]; then
  exec "$PY" -m accelerate.commands.launch --num_processes 2 --main_process_port 29615 \
    "$ENTRY" --config "$CFG" \
    run_id=DP_1015_h20_aug_h16_smoke \
    trainer.max_train_steps=12 trainer.save_interval=999999 trainer.eval_interval=999999 \
    trainer.checkpoint_gt_eval=false trainer.batch_eval_enabled=false
else
  exec "$PY" -m accelerate.commands.launch --num_processes 2 --main_process_port 29615 \
    "$ENTRY" --config "$CFG"
fi

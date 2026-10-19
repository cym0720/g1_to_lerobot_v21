#!/bin/bash
set -euo pipefail
ROOT=/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014
source "$ROOT/scripts/env_baselines.sh"
MODE=${1:-formal}
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
CFG=$ROOT/baselines/starvla_local/examples/realRobots/PickAppleACT/train_files/oft_20261016_30k.yaml
cd "$ROOT/baselines/starvla_local"
if [ "$MODE" = "smoke" ]; then
  exec "$PY" -m accelerate.commands.launch --num_processes 4 --main_process_port 29616 \
    starVLA/training/train_starvla.py --config_yaml "$CFG" \
    --run_id OFT_20261016_smoke20 \
    --trainer.max_train_steps 20 --trainer.save_interval 10000 --trainer.eval_interval 10000
else
  exec "$PY" -m accelerate.commands.launch --num_processes 4 --main_process_port 29616 \
    starVLA/training/train_starvla.py --config_yaml "$CFG"
fi

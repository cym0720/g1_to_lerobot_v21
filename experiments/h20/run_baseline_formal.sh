#!/bin/bash
set -euo pipefail
ROOT=/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014
source "$ROOT/scripts/env_baselines.sh"
case "$1" in ACT) GPU=1; CONFIG=act_train.yaml;; DP) GPU=2; CONFIG=dp_train.yaml;; *) exit 2;; esac
MEM=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$MEM" -gt 4096 ]; then echo "Assigned GPU unexpectedly busy: $MEM"; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT/baselines"
if [ "${2:-}" = "20k" ]; then
  case "$1" in
    ACT) RUN_ID=ACT_1008_h20_aug_h16_20k; CHECKPOINT="$ROOT/runs/ACT_1008_h20_aug_h16_10k/checkpoints/steps_10000_pytorch_model.pt";;
    DP) RUN_ID=DP_1008_h20_aug_h16_20k; CHECKPOINT="$ROOT/runs/DP_1008_h20_aug_h16_10k/checkpoints/steps_10000_pytorch_model.pt";;
  esac
  "$ROOT/.envs/baselines/bin/python" experiment/1008_only_10k/train_single_gpu.py --config "configs/$CONFIG" \
    trainer.max_train_steps=20000 run_id="$RUN_ID" trainer.continuation_checkpoint="$CHECKPOINT" trainer.continuation_start_step=10000 \
    trainer.gt_output_root="$ROOT/runs/$RUN_ID/gt" trainer.final_proof_output="$ROOT/runs/$RUN_ID/final_inference_proof.json" \
    trainer.diagnostic_output_root="$ROOT/runs/$RUN_ID/train_diagnostics"
elif [ "${2:-}" = "40k" ]; then
  case "$1" in
    DP) RUN_ID=DP_1008_h20_aug_h16_40k; CHECKPOINT="$ROOT/runs/DP_1008_h20_aug_h16_20k_retry1/checkpoints/steps_20000_pytorch_model.pt";;
    *) echo "40k continuation is defined for DP only"; exit 3;;
  esac
  "$ROOT/.envs/baselines/bin/python" experiment/1008_only_10k/train_single_gpu.py --config "configs/$CONFIG" \
    trainer.max_train_steps=40000 run_id="$RUN_ID" trainer.continuation_checkpoint="$CHECKPOINT" trainer.continuation_start_step=20000 \
    trainer.gt_output_root="$ROOT/runs/$RUN_ID/gt" trainer.final_proof_output="$ROOT/runs/$RUN_ID/final_inference_proof.json" \
    trainer.diagnostic_output_root="$ROOT/runs/$RUN_ID/train_diagnostics"
else
  "$ROOT/.envs/baselines/bin/python" experiment/1008_only_10k/train_single_gpu.py --config "configs/$CONFIG"
fi

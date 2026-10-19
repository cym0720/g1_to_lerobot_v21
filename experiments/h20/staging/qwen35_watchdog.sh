#!/bin/bash
ROOT=/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local
R="$ROOT/playground/Pretrained_models/qwen35_0.8b_download_receipt.json"
cd "$ROOT"
while :; do
  st=$(python3 -c "import json;print(json.load(open('$R')).get('status',''))" 2>/dev/null)
  [ "$st" = "PASS" ] && break
  /data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/.envs/baselines/bin/python scripts/download_qwen35_starvla.py >> playground/Pretrained_models/qwen35_download.log 2>&1
  sleep 10
done
echo WATCHDOG_QWEN35_PASS

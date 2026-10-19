#!/bin/bash
cd /data2/cym_ws/qwen35_oft_20261015
while :; do
  st=$(python3 -c "import json;print(json.load(open('audits/download_receipt.json')).get('status',''))" 2>/dev/null)
  [ "$st" = "PASS" ] && break
  [ "$st" = "FAIL" ] && sleep 60 && continue
  /data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/.envs/baselines/bin/python scripts/download_qwen.py >> audits/download.log 2>&1
  sleep 10
done
echo WATCHDOG_QWEN_PASS

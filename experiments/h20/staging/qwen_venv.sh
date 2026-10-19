#!/bin/bash
set -x
cd /data2/cym_ws/qwen35_oft_20261015
export HTTPS_PROXY=http://127.0.0.1:18193
export HTTP_PROXY=http://127.0.0.1:18193
python3 -m venv .venv || exit 1
.venv/bin/pip install --upgrade pip || exit 1
ok=0
for i in $(seq 1 12); do
  .venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && ok=1 && break
  echo PIP_TORCH_RETRY $i
  sleep 20
done
[ $ok -eq 1 ] || { echo VENV_TORCH_FAIL; exit 1; }
ok=0
for i in $(seq 1 12); do
  .venv/bin/pip install "transformers>=4.50" accelerate peft && ok=1 && break
  echo PIP_HF_RETRY $i
  sleep 20
done
[ $ok -eq 1 ] || { echo VENV_HF_FAIL; exit 1; }
.venv/bin/pip freeze > requirements.lock
echo VENV_SETUP_DONE

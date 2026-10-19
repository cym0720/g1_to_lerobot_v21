#!/bin/bash
# 下载 DP_1015 三步模型 bundle (10k/15k/20k) —— 分块 scp + SHA256 校验，落到 D:\go work\models
DEST="/d/go work/models"
RDIR="/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/chunks1015_3step"
OUT="$DEST/dp_1015_10k15k20k_server_bundle.tar.gz"
EXPECT="03947255c3414d2a3725911779bc2c74821ee8238d02e7e6fad35bdfe2bcaccd"
mkdir -p "$DEST"
NAMES=$(ssh -o ClearAllForwardings=yes -o ConnectTimeout=25 H20 "ls $RDIR | grep -v SHA256" | tr -d '\r')
for f in $NAMES; do
  ok=0
  for try in $(seq 1 40); do
    scp -o ClearAllForwardings=yes -o ConnectTimeout=25 "H20:$RDIR/$f" "$DEST/" >/dev/null 2>&1 && ok=1 && break
    sleep 8
  done
  [ $ok -eq 1 ] && echo "chunk $f done" || { echo "chunk $f FAILED after 40 tries"; exit 1; }
done
cat "$DEST"/part_* > "$OUT"
GOT=$(sha256sum "$OUT" | cut -d" " -f1)
if [ "$GOT" = "$EXPECT" ]; then
  echo "SHA256_OK $OUT"
  rm -f "$DEST"/part_*
  echo "CHUNKS_CLEANED"
else
  echo "SHA256_MISMATCH got=$GOT"
  exit 1
fi

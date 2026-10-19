#!/bin/bash
# 下载 DP_1015(30k, best step20000) 服务 bundle —— 分块 scp + SHA256 校验
DEST="/d/go work/tmp"
RDIR="/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/chunks1015dp"
OUT="$DEST/dp_1015_best_server_bundle.tar.gz"
EXPECT="117b4cd5196781c7de15c480d8afcacc2b8315d40af38413ce1ea741fbded575"
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

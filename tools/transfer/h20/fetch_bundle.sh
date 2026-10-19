#!/bin/bash
DEST="/d/go work/tmp"
RDIR="/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/chunks40k"
OUT="$DEST/dp_40k_best_server_bundle.tar.gz"
EXPECT="90d51a3d801ade5092adc325ad4ce9c678c1cea06049180bc51142a25cb6d294"
mkdir -p "$DEST"
NAMES=$(ssh -o ClearAllForwardings=yes -o ConnectTimeout=25 H20 "ls $RDIR" | tr -d '\r')
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

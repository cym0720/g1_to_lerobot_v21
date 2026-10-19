#!/bin/bash
DEST="/d/go work/tmp"
RDIR="/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/chunks3w"
OUT="$DEST/dp3w_server_bundle.tar.gz"
EXPECT="b7a455c2fba4b274455db19598856084d333fc68a39550a19df6e8a4ea0de778"
rm -f "$DEST"/part_*
mkdir -p "$DEST"
NAMES=$(ssh -o ClearAllForwardings=yes -o ConnectTimeout=25 H20 "ls $RDIR" | tr -d '\r')
TOTAL=$(echo "$NAMES" | wc -w)
i=0
for f in $NAMES; do
  i=$((i+1))
  ok=0
  for try in $(seq 1 60); do
    scp -o ClearAllForwardings=yes -o ConnectTimeout=25 "H20:$RDIR/$f" "$DEST/" >/dev/null 2>&1 && ok=1 && break
    sleep 8
  done
  [ $ok -eq 1 ] && echo "[$i/$TOTAL] $f done" || { echo "chunk $f FAILED after 60 tries"; exit 1; }
done
cat "$DEST"/part_* > "$OUT"
GOT=$(sha256sum "$OUT" | cut -d" " -f1)
if [ "$GOT" = "$EXPECT" ]; then
  echo "SHA256_OK"
  rm -f "$DEST"/part_*
  echo "ALL_DONE"
else
  echo "SHA256_MISMATCH got=$GOT"
  exit 1
fi

#!/bin/bash
# 并行分块下载 DP_rightmask 20k+30k bundle（xargs -P6, 每块独立重试, SHA256 校验）
set -euo pipefail
DEST="/d/go work/models"
RDIR="/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/chunks_rm2"
OUT="$DEST/dp_rightmask_20k30k_server_bundle.tar.gz"
EXPECT="ddd65f48a66f3f42fba8c85a080fc7254c043f1bf16d37046285208bb32d1e74"
mkdir -p "$DEST"

# worker: 已存在且大小对 = 跳过（断点续传语义）；否则重试 scp
WORKER="$DEST/.chunk_worker.sh"
cat > "$WORKER" <<'EOS'
#!/bin/bash
# xargs 调用形如: bash worker "$DEST" "$RDIR" <name>  → 块名在 $3
DEST="$1"; RDIR="$2"; f="$3"
RS=$(ssh -o ClearAllForwardings=yes -o ConnectTimeout=25 H20 "stat -c %s $RDIR/$f" 2>/dev/null | tr -d '\r')
[ -z "$RS" ] && { echo "$f NO_REMOTE_SIZE"; exit 1; }
LS=$(stat -c %s "$DEST/$f" 2>/dev/null || echo 0)
if [ "$LS" = "$RS" ]; then echo "$f SKIP"; exit 0; fi
for t in $(seq 1 40); do
  rm -f "$DEST/$f.part"
  scp -o ClearAllForwardings=yes -o ConnectTimeout=25 "H20:$RDIR/$f" "$DEST/$f.part" >/dev/null 2>&1 || { sleep 6; continue; }
  NS=$(stat -c %s "$DEST/$f.part" 2>/dev/null || echo 0)
  if [ "$NS" = "$RS" ]; then mv "$DEST/$f.part" "$DEST/$f"; echo "$f done"; exit 0; fi
  sleep 6
done
echo "$f FAILED"
exit 1
EOS
chmod +x "$WORKER"

NAMES=$(ssh -o ClearAllForwardings=yes -o ConnectTimeout=25 H20 "ls $RDIR" | tr -d '\r')
echo "$NAMES" | xargs -P 6 -n 1 bash "$WORKER" "$DEST" "$RDIR"

# 拼接 + 校验
cat "$DEST"/part_* > "$OUT"
GOT=$(sha256sum "$OUT" | cut -d" " -f1)
if [ "$GOT" = "$EXPECT" ]; then
  echo "SHA256_OK $OUT"
  rm -f "$DEST"/part_* "$WORKER"
  echo "CHUNKS_CLEANED"
else
  echo "SHA256_MISMATCH got=$GOT"
  exit 1
fi

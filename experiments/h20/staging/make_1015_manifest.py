#!/usr/bin/env python3
"""Generate DP training_manifest.json for the 1015 letterbox dataset on H20.

Reads split_manifest.json, maps source_directory -> source_dataset (the GT
eval contract requires source_dataset), validates every entry length against
the per-split episodes.jsonl, and writes training_manifest.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014")
DATA = ROOT / "data/training_split_1015_letterbox224_v1"

split = json.loads((DATA / "split_manifest.json").read_text())
entries_in = split["entries"]

out_entries = []
for e in entries_in:
    ee = dict(e)
    if "source_dataset" not in ee and "source_directory" in ee:
        ee["source_dataset"] = ee.pop("source_directory")
    ee.setdefault("source_group", "1015")
    if "source_dataset" not in ee:
        print("NO_SOURCE_DATASET", ee)
        sys.exit(3)
    out_entries.append(ee)

# validate length against episodes.jsonl per split
eps = {}
for sp in ("train", "val"):
    lines = [json.loads(l) for l in (DATA / sp / "meta" / "episodes.jsonl").read_text().splitlines()]
    eps[sp] = {}
    for x in lines:
        length = x.get("length", x.get("episode_length"))
        eps[sp][int(x["episode_index"])] = int(length)

problems = []
for e in out_entries:
    sp, idx, ln = e["split"], int(e["episode_index"]), int(e["length"])
    if sp not in eps or eps[sp].get(idx) != ln:
        problems.append((sp, idx, ln, eps.get(sp, {}).get(idx)))
if problems:
    print("LENGTH_MISMATCH total=%d" % len(problems))
    for p in problems[:10]:
        print("  ", p)
    sys.exit(2)

req = ["split", "episode_index", "length", "original_episode_id", "source_dataset", "source_episode_index"]
missing = [k for k in req if k not in out_entries[0]]
if missing:
    print("MISSING_KEYS", missing)
    sys.exit(4)

train_n = sum(1 for e in out_entries if e["split"] == "train")
val_n = sum(1 for e in out_entries if e["split"] == "val")
train_frames = sum(int(e["length"]) for e in out_entries if e["split"] == "train")
val_frames = sum(int(e["length"]) for e in out_entries if e["split"] == "val")
val_ids = [e["original_episode_id"] for e in out_entries if e["split"] == "val"]

manifest = {
    "status": "PASS",
    "train_episodes": train_n,
    "val_episodes": val_n,
    "validation_episodes": val_n,
    "stats_policy": "train stats only used for training/inference; val stats are diagnostic only",
    "val_id": val_ids[0] if len(val_ids) == 1 else ",".join(val_ids),
    "train_frames": train_frames,
    "val_frames": val_frames,
    "entries": out_entries,
    "train_val_disjoint": bool(split.get("train_val_disjoint", True)),
}
(DATA / "training_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
print("PASS train=%d val=%d train_frames=%d val_frames=%d" % (train_n, val_n, train_frames, val_frames))

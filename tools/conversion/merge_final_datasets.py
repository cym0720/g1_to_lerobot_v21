"""Concat-merge two FINAL LeRobot v2.1 datasets without touching any numeric values.

Unlike `merge_pick_apple_1008_1015.py` (which re-synthesises `action = state[t+3]` and
re-orders the body), this tool assumes both inputs are already finished products:
same column names, same body order, same gripper conventions. It only:

  * renumbers `episode_index` (and the global `index` column) across both sources
  * hardlinks (falls back to copy) episode videos under the new episode ids
  * concatenates train/ and val/ separately
  * writes meta/info.json with statistics_status=pending (run 05_recompute_metadata_stats after)
  * records merge_provenance incl. the detected body order of each source

Usage:
  python tools/conversion/merge_final_datasets.py --a ROOT_A --b ROOT_B --output NEW_ROOT
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.body_order import detect_body_order

STATE_CANDIDATES = ("observation.state", "observations.state.qpos")


def state_col(schema) -> str:
    for name in STATE_CANDIDATES:
        if name in schema.names:
            return name
    raise ValueError(f"no state column in {schema.names}")


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def source_episodes(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def merge_split(split: str, sources: list[tuple[str, Path]], out_split: Path) -> dict:
    """Concatenate one split (train/val) from several source roots."""
    data_dir = out_split / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_split / "videos" / "chunk-000"

    ep_index = 0
    global_index = 0
    total_frames = 0
    episodes_meta: list[dict] = []
    video_keys: set[str] = set()
    tasks: dict[int, str] = {}
    body_orders: dict[str, str | None] = {}
    detection: dict[str, list[str]] = {}

    for tag, root in sources:
        files = source_episodes(root / split)
        if not files:
            continue
        # tasks
        task_file = root / split / "meta" / "tasks.jsonl"
        if task_file.exists():
            for line in task_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    tasks[int(rec.get("task_index", 0))] = rec.get("task", "")

        detected_here: list[str] = []
        for src_pq in files:
            table = pq.read_table(src_pq)
            n = table.num_rows
            if n == 0:
                continue

            st_name = state_col(table.schema)
            st = np.asarray(table.column(st_name).to_pylist(), np.float64)
            order = detect_body_order(st)
            if order:
                detected_here.append(order)

            # renumber episode_index + global index, leave every numeric value untouched
            cols = list(table.column_names)
            table = table.set_column(
                cols.index("episode_index"), "episode_index", pa.array([ep_index] * n, pa.int64())
            )
            if "index" in cols:
                table = table.set_column(
                    cols.index("index"), "index",
                    pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                )
            pq.write_table(table, data_dir / f"episode_{ep_index:06d}.parquet")

            # videos: episode_{src_i:06d}.mp4 -> episode_{ep_index:06d}.mp4
            src_i = int(src_pq.stem.split("_")[1])
            if videos_dir.exists() or (root / split / "videos").exists():
                for cam_dir in sorted((root / split / "videos").glob("chunk-*/*")):
                    if not cam_dir.is_dir():
                        continue
                    video_keys.add(cam_dir.name)
                    src_video = cam_dir / f"episode_{src_i:06d}.mp4"
                    if src_video.exists():
                        link_or_copy(
                            src_video,
                            videos_dir / cam_dir.name / f"episode_{ep_index:06d}.mp4",
                        )

            episodes_meta.append({"episode_index": ep_index, "length": n, "task_index": 0})
            ep_index += 1
            global_index += n
            total_frames += n

        detection[tag] = sorted(set(detected_here))
        body_orders[tag] = detection[tag][0] if len(detection[tag]) == 1 else (
            "mixed" if detection[tag] else None
        )

    (out_split / "meta").mkdir(parents=True, exist_ok=True)
    with (out_split / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_split / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for idx in sorted(tasks) or [0]:
            f.write(json.dumps({"task_index": idx, "task": tasks.get(idx, "")}, ensure_ascii=False) + "\n")

    return {
        "episodes": ep_index,
        "frames": total_frames,
        "video_keys": sorted(video_keys),
        "body_order": body_orders,
        "detection": detection,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Concat-merge two FINAL LeRobot v2.1 datasets (no numeric change).")
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    sources = [("a", args.a), ("b", args.b)]
    info_a = json.loads((args.a / "train" / "meta" / "info.json").read_text(encoding="utf-8"))

    report: dict[str, dict] = {}
    for split in ("train", "val"):
        if not any(source_episodes(root / split) for _, root in sources):
            continue
        report[split] = merge_split(split, sources, args.output / split)

    total_eps = sum(r["episodes"] for r in report.values())
    total_frames = sum(r["frames"] for r in report.values())
    info = dict(info_a)
    info["total_episodes"] = total_eps
    info["total_frames"] = total_frames
    info["total_videos"] = total_eps * 3
    info["splits"] = {name: f"0:{r['episodes']}" for name, r in report.items()}
    info["statistics_status"] = "pending"
    info["merge_provenance"] = {
        "tool": "merge_final_datasets.py",
        "numeric_values_unchanged": True,
        "action_resynthesis": False,
        "body_reorder": False,
        "sources": {"a": str(args.a), "b": str(args.b)},
        "per_source_body_order": {k: v for k, v in report.get("train", {}).get("body_order", {}).items()},
        "note": "成品级拼接：仅重编号 episode_index/index 与视频链接；统计由 05_recompute_metadata_stats 重建。",
    }

    for split, rep in report.items():
        split_info = dict(info)
        split_info["total_episodes"] = rep["episodes"]
        split_info["total_frames"] = rep["frames"]
        split_info["splits"] = {split: f"0:{rep['episodes']}"}
        (args.output / split / "meta").mkdir(parents=True, exist_ok=True)
        (args.output / split / "meta" / "info.json").write_text(
            json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "output": str(args.output),
        "splits": {k: {"episodes": v["episodes"], "frames": v["frames"]} for k, v in report.items()},
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "per_source_body_order": {k: v["body_order"] for k, v in report.items()},
        "detected_orders": {k: v["detection"] for k, v in report.items()},
        "numeric_values_unchanged": True,
    }
    (args.output / "merged_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# 用途（合并步骤）：把 1008 最终版（含 multigrasp）与 1015 最终版合并为单一 LeRobot v2.1 合集。
# 1008 行字节不动；1015 行统一重推导为 1008 口径（身体顺序 [lift, waist_pitch]、
# action=state[min(t+3,n-1)]、夹爪=同帧记录指令>=0.5 二值）。划分与统计重建留给后续确认步骤。
"""合并 1008 + 1015 pick apple 数据集。

- 输出为单一任务根（197 条），splits={"all": "0:N"}，statistics/split 均标记 pending；
- 视频硬链接（同盘零拷贝），parquet 重写（episode_index/global index 重编号，1015 行做口径统一）；
- 写 merged_manifest.json 与 merge_provenance。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.lerobot_v21 import episodes  # noqa: E402

PROMPT = "pick up the apple and put it into the plate"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def canonicalize_1015(table) -> tuple[np.ndarray, np.ndarray]:
    """1015 EVA 行 -> 1008 口径：身体换位 + 未来三帧状态动作 + 二值夹爪。

    注意：身体换位仅适用于 2026-10-18 契约修正前转换的 1015 产物
    （waist_pitch-first）。当前 EVA 转换契约是 [waist_pitch, lift]；本脚本
    输出的是历史 1008 训练口径，不能用作当前契约的通用合并入口。
    重跑前必须核对来源的实际身体顺序，否则会二次换位。
    """
    s = np.asarray(table["observation.state"].to_pylist(), np.float32)
    a_cmd = np.asarray(table["action"].to_pylist(), np.float32)
    n = len(s)
    body = s[:, 18:20].copy()
    s2 = s.copy()
    s2[:, 18] = body[:, 1]  # EVA: waist_pitch,lift -> 1008: lift,waist_pitch
    s2[:, 19] = body[:, 0]
    command_grip = a_cmd[:, [7, 15]].copy()
    a2 = s2[np.minimum(np.arange(n) + 3, n - 1)].copy()
    a2[:, [7, 15]] = (command_grip >= 0.5).astype(np.float32)
    for name, arr in (("state", s2), ("action", a2)):
        assert np.isfinite(arr).all(), f"{name} 非有限值"
        assert np.all((arr[:, [7, 15]] >= 0) & (arr[:, [7, 15]] <= 1)), f"{name} 夹爪越界"
    return s2, a2


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="1008 最终版根目录（含 train/ val/）")
    p.add_argument("--b", type=Path, required=True, help="1015 最终版根目录（含 train/ val/）")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"拒绝覆盖已有目录: {a.output}")

    sources = [
        ("1008_letterbox224_v1", a.a / "train", False),
        ("1008_letterbox224_v1", a.a / "val", False),
        ("1015_trim_letterbox224_v1", a.b / "train", True),
        ("1015_trim_letterbox224_v1", a.b / "val", True),
    ]
    a_info = read_json(sources[0][1] / "meta/info.json")
    b_info = read_json(sources[2][1] / "meta/info.json")
    features = {}
    for key, value in a_info["features"].items():
        features[key] = json.loads(json.dumps(value))
    for key, value in b_info["features"].items():
        if value["dtype"] == "video":  # 用 1015 的规范 224 视频元数据
            features[key] = json.loads(json.dumps(value))

    a.output.mkdir(parents=True)
    (a.output / "meta").mkdir()
    merged_rows = []
    manifest = []
    offset = 0
    canonicalized = []
    for tag, src, is_1015 in sources:
        task_rows = [json.loads(x) for x in (src / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert [r["task"] for r in task_rows] == [PROMPT], f"{src}: prompt 不一致"
        cameras = [k for k, v in features.items() if v["dtype"] == "video"]
        for ep in episodes(src):
            i = len(merged_rows)
            old = ep["episode_index"]
            n = ep["length"]
            table = pq.read_table(src / f"data/chunk-000/episode_{old:06d}.parquet")
            assert table.num_rows == n
            if is_1015:
                s2, a2 = canonicalize_1015(table)
                state_type = table.schema.field("observation.state").type
                action_type = table.schema.field("action").type
                cols = {
                    "timestamp": table["timestamp"],
                    "frame_index": table["frame_index"],
                    "episode_index": pa.array(np.full(n, i, np.int64)),
                    "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
                    "task_index": table["task_index"],
                    "observation.state": pa.array(s2.tolist(), type=state_type),
                    "action": pa.array(a2.tolist(), type=action_type),
                }
                canonicalized.append(ep.get("original_episode_id"))
            else:
                keep = {c: table[c] for c in table.column_names if c not in ("episode_index", "index")}
                cols = dict(keep)
                cols["episode_index"] = pa.array(np.full(n, i, np.int64))
                cols["index"] = pa.array(np.arange(offset, offset + n, dtype=np.int64))
            dest = a.output / f"data/chunk-000/episode_{i:06d}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(cols), dest, compression="snappy")
            for cam in cameras:
                link_or_copy(
                    src / f"videos/chunk-000/{cam}/episode_{old:06d}.mp4",
                    a.output / f"videos/chunk-000/{cam}/episode_{i:06d}.mp4",
                )
            row = dict(ep)
            row.update(
                episode_index=i, dataset_from_index=offset, dataset_to_index=offset + n,
                merged_source=tag, merged_source_split=src.name, merged_source_episode_index=old,
                canonicalized_from_1015=is_1015,
            )
            merged_rows.append(row)
            manifest.append({
                "episode_index": i, "length": n, "merged_source": tag,
                "merged_source_split": src.name, "merged_source_episode_index": old,
                "original_episode_id": ep.get("original_episode_id"),
            })
            offset += n

    info = {
        "codebase_version": "v2.1",
        "robot_type": "G1",
        "fps": 30,
        "total_episodes": len(merged_rows),
        "total_frames": offset,
        "total_tasks": 1,
        "total_videos": sum(1 for v in features.values() if v["dtype"] == "video") * len(merged_rows),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"all": f"0:{len(merged_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "statistics_status": "pending (合并后未重算，待用户确认划分后执行)",
        "split_status": "pending (单一合集，待用户确认划分方案)",
        "merge_provenance": {
            "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": [
                {"root": str(src), "tag": tag, "episodes": sum(1 for m in manifest if m["merged_source_split"] == src.name and m["merged_source"] == tag)}
                for tag, src, _ in sources
            ],
            "canonicalization_1015": [
                "body order: EVA [waist_pitch, lift] -> 1008 [lift, waist_pitch]",
                "action = state[min(t+3, n-1)]",
                "gripper action = (same-frame recorded command >= 0.5) binarized",
                "columns kept as observation.state / action (1008/评估脚本口径)",
            ],
            "videos": "hardlinked from sources (byte-identical, zero-copy)",
        },
    }
    dump(a.output / "meta/info.json", info)
    (a.output / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged_rows), encoding="utf-8")
    (a.output / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": PROMPT}, ensure_ascii=False) + "\n", encoding="utf-8")
    dump(a.output / "merged_manifest.json", {
        "total": len(merged_rows),
        "per_source": {tag: sum(1 for m in manifest if m["merged_source"] == tag) for tag, _, _ in sources},
        "canonicalized_1015_episodes": len(canonicalized),
        "statistics_status": "pending",
        "split_status": "pending",
        "entries": manifest,
    })
    print(f"MERGED total={len(merged_rows)} frames={offset} canonicalized_1015={len(canonicalized)}", flush=True)


if __name__ == "__main__":
    main()

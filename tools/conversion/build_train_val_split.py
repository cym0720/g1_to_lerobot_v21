# 用途（主流程划分步骤）：按完整轨迹把一个转换/裁剪后的 LeRobot v2.1 数据集切成 train/val，
# 整条隔离防泄漏，逐条重编号并重建全局索引；统计留给 05_recompute_metadata_stats。
"""按完整轨迹构建 train/val 划分。

约定：
- 只移动完整 episode（不裁剪、不改数值内容，仅重写 episode_index 与全局 index 列）；
- val 默认取裁剪后最短的 N 条（--val-count），并用 --read-validation 只在读校验通过的条目中选；
- 输出 train/、val/ 各为独立 LeRobot v2.1 数据集根目录；
- episodes_stats.jsonl / stats.json 不复制，info 标记 statistics_status=pending，由 05 重建。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.lerobot_v21 import episodes, task_roots  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os_link = __import__("os").link
        os_link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="任务父目录或单任务根目录（划分来源）")
    p.add_argument("--output", type=Path, required=True, help="划分输出根目录，不得已存在")
    p.add_argument("--val-count", type=int, default=4, help="按长度选最短 N 条做验证集")
    p.add_argument(
        "--from-manifest",
        type=Path,
        help="按合并清单 entries 的 merged_source_split('train'/'val') 分配划分；设置时忽略 --val-count",
    )
    p.add_argument(
        "--read-validation",
        type=Path,
        help="01_validate 的报告 CSV；只在 status=ok 的条目中划分",
    )
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"拒绝覆盖已有目录: {a.output}")

    roots = task_roots(a.input)
    if not roots:
        raise SystemExit(f"未找到数据集根目录: {a.input}")

    task_texts = set()
    for root in roots:
        rows = lines(root / "meta/tasks.jsonl")
        if len(rows) != 1:
            raise SystemExit(f"{root}: 期望单任务，实际 {len(rows)}")
        task_texts.add(rows[0]["task"])
    if len(task_texts) != 1:
        raise SystemExit(f"多任务文本暂不支持: {sorted(task_texts)}")
    task_text = task_texts.pop()

    good = None
    excluded = []
    if a.read_validation:
        with a.read_validation.open(encoding="utf-8") as f:
            validation = list(csv.DictReader(f))
        good = {(r["task_directory"], int(r["episode_index"])) for r in validation if r["status"] == "ok"}
        excluded = [r for r in validation if r["status"] != "ok"]

    entries = []
    for root in roots:
        for ep in episodes(root):
            if good is not None and (root.name, ep["episode_index"]) not in good:
                excluded.append({"task_directory": root.name, "episode_index": ep["episode_index"], "status": "skipped_not_ok"})
                continue
            entries.append({"root": root, "ep": ep})
    if not entries:
        raise SystemExit("没有可划分的 episode")

    if a.from_manifest:
        manifest_rows = [
            json.loads(x)
            for x in a.from_manifest.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ] if a.from_manifest.suffix == ".jsonl" else json.loads(a.from_manifest.read_text(encoding="utf-8")).get("entries", [])
        split_of = {row["episode_index"]: row["merged_source_split"] for row in manifest_rows}
        val_slots = set()
        for i, e in enumerate(entries):
            src_split = split_of.get(e["ep"]["episode_index"])
            if src_split not in ("train", "val"):
                raise SystemExit(f"manifest 未覆盖 episode {e['ep']['episode_index']} 或划分非法: {src_split}")
            if src_split == "val":
                val_slots.add(i)
        if not val_slots or len(val_slots) == len(entries):
            raise SystemExit("manifest 划分异常")
    else:
        order = sorted(
            range(len(entries)),
            key=lambda i: (entries[i]["ep"]["length"], entries[i]["root"].name, entries[i]["ep"].get("original_episode_id", "")),
        )
        if a.val_count < 1 or a.val_count >= len(entries):
            raise SystemExit(f"--val-count 必须在 1..{len(entries) - 1}，当前 {a.val_count}")
        val_slots = set(order[: a.val_count])

    a.output.mkdir(parents=True)
    manifest = []
    for split in ("train", "val"):
        items = [e for i, e in enumerate(entries) if (i in val_slots) == (split == "val")]
        out = a.output / split
        (out / "meta").mkdir(parents=True)
        src_info = read_json(items[0]["root"] / "meta/info.json")
        cameras = [k for k, v in src_info["features"].items() if v["dtype"] == "video"]
        offset = 0
        new_eps = []
        for i, item in enumerate(items):
            source, ep = item["root"], item["ep"]
            old, n = ep["episode_index"], ep["length"]
            table = pq.read_table(source / f"data/chunk-000/episode_{old:06d}.parquet")
            if table.num_rows != n:
                raise SystemExit(f"{source.name} episode {old}: parquet 行数 {table.num_rows} != 声明 {n}")
            table = table.set_column(
                table.schema.get_field_index("episode_index"), "episode_index",
                pa.array(np.full(table.num_rows, i, dtype=np.int64)))
            table = table.set_column(
                table.schema.get_field_index("index"), "index",
                pa.array(np.arange(offset, offset + table.num_rows, dtype=np.int64)))
            dest = out / f"data/chunk-000/episode_{i:06d}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest, compression="snappy")
            for key in cameras:
                link_or_copy(
                    source / f"videos/chunk-000/{key}/episode_{old:06d}.mp4",
                    out / f"videos/chunk-000/{key}/episode_{i:06d}.mp4",
                )
            row = dict(ep)
            row.update(episode_index=i, dataset_from_index=offset, dataset_to_index=offset + n)
            new_eps.append(row)
            manifest.append({
                "split": split, "episode_index": i, "length": n,
                "original_episode_id": ep.get("original_episode_id"),
                "source_directory": source.name, "source_episode_index": old,
            })
            offset += n
        info = json.loads(json.dumps(src_info))
        info.update(
            total_episodes=len(items),
            total_frames=offset,
            total_videos=len(cameras) * len(items),
            splits={split: f"0:{len(items)}"},
            statistics_status="pending full recomputation",
        )
        info["split_provenance"] = {
            "source": str(a.input),
            "rule": (
                f"whole-episode split; from merged manifest {a.from_manifest.name}"
                if a.from_manifest
                else f"whole-episode split; val = shortest {a.val_count} by post-trim length"
            ),
            "read_validation": str(a.read_validation) if a.read_validation else None,
        }
        dump(out / "meta/info.json", info)
        (out / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in new_eps), encoding="utf-8")
        shutil.copy2(items[0]["root"] / "meta/tasks.jsonl", out / "meta/tasks.jsonl")

    train_ids = {m["original_episode_id"] for m in manifest if m["split"] == "train"}
    val_ids = {m["original_episode_id"] for m in manifest if m["split"] == "val"}
    if train_ids & val_ids:
        raise SystemExit(f"train/val 存在泄漏: {sorted(train_ids & val_ids)[:5]}")
    dump(a.output / "split_manifest.json", {
        "train_episodes": len(train_ids),
        "validation_episodes": len(val_ids),
        "train_val_disjoint": True,
        "val_selection": [
            {"id": entries[i]["ep"].get("original_episode_id"), "frames": entries[i]["ep"]["length"]}
            for i in sorted(val_slots)
        ],
        "excluded": excluded,
        "task": task_text,
        "entries": manifest,
        "statistics_status": "pending",
    })
    print(f"SPLIT_WRITTEN train={len(train_ids)} val={len(val_ids)}", flush=True)


if __name__ == "__main__":
    main()

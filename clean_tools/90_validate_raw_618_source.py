# 用途（辅助）：检查 618 原始 episode 的关键文件完整性、任务映射和最短时长；默认只生成审计报告，不复制原始数据。
"""Validate and group raw 618 episodes for conversion.

The source is untouched by default. Use --mode move only after checking the
generated manifest; it moves each full episode directory into a new root.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def load_map(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("task map 必须有 tasks 对象")
    directories = [x.get("directory") for x in tasks.values()]
    if len(directories) != len(set(directories)):
        raise ValueError("task map 中 directory 必须唯一")
    return tasks


def validate_episode(path: Path, min_duration_s: float) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    meta_path = path / "meta_info.json"
    if not meta_path.is_file():
        return None, ["missing_meta_info.json"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid_meta_info.json:{exc}"]
    required = [
        "record/raw_joints.h5",
        "camera/head_color/head_color.h265",
        "camera/head_color/head_color.txt",
    ]
    for relative in required:
        if not (path / relative).is_file():
            errors.append(f"missing_{relative}")
    for wrist in ("hand_left", "hand_right"):
        if not list((path / "camera" / wrist / "color").glob("*.jpg")):
            errors.append(f"missing_{wrist}_jpg")
    timestamp_file = path / "camera" / "head_color" / "head_color.txt"
    if timestamp_file.is_file():
        values = [int(line.split()[0]) for line in timestamp_file.read_text(encoding="utf8").splitlines() if line.strip()]
        duration_s = (values[-1] - values[0]) / 1e9 if len(values) >= 2 else 0.0
        if duration_s < min_duration_s:
            # This excludes it from the validated output; source data remains untouched.
            errors.append(f"too_short_head_duration_s:{duration_s:.3f}<{min_duration_s:g}")
    return meta, errors


def transfer(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(source, destination)
    else:
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="按 task_id 分组 618 原始 episode，并隔离缺陷数据。")
    parser.add_argument("--input", type=Path, required=True, help="dataset/618data")
    parser.add_argument("--output", type=Path, required=True, help="复制/移动模式的分组根目录；dry-run 时仅作为报告默认位置")
    parser.add_argument("--task-map", type=Path, default=Path(__file__).parent.parent / "config" / "task_id_map.json")
    parser.add_argument("--min-duration-s", type=float, default=8.0, help="头部 RGB 时间跨度少于该秒数的轨迹直接归入 00_defective")
    parser.add_argument("--mode", choices=("copy", "move"), default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="审计 CSV 路径；不传时为 output/organize_manifest.csv")
    args = parser.parse_args()
    if not args.dry_run and args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"输出目录必须不存在或为空: {args.output}")
    mapping = load_map(args.task_map)
    rows = []
    for episode in sorted(x for x in args.input.iterdir() if x.is_dir()):
        meta, errors = validate_episode(episode, args.min_duration_s)
        if errors:
            bucket, task_id, task_name = "00_defective", "", "defective"
        else:
            task_id = str(meta.get("task_id", ""))
            spec = mapping.get(task_id)
            if spec is None:
                bucket, task_name = "00_defective", "unknown_task_id"
                errors.append(f"unmapped_task_id:{task_id}")
            else:
                bucket, task_name = spec["directory"], spec["name"]
        destination = args.output / bucket / episode.name
        rows.append([episode.name, task_id, task_name, bucket, ";".join(errors) or "ok", str(destination)])
        if not args.dry_run:
            transfer(episode, destination, args.mode)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.task_map, args.output / "task_id_map.json")
    manifest = args.report or args.output / "organize_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf8") as file:
        csv.writer(file).writerows([["episode_uuid", "task_id", "task_name", "bucket", "validation", "destination"], *rows])
    for row in rows:
        print(" | ".join(row[:5]))


if __name__ == "__main__":
    main()

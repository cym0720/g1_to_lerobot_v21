# 用途：只读完整 LeRobot v2.1 数据集，逐任务加载元数据、全部 Parquet 行与所有视频字段的每一帧；不修改任何数据。
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()
    ]


def decode_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))


def check_episode(root: Path, episode: dict, cameras: tuple[str, ...]) -> dict:
    index = int(episode["episode_index"])
    expected = int(episode["length"])
    parquet = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
    table = pq.read_table(parquet)  # Reads every column and row, not metadata only.
    errors = []
    if table.num_rows != expected:
        errors.append(f"parquet_rows={table.num_rows},expected={expected}")
    for column, value in (("episode_index", index), ("task_index", 0)):
        values = table[column].to_pylist()
        if any(x != value for x in values):
            errors.append(f"{column}_mismatch")
    frame_index = table["frame_index"].to_pylist()
    if frame_index != list(range(expected)):
        errors.append("frame_index_not_contiguous")
    for camera in cameras:
        video = root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
        frames = decode_count(video)  # Fully decodes every frame through PyAV.
        if frames != expected:
            errors.append(f"{camera}_frames={frames},expected={expected}")
    return {
        "episode_index": index,
        "task": episode["tasks"][0] if episode.get("tasks") else "",
        "frames": expected,
        "status": ";".join(errors) or "ok",
    }


def check_episode_job(job: tuple[Path, dict, tuple[str, ...]]) -> dict:
    return check_episode(*job)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only full LeRobot v2.1 integrity test.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in sorted(p for p in args.input.iterdir() if (p / "meta/info.json").is_file()):
        info = json.loads((root / "meta/info.json").read_text(encoding="utf8"))
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root}: not v2.1")
        task_rows = read_jsonl(root / "meta/tasks.jsonl")
        episodes = read_jsonl(root / "meta/episodes.jsonl")
        if len(task_rows) != 1:
            raise ValueError(f"{root}: expected one task, got {len(task_rows)}")
        if any(ep.get("tasks") != [task_rows[0]["task"]] for ep in episodes):
            raise ValueError(f"{root}: episode task map mismatch")
        cameras = tuple(key for key, value in info["features"].items() if value["dtype"] == "video")
        jobs = [(root, episode, cameras) for episode in episodes]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            checked = list(pool.map(check_episode_job, jobs))
        for item in checked:
            item["task_directory"] = root.name
        rows.extend(checked)
        print(
            f'{root.name}: task={task_rows[0]["task"]!r}, episodes={len(checked)}, bad={sum(x["status"]!="ok" for x in checked)}'
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["task_directory", "episode_index", "task", "frames", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)
    bad = sum(row["status"] != "ok" for row in rows)
    print(f"total_episodes={len(rows)} bad={bad} report={args.report}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

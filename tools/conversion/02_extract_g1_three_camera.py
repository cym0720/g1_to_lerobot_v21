# Purpose: copy a LeRobot v2.1 G1 dataset into an independent three-camera training subset without altering the source dataset.
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.lerobot_v21 import episodes, tasks

CAMERAS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an independent head + left/right wrist LeRobot v2.1 subset.")
    parser.add_argument("--input", type=Path, required=True, help="Source LeRobot v2.1 dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="New output root; must not exist.")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    roots = tasks(args.input)
    if not roots:
        raise SystemExit("input contains no LeRobot v2.1 task directories")
    report = []
    for source_root in roots:
        destination_root = args.output / source_root.name
        info = json.loads((source_root / "meta/info.json").read_text(encoding="utf8"))
        missing = [key for key in CAMERAS if key not in info["features"]]
        if missing:
            raise ValueError(f"{source_root.name}: missing requested camera fields: {missing}")
        for path in (source_root / "data").rglob("*.parquet"):
            copy_file(path, destination_root / path.relative_to(source_root))
        for path in (source_root / "meta").iterdir():
            copy_file(path, destination_root / "meta" / path.name)
        episode_rows = episodes(source_root)
        videos = 0
        for episode in episode_rows:
            index = int(episode["episode_index"])
            for camera in CAMERAS:
                source_video = source_root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
                if not source_video.is_file():
                    raise FileNotFoundError(source_video)
                copy_file(source_video, destination_root / source_video.relative_to(source_root))
                videos += 1
        info["features"] = {key: value for key, value in info["features"].items() if value.get("dtype") != "video" or key in CAMERAS}
        info["total_videos"] = videos
        (destination_root / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
        stats_path = destination_root / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf8"))
        stats = {key: value for key, value in stats.items() if key in info["features"]}
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf8")
        per_episode_path = destination_root / "meta/episodes_stats.jsonl"
        rows = [json.loads(line) for line in per_episode_path.read_text(encoding="utf8").splitlines() if line.strip()]
        for row in rows:
            row["stats"] = {key: value for key, value in row["stats"].items() if key in info["features"]}
        per_episode_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
        report.append({"task": source_root.name, "episodes": len(episode_rows), "videos": videos, "cameras": list(CAMERAS)})
        print(f"{source_root.name}: episodes={len(episode_rows)} videos={videos}", flush=True)
    (args.output / "three_camera_extraction_report.json").write_text(json.dumps({"source": str(args.input), "cameras": list(CAMERAS), "tasks": report}, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()

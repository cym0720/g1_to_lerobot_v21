# 用途（辅助）：从既有 LeRobot 数据集筛除短/缺陷 episode，并保留完整审计清单；主流程已在转换入口按秒数筛除。
from __future__ import annotations
import argparse, csv, shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from packages.lerobot_v21 import (
    tasks,
    episodes,
    copy_dataset_skeleton,
    episode_parquet,
    video_files,
    write_episodes,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-duration-s", type=float, default=8.0, help="按 LeRobot fps 计算的最短轨迹时长（秒）")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = []
    for src in tasks(a.input):
        dst = a.output / src.name
        copy_dataset_skeleton(src, dst)
        fps = float(
            __import__("json").loads((dst / "meta/info.json").read_text(encoding="utf8"))["fps"]
        )
        kept = []
        for e in episodes(dst):
            duration_s = e["length"] / fps
            bad = duration_s < a.min_duration_s
            report.append(
                [
                    src.name,
                    e["episode_index"],
                    e["length"],
                    f"{duration_s:.3f}",
                    "rejected_short" if bad else "kept",
                ]
            )
            if bad:
                reject = (
                    a.output / "rejected_short" / src.name / f"episode_{e['episode_index']:06d}"
                )
                reject.mkdir(parents=True, exist_ok=True)
                for f in [
                    episode_parquet(dst, e["episode_index"]),
                    *video_files(dst, e["episode_index"]),
                ]:
                    if f.exists():
                        shutil.move(str(f), reject / f.name)
            else:
                kept.append(e)
        write_episodes(dst, kept)
    with (a.output / "short_filter_report.csv").open("w", newline="", encoding="utf8") as f:
        csv.writer(f).writerows([["task", "episode", "frames", "duration_s", "result"], *report])


if __name__ == "__main__":
    main()

# 用途（主流程第 02 步）：依据左右机械臂全部关节的运动量，裁剪轨迹首尾多余静止帧并保持多模态时间对齐。
from __future__ import annotations
import argparse, csv, shutil, subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from packages.lerobot_v21 import (
    tasks,
    episodes,
    copy_dataset_skeleton,
    episode_parquet,
    video_files,
    write_episodes,
    table,
)

# qpos20 layout: left arm [0:7], right arm [8:15]; grippers/head/body are excluded.
DEFAULT_ARM_JOINT_INDICES = "0,1,2,3,4,5,6,8,9,10,11,12,13,14"
DEFAULT_MOTION_THRESHOLD = 1e-3
DEFAULT_PADDING_FRAMES = 5
DEFAULT_MIN_MOTION_RUN = 3


def trim_video(path, start, end, fps):
    tmp = path.with_suffix(".trim.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=between(n\\,{start}\\,{end}),setpts=N/({fps}*TB)",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)


def trim_episode(job):
    dst, episode, idx, threshold, padding, min_motion_run, fps = job
    episode_index = int(episode["episode_index"])
    current = table(dst, episode_index)
    q = np.asarray(current["observations.state.qpos"].to_pylist(), dtype=np.float32)
    moving = np.r_[False, np.max(np.abs(np.diff(q[:, idx], axis=0)), axis=1) > threshold]
    valid = np.convolve(moving.astype(int), np.ones(min_motion_run, dtype=int), "same") >= min_motion_run
    if not valid.any():
        return episode, [dst.name, episode_index, len(q), "all_static_skip"]
    start = max(0, int(np.flatnonzero(valid)[0]) - padding)
    end = min(len(q) - 1, int(np.flatnonzero(valid)[-1]) + padding)
    trimmed = current.slice(start, end - start + 1)
    length = end - start + 1
    for column, values in (("timestamp", pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32())), ("frame_index", pa.array(np.arange(length, dtype=np.int64)))):
        trimmed = trimmed.set_column(trimmed.schema.get_field_index(column), column, values)
    pq.write_table(trimmed, episode_parquet(dst, episode_index), compression="snappy")
    for video in video_files(dst, episode_index):
        trim_video(video, start, end, fps)
    updated = dict(episode)
    updated.update({"length": length, "dataset_from_index": 0, "dataset_to_index": length, "trim_source_start": start, "trim_source_end": end})
    return updated, [dst.name, episode_index, len(q), f"{start}:{end}"]


def main():
    started=time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm-joint-indices", default=DEFAULT_ARM_JOINT_INDICES)
    p.add_argument("--threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    p.add_argument("--padding-frames", type=int, default=DEFAULT_PADDING_FRAMES)
    p.add_argument("--min-motion-run", type=int, default=DEFAULT_MIN_MOTION_RUN)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    idx = np.array([int(x) for x in a.arm_joint_indices.split(",")])
    a.output.mkdir(parents=True, exist_ok=False)
    report = []
    for src in tasks(a.input):
        dst = a.output / src.name
        copy_dataset_skeleton(src, dst)
        fps = float(__import__("json").loads((dst / "meta/info.json").read_text())["fps"])
        jobs = [(dst, episode, idx, a.threshold, a.padding_frames, a.min_motion_run, fps) for episode in episodes(dst)]
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
            results = list(pool.map(trim_episode, jobs))
        new = [result[0] for result in results]
        report.extend(result[1] for result in results)
        # Rebuild task-local global indices after variable-length trims.
        offset = 0
        for e in new:
            path = episode_parquet(dst, e["episode_index"])
            current = pq.read_table(path)
            length = current.num_rows
            current = current.set_column(
                current.schema.get_field_index("index"),
                "index",
                pa.array(np.arange(length, dtype=np.int64) + offset),
            )
            pq.write_table(current, path, compression="snappy")
            e["dataset_from_index"] = offset
            e["dataset_to_index"] = offset + length
            offset += length
        info = __import__("json").loads((dst / "meta/info.json").read_text())
        info["total_frames"] = offset
        info["total_episodes"] = len(new)
        info["total_videos"] = 3 * len(new)
        (dst / "meta/info.json").write_text(
            __import__("json").dumps(info, indent=2), encoding="utf8"
        )
        write_episodes(dst, new)
    with (a.output / "trim_report.csv").open("w", newline="", encoding="utf8") as f:
        csv.writer(f).writerows([["task", "episode", "source_frames", "result"], *report])
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}", flush=True)


if __name__ == "__main__":
    main()

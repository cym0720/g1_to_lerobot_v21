# 用途（主流程第 02 步）：依据左右机械臂全部关节的运动量，裁剪轨迹首尾多余静止帧并保持多模态时间对齐。
from __future__ import annotations
import argparse, csv, shutil, subprocess
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from common_lerobot import (
    tasks,
    episodes,
    copy_dataset_skeleton,
    episode_parquet,
    video_files,
    write_episodes,
    table,
)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm-joint-indices", required=True)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--padding-frames", type=int, default=5)
    p.add_argument("--min-motion-run", type=int, default=3)
    a = p.parse_args()
    idx = np.array([int(x) for x in a.arm_joint_indices.split(",")])
    a.output.mkdir(parents=True, exist_ok=False)
    report = []
    for src in tasks(a.input):
        dst = a.output / src.name
        copy_dataset_skeleton(src, dst)
        fps = float(__import__("json").loads((dst / "meta/info.json").read_text())["fps"])
        new = []
        for e in episodes(dst):
            t = table(dst, e["episode_index"])
            q = np.asarray(t["observation.state.joint_position"].to_pylist(), dtype=np.float32)
            moving = np.r_[False, np.max(np.abs(np.diff(q[:, idx], axis=0)), axis=1) > a.threshold]
            # A short isolated movement is noise: require a run of valid movement frames.
            valid = (
                np.convolve(moving.astype(int), np.ones(a.min_motion_run, dtype=int), "same")
                >= a.min_motion_run
            )
            if not valid.any():
                report.append([src.name, e["episode_index"], len(q), "all_static_skip"])
                new.append(e)
                continue
            start = max(0, int(np.flatnonzero(valid)[0]) - a.padding_frames)
            end = min(len(q) - 1, int(np.flatnonzero(valid)[-1]) + a.padding_frames)
            trimmed=t.slice(start, end - start + 1)
            length=end-start+1
            # The output video begins at frame/time zero, so its corresponding
            # parquet columns must use the same new CFR clock.
            for column, values in (
                ('timestamp', pa.array(np.arange(length,dtype=np.float32)/fps, type=pa.float32())),
                ('frame_index', pa.array(np.arange(length,dtype=np.int64))),
            ):
                trimmed=trimmed.set_column(trimmed.schema.get_field_index(column),column,values)
            pq.write_table(
                trimmed,
                episode_parquet(dst, e["episode_index"]),
                compression="snappy",
            )
            for v in video_files(dst, e["episode_index"]):
                trim_video(v, start, end, fps)
            e.update(
                {
                    "length": length,
                    "dataset_from_index": 0,
                    "dataset_to_index": end - start + 1,
                    "trim_source_start": start,
                    "trim_source_end": end,
                }
            )
            new.append(e)
            report.append([src.name, e["episode_index"], len(q), f"{start}:{end}"])
        # Rebuild task-local global indices after variable-length trims.
        offset=0
        for e in new:
            path=episode_parquet(dst,e['episode_index']); current=pq.read_table(path); length=current.num_rows
            current=current.set_column(current.schema.get_field_index('index'),'index',pa.array(np.arange(length,dtype=np.int64)+offset))
            pq.write_table(current,path,compression='snappy')
            e['dataset_from_index']=offset; e['dataset_to_index']=offset+length
            offset += length
        info=__import__('json').loads((dst/'meta/info.json').read_text())
        info['total_frames']=offset; info['total_episodes']=len(new); info['total_videos']=3*len(new)
        (dst/'meta/info.json').write_text(__import__('json').dumps(info,indent=2),encoding='utf8')
        write_episodes(dst, new)
    with (a.output / "trim_report.csv").open("w", newline="", encoding="utf8") as f:
        csv.writer(f).writerows([["task", "episode", "source_frames", "result"], *report])


if __name__ == "__main__":
    main()

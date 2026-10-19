#!/usr/bin/env python3
"""CPU-only native decoder timing and exact timestamps/pixels comparison."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import functools
import json
from pathlib import Path
import time

import torch
import torchvision
from lerobot.datasets.video_utils import decode_video_frames

ROOT = Path(__file__).resolve().parents[2] / "output/lerobot_act_control_20260929/native_train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "decoder_benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    reader = torchvision.io.VideoReader
    videos = sorted((ROOT / "dataset/videos/chunk-000").glob("*/episode_000000.mp4"))
    references, report = {}, dict(device="CPU", samples={}, modes={})
    for name, backend, threads in (("pyav_default", "pyav", 0), ("pyav_one_thread", "pyav", 1),
                                   ("native_torchcodec", "torchcodec", None)):
        torchvision.io.VideoReader = reader if threads in (0, None) else functools.partial(reader, num_threads=threads)
        rows = []
        for path in videos:
            for frame in (52, 104, 220):
                key = f"{path.parent.name}/{frame}"
                started = time.monotonic()
                try:
                    value = decode_video_frames(path, [frame / 30], tolerance_s=1e-4, backend=backend)[0]
                    seconds = time.monotonic() - started
                    if name == "pyav_default":
                        references[key] = value.clone()
                    difference = torch.abs(value - references[key])
                    row = dict(video=str(path), frame=frame, seconds=seconds, shape=list(value.shape),
                               max_rgb_difference=float(difference.max()), mean_rgb_difference=float(difference.mean()),
                               timestamp_tolerance_passed=True, status="ok")
                except Exception as e:
                    row = dict(video=str(path), frame=frame, seconds=time.monotonic() - started, status="error", error=repr(e))
                rows.append(row)
                print("DECODE_BENCH", name, json.dumps(row), flush=True)
        report["modes"][name] = dict(rows=rows, total_seconds=sum(r["seconds"] for r in rows),
                                     all_timestamp_checks_passed=all(r["status"] == "ok" for r in rows),
                                     all_pixels_identical=all(r.get("max_rgb_difference") == 0 for r in rows))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torchvision.io.VideoReader = reader
    print("NATIVE_DECODER_BENCH_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

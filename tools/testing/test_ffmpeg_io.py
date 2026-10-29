# Purpose: smoke-test timestamp-aligned FFmpeg decode/encode on a real H.265 stream.
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from packages.ffmpeg_io import encode_h265_aligned, video_frame_count

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--timestamps", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--frames", type=int, default=20)
a = p.parse_args()
t = np.asarray(
    [int(x.split()[0]) for x in a.timestamps.read_text().splitlines() if x.strip()], dtype=np.int64
)[: a.frames]
backend = encode_h265_aligned(
    a.source, t, t, a.output, 30, crop={"x": 0, "y": 40, "width": 1280, "height": 720}
)
assert video_frame_count(a.output) == len(t)
print("backend={} frames={} output={}".format(backend, len(t), a.output))

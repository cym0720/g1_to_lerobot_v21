# Purpose: compare PyAV and FFmpeg RGB-statistics decoding on identical local videos; read-only.
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from packages.ffmpeg_io import rgb_video_stats


def load_stats_step():
    path = REPO_ROOT / "src" / "clean_tools" / "05_recompute_metadata_stats.py"
    spec = importlib.util.spec_from_file_location("stats_step", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only PyAV vs FFmpeg RGB-statistics benchmark."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--camera", help="Optional one camera field; default benchmarks all cameras."
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text(encoding="utf8"))
    cameras = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    if args.camera:
        cameras = [args.camera]
    step = load_stats_step()
    rows = []
    for camera in cameras:
        video = (
            args.dataset_root
            / "videos"
            / "chunk-000"
            / camera
            / "episode_{:06d}.mp4".format(args.episode)
        )
        start = time.perf_counter()
        old = step.video_stats_pyav(video)
        pyav_s = time.perf_counter() - start
        start = time.perf_counter()
        new = rgb_video_stats(video)
        ffmpeg_s = time.perf_counter() - start
        if old["count"] != new["count"]:
            raise RuntimeError(
                "frame mismatch for {}: {} vs {}".format(camera, old["count"], new["count"])
            )
        max_mean_delta = max(
            abs(float(old["mean"][i][0][0]) - float(new["mean"][i][0][0])) for i in range(3)
        )
        rows.append(
            {
                "camera": camera,
                "frames": old["count"][0],
                "pyav_seconds": pyav_s,
                "ffmpeg_seconds": ffmpeg_s,
                "speedup_pyav_over_ffmpeg": pyav_s / ffmpeg_s if ffmpeg_s else 0.0,
                "max_mean_delta": max_mean_delta,
            }
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2), encoding="utf8")
    for row in rows:
        print(
            "{camera}: frames={frames} pyav={pyav_seconds:.3f}s ffmpeg={ffmpeg_seconds:.3f}s speedup={speedup_pyav_over_ffmpeg:.2f}x delta={max_mean_delta:.3g}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()

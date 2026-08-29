# Purpose: compare startup action jitter before/after low-pass filtering; read-only.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.lerobot_v21 import episodes, episode_parquet, tasks


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare startup action jitter before and after filtering.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=25)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for before_root in tasks(args.before):
        after_root = args.after / before_root.name
        for episode in episodes(before_root):
            index = int(episode["episode_index"])
            before = pq.read_table(episode_parquet(before_root, index))
            after = pq.read_table(episode_parquet(after_root, index))
            raw = np.asarray(before["action"].to_pylist(), dtype=np.float64)
            smooth = np.asarray(after["action"].to_pylist(), dtype=np.float64)
            window = min(args.window_frames, len(raw))
            raw_velocity = np.diff(raw[:window], axis=0)
            smooth_velocity = np.diff(smooth[:window], axis=0)
            raw_acceleration = np.diff(raw_velocity, axis=0)
            smooth_acceleration = np.diff(smooth_velocity, axis=0)
            boundary = min(window, len(raw) - 1)
            rows.append({
                "task": before_root.name,
                "episode_index": index,
                "window_frames": window,
                "velocity_rms_before": rms(raw_velocity),
                "velocity_rms_after": rms(smooth_velocity),
                "acceleration_rms_before": rms(raw_acceleration),
                "acceleration_rms_after": rms(smooth_acceleration),
                "boundary_jump_before": rms(raw[boundary] - raw[boundary - 1]),
                "boundary_jump_after": rms(smooth[boundary] - smooth[boundary - 1]),
                "outside_window_max_abs_delta": float(np.max(np.abs(raw[window:] - smooth[window:]))) if window < len(raw) else 0.0,
            })
    summary = {
        "episodes": len(rows),
        "velocity_rms_reduction_percent": 100.0 * (1.0 - np.mean([row["velocity_rms_after"] for row in rows]) / np.mean([row["velocity_rms_before"] for row in rows])),
        "acceleration_rms_reduction_percent": 100.0 * (1.0 - np.mean([row["acceleration_rms_after"] for row in rows]) / np.mean([row["acceleration_rms_before"] for row in rows])),
        "boundary_jump_change_percent": 100.0 * (np.mean([row["boundary_jump_after"] for row in rows]) / np.mean([row["boundary_jump_before"] for row in rows]) - 1.0),
        "outside_window_max_abs_delta": max(row["outside_window_max_abs_delta"] for row in rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

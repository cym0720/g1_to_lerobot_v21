# Step 03: smooth startup action values over preserved pre-roll, motion, and post-roll; state/video/timestamps remain unchanged.
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from packages.filters import smooth_startup_action
from packages.lerobot_v21 import episodes, episode_parquet, tasks

def arrow_array(values: np.ndarray, field: pa.Field) -> pa.Array:
    return pa.array(values.tolist(), type=field.type) if pa.types.is_list(field.type) else pa.array(values.reshape(-1), type=field.type)

def main() -> None:
    parser = argparse.ArgumentParser(description="In-place startup action smoothing for LeRobot v2.1.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--motion-duration-s", type=float, default=0.5)
    parser.add_argument("--cutoff-hz", type=float, default=2.5)
    parser.add_argument("--pre-roll-frames", type=int, default=5)
    parser.add_argument("--post-roll-frames", type=int, default=5)
    args = parser.parse_args()
    if args.motion_duration_s <= 0 or args.cutoff_hz <= 0:
        raise SystemExit("motion-duration-s and cutoff-hz must be positive")
    report = []
    for root in tasks(args.input):
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf8"))
        named = [key for key in info["features"] if key.startswith("action.") and key != "action"]
        for episode in episodes(root):
            path = episode_parquet(root, int(episode["episode_index"]))
            table = pq.read_table(path)
            changed = []
            window = {"window_frames": 0, "pre_roll_frames": 0, "motion_frames": 0, "post_roll_frames": 0}
            for key in named:
                index = table.schema.get_field_index(key)
                if index < 0:
                    continue
                raw = np.asarray(table[key].to_pylist(), dtype=np.float32)
                smooth, window = smooth_startup_action(raw, float(info["fps"]), args.motion_duration_s, args.cutoff_hz, args.pre_roll_frames, args.post_roll_frames)
                table = table.set_column(index, key, arrow_array(smooth, table.schema.field(index)))
                changed.append(key)
            if "action" in table.column_names and named:
                canonical = np.concatenate([np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1) for key in named], axis=1)
                index = table.schema.get_field_index("action")
                table = table.set_column(index, "action", arrow_array(canonical, table.schema.field(index)))
            pq.write_table(table, path, compression="snappy")
            report.append([root.name, episode["episode_index"], window["pre_roll_frames"], window["motion_frames"], window["post_roll_frames"], window["window_frames"], ";".join(changed)])
    with (args.input / "startup_action_smoothing_report.csv").open("w", newline="", encoding="utf8") as file:
        csv.writer(file).writerows([["task", "episode_index", "pre_roll_frames", "motion_frames", "post_roll_frames", "window_frames", "fields"], *report])
if __name__ == "__main__":
    main()

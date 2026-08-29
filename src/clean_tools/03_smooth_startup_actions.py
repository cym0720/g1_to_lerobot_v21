# 用途（主流程第 03 步）：只平滑轨迹开头的遥操作 action 抖动；不改 state、视频、时间戳或帧对齐。
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.clean_tools.common_lerobot import episodes, episode_parquet, tasks


def smooth_start(
    values: np.ndarray, fps: float, duration_s: float, cutoff_hz: float, blend_frames: int
) -> tuple[np.ndarray, int]:
    """Causal EMA low-pass limited to the startup window, then crossfade to raw values."""
    n = min(len(values), max(1, round(fps * duration_s)))
    if n < 3:
        return values, n
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps)
    filtered = values[:n].copy()
    for i in range(1, n):
        filtered[i] = alpha * values[i] + (1.0 - alpha) * filtered[i - 1]
    blend = min(blend_frames, n)
    if blend:
        weights = np.linspace(0.0, 1.0, blend, dtype=np.float32).reshape(
            blend, *([1] * (values.ndim - 1))
        )
        filtered[-blend:] = (1.0 - weights) * filtered[-blend:] + weights * values[n - blend : n]
    output = values.copy()
    output[:n] = filtered
    return output, n


def arrow_array(values: np.ndarray, field: pa.Field) -> pa.Array:
    if pa.types.is_list(field.type):
        return pa.array(values.tolist(), type=field.type)
    return pa.array(values.reshape(-1), type=field.type)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-place startup action smoothing for LeRobot v2.1."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--cutoff-hz", type=float, default=2.5)
    parser.add_argument("--blend-frames", type=int, default=10)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.cutoff_hz <= 0:
        raise SystemExit("duration-s 和 cutoff-hz 必须大于 0")
    report = []
    for root in tasks(args.input):
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf8"))
        fps = float(info["fps"])
        named_actions = [
            key for key in info["features"] if key.startswith("action.") and key != "action"
        ]
        for episode in episodes(root):
            path = episode_parquet(root, int(episode["episode_index"]))
            table = pq.read_table(path)
            changed = []
            for key in named_actions:
                index = table.schema.get_field_index(key)
                if index < 0:
                    continue
                raw = np.asarray(table[key].to_pylist(), dtype=np.float32)
                smooth, count = smooth_start(
                    raw, fps, args.duration_s, args.cutoff_hz, args.blend_frames
                )
                table = table.set_column(index, key, arrow_array(smooth, table.schema.field(index)))
                changed.append(key)
            # The canonical ACT action must equal the concatenated named action streams.
            if "action" in table.column_names and named_actions:
                canonical = np.concatenate(
                    [
                        np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
                        for key in named_actions
                    ],
                    axis=1,
                )
                index = table.schema.get_field_index("action")
                table = table.set_column(
                    index, "action", arrow_array(canonical, table.schema.field(index))
                )
            pq.write_table(table, path, compression="snappy")
            report.append([root.name, episode["episode_index"], count, ";".join(changed)])
    with (args.input / "startup_action_smoothing_report.csv").open(
        "w", newline="", encoding="utf8"
    ) as file:
        csv.writer(file).writerows(
            [["task", "episode_index", "smoothed_frames", "fields"], *report]
        )


if __name__ == "__main__":
    main()

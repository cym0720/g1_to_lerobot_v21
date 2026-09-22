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

# 14 joint-position dimensions: left arm 0-6, right arm 7-13
JOINT_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow_pitch", "left_wrist_roll", "left_wrist_yaw", "left_gripper",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow_pitch", "right_wrist_roll", "right_wrist_yaw", "right_gripper",
]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def save_comparison_plot(path: Path, title: str, raw: np.ndarray, smooth: np.ndarray) -> None:
    """Write one dependency-light PNG; Pillow is imported only in benchmark plot mode."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(image)
    try:
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 23)
    except OSError:
        label_font = title_font = ImageFont.load_default()
    panels = [
        ("action", raw, smooth),
        ("first difference (velocity proxy)", np.diff(raw), np.diff(smooth)),
        ("second difference (jitter proxy)", np.diff(raw, n=2), np.diff(smooth, n=2)),
    ]
    draw.text((45, 18), title, fill="#111111", font=title_font)
    for panel_index, (name, before, after) in enumerate(panels):
        left, top, right, bottom = 75, 75 + panel_index * 270, 1360, 300 + panel_index * 270
        values = np.concatenate((before, after))
        low, high = float(values.min()), float(values.max())
        margin = max((high - low) * 0.08, 1e-9)
        low, high = low - margin, high + margin
        for tick in range(5):
            y = top + tick * (bottom - top) / 4
            draw.line((left, y, right, y), fill="#dddddd")
        for data, color in ((before, "#2878b5"), (after, "#e45756")):
            points = [
                (left + i * (right - left) / max(1, len(data) - 1), bottom - (float(value) - low) * (bottom - top) / (high - low))
                for i, value in enumerate(data)
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
        draw.rectangle((left, top, right, bottom), outline="#555555", width=2)
        draw.text((left, top - 27), name, fill="#222222", font=label_font)
    draw.line((1040, 42, 1080, 42), fill="#2878b5", width=5)
    draw.text((1090, 31), "raw", fill="#222222", font=label_font)
    draw.line((1190, 42, 1230, 42), fill="#e45756", width=5)
    draw.text((1240, 31), "filtered", fill="#222222", font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_joint_panel_plot(path: Path, title: str, joint_names: list[str], raw: np.ndarray, smooth: np.ndarray) -> None:
    """Plot each joint in its own stacked panel; raw=blue, filtered=red.

    Parameters
    ----------
    raw, smooth : ndarray of shape (T, n_joints)
    """
    from PIL import Image, ImageDraw, ImageFont

    n_joints = len(joint_names)
    panel_height = 120
    header = 60
    footer = 40
    width = 1600
    height = header + n_joints * panel_height + footer
    left_margin = 200
    right_margin = 40
    top_margin = header

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
        joint_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except OSError:
        label_font = title_font = joint_font = ImageFont.load_default()

    draw.text((45, 15), title, fill="#111111", font=title_font)

    # legend
    lx = width - right_margin - 260
    draw.line((lx, 18, lx + 30, 18), fill="#2878b5", width=4)
    draw.text((lx + 38, 8), "raw", fill="#222222", font=label_font)
    draw.line((lx + 110, 18, lx + 140, 18), fill="#e45756", width=4)
    draw.text((lx + 148, 8), "filtered", fill="#222222", font=label_font)

    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left

    for ji, jname in enumerate(joint_names):
        top = top_margin + ji * panel_height
        bottom = top + panel_height - 8

        raw_col = raw[:, ji].astype(np.float64)
        smooth_col = smooth[:, ji].astype(np.float64)
        vals = np.concatenate((raw_col, smooth_col))
        lo, hi = float(vals.min()), float(vals.max())
        margin = max((hi - lo) * 0.1, 1e-9)
        lo, hi = lo - margin, hi + margin

        # grid
        for tick in range(5):
            y = top + tick * (bottom - top) / 4
            draw.line((plot_left, y, plot_right, y), fill="#eeeeee")

        # curves
        n = len(raw_col)
        for data, color in ((raw_col, "#2878b5"), (smooth_col, "#e45756")):
            points = [
                (plot_left + i * plot_width / max(1, n - 1), bottom - (float(v) - lo) * (bottom - top) / (hi - lo))
                for i, v in enumerate(data)
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)

        # panel border
        draw.rectangle((plot_left, top, plot_right, bottom), outline="#cccccc", width=1)

        # joint label
        draw.text((8, top + (bottom - top) // 2 - 10), jname, fill="#333333", font=joint_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _episode_index_map(root: Path) -> dict[str, int]:
    """Map original_episode_id → LeRobot episode_index for a task dataset."""
    mapping: dict[str, int] = {}
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            original = entry.get("original_episode_id", "")
            idx = entry.get("episode_index")
            if original and idx is not None:
                mapping[original] = int(idx)
    return mapping


def _load_trim_offsets(after_dir: Path) -> dict[tuple[str, int], int]:
    """Load trim start offsets from step02 report.  Returns {(task, clean_idx): raw_start_frame}."""
    report_path = after_dir / "trim_report.csv"
    offsets: dict[tuple[str, int], int] = {}
    if not report_path.is_file():
        return offsets
    import csv
    with report_path.open(newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            task = row["task"]
            episode = int(row["episode"])
            result = row["result"]
            if ":" in result:
                start = int(result.split(":")[0])
                offsets[(task, episode)] = start
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare startup action jitter before and after filtering.")
    parser.add_argument("--before", type=Path, required=True, help="Raw converted dataset (step 00 output).")
    parser.add_argument("--after", type=Path, required=True, help="Cleaned dataset (step 02+03 output).")
    parser.add_argument("--window-frames", type=int, default=25)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, help="Optional benchmark-only PNG output directory.")
    parser.add_argument("--joint-plots-dir", type=Path, help="Optional directory for per-hand 14-joint panel plots.")
    args = parser.parse_args()
    trim_offsets = _load_trim_offsets(args.after)
    rows = []
    for after_root in tasks(args.after):
        before_root = args.before / after_root.name
        if not before_root.is_dir():
            print(f"SKIP {after_root.name}: no matching raw task")
            continue
        raw_map = _episode_index_map(before_root)
        for episode in episodes(after_root):
            clean_idx = int(episode["episode_index"])
            original_id = episode.get("original_episode_id", "")
            raw_idx = raw_map.get(original_id)
            if raw_idx is None:
                print(f"SKIP {after_root.name} episode {clean_idx} ({original_id}): not found in raw")
                continue
            before = pq.read_table(episode_parquet(before_root, raw_idx))
            after = pq.read_table(episode_parquet(after_root, clean_idx))
            raw_full = np.asarray(before["action"].to_pylist(), dtype=np.float64)
            smooth = np.asarray(after["action"].to_pylist(), dtype=np.float64)
            # Align: use trim offset to start raw at the same physical frame as cleaned
            start = trim_offsets.get((after_root.name, clean_idx), 0)
            raw = raw_full[start:]
            window = min(args.window_frames, len(raw), len(smooth))
            raw_velocity = np.diff(raw[:window], axis=0)
            smooth_velocity = np.diff(smooth[:window], axis=0)
            raw_acceleration = np.diff(raw_velocity, axis=0)
            smooth_acceleration = np.diff(smooth_velocity, axis=0)
            channel = int(np.argmax(np.sqrt(np.mean(raw_acceleration ** 2, axis=0)) - np.sqrt(np.mean(smooth_acceleration ** 2, axis=0))))
            if args.plots_dir:
                save_comparison_plot(
                    args.plots_dir / after_root.name / f"episode_{clean_idx:06d}.png",
                    f"{after_root.name} / episode {clean_idx:06d} / action channel {channel}",
                    raw[:window, channel],
                    smooth[:window, channel],
                )
            if args.joint_plots_dir:
                n_joints = len(JOINT_NAMES)
                raw_joints = raw[:window, :n_joints]
                smooth_joints = smooth[:window, :n_joints]
                ep_dir = args.joint_plots_dir / after_root.name
                save_joint_panel_plot(
                    ep_dir / f"episode_{clean_idx:06d}_left_arm.png",
                    f"{after_root.name} / episode {clean_idx:06d} — left arm (7 joints)",
                    JOINT_NAMES[:7],
                    raw_joints[:, :7],
                    smooth_joints[:, :7],
                )
                save_joint_panel_plot(
                    ep_dir / f"episode_{clean_idx:06d}_right_arm.png",
                    f"{after_root.name} / episode {clean_idx:06d} — right arm (7 joints)",
                    JOINT_NAMES[7:],
                    raw_joints[:, 7:],
                    smooth_joints[:, 7:],
                )
            boundary = min(window, len(raw) - 1)
            rows.append({
                "task": after_root.name,
                "episode_index": clean_idx,
                "original_episode_id": original_id,
                "window_frames": window,
                "plotted_action_channel": channel,
                "velocity_rms_before": rms(raw_velocity),
                "velocity_rms_after": rms(smooth_velocity),
                "acceleration_rms_before": rms(raw_acceleration),
                "acceleration_rms_after": rms(smooth_acceleration),
                "boundary_jump_before": rms(raw[boundary] - raw[boundary - 1]),
                "boundary_jump_after": rms(smooth[boundary] - smooth[boundary - 1]),
                "outside_window_max_abs_delta": float(np.max(np.abs(raw[window:len(smooth)] - smooth[window:]))) if window < len(smooth) and len(raw) >= len(smooth) else 0.0,
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

# Purpose: copy a LeRobot v2.1 G1 dataset into an independent three-camera training subset without altering the source dataset.
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.lerobot_v21 import episodes, tasks

CAMERAS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def append_robot_velocity(source: Path, destination: Path) -> None:
    """Append the recorded two-axis base command to the canonical action."""
    data = pq.read_table(source)
    required = ("action", "action.robot_velocity")
    missing = [key for key in required if key not in data.column_names]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    action = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    velocity = np.asarray(data["action.robot_velocity"].to_pylist(), dtype=np.float32)
    if action.ndim != 2 or velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(
            f"{source}: expected action [N,D] and robot velocity [N,2], got "
            f"{action.shape} and {velocity.shape}"
        )
    if len(action) != len(velocity):
        raise ValueError(f"{source}: action/velocity frame count mismatch")
    merged = np.concatenate((action, velocity), axis=1)
    column = pa.array(merged.tolist(), type=pa.list_(pa.float32(), merged.shape[1]))
    data = data.set_column(data.schema.get_field_index("action"), "action", column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, destination, compression="snappy")


def rebuild_g1_canonical_20(source: Path, destination: Path) -> None:
    """Build G1's canonical 20-D vectors from named arm, gripper, head and waist fields."""
    data = pq.read_table(source)
    required = (
        "observation.state.joint_position",
        "observation.state.left_effector_position",
        "observation.state.right_effector_position",
        "observation.state.head_position",
        "observation.state.waist_position",
        "action.joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
    )
    missing = [key for key in required if key not in data.column_names]
    if missing:
        raise ValueError(f"{source}: missing canonical G1 fields: {missing}")
    def vector(key: str) -> np.ndarray:
        return np.asarray(data[key].to_pylist(), dtype=np.float32).reshape(len(data), -1)
    state = np.concatenate((
        vector("observation.state.joint_position"),
        vector("observation.state.left_effector_position"),
        vector("observation.state.right_effector_position"),
        vector("observation.state.head_position"),
        vector("observation.state.waist_position"),
    ), axis=1)
    # The recorder's action head/waist streams are often absent.  The prior
    # G1 convention uses observed head position and zero waist targets here.
    action = np.concatenate((
        vector("action.joint_position"),
        vector("action.left_effector_position"),
        vector("action.right_effector_position"),
        vector("observation.state.head_position"),
        np.zeros((len(data), 2), dtype=np.float32),
    ), axis=1)
    if state.shape[1] != 20 or action.shape[1] != 20:
        raise ValueError(f"{source}: expected canonical 20-D vectors, got {state.shape}/{action.shape}")
    for key, values in (("observation.state", state), ("action", action)):
        column = pa.array(values.tolist(), type=pa.list_(pa.float32(), 20))
        data = data.set_column(data.schema.get_field_index(key), key, column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, destination, compression="snappy")


def normalize_gripper_state(source: Path, destination: Path) -> None:
    """Convert closing encoder to opening: 120 (closed) -> 0, 0 -> 1."""
    data = pq.read_table(source)
    fields = (
        "observation.state",
        "observation.state.left_effector_position",
        "observation.state.right_effector_position",
    )
    missing = [key for key in fields if key not in data.column_names]
    if missing:
        raise ValueError(f"{source}: missing gripper-state columns: {missing}")
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 2:
        raise ValueError(f"{source}: unexpected observation.state shape {state.shape}")
    state[:, -2:] = 1.0 - np.clip(state[:, -2:] / 120.0, 0.0, 1.0)
    state_column = pa.array(state.tolist(), type=pa.list_(pa.float32(), state.shape[1]))
    data = data.set_column(data.schema.get_field_index("observation.state"), "observation.state", state_column)
    for key in fields[1:]:
        values = np.asarray(data[key].to_pylist(), dtype=np.float32).reshape(-1)
        values = 1.0 - np.clip(values / 120.0, 0.0, 1.0)
        column = pa.array(values.tolist(), type=pa.float32())
        data = data.set_column(data.schema.get_field_index(key), key, column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, destination, compression="snappy")


def append_robot_velocity_to_state(source: Path, destination: Path) -> None:
    """Expose recorded base [vx, wz] commands as the final two state values."""
    data = pq.read_table(source)
    required = ("observation.state", "action.robot_velocity")
    missing = [key for key in required if key not in data.column_names]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    velocity = np.asarray(data["action.robot_velocity"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or velocity.ndim != 2 or velocity.shape != (len(state), 2):
        raise ValueError(f"{source}: invalid state/robot velocity shapes {state.shape}/{velocity.shape}")
    state = np.concatenate((state, velocity), axis=1)
    state_column = pa.array(state.tolist(), type=pa.list_(pa.float32(), state.shape[1]))
    data = data.set_column(data.schema.get_field_index("observation.state"), "observation.state", state_column)
    velocity_column = pa.array(velocity.tolist(), type=pa.list_(pa.float32(), 2))
    if "observation.state.robot_velocity" in data.column_names:
        data = data.set_column(data.schema.get_field_index("observation.state.robot_velocity"), "observation.state.robot_velocity", velocity_column)
    else:
        data = data.append_column("observation.state.robot_velocity", velocity_column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, destination, compression="snappy")


def extract_episode(job: tuple[Path, Path, int, bool, bool, bool, bool]) -> int:
    source_root, destination_root, index, canonical_g1_20, append_velocity, normalize_gripper, append_velocity_state = job
    source_parquet = source_root / "data/chunk-000" / f"episode_{index:06d}.parquet"
    destination_parquet = destination_root / source_parquet.relative_to(source_root)
    if canonical_g1_20 or append_velocity or normalize_gripper or append_velocity_state:
        if canonical_g1_20:
            rebuild_g1_canonical_20(source_parquet, destination_parquet)
            working_parquet = destination_parquet
        else:
            working_parquet = source_parquet
        if append_velocity:
            append_robot_velocity(working_parquet, destination_parquet)
            working_parquet = destination_parquet
        else:
            working_parquet = source_parquet
        if normalize_gripper:
            normalize_gripper_state(working_parquet, destination_parquet)
            working_parquet = destination_parquet
        if append_velocity_state:
            append_robot_velocity_to_state(working_parquet, destination_parquet)
    else:
        copy_file(source_parquet, destination_parquet)
    for camera in CAMERAS:
        source_video = source_root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        copy_file(source_video, destination_root / source_video.relative_to(source_root))
    return index


def vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
    }


def combined_vector_stats(parts: list[dict]) -> dict:
    count = 0
    mean = m2 = lo = hi = None
    for part in parts:
        n = int(part["count"][0])
        part_mean = np.asarray(part["mean"], dtype=np.float64)
        part_std = np.asarray(part["std"], dtype=np.float64)
        part_lo = np.asarray(part["min"], dtype=np.float64)
        part_hi = np.asarray(part["max"], dtype=np.float64)
        if count == 0:
            count, mean, m2, lo, hi = n, part_mean, part_std**2 * n, part_lo, part_hi
            continue
        total = count + n
        delta = part_mean - mean
        m2 = m2 + part_std**2 * n + delta**2 * count * n / total
        mean = mean + delta * n / total
        count, lo, hi = total, np.minimum(lo, part_lo), np.maximum(hi, part_hi)
    return {"min": lo.tolist(), "max": hi.tolist(), "mean": mean.tolist(), "std": np.sqrt(m2 / count).tolist(), "count": [count]}


def main() -> None:
    started=time.perf_counter()
    parser = argparse.ArgumentParser(description="Create an independent head + left/right wrist LeRobot v2.1 subset.")
    parser.add_argument("--input", type=Path, required=True, help="Source LeRobot v2.1 dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="New output root; must not exist.")
    parser.add_argument("--max-episodes-per-task", type=int, help="Copy only the first N episodes of each task (for smoke tests).")
    parser.add_argument("--append-robot-velocity", action="store_true", help="Append action.robot_velocity [vx, wz] to canonical action.")
    parser.add_argument("--normalize-gripper-state", action="store_true", help="Map gripper state from 0..120 mm to 0..1.")
    parser.add_argument("--append-robot-velocity-to-state", action="store_true", help="Append recorded [vx, wz] to canonical state.")
    parser.add_argument("--canonical-g1-20", action="store_true", help="Rebuild canonical state/action as G1's arm/gripper/head/waist 20-D vectors.")
    parser.add_argument("--workers", type=int, default=4, help="Process workers for independent episodes.")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task < 1:
        raise SystemExit("--max-episodes-per-task must be positive")
    roots = tasks(args.input)
    if not roots:
        raise SystemExit("input contains no LeRobot v2.1 task directories")
    report = []
    for source_root in roots:
        destination_root = args.output / source_root.name
        info = json.loads((source_root / "meta/info.json").read_text(encoding="utf8"))
        episode_rows = episodes(source_root)
        if not episode_rows:
            print(f"{source_root.name}: skipped empty task", flush=True)
            continue
        missing = [key for key in CAMERAS if key not in info["features"]]
        if missing:
            raise ValueError(f"{source_root.name}: missing requested camera fields: {missing}")
        if args.max_episodes_per_task is not None:
            episode_rows = episode_rows[: args.max_episodes_per_task]
        jobs = [(source_root, destination_root, int(episode["episode_index"]), args.canonical_g1_20, args.append_robot_velocity, args.normalize_gripper_state, args.append_robot_velocity_to_state) for episode in episode_rows]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(extract_episode, jobs))
        for path in (source_root / "meta").iterdir():
            copy_file(path, destination_root / "meta" / path.name)
        (destination_root / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in episode_rows),
            encoding="utf8",
        )
        videos = 0
        videos = len(episode_rows) * len(CAMERAS)
        info["features"] = {key: value for key, value in info["features"].items() if value.get("dtype") != "video" or key in CAMERAS}
        if args.canonical_g1_20:
            info["features"]["observation.state"]["shape"] = [20]
            info["features"]["action"]["shape"] = [20]
        if args.append_robot_velocity:
            action_feature = info["features"]["action"]
            previous_shape = action_feature["shape"]
            if len(previous_shape) != 1:
                raise ValueError(f"{source_root.name}: unexpected action shape {previous_shape}")
            action_feature["shape"] = [int(previous_shape[0]) + 2]
        if args.append_robot_velocity_to_state:
            state_feature = info["features"]["observation.state"]
            previous_shape = state_feature["shape"]
            if len(previous_shape) != 1:
                raise ValueError(f"{source_root.name}: unexpected state shape {previous_shape}")
            state_feature["shape"] = [int(previous_shape[0]) + 2]
            info["features"]["observation.state.robot_velocity"] = {"dtype": "float32", "shape": [2], "names": ["linear_x", "angular_z"]}
        info["total_videos"] = videos
        info["total_episodes"] = len(episode_rows)
        info["total_frames"] = sum(int(episode["length"]) for episode in episode_rows)
        info["splits"] = {"train": f"0:{len(episode_rows)}"}
        (destination_root / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
        stats_path = destination_root / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf8"))
        stats = {key: value for key, value in stats.items() if key in info["features"]}
        per_episode_path = destination_root / "meta/episodes_stats.jsonl"
        rows = [json.loads(line) for line in per_episode_path.read_text(encoding="utf8").splitlines() if line.strip()]
        rows = [row for row in rows if int(row["episode_index"]) in {int(episode["episode_index"]) for episode in episode_rows}]
        for row in rows:
            row["stats"] = {key: value for key, value in row["stats"].items() if key in info["features"]}
        if args.canonical_g1_20 or args.append_robot_velocity or args.normalize_gripper_state or args.append_robot_velocity_to_state:
            for row in rows:
                index = int(row["episode_index"])
                parquet = destination_root / "data/chunk-000" / f"episode_{index:06d}.parquet"
                fields = []
                if args.canonical_g1_20:
                    fields.extend(("observation.state", "action"))
                if args.append_robot_velocity:
                    fields.append("action")
                if args.normalize_gripper_state:
                    fields.extend(("observation.state", "observation.state.left_effector_position", "observation.state.right_effector_position"))
                if args.append_robot_velocity_to_state:
                    fields.extend(("observation.state", "observation.state.robot_velocity"))
                fields = list(dict.fromkeys(fields))
                data = pq.read_table(parquet, columns=fields)
                for key in fields:
                    row["stats"][key] = vector_stats(np.asarray(data[key].to_pylist(), dtype=np.float32))
            changed = []
            if args.canonical_g1_20:
                changed.extend(("observation.state", "action"))
            if args.append_robot_velocity:
                changed.append("action")
            if args.normalize_gripper_state:
                changed.extend(("observation.state", "observation.state.left_effector_position", "observation.state.right_effector_position"))
            if args.append_robot_velocity_to_state:
                changed.extend(("observation.state", "observation.state.robot_velocity"))
            changed = list(dict.fromkeys(changed))
            for key in changed:
                stats[key] = combined_vector_stats([row["stats"][key] for row in rows])
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf8")
        per_episode_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
        report.append({"task": source_root.name, "episodes": len(episode_rows), "videos": videos, "cameras": list(CAMERAS), "action_dim": info["features"]["action"]["shape"][0]})
        print(f"{source_root.name}: episodes={len(episode_rows)} videos={videos}", flush=True)
    (args.output / "three_camera_extraction_report.json").write_text(json.dumps({"source": str(args.input), "cameras": list(CAMERAS), "tasks": report}, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"output={args.output}", flush=True)
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}", flush=True)


if __name__ == "__main__":
    main()

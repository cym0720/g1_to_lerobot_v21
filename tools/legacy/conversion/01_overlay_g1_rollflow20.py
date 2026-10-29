# Purpose: replace a 618 LeRobot v2.1 dataset's canonical state/action with RollFlow's native G1 20D layout, while retaining all named source columns.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.lerobot_v21 import episode_parquet, episodes, tasks


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def nearest(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(values, target, side="left")
    left = np.clip(right - 1, 0, len(values) - 1)
    right = np.clip(right, 0, len(values) - 1)
    return np.where(np.abs(values[right] - target) < np.abs(values[left] - target), right, left)


def h5_series(handle: h5py.File, key: str, target: np.ndarray, width: int) -> np.ndarray:
    base = key.rsplit("/", 1)[0]
    values, stamps = handle.get(key), handle.get(base + "/timestamp")
    if values is None or stamps is None or len(values) == 0 or len(stamps) == 0:
        raise ValueError(f"missing or empty raw source: {key}")
    count = min(len(values), len(stamps))
    return np.asarray(values[:count][nearest(np.asarray(stamps[:count]), target)], dtype=np.float32).reshape(len(target), width)


def raw_g1(raw_episode: Path) -> tuple[np.ndarray, np.ndarray]:
    head_stamps = np.asarray([int(line.split()[0]) for line in (raw_episode / "camera/head_color/head_color.txt").read_text().splitlines() if line.strip()])
    with h5py.File(raw_episode / "record/raw_joints.h5", "r") as handle:
        state_joint = h5_series(handle, "state/joint/position", head_stamps, 14)
        state_waist = h5_series(handle, "state/waist/position", head_stamps, 2)
        state_head = h5_series(handle, "state/head/position", head_stamps, 2)
        state_left = h5_series(handle, "state/left_effector/position", head_stamps, 1)
        state_right = h5_series(handle, "state/right_effector/position", head_stamps, 1)
        action_joint = h5_series(handle, "action/joint/position", head_stamps, 14)
        action_left = h5_series(handle, "action/left_effector/position", head_stamps, 1)
        action_right = h5_series(handle, "action/right_effector/position", head_stamps, 1)
    state = np.concatenate((state_joint, state_waist, state_head, state_left, state_right), axis=1)
    action = np.concatenate((action_joint, state_waist, state_head, action_left, action_right), axis=1)
    return state, action


def locate_window(full: np.ndarray, observed: np.ndarray) -> int:
    """Locate a possibly trimmed LeRobot trajectory in its raw source by 14D joint state."""
    if len(observed) > len(full):
        raise ValueError(f"dataset episode has {len(observed)} frames, raw source has only {len(full)}")
    candidates = np.flatnonzero(np.all(np.isclose(full[:, :14], observed[0, :14], atol=1e-5, rtol=1e-5), axis=1))
    probes = sorted({0, len(observed) // 2, len(observed) - 1})
    for start in candidates:
        if start + len(observed) > len(full):
            continue
        if all(np.allclose(full[start + probe, :14], observed[probe, :14], atol=1e-5, rtol=1e-5) for probe in probes):
            return int(start)
    raise ValueError("could not align LeRobot frames to raw H5 (joint-state probes did not match)")


def stats(values: np.ndarray) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"min": data.min(0).tolist(), "max": data.max(0).tolist(), "mean": data.mean(0).tolist(), "std": data.std(0).tolist(), "count": [int(len(data))]}


def combine(parts: list[dict]) -> dict:
    count = sum(int(part["count"][0]) for part in parts)
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    variances = [np.asarray(part["std"], dtype=np.float64) ** 2 for part in parts]
    weights = np.asarray([part["count"][0] for part in parts], dtype=np.float64)
    mean = sum(value * weight for value, weight in zip(means, weights)) / count
    variance = sum(weight * (var + (value - mean) ** 2) for value, var, weight in zip(means, variances, weights)) / count
    return {"min": np.minimum.reduce([np.asarray(part["min"]) for part in parts]).tolist(), "max": np.maximum.reduce([np.asarray(part["max"]) for part in parts]).tolist(), "mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": [count]}


def list_column(values: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(values, dtype=np.float32).tolist(), type=pa.list_(pa.float32()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay native G1 RollFlow 20D canonical state/action into existing LeRobot v2.1 task datasets.")
    parser.add_argument("--input", type=Path, required=True, help="LeRobot v2.1 dataset root containing task directories; updated in place only with --apply.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw 618data directory containing UUID episode folders.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/contracts/g1_rollflow_20d.json")
    parser.add_argument("--tasks", nargs="*", help="Optional task directory names.")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite Parquet and metadata. Omit for a raw-alignment dry run.")
    args = parser.parse_args()
    spec = read_json(args.config)
    selected = set(args.tasks or [])
    roots = [root for root in tasks(args.input) if not selected or root.name in selected]
    unknown = selected - {root.name for root in roots}
    if unknown:
        raise SystemExit(f"unknown task directories: {sorted(unknown)}")
    report = []
    for root in roots:
        info = read_json(root / "meta/info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root}: not LeRobot v2.1")
        required = set(spec["state_sources"] + spec["action_sources"])
        missing = required - set(info["features"])
        if missing:
            raise ValueError(f"{root}: missing named source columns {sorted(missing)}")
        episode_rows = episodes(root)
        updated_stats: dict[int, dict] = {}
        for episode in episode_rows:
            index = int(episode["episode_index"])
            raw_id = episode.get("original_episode_id")
            raw_path = args.raw_root / str(raw_id)
            if not raw_path.is_dir():
                raise FileNotFoundError(f"{root.name} episode {index}: raw source missing: {raw_path}")
            table = pq.read_table(episode_parquet(root, index))
            named_state = np.asarray(table["observation.state.joint_position"].to_pylist(), dtype=np.float32)
            raw_state, raw_action = raw_g1(raw_path)
            start = locate_window(raw_state, named_state)
            state = raw_state[start:start + len(table)]
            action = raw_action[start:start + len(table)]
            if state.shape != (len(table), spec["dimension"]) or action.shape != state.shape:
                raise AssertionError(f"{root.name} episode {index}: invalid overlay shape")
            report.append({"task": root.name, "episode_index": index, "original_episode_id": raw_id, "raw_start_frame": start, "frames": len(table), "status": "would_update" if not args.apply else "updated"})
            updated_stats[index] = {"observation.state": stats(state), "action": stats(action)}
            if args.apply:
                table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", list_column(state))
                table = table.set_column(table.schema.get_field_index("action"), "action", list_column(action))
                path = episode_parquet(root, index)
                temporary = path.with_suffix(".overlay.tmp.parquet")
                pq.write_table(table, temporary, compression="snappy")
                temporary.replace(path)
        if args.apply:
            for key in ("observation.state", "action"):
                info["features"][key] = {"dtype": "float32", "shape": [spec["dimension"]], "names": spec["names"]}
            (root / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
            stats_rows = [json.loads(line) for line in (root / "meta/episodes_stats.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
            for row in stats_rows:
                row["stats"].update(updated_stats[int(row["episode_index"])])
            (root / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stats_rows), encoding="utf8")
            global_stats = read_json(root / "meta/stats.json")
            for key in ("observation.state", "action"):
                global_stats[key] = combine([updated_stats[int(row["episode_index"])][key] for row in stats_rows])
            (root / "meta/stats.json").write_text(json.dumps(global_stats, ensure_ascii=False, indent=2), encoding="utf8")
    report_path = args.input / "g1_rollflow20_overlay_report.json"
    report_path.write_text(json.dumps({"apply": args.apply, "dimension": spec["dimension"], "episodes": report}, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"mode={'apply' if args.apply else 'dry-run'} episodes={len(report)} report={report_path}")


if __name__ == "__main__":
    main()

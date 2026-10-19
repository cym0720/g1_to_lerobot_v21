"""Restore G1 gripper action values divided by 120 during conversion.

This one-off repair is intentionally limited to the three rebuilt datasets.
It backs up each touched Parquet/statistics file before modifying it and never
changes state.qpos or videos. Run with --apply to make changes; default is a
read-only audit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOTS = (
    REPO_ROOT / "output/rebuild_20260924_g1_retasked/final/0923_g1_pick_apple",
    REPO_ROOT / "output/rebuild_20260924_g1_retasked/final/record_0904",
    REPO_ROOT / "output/rebuild_20260924_g1_retasked/final/618data",
)
GRIPPER_INDICES = (7, 15)
DIVISOR = 120.0
MAX_EXPECTED_BROKEN_VALUE = 1.0 / DIVISOR + 1e-6


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def task_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in dataset_root.rglob("meta/info.json")
        if (path.parent.parent / "data").is_dir()
    )


def load_action(path: Path) -> tuple[pq.ParquetFile, np.ndarray]:
    parquet = pq.ParquetFile(path)
    column = parquet.read(columns=["action.qpos"])["action.qpos"].combine_chunks()
    values = np.asarray(column.values.to_numpy(), dtype=np.float32).reshape(-1, 20)
    return parquet, values


def audit_task(task: Path) -> tuple[list[Path], dict[int, float]]:
    files = sorted((task / "data").rglob("episode_*.parquet"))
    if not files:
        raise RuntimeError(f"no episode Parquet files found under {task}")
    channels: dict[int, list[np.ndarray]] = {index: [] for index in GRIPPER_INDICES}
    for path in files:
        _, action = load_action(path)
        if action.shape[1] != 20 or not np.isfinite(action).all():
            raise RuntimeError(f"invalid action.qpos shape/values in {path}")
        for index in GRIPPER_INDICES:
            channels[index].append(action[:, index])

    maxima: dict[int, float] = {}
    for index, parts in channels.items():
        values = np.concatenate(parts)
        minimum, maximum = float(values.min()), float(values.max())
        if minimum < -1e-7:
            raise RuntimeError(f"action gripper[{index}] has negative value {minimum} in {task}")
        if maximum > MAX_EXPECTED_BROKEN_VALUE:
            raise RuntimeError(
                f"action gripper[{index}] max={maximum:.8g} is not the known /120 signature; "
                f"refusing to alter {task}"
            )
        maxima[index] = maximum
    return files, maxima


def backup_files(task: Path, files: list[Path]) -> Path:
    backup = task / "_gripper_action_repair_backup"
    if backup.exists():
        raise RuntimeError(f"backup already exists; refusing to overwrite: {backup}")
    backup.mkdir()
    for source in files:
        target = backup / source.relative_to(task)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return backup


def rewrite_parquet(path: Path) -> None:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index("action.qpos")
    old = table.column(column_index).combine_chunks()
    actions = np.asarray(old.values.to_numpy(), dtype=np.float32).reshape(-1, 20).copy()
    actions[:, GRIPPER_INDICES] *= np.float32(DIVISOR)
    new_column = pa.array(actions.tolist(), type=table.schema.field(column_index).type)
    updated = table.set_column(column_index, table.schema.field(column_index), new_column)
    temp = path.with_suffix(path.suffix + ".repair-tmp")
    pq.write_table(updated, temp, compression="zstd")
    temp.replace(path)


def refresh_stats(task: Path, files: list[Path]) -> None:
    episode_stats_path = task / "meta/episodes_stats.jsonl"
    episodes = [json.loads(line) for line in episode_stats_path.read_text(encoding="utf-8").splitlines() if line]
    per_episode: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    for path in files:
        _, action = load_action(path)
        per_episode.append(action)
        all_action.append(action)

    if len(episodes) != len(per_episode):
        raise RuntimeError(f"episode stats count does not match Parquet files under {task}")
    for record, action in zip(episodes, per_episode):
        record["stats"]["action.qpos"] = vector_stats(action)
    episode_stats_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes), encoding="utf-8"
    )

    stats_path = task / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["action.qpos"] = vector_stats(np.concatenate(all_action, axis=0))
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the repair; default only audits")
    args = parser.parse_args()

    tasks: list[tuple[Path, list[Path], dict[int, float]]] = []
    for root in DATASET_ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(f"expected dataset root does not exist: {root}")
        for task in task_dirs(root):
            files, maxima = audit_task(task)
            tasks.append((task, files, maxima))
            print(
                f"{root.name}/{task.name}: episodes={len(files)} "
                f"action.gripper[7] max={maxima[7]:.8g}, [15] max={maxima[15]:.8g}"
            )

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}; tasks={len(tasks)}")
    if not args.apply:
        print("No files changed. Re-run with --apply to back up and restore these action fields.")
        return

    for task, files, _ in tasks:
        protected = [*files, task / "meta/stats.json", task / "meta/episodes_stats.jsonl"]
        backup = backup_files(task, protected)
        try:
            for path in files:
                rewrite_parquet(path)
            refresh_stats(task, files)
        except Exception:
            print(f"Repair failed for {task}; original files are backed up at {backup}")
            raise
        print(f"REPAIRED {task}; recoverable backups: {backup}")


if __name__ == "__main__":
    main()

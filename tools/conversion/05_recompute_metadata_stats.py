"""Recompute LeRobot v2.1 episode/global metadata statistics from Parquet rows.

Video statistics keep the converter's conservative normalized-RGB contract; only
their frame counts are refreshed. Numeric statistics are recomputed exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from packages.lerobot_v21 import episodes, task_roots, episode_parquet


def numeric_stats(values: np.ndarray) -> dict:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(len(array))],
    }


def combine(parts: list[dict]) -> dict:
    count = sum(int(part["count"][0]) for part in parts)
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    stds = [np.asarray(part["std"], dtype=np.float64) for part in parts]
    weights = [int(part["count"][0]) for part in parts]
    mean = sum(value * weight for value, weight in zip(means, weights)) / count
    variance = sum(
        (std**2 + (value - mean) ** 2) * weight
        for value, std, weight in zip(means, stds, weights)
    ) / count
    return {
        "min": np.minimum.reduce([np.asarray(p["min"], dtype=np.float64) for p in parts]).tolist(),
        "max": np.maximum.reduce([np.asarray(p["max"], dtype=np.float64) for p in parts]).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [count],
    }


def episode_job(job: tuple[Path, int, tuple[str, ...], tuple[str, ...]]) -> tuple[int, int, dict]:
    root, index, numeric_columns, video_columns = job
    table = pq.read_table(episode_parquet(root, index), columns=list(numeric_columns))
    result = {key: numeric_stats(np.asarray(table[key].to_pylist())) for key in numeric_columns}
    n = table.num_rows
    for key in video_columns:
        result[key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [n],
        }
    return index, n, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="task root or parent of task roots")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-stats-for-val", action="store_true", help="archive val distribution stats and use train stats as val normalization")
    args = parser.parse_args()
    roots = task_roots(args.input)
    if not roots:
        raise SystemExit(f"no LeRobot v2.1 task roots found under {args.input}")
    for root in roots:
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        feature_map = info["features"]
        videos = tuple(key for key, value in feature_map.items() if value["dtype"] == "video")
        columns = tuple(key for key, value in feature_map.items() if value["dtype"] != "video")
        eps = episodes(root)
        jobs = [(root, int(ep["episode_index"]), columns, videos) for ep in eps]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            computed = list(pool.map(episode_job, jobs))
        stats_by_index = {index: stats for index, _, stats in computed}
        episode_rows = [
            {"episode_index": int(ep["episode_index"]), "stats": stats_by_index[int(ep["episode_index"])]}
            for ep in eps
        ]
        all_keys = list(episode_rows[0]["stats"])
        global_stats = {key: combine([row["stats"][key] for row in episode_rows]) for key in all_keys}
        (root / "meta/episodes_stats.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows), encoding="utf-8"
        )
        (root / "meta/stats.json").write_text(
            json.dumps(global_stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        info["statistics_status"] = "recomputed from current parquet rows; video RGB bounds retained from converter contract"
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"STATS_RECOMPUTED {root} episodes={len(eps)} frames={sum(n for _, n, _ in computed)}", flush=True)

    if args.train_stats_for_val:
        train, val = args.input / "train", args.input / "val"
        train_stats_path, val_stats_path = train / "meta/stats.json", val / "meta/stats.json"
        if not train_stats_path.exists() or not val_stats_path.exists():
            raise SystemExit("--train-stats-for-val requires train/ and val/ LeRobot roots")
        diagnostic = val / "meta/validation_distribution_stats.json"
        if not diagnostic.exists():
            shutil.copy2(val_stats_path, diagnostic)
        else:
            print(f"PRESERVING_EXISTING_VALIDATION_DISTRIBUTION {diagnostic}", flush=True)
        shutil.copy2(train_stats_path, val_stats_path)
        digest = hashlib.sha256(train_stats_path.read_bytes()).hexdigest()
        for split_root in (train, val):
            info_path = split_root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["statistics_status"] = "complete"
            info["normalization_statistics_source"] = "train only"
            info["normalization_statistics_sha256"] = digest
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = args.input / "split_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics_status"] = "complete; validation uses train-only normalization"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"VAL_NORMALIZATION_FROM_TRAIN sha256={digest}", flush=True)


if __name__ == "__main__":
    main()

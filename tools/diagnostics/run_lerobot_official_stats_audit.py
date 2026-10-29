"""Run Hugging Face LeRobot's upstream compute_stats implementation on local G1 data.

The official module source is downloaded separately to output/ and checked by SHA256.
Only numeric feature functions are extracted from that module's AST, avoiding imports
of unrelated robot/video runtime dependencies. Episodes are processed and aggregated
using the same official functions used by LeRobotDataset metadata recomputation.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SOURCE = ROOT / "experiments/statistics_audit_20261028/official_compute_stats_main_20261028.py"
OUT = ROOT / "output/diagnostics/statistics_audit_20261028"
DATASETS = {
    "wrist_lr_railfix": ROOT / "output/merged/wrist_lr/1020+1026+1027_only_pick_apple_qpos20_v1_railfix_20261028/train",
    "head_color_fisheye_lr": ROOT / "output/merged/head_color_fisheye_lr/1020+1026+1027_pick_apple_qpos20_v1/train",
}
FEATURES = ("observation.state", "action")


def load_official_functions():
    source = OFFICIAL_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OFFICIAL_SOURCE))
    wanted = {
        "RunningQuantileStats", "_prepare_array_for_stats", "_compute_basic_stats",
        "_reshape_stats_by_axis", "_reshape_for_image_stats", "_reshape_for_vector_stats",
        "_reshape_for_feature_stats", "_reshape_for_global_stats", "_reshape_single_stat",
        "get_feature_stats", "aggregate_feature_stats",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_QUANTILES" for target in node.targets
        ):
            selected.append(node)
    found = {node.name for node in selected if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Official source is missing expected symbols: {sorted(missing)}")
    ns = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(OFFICIAL_SOURCE), "exec"), ns)
    return ns["get_feature_stats"], ns["aggregate_feature_stats"]


def load_rows(root: Path):
    records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))
    groups = defaultdict(lambda: {key: [] for key in FEATURES})
    episode_groups = defaultdict(list)
    source_by_episode = {}
    all_values = {key: [] for key in FEATURES}
    for rec in records:
        idx = int(rec["episode_index"])
        source_batches = rec.get("source_batches") or [rec.get("source_batch", "unknown")]
        if len(source_batches) != 1:
            raise ValueError(f"Episode {idx} has ambiguous source_batches={source_batches}")
        batch = str(source_batches[0])
        source_by_episode[idx] = batch
        path = root / "data" / f"chunk-{idx // chunk_size:03d}" / f"episode_{idx:06d}.parquet"
        table = pq.read_table(path, columns=list(FEATURES))
        for key in FEATURES:
            col = table[key].combine_chunks()
            dim = len(col[0].as_py())
            arr = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, dim)
            if arr.shape[1] != 20 or not np.isfinite(arr).all():
                raise ValueError(f"Bad vector data: {path} {key} shape={arr.shape}")
            groups[batch][key].append(arr)
            all_values[key].append(arr)
            episode_groups[batch].append((idx, key, arr))
    pipeline_episode_rows = [json.loads(line) for line in (root / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    pipeline_batches = defaultdict(lambda: {key: [] for key in FEATURES})
    for row in pipeline_episode_rows:
        batch = source_by_episode[int(row["episode_index"])]
        for key in FEATURES:
            pipeline_batches[batch][key].append(row["stats"][key])
    return groups, episode_groups, all_values, len(records), pipeline_batches


def official_episode_aggregate(arrays, get_feature_stats, aggregate_feature_stats):
    stats_per_episode = [get_feature_stats(arr, axis=0, keepdims=False) for arr in arrays]
    return aggregate_feature_stats(stats_per_episode)


def serializable(stats):
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def pipeline_aggregate(parts):
    counts = np.asarray([int(x["count"][0]) for x in parts], dtype=np.float64)
    total = counts.sum()
    means = np.stack([np.asarray(x["mean"], dtype=np.float64) for x in parts])
    stds = np.stack([np.asarray(x["std"], dtype=np.float64) for x in parts])
    mean = np.average(means, axis=0, weights=counts)
    variance = np.average(stds**2 + (means - mean) ** 2, axis=0, weights=counts)
    return {
        "min": np.min(np.stack([x["min"] for x in parts]), axis=0),
        "max": np.max(np.stack([x["max"] for x in parts]), axis=0),
        "mean": mean,
        "std": np.sqrt(variance),
        "count": np.asarray([total]),
    }


def run_one(name: str, root: Path, get_feature_stats, aggregate_feature_stats):
    groups, _, all_values, total_episodes, pipeline_batches = load_rows(root)
    stats_path = root / "meta/stats.json"
    pipeline = json.loads(stats_path.read_text(encoding="utf-8"))
    batch_out = {}
    for batch in sorted(groups, key=int):
        batch_out[batch] = {"episodes": len(groups[batch][FEATURES[0]]), "features": {}}
        for feature in FEATURES:
            arrays = groups[batch][feature]
            official = official_episode_aggregate(arrays, get_feature_stats, aggregate_feature_stats)
            pipe_stats = pipeline_aggregate(pipeline_batches[batch][feature])
            concat = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
            official_full_frame = get_feature_stats(concat, axis=0, keepdims=False)
            direct = {
                "min": concat.min(0), "max": concat.max(0), "mean": concat.mean(0),
                "std": concat.std(0), "count": np.array([len(concat)]),
                "q01_exact_all_frames": np.quantile(concat, .01, axis=0),
                "q99_exact_all_frames": np.quantile(concat, .99, axis=0),
            }
            common = {k: max_abs_diff(official[k], pipe_stats[k]) for k in ("min", "max", "mean", "std", "count")}
            batch_out[batch]["features"][feature] = {
                "official": serializable(official),
                "official_single_pass_over_all_frames": serializable(official_full_frame),
                "pipeline_episode_stats_aggregate": serializable(pipe_stats),
                "pipeline_direct_frame_stats": serializable(direct),
                "max_abs_diff_official_vs_pipeline": common,
                "official_q99_vs_exact_all_frame_q99_max_abs_diff": max_abs_diff(
                    official["q99"], direct["q99_exact_all_frames"]
                ),
                "official_single_pass_q99_vs_exact_all_frame_q99_max_abs_diff": max_abs_diff(
                    official_full_frame["q99"], direct["q99_exact_all_frames"]
                ),
            }
    overall = {"episodes": total_episodes, "features": {}}
    for feature in FEATURES:
        arrays = [a for batch in sorted(groups, key=int) for a in groups[batch][feature]]
        official = official_episode_aggregate(arrays, get_feature_stats, aggregate_feature_stats)
        concat = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
        official_full_frame = get_feature_stats(concat, axis=0, keepdims=False)
        direct = {
            "min": concat.min(0), "max": concat.max(0), "mean": concat.mean(0),
            "std": concat.std(0), "count": np.array([len(concat)]),
            "q01_exact_all_frames": np.quantile(concat, .01, axis=0),
            "q99_exact_all_frames": np.quantile(concat, .99, axis=0),
        }
        pipe_stat = pipeline[feature]
        overall["features"][feature] = {
            "official": serializable(official),
            "official_single_pass_over_all_frames": serializable(official_full_frame),
            "pipeline_stats_json": pipe_stat,
            "pipeline_direct_frame_stats": serializable(direct),
            "max_abs_diff_official_vs_pipeline": {
                k: max_abs_diff(official[k], pipe_stat[k]) for k in ("min", "max", "mean", "std", "count")
            },
            "official_q99_vs_exact_all_frame_q99_max_abs_diff": max_abs_diff(
                official["q99"], direct["q99_exact_all_frames"]
            ),
            "official_single_pass_q99_vs_exact_all_frame_q99_max_abs_diff": max_abs_diff(
                official_full_frame["q99"], direct["q99_exact_all_frames"]
            ),
        }
    return {"dataset": name, "train_root": str(root), "source_batches": sorted(groups, key=int),
            "train_episodes": total_episodes, "batches": batch_out, "overall": overall}


def main():
    if not OFFICIAL_SOURCE.is_file():
        raise FileNotFoundError(OFFICIAL_SOURCE)
    get_feature_stats, aggregate_feature_stats = load_official_functions()
    results = []
    for name, root in DATASETS.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        results.append(run_one(name, root, get_feature_stats, aggregate_feature_stats))
    OUT.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(OFFICIAL_SOURCE.read_bytes()).hexdigest()
    report = {
        "status": "COMPLETE",
        "official_source": "https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py",
        "official_source_sha256": source_sha,
        "official_implementation": "compute_stats.py RunningQuantileStats + get_feature_stats + aggregate_feature_stats, AST-loaded without modifying implementation",
        "statistic_scope": "state/action numeric features; train split only; episodes grouped by source_batches",
        "q99_note": "Official q01/q99 are 5000-bin histogram approximations computed per episode; the current upstream aggregate uses a conservative envelope (minimum for q<=50, maximum for q>50), not a weighted mean or exact whole-dataset quantile. pipeline_direct_frame_stats q01/q99 are exact NumPy quantiles over all frames.",
        "results": results,
    }
    path = OUT / "official_vs_pipeline_statistics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = OUT / "summary.md"
    lines = ["# LeRobot official statistics vs local pipeline", "", f"Official source SHA256: `{source_sha}`", "",
             "Train state/action only. Common-stat diffs compare official per-episode stats aggregated exactly as LeRobotDataset does. The q99 column compares a single official 5000-bin estimate over all frames with exact all-frame NumPy q99; JSON also preserves the official aggregate envelope values used in stats.json.", ""]
    for ds in results:
        lines += [f"## {ds['dataset']}", "", f"Episodes: {ds['train_episodes']} (batches: {', '.join(ds['source_batches'])})", "",
                  "| Source | Episodes | Frames | Feature | max abs diff min/max/mean/std | official q01/q99 vs exact q01/q99 |",
                  "|---|---:|---:|---|---:|---|"]
        for batch, info in ds["batches"].items():
            for feature, values in info["features"].items():
                frames = int(values["official"]["count"][0])
                diffs = values["max_abs_diff_official_vs_pipeline"]
                v = max(diffs[k] for k in ("min", "max", "mean", "std"))
                # For the batch direct exact q99 is kept in pipeline_direct_frame_stats.
                qdelta = values["official_single_pass_q99_vs_exact_all_frame_q99_max_abs_diff"]
                lines.append(f"| {batch} | {info['episodes']} | {frames} | {feature} | {v:.3g} | {qdelta:.3g} max q99 delta |")
        for feature, values in ds["overall"]["features"].items():
            diffs = values["max_abs_diff_official_vs_pipeline"]
            v = max(diffs[k] for k in ("min", "max", "mean", "std"))
            lines.append(f"| combined | {ds['train_episodes']} | {values['official']['count'][0]} | {feature} | {v:.3g} | {values['official_single_pass_q99_vs_exact_all_frame_q99_max_abs_diff']:.3g} max q99 delta |")
        lines.append("")
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"AUDIT_COMPLETE {path}")
    print(f"SUMMARY {summary}")
    print(f"OFFICIAL_SOURCE_SHA256 {source_sha}")


if __name__ == "__main__":
    main()

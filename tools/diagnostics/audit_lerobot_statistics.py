"""Read-only audit of numeric LeRobot statistics against a train episode manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def values_from_episode(root: Path, info: dict, episode_index: int, keys: list[str]) -> dict[str, np.ndarray]:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(path, columns=keys)
    result = {}
    for key in keys:
        col = table[key].combine_chunks()
        first = col[0].as_py()
        if isinstance(first, (list, tuple)):
            dim = len(first)
            data = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, dim)
        else:
            data = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1)
        if not np.isfinite(data).all():
            raise ValueError(f"non-finite values: episode={episode_index} key={key}")
        result[key] = data
    return result


def stats(data: np.ndarray) -> dict:
    return {
        "count": int(data.shape[0]), "min": data.min(0), "max": data.max(0),
        "mean": data.mean(0), "std": data.std(0, ddof=0),
        "q01": np.quantile(data, 0.01, axis=0), "q99": np.quantile(data, 0.99, axis=0),
    }


def compare(saved: dict, actual: dict) -> dict:
    out = {"saved_count": saved.get("count", [None])[0], "actual_count": actual["count"]}
    for key in ("min", "max", "mean", "std"):
        lhs = np.asarray(saved[key], dtype=np.float64).reshape(-1)
        rhs = np.asarray(actual[key]).reshape(-1)
        if lhs.shape != rhs.shape:
            out[key] = {"shape_match": False, "saved_shape": list(lhs.shape), "actual_shape": list(rhs.shape)}
        else:
            diff = np.abs(lhs - rhs)
            out[key] = {"shape_match": True, "max_abs_error": float(diff.max(initial=0)),
                        "max_relative_error": float((diff / np.maximum(np.abs(rhs), 1e-12)).max(initial=0))}
    return out


def serializable(item):
    if isinstance(item, dict):
        return {k: serializable(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [serializable(v) for v in item]
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, (np.integer, np.floating)):
        return item.item()
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="LeRobot train root")
    parser.add_argument("--heldout", type=Path, help="optional val/test LeRobot root; used only for coverage")
    parser.add_argument("--out", type=Path, required=True, help="new report path; refuses overwrite")
    parser.add_argument("--keys", nargs="*", help="numeric feature names; defaults to all float vectors")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    info = json.loads((args.train / "meta/info.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(args.train / "meta/episodes.jsonl")
    features = info["features"]
    keys = args.keys or [k for k, v in features.items() if v.get("dtype") == "float32" and len(v.get("shape", [])) == 1]
    if not episodes or not keys:
        raise ValueError("train manifest or numeric feature list is empty")
    per_episode_saved = {int(x["episode_index"]): x["stats"] for x in read_jsonl(args.train / "meta/episodes_stats.jsonl")}
    global_saved = json.loads((args.train / "meta/stats.json").read_text(encoding="utf-8"))
    concatenated = {key: [] for key in keys}
    per_episode_report = []
    for record in episodes:
        idx = int(record["episode_index"])
        actual_values = values_from_episode(args.train, info, idx, keys)
        row = {"episode_index": idx, "manifest_frames": int(record["length"])}
        for key in keys:
            arr = actual_values[key]
            concatenated[key].append(arr)
            row[key] = compare(per_episode_saved[idx][key], stats(arr))
        per_episode_report.append(row)

    train_actual = {key: stats(np.concatenate(parts)) for key, parts in concatenated.items()}
    global_report = {key: compare(global_saved[key], train_actual[key]) for key in keys}
    coverage = {}
    if args.heldout:
        held_info = json.loads((args.heldout / "meta/info.json").read_text(encoding="utf-8"))
        held_eps = read_jsonl(args.heldout / "meta/episodes.jsonl")
        held_values = {key: [] for key in keys}
        for record in held_eps:
            vals = values_from_episode(args.heldout, held_info, int(record["episode_index"]), keys)
            for key in keys:
                held_values[key].append(vals[key])
        for key in keys:
            train_stat = train_actual[key]
            held = np.concatenate(held_values[key])
            qlo, qhi = train_stat["q01"], train_stat["q99"]
            coverage[key] = {
                "heldout_count": len(held), "train_q01": qlo, "train_q99": qhi,
                "heldout_below_train_q01_fraction": np.mean(held < qlo, axis=0),
                "heldout_above_train_q99_fraction": np.mean(held > qhi, axis=0),
                "heldout_min": held.min(0), "heldout_max": held.max(0),
            }

    report = {
        "status": "AUDIT_COMPLETE",
        "train_root": str(args.train.resolve()), "heldout_root": str(args.heldout.resolve()) if args.heldout else None,
        "train_episode_count": len(episodes), "train_manifest_frames": sum(int(x["length"]) for x in episodes),
        "feature_keys": keys, "global_numeric_stats_comparison": global_report,
        "heldout_coverage_diagnostic_only": coverage, "per_episode": per_episode_report,
        "std_definition": "population std ddof=0", "source_mutated": False,
        "limitations": ["Video RGB statistics and real training-loader consumption are not inspected by this tool."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(serializable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out.resolve()), "features": keys,
                      "global_numeric_stats_comparison": global_report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

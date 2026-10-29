"""Read-only quality preflight for one G1 raw recording batch.

Checks the named numeric streams, head-clock alignment, required RGB/fisheye
video decodability, source task metadata, and inferred qpos body order. Writes
one manifest plus every rejection reason. No source files are changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.body_order import detect_body_order
from packages.eva_qpos import assemble_g1_qpos

FISHEYE_CAMERAS = ("head_color", "head_left_fisheye", "head_right_fisheye")
WRIST_CAMERAS = ("head_color", "hand_left", "hand_right")

def camera_specs_for(variant: str):
    return ([(x, "h265") for x in FISHEYE_CAMERAS] if variant == "head_color_fisheye_lr" else
            [("head_color", "h265"), ("hand_left", "jpg"), ("hand_right", "jpg")])
FIELDS = {
    "state/joint/position": (14, True),
    "state/left_effector/position": (1, True),
    "state/right_effector/position": (1, True),
    "state/head/position": (2, True),
    "state/waist/position": (2, True),
    "action/joint/position": (14, True),
    "action/left_effector/position": (1, True),
    "action/right_effector/position": (1, True),
    "action/head/position": (2, False),
    "action/waist/position": (2, False),
}


def read_ts(path: Path) -> np.ndarray:
    return np.asarray(
        [int(line.split()[0]) for line in path.read_text(encoding="utf8").splitlines() if line.strip()],
        dtype=np.int64,
    )


def nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(source, target), 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    return np.where(np.abs(source[right] - target) < np.abs(source[left] - target), right, left)


def check_episode(task: tuple[str, str]) -> dict:
    path_text, variant = task
    path = Path(path_text)
    result = {"episode": path.name, "path": str(path), "errors": [], "warnings": [], "action_fallbacks": []}
    err = result["errors"]
    try:
        meta = json.loads((path / "meta_info.json").read_text(encoding="utf8"))
        result["task_id"] = str(meta.get("task_id", ""))
        result["task_text"] = str(meta.get("text", ""))
        result["metadata_duration_s"] = meta.get("duration")
        result["record_validation_flags"] = {
            "data_validate": meta.get("data_validate"),
            "integrity": meta.get("integrity"),
            "fps_validate": meta.get("fps_validate"),
        }
    except Exception as exc:
        return {**result, "errors": [f"invalid_meta_info:{type(exc).__name__}:{exc}"]}

    try:
        head_ts = read_ts(path / "camera/head_color/head_color.txt")
        if len(head_ts) < 2 or np.any(np.diff(head_ts) <= 0):
            raise ValueError("head timestamps must be at least two and strictly increasing")
        result["head_timestamp_count"] = len(head_ts)
        result["head_duration_s"] = float((head_ts[-1] - head_ts[0]) / 1e9)
        result["head_timestamp_period_ms_median"] = float(np.median(np.diff(head_ts)) / 1e6)
    except Exception as exc:
        err.append(f"head_clock:{type(exc).__name__}:{exc}")
        return result

    numeric = {}
    try:
        with h5py.File(path / "record/raw_joints.h5", "r") as h5:
            for key, (width, required) in FIELDS.items():
                base = key.rsplit("/", 1)[0]
                tkey = base + "/timestamp"
                exists = key in h5 and tkey in h5 and h5[key].size > 0 and h5[tkey].size > 0
                if not exists:
                    if required:
                        err.append(f"missing_required_h5:{key}")
                    elif key.startswith("action/"):
                        fallback = "state/" + key.split("/", 1)[1]
                        if fallback in h5 and base + "/timestamp" in h5 and h5[fallback].size:
                            result["action_fallbacks"].append({
                                "field": key, "source": fallback,
                                "policy": "copy same-frame state without scaling/interpolation",
                            })
                        else:
                            err.append(f"missing_fallback_source:{fallback}")
                    continue
                values = np.asarray(h5[key])
                stamps = np.asarray(h5[tkey], dtype=np.int64).reshape(-1)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                if values.ndim != 2 or values.shape[1] != width:
                    err.append(f"shape:{key}:{values.shape}:expected[N,{width}]")
                if len(values) != len(stamps):
                    err.append(f"length_mismatch:{key}:{len(values)}:{len(stamps)}")
                if not np.isfinite(values).all():
                    err.append(f"nonfinite:{key}")
                if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
                    err.append(f"timestamps_not_strict:{tkey}")
                if key in ("state/joint/position", "state/left_effector/position", "state/right_effector/position",
                           "state/head/position", "state/waist/position", "action/joint/position",
                           "action/left_effector/position", "action/right_effector/position"):
                    numeric[key] = (values, stamps)
                if "effector/position" in key and values.size:
                    result.setdefault("gripper_raw_ranges", {})[key] = [float(values.min()), float(values.max())]

        # Explicitly construct one same-frame qpos sample so the batch's body
        # order is measured rather than assumed from a name or date.
        cols = {}
        source_map = {
            "observation.state.joint_position": "state/joint/position",
            "observation.state.left_effector_position": "state/left_effector/position",
            "observation.state.right_effector_position": "state/right_effector/position",
            "observation.state.head_position": "state/head/position",
            "observation.state.waist_position": "state/waist/position",
            "action.joint_position": "action/joint/position",
            "action.left_effector_position": "action/left_effector/position",
            "action.right_effector_position": "action/right_effector/position",
            "action.head_position": "state/head/position",
            "action.waist_position": "state/waist/position",
        }
        for out, key in source_map.items():
            vals, stamps = numeric[key]
            cols[out] = vals[nearest(stamps, head_ts)]
        qpos = assemble_g1_qpos(
            cols, gripper_convention="raw_encoder_0_120", action_gripper_convention="closed_0_1",
            allow_state_action_fallback=True,
        )
        result["body_order_detected"] = detect_body_order(qpos.state)
        result["state_qpos_shape"] = list(qpos.state.shape)
        result["state_qpos_finite"] = bool(np.isfinite(qpos.state).all())
        result["state_waist_medians"] = np.median(qpos.state[:, 18:20], axis=0).astype(float).tolist()
        for key in ("state/left_effector/position", "state/right_effector/position"):
            vals, _ = numeric[key]
            if np.min(vals) < -1e-3 or np.max(vals) > 120.001:
                result["warnings"].append(f"state_gripper_outside_encoder_0_120:{key}")
        for key in ("action/left_effector/position", "action/right_effector/position"):
            vals = np.asarray(h5_value(path, key))
            if np.min(vals) < -1e-3 or np.max(vals) > 1.001:
                result["warnings"].append(f"action_gripper_outside_closed_0_1:{key}")
    except Exception as exc:
        err.append(f"h5_or_qpos:{type(exc).__name__}:{exc}")

    cameras = {}
    for camera, kind in camera_specs_for(variant):
        if kind == "jpg":
            folder = path / "camera" / camera / "color"
            try:
                files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
                if len(files) < 2:
                    raise ValueError(f"only {len(files)} JPG frames; at least two are required for timestamp alignment")
                stamps = np.asarray([int(p.stem) for p in files], dtype=np.int64)
                if np.any(np.diff(stamps) <= 0):
                    raise ValueError("JPG timestamp filenames must be strictly increasing")
                selected = nearest(stamps, head_ts)
                shapes = set()
                for p in files:
                    with Image.open(p) as image:
                        image.verify()
                    with Image.open(p) as image:
                        shapes.add((image.height, image.width, 3))
                if len(shapes) != 1:
                    raise ValueError(f"JPG geometry changed: {sorted(shapes)}")
                shape = list(next(iter(shapes)))
                delta = np.abs(stamps[selected] - head_ts) / 1e6
                cameras[camera] = {"frames_decoded": len(files), "timestamp_rows": len(stamps), "shape": shape,
                                   "max_head_clock_skew_ms": float(delta.max()), "p99_head_clock_skew_ms": float(np.percentile(delta, 99)),
                                   "kind": "timestamped_jpg"}
            except Exception as exc:
                err.append(f"camera_{camera}:{type(exc).__name__}:{exc}")
            continue
        video = path / "camera" / camera / f"{camera}.h265"
        stamp_path = video.with_suffix(".txt")
        try:
            stamps = read_ts(stamp_path)
            if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
                raise ValueError("timestamps must be at least two and strictly increasing")
            selected = nearest(stamps, head_ts)
            decoded = 0
            shape = None
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "NONE"
                stream.thread_count = 1
                shape = [int(stream.height), int(stream.width), 3]
                for frame in container.decode(video=0):
                    if frame.to_ndarray(format="rgb24").shape != (shape[0], shape[1], 3):
                        raise ValueError("decoded frame geometry changed")
                    decoded += 1
            if decoded <= int(selected.max()):
                raise ValueError(f"head clock selects missing video frame {int(selected.max())}/{decoded}")
            nearest_delta_ms = np.abs(stamps[selected] - head_ts) / 1e6
            cameras[camera] = {
                "frames_decoded": decoded, "timestamp_rows": len(stamps), "shape": shape,
                "max_head_clock_skew_ms": float(nearest_delta_ms.max()),
                "p99_head_clock_skew_ms": float(np.percentile(nearest_delta_ms, 99)),
            }
            if decoded != len(stamps):
                result["warnings"].append(f"video_timestamp_count_diff:{camera}:{decoded}:{len(stamps)}")
        except Exception as exc:
            err.append(f"camera_{camera}:{type(exc).__name__}:{exc}")
    result["cameras"] = cameras
    return result


def h5_value(path: Path, key: str) -> np.ndarray:
    with h5py.File(path / "record/raw_joints.h5", "r") as h5:
        return np.asarray(h5[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("wrist_lr", "head_color_fisheye_lr"), default="head_color_fisheye_lr")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {args.output}")
    args.input = args.input.resolve()
    episodes = sorted(p for p in args.input.iterdir() if p.is_dir() and (p / "meta_info.json").is_file())
    if not episodes:
        raise RuntimeError(f"No raw episode directories under {args.input}")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_episode, (str(p.resolve()), args.variant)): p for p in episodes}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"preflight {len(results)}/{len(episodes)} {row['episode']} errors={len(row['errors'])}", flush=True)
            with (args.output / "progress.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    results.sort(key=lambda x: x["episode"])
    for row in results:
        row["status"] = "REJECT" if row["errors"] else "ACCEPT"
    (args.output / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    with (args.output / "rejections.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "task_id", "error", "warning"])
        for row in results:
            for error in (row["errors"] or [""]):
                writer.writerow([row["episode"], row.get("task_id", ""), error, ";".join(row["warnings"])])
    task_ids = {}
    for row in results:
        task_ids.setdefault(row.get("task_id", ""), {"count": 0, "texts": set()})
        task_ids[row.get("task_id", "")]["count"] += 1
        task_ids[row.get("task_id", "")]["texts"].add(row.get("task_text", ""))
    summary = {
        "input": str(args.input.resolve()), "raw_episodes": len(results),
        "accepted": sum(not row["errors"] for row in results),
        "rejected": sum(bool(row["errors"]) for row in results),
        "warnings": sum(len(row["warnings"]) for row in results),
        "action_fallback_episodes": sum(bool(row["action_fallbacks"]) for row in results),
        "action_fallbacks_by_field": sorted({x["field"] for row in results for x in row["action_fallbacks"]}),
        "body_order_counts": {k: sum(row.get("body_order_detected") == k for row in results) for k in ("waist_first", "lift_first")},
        "body_order_ambiguous": sum(row.get("body_order_detected") is None for row in results),
        "tasks": {k: {"count": v["count"], "texts": sorted(v["texts"])} for k, v in task_ids.items()},
        "variant": args.variant,
        "camera_shapes": {camera: sorted({tuple(row.get("cameras", {}).get(camera, {}).get("shape", [])) for row in results}) for camera, _ in camera_specs_for(args.variant)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["rejected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

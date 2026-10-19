"""Restore G1 camera geometry for accepted multi-grasp data and append it to ACT training.

The source accepted dataset is read-only. New, versioned outputs are created under
output/. Numeric observations remain unchanged. The appended ACT labels follow the
existing training convention: continuous channels use state[t+3] clamped within
the episode; gripper channels use the source same-frame action threshold >= 0.5.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
SOURCE_DEFAULT = Path(
    r"D:\go work\EVA-CLIENT\work_dirs\collection\agibot_g1\pick_up_the_apple\export\lerobot_v21_30hz\accepted"
)
RAW_DEFAULT = Path(
    r"D:\go work\EVA-CLIENT\work_dirs\collection\agibot_g1\pick_up_the_apple\raw"
)
REFERENCE_DEFAULT = Path(r"D:\go work\deploy_g1\examples\dataset\empty_text_task_1")
TRAIN_DEFAULT = REPO / "output/pick_apple_merged_20261006/training_100_6/train100"
VAL_DEFAULT = REPO / "output/pick_apple_merged_20261006/training_100_6/val6"
OUTPUT_DEFAULT = REPO / "output/pick_apple_multigrasp_20261007"
CANONICAL_STATE_NAMES = [
    *(f"left_arm_joint{i}" for i in range(1, 8)),
    "left_gripper",
    *(f"right_arm_joint{i}" for i in range(1, 8)),
    "right_gripper",
    "head_yaw",
    "head_pitch",
    "lift",
    "waist_pitch",
]
VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
FEATURE_RENAMES = {
    "observations.state.qpos": "observation.state",
    "action.qpos": "action",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stats_for(values: np.ndarray) -> dict:
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


def combine_stats(parts: list[dict]) -> dict:
    count = sum(int(part["count"][0]) for part in parts)
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    stds = [np.asarray(part["std"], dtype=np.float64) for part in parts]
    weights = [int(part["count"][0]) for part in parts]
    mean = sum(x * w for x, w in zip(means, weights)) / count
    variance = sum((s**2 + (m - mean) ** 2) * w for m, s, w in zip(means, stds, weights)) / count
    return {
        "min": np.minimum.reduce([np.asarray(x["min"], dtype=np.float64) for x in parts]).tolist(),
        "max": np.maximum.reduce([np.asarray(x["max"], dtype=np.float64) for x in parts]).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [count],
    }


def video_stats(path: Path, expected_frames: int) -> dict:
    total = np.zeros(3, np.float64)
    squared = np.zeros(3, np.float64)
    minimum = np.full(3, 255.0)
    maximum = np.zeros(3)
    pixels = frame_count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.width == 224 and stream.codec_context.height == 224, path
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24").reshape(-1, 3)
            values = image.astype(np.float64)
            total += values.sum(axis=0)
            squared += np.einsum("ij,ij->j", values, values)
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
            pixels += len(image)
            frame_count += 1
    assert frame_count == expected_frames, (path, frame_count, expected_frames)
    mean = total / pixels
    std = np.sqrt(np.maximum(0.0, squared / pixels - mean**2))
    shaped = lambda x: (x / 255.0).reshape(3, 1, 1).tolist()
    return {"min": shaped(minimum), "max": shaped(maximum), "mean": shaped(mean), "std": shaped(std), "count": [frame_count]}


def convert_video(job: tuple[str, str, int, int, int]) -> str:
    source, destination, width, height, expected_frames = job
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ratio = 224 / max(width, height)
    out_w, out_h = round(width * ratio), round(height * ratio)
    x, y = (224 - out_w) // 2, (224 - out_h) // 2
    # Source frames are 224x224 stretched. Reverse that geometric stretch into
    # the verified source-camera aspect, then match the training black letterbox.
    vf = f"scale={out_w}:{out_h}:flags=bilinear,pad=224:224:{x}:{y}:color=black,setsar=1"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vf", vf,
            "-fps_mode", "passthrough", "-c:v", "libx264", "-threads", "1",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(destination),
        ],
        check=True,
    )
    stats = video_stats(destination, expected_frames)
    return json.dumps({"path": str(destination), "stats": stats})


def fixed_float(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def preflight(source: Path, raw: Path, reference: Path):
    info = read_json(source / "meta/info.json")
    ref = read_json(reference / "meta/info.json")
    accepted_eps = jsonl(source / "meta/episodes.jsonl")
    raw_eps = jsonl(raw / "meta/episodes.jsonl")
    split = read_json(source / "meta/quality_split.json")
    accepted_source_ids = set(split["source_episode_indices"])
    assert info["codebase_version"] == "v2.1" and info["fps"] == 30
    assert len(accepted_eps) == info["total_episodes"] == 33
    assert info["total_frames"] == sum(ep["length"] for ep in accepted_eps) == 30651
    assert all(ep.get("quality") == "green" and not ep.get("quality_issues") for ep in accepted_eps)
    assert all(k in info["features"] for k in VIDEO_KEYS)
    expected_shapes = {}
    for key in VIDEO_KEYS:
        source_shape = info["features"][key]["shape"]
        target_shape = ref["features"][key]["shape"]
        assert source_shape == [224, 224, 3], (key, source_shape)
        expected_shapes[key] = target_shape
    assert expected_shapes[VIDEO_KEYS[0]][:2] == [720, 1280]
    assert expected_shapes[VIDEO_KEYS[1]][:2] == [480, 848]
    assert expected_shapes[VIDEO_KEYS[2]][:2] == [480, 848]
    assert len(raw_eps) == 42
    missing = []
    rejected = []
    raw_by_index = {ep["episode_index"]: ep for ep in raw_eps}
    for source_id in sorted(accepted_source_ids):
        if source_id not in raw_by_index:
            missing.append(source_id)
    for ep in raw_eps:
        if ep["episode_index"] not in accepted_source_ids:
            rejected.append({
                "episode_index": ep["episode_index"],
                "qc_verdict": ep.get("qc_verdict", ""),
                "quality": ep.get("quality", ""),
                "quality_issues": ";".join(ep.get("quality_issues", [])),
                "qc_note": ep.get("qc_note", ""),
            })
    assert not missing
    assert len(rejected) == len(raw_eps) - len(accepted_eps) == 9
    return info, ref, accepted_eps, expected_shapes, rejected


def build_corrected_candidate(source: Path, output: Path, reference: Path, info: dict, ref: dict,
                              episodes: list[dict], source_shapes: dict, workers: int):
    assert not output.exists(), f"Refusing to overwrite {output}"
    started = time.monotonic()
    output.mkdir(parents=True)
    (output / "meta").mkdir()
    source_stats_rows = jsonl(source / "meta/episodes_stats.jsonl")
    source_stats = {row["episode_index"]: row["stats"] for row in source_stats_rows}
    video_jobs = []
    episode_rows = []
    episode_stat_rows = []
    offset = 0

    for new_index, ep in enumerate(episodes):
        old_index, length = ep["episode_index"], ep["length"]
        in_path = source / f"data/chunk-000/episode_{old_index:06d}.parquet"
        table = pq.read_table(in_path)
        assert table.num_rows == length
        source_state = np.stack(table["observations.state.qpos"].to_numpy()).astype(np.float32)
        source_action = np.stack(table["action.qpos"].to_numpy()).astype(np.float32)
        assert source_state.shape == source_action.shape == (length, 20)
        assert np.isfinite(source_state).all() and np.isfinite(source_action).all()
        assert np.array_equal(source_state, source_action), f"Unexpected action/state semantics change in ep {old_index}"
        assert np.array_equal(table["frame_index"].to_numpy(), np.arange(length))
        assert np.array_equal(table["episode_index"].to_numpy(), np.full(length, old_index))
        assert np.array_equal(table["index"].to_numpy(), np.arange(offset, offset + length))
        assert np.allclose(np.diff(table["timestamp"].to_numpy()), 1 / 30, atol=1e-6)

        renames = [FEATURE_RENAMES.get(name, name) for name in table.column_names]
        converted = table.rename_columns(renames)
        out_table_path = output / f"data/chunk-000/episode_{new_index:06d}.parquet"
        out_table_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted, out_table_path, compression="snappy")

        row_stats = {}
        for feature_key in converted.column_names:
            if feature_key in ("observation.state", "action") or feature_key in (
                "observations.state.eef", "action.eef", "timestamp", "capture_time",
                "frame_index", "episode_index", "index", "task_index",
            ):
                arr = converted[feature_key].to_numpy(zero_copy_only=False)
                if feature_key in ("observation.state", "action", "observations.state.eef", "action.eef"):
                    arr = np.stack(converted[feature_key].to_numpy()).astype(np.float32)
                row_stats[feature_key] = stats_for(arr)

        for camera in VIDEO_KEYS:
            source_video = source / f"videos/chunk-000/{camera}/episode_{old_index:06d}.mp4"
            output_video = output / f"videos/chunk-000/{camera}/episode_{new_index:06d}.mp4"
            h, w = source_shapes[camera][:2]
            video_jobs.append((str(source_video), str(output_video), w, h, length))

        enriched = dict(ep)
        enriched["episode_index"] = new_index
        enriched["dataset_from_index"] = offset
        enriched["dataset_to_index"] = offset + length
        enriched["source_episode_index"] = read_json(source / "meta/quality_split.json")["source_episode_indices"][new_index]
        enriched["source_dataset"] = str(source)
        enriched["image_geometry_recovery"] = "assumed uncropped; inverse vertical stretch to camera aspect, then 224 black letterbox"
        episode_rows.append(enriched)
        episode_stat_rows.append({"episode_index": new_index, "stats": row_stats})
        offset += length

    with ProcessPoolExecutor(max_workers=workers) as pool:
        video_results = [json.loads(x) for x in pool.map(convert_video, video_jobs)]
    by_video = {Path(row["path"]).as_posix(): row["stats"] for row in video_results}
    for row in episode_stat_rows:
        ep_index = row["episode_index"]
        ep = episodes[ep_index]
        for camera in VIDEO_KEYS:
            path = (output / f"videos/chunk-000/{camera}/episode_{ep_index:06d}.mp4").as_posix()
            row["stats"][camera] = by_video[path]
        assert set(VIDEO_KEYS).issubset(row["stats"])

    out_features = {}
    for key, feature in info["features"].items():
        new_key = FEATURE_RENAMES.get(key, key)
        cloned = dict(feature)
        if new_key in ("observation.state", "action"):
            cloned["names"] = CANONICAL_STATE_NAMES
        if key in VIDEO_KEYS:
            h, w = source_shapes[key][:2]
            scale = 224 / max(h, w)
            cloned["shape"] = [224, 224, 3]
            cloned["info"] = dict(cloned.get("info", {}), **{
                "video.height": 224, "video.width": 224, "video.fps": 30.0,
                "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False,
            })
            cloned["info"]["source.height"] = h
            cloned["info"]["source.width"] = w
            cloned["info"]["letterbox.content.height"] = round(h * scale)
            cloned["info"]["letterbox.content.width"] = round(w * scale)
        out_features[new_key] = cloned

    out_info = dict(info)
    out_info.update(robot_type="G1", features=out_features, total_videos=len(episodes) * 3,
                    splits={"train": f"0:{len(episodes)}"})
    out_info["conversion_provenance"] = {
        "source_dataset": str(source),
        "reference_camera_geometry_dataset": str(reference),
        "camera_source_shapes": {k: source_shapes[k][:2] for k in VIDEO_KEYS},
        "output_shape": [224, 224, 3],
        "image_transform": "inverse nonuniform square stretch by resizing to verified camera aspect, then bilinear resize and black letterbox",
        "assumption": "source images were not cropped before the 224x224 stretch; geometric restoration cannot recover information lost by earlier downsampling/H.264",
        "numeric_source_values_preserved": True,
        "source_action_qpos_equals_state_qpos": True,
        "source_fps": 10,
        "stored_fps": 30,
        "video_resampling": "source videos are zero-order-held from 10 Hz to 30 Hz per RESAMPLE_MANIFEST.json",
    }
    (output / "meta/info.json").write_text(json.dumps(out_info, indent=2), encoding="utf-8")
    shutil.copy2(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")
    shutil.copy2(source / "meta/quality_split.json", output / "meta/quality_split.json")
    (output / "meta/episodes.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in episode_rows), encoding="utf-8")
    (output / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episode_stat_rows), encoding="utf-8")
    keys = episode_stat_rows[0]["stats"].keys()
    global_stats = {key: combine_stats([row["stats"][key] for row in episode_stat_rows]) for key in keys}
    (output / "meta/stats.json").write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
    (output / "meta/feature_mapping.json").write_text(json.dumps({
        "observations.state.qpos": "observation.state",
        "action.qpos": "action",
        "qpos_order": CANONICAL_STATE_NAMES,
        "state_action_values_changed": False,
        "auxiliary_eef_features_preserved": True,
    }, indent=2), encoding="utf-8")
    validation = {
        "status": "PASS",
        "episodes": len(episode_rows),
        "frames": offset,
        "videos_expected_and_decoded": len(video_results),
        "video_frame_count_failures": 0,
        "camera_geometry": {k: {"source": source_shapes[k][:2], "output_content": [
            round(source_shapes[k][0] * 224 / max(source_shapes[k][:2])),
            round(source_shapes[k][1] * 224 / max(source_shapes[k][:2])),
        ], "canvas": [224, 224]} for k in VIDEO_KEYS},
        "action_qpos_exactly_equals_state_qpos": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "meta/conversion_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return episode_rows, episode_stat_rows, out_info, validation


def append_training(candidate: Path, train_source: Path, val_source: Path, output: Path,
                    new_episodes: list[dict], new_stats: list[dict], workers: int):
    training_out = output / "training"
    assert not training_out.exists(), f"Refusing to overwrite {training_out}"
    train_out = training_out / "train133"
    val_out = training_out / "val6"
    train_out.mkdir(parents=True)
    shutil.copytree(val_source, val_out)
    started = time.monotonic()

    old_info = read_json(train_source / "meta/info.json")
    old_episodes = jsonl(train_source / "meta/episodes.jsonl")
    old_stats = jsonl(train_source / "meta/episodes_stats.jsonl")
    task_rows = jsonl(train_source / "meta/tasks.jsonl")
    new_tasks = jsonl(candidate / "meta/tasks.jsonl")
    old_task_texts = {row["task_index"]: row["task"] for row in task_rows}
    new_task_text = new_tasks[0]["task"]
    assert new_task_text not in old_task_texts.values()
    new_task_index = max(old_task_texts) + 1

    for subdir in ("data/chunk-000",):
        (train_out / subdir).mkdir(parents=True)
    old_video_keys = [k for k, v in old_info["features"].items() if v.get("dtype") == "video"]
    for key in old_video_keys:
        (train_out / "videos/chunk-000" / key).mkdir(parents=True)

    episode_rows = []
    stats_rows = []
    frames_offset = 0
    for i, ep in enumerate(old_episodes):
        source_parquet = train_source / f"data/chunk-000/episode_{i:06d}.parquet"
        dest_parquet = train_out / f"data/chunk-000/episode_{i:06d}.parquet"
        shutil.copy2(source_parquet, dest_parquet)
        table = pq.read_table(dest_parquet)
        assert table.num_rows == ep["length"]
        assert np.array_equal(table["index"].to_numpy(), np.arange(frames_offset, frames_offset + ep["length"]))
        for key in old_video_keys:
            src = train_source / f"videos/chunk-000/{key}/episode_{i:06d}.mp4"
            dst = train_out / f"videos/chunk-000/{key}/episode_{i:06d}.mp4"
            shutil.copy2(src, dst)
        episode_rows.append(ep)
        stats_rows.append(old_stats[i])
        frames_offset += ep["length"]

    task_features = {"timestamp", "frame_index", "episode_index", "index", "task_index", "observation.state", "action"}
    for j, ep in enumerate(new_episodes):
        new_index = len(old_episodes) + j
        source_table = pq.read_table(candidate / f"data/chunk-000/episode_{j:06d}.parquet")
        state = np.stack(source_table["observation.state"].to_numpy()).astype(np.float32)
        original_action = np.stack(source_table["action"].to_numpy()).astype(np.float32)
        n = len(state)
        assert state.shape == original_action.shape == (n, 20)
        assert np.array_equal(state, original_action)
        future = state[np.minimum(np.arange(n) + 3, n - 1)].copy()
        future[:, [7, 15]] = (original_action[:, [7, 15]] >= 0.5).astype(np.float32)
        values = {
            "timestamp": pa.array(source_table["timestamp"].to_numpy(), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, new_index, dtype=np.int64)),
            "index": pa.array(np.arange(frames_offset, frames_offset + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, new_task_index, dtype=np.int64)),
            "observation.state": fixed_float(state, 20),
            "action": fixed_float(future, 20),
        }
        out_table = pa.table(values)
        dest_parquet = train_out / f"data/chunk-000/episode_{new_index:06d}.parquet"
        pq.write_table(out_table, dest_parquet, compression="snappy")
        read_back = pq.read_table(dest_parquet)
        assert read_back.num_rows == n
        np.testing.assert_array_equal(np.stack(read_back["observation.state"].to_numpy()), state)
        np.testing.assert_array_equal(np.stack(read_back["action"].to_numpy()), future)
        assert set(np.unique(future[:, [7, 15]])).issubset({0.0, 1.0})

        episode_row = dict(ep)
        episode_row.update(episode_index=new_index, dataset_from_index=frames_offset,
                           dataset_to_index=frames_offset + n, tasks=[new_task_text],
                           task_index=new_task_index,
                           action_label_policy="continuous=state[t+3] clamped; grippers=source_action[t]>=0.5")
        episode_rows.append(episode_row)
        row_stats = {key: stats_for(np.asarray(read_back[key].to_numpy())) for key in
                     ("timestamp", "frame_index", "episode_index", "index", "task_index")}
        row_stats["observation.state"] = stats_for(state)
        row_stats["action"] = stats_for(future)
        for key in old_video_keys:
            source_video = candidate / f"videos/chunk-000/{key}/episode_{j:06d}.mp4"
            output_video = train_out / f"videos/chunk-000/{key}/episode_{new_index:06d}.mp4"
            shutil.copy2(source_video, output_video)
            row_stats[key] = new_stats[j]["stats"][key]
        stats_rows.append({"episode_index": new_index, "stats": row_stats})
        frames_offset += n

    for i, (ep, stats) in enumerate(zip(episode_rows[:len(old_episodes)], stats_rows[:len(old_episodes)])):
        assert ep["episode_index"] == i and stats["episode_index"] == i
    (train_out / "meta").mkdir()
    info = dict(old_info)
    info.update(total_episodes=len(episode_rows), total_frames=frames_offset,
                total_videos=len(episode_rows) * len(old_video_keys), total_tasks=2,
                splits={"train": f"0:{len(episode_rows)}"})
    info["multi_grasp_append_provenance"] = {
        "candidate_dataset": str(candidate),
        "source_episode_count": len(new_episodes),
        "new_task_index": new_task_index,
        "numeric_state_values_unchanged": True,
        "continuous_action_policy": "state[t+3] clamped at episode end, matching existing ACT train100 builder",
        "gripper_action_policy": "original same-frame qpos >= 0.5 to binary 0/1, matching existing ACT train100 builder",
        "source_action_qpos_equals_state_qpos": True,
        "eef_auxiliary_fields_dropped_from_act_projection": True,
    }
    (train_out / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (train_out / "meta/tasks.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in [
        *task_rows, {"task_index": new_task_index, "task": new_task_text}
    ]), encoding="utf-8")
    (train_out / "meta/episodes.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in episode_rows), encoding="utf-8")
    (train_out / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(x) + "\n" for x in stats_rows), encoding="utf-8")
    global_stats = {key: combine_stats([row["stats"][key] for row in stats_rows]) for key in stats_rows[0]["stats"]}
    (train_out / "meta/stats.json").write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
    shutil.copy2(train_source / "meta/modality.json", train_out / "meta/modality.json")
    shutil.copy2(train_source / "meta/label_validation.json", train_out / "meta/label_validation.json")
    modality = read_json(train_out / "meta/modality.json")
    modality["annotation"]["human.action.task_description"]["original_key"] = "task_index"
    (train_out / "meta/modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    split = {
        "status": "PASS",
        "training_episodes": len(episode_rows),
        "training_frames": frames_offset,
        "validation_episodes": 6,
        "validation_source": str(val_source),
        "validation_unchanged": True,
        "train_old_and_1004": {"episodes": len(old_episodes), "frames": old_info["total_frames"]},
        "train_multigrasp": {"episodes": len(new_episodes), "frames": sum(x["length"] for x in new_episodes)},
        "task_index_map": old_task_texts | {new_task_index: new_task_text},
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "training/split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "training/README.md").write_text(
        "ACT-ready smoke dataset created as a new version. train133 contains the existing train100 "
        "(86 old + 14 1004) plus 33 accepted multi-grasp episodes. val6 is copied unchanged. "
        "The multi-grasp source keeps its task text as task_index 1. Numeric observations are unchanged; "
        "continuous action labels follow state[t+3], while grippers use same-frame source action >= 0.5. "
        "Source accepted data and the previous training datasets remain untouched.\n",
        encoding="utf-8",
    )
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--val", type=Path, default=VAL_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.workers >= 1
    assert not args.output.exists(), f"Refusing to overwrite {args.output}"
    preflight_started = time.monotonic()
    info, ref, episodes, source_shapes, rejected = preflight(args.input, args.raw, args.reference)
    assert args.train.exists() and args.val.exists()
    args.output.mkdir(parents=True)
    candidate_out = args.output / "candidate_g1style"
    rows, stats, _, conversion_report = build_corrected_candidate(
        args.input, candidate_out, args.reference, info, ref, episodes, source_shapes, args.workers
    )
    training_report = append_training(candidate_out, args.train, args.val, args.output, rows, stats, args.workers)

    raw_eps = jsonl(args.raw / "meta/episodes.jsonl")
    accepted_ids = set(read_json(args.input / "meta/quality_split.json")["source_episode_indices"])
    with (args.output / "rejection_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "qc_verdict", "quality", "quality_issues", "qc_note"])
        writer.writeheader()
        writer.writerows(rejected)
    audit = {
        "raw_episodes": len(raw_eps),
        "accepted_episodes": len(episodes),
        "rejected_episodes": len(rejected),
        "accepted_source_indices": sorted(accepted_ids),
        "rejection_report": str(args.output / "rejection_report.csv"),
        "candidate_conversion": conversion_report,
        "training_merge": training_report,
        "state_action_audit": {
            "source_action_qpos_equals_state_qpos_exactly_all_frames": True,
            "converted_training_continuous_labels": "state[t+3] clamped at episode end",
            "converted_training_gripper_labels": "source same-frame action >= 0.5",
            "source_numeric_state_preserved": True,
            "waist_numeric_order": "index 18 matches lift (~0.2196); index 19 matches waist_pitch (~0.1745 rad); no swap applied",
        },
        "preflight_elapsed_seconds": time.monotonic() - preflight_started,
    }
    (args.output / "audit_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "CONVERSION_AND_MERGE_PASS", "output": str(args.output),
                      "train_episodes": training_report["training_episodes"],
                      "train_frames": training_report["training_frames"],
                      "candidate_videos_checked": conversion_report["videos_expected_and_decoded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

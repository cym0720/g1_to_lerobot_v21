# 用途（主流程第 00 步）：将原始 618data 直接转换为只含三路 RGB 的 LeRobot v2.1 数据集；输出永不回写。
from __future__ import annotations

import argparse
import sys
import csv
import json
import time
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.ffmpeg_io import (
    encode_h265_aligned as encode_h265_ffmpeg,
    encode_jpeg_sequence,
    video_frame_count,
)
from packages.execution import (
    ProcessLane,
    run_process_job_groups,
    run_routed_process_jobs,
    video_process_lanes,
)
from packages.contracts import load_qpos_contract
from packages.eva_qpos import assemble_g1_qpos
from packages.raw_preflight import (
    add_empty_text_task_mappings,
    collect_raw_episodes,
    h5_stream_present,
    read_json,
    timestamps,
)

# Throughput tuning defaults. They are implementation policy rather than data
# semantics, so callers normally should not need to repeat them.
DEFAULT_WORKERS = 8
DEFAULT_CUDA_WORKERS = 4
DEFAULT_QSV_WORKERS = 0


def nearest(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(values, target, side="left")
    left = np.clip(right - 1, 0, len(values) - 1)
    right = np.clip(right, 0, len(values) - 1)
    return np.where(np.abs(values[right] - target) < np.abs(values[left] - target), right, left)


def h5_series_required(
    h5: h5py.File, data_key: str, target: np.ndarray, width: int, required: bool = True
) -> np.ndarray | None:
    """Read a named G1 component; no fabricated zeros for missing contract fields."""
    if not h5_stream_present(h5, data_key):
        if required:
            raise ValueError(f"missing or empty required H5 stream: {data_key}")
        return None
    base = data_key.rsplit("/", 1)[0]
    values = np.asarray(h5[data_key])
    times = np.asarray(h5[base + "/timestamp"])
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
    result = np.asarray(values[nearest(times, target)], dtype=np.float32).reshape(len(target), -1)
    if result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{data_key}: expected finite width {width}, got {result.shape}")
    return result


def read_g1_qpos_inputs(
    h5: h5py.File, target: np.ndarray, raw_adapter: dict
) -> dict[str, np.ndarray]:
    widths = {
        "observation.state.joint_position": 14,
        "observation.state.left_effector_position": 1,
        "observation.state.right_effector_position": 1,
        "observation.state.head_position": 2,
        "observation.state.waist_position": 2,
        "action.joint_position": 14,
        "action.left_effector_position": 1,
        "action.right_effector_position": 1,
        "action.head_position": 2,
        "action.waist_position": 2,
    }
    columns: dict[str, np.ndarray] = {}
    for group in ("state", "action"):
        for name, data_key in raw_adapter[group].items():
            values = h5_series_required(
                h5,
                data_key,
                target,
                widths[name],
                required=name not in {"action.head_position", "action.waist_position"},
            )
            if values is not None:
                columns[name] = values
    return columns


def encode_h265_aligned_pyav(
    ep: Path, spec: dict, target: np.ndarray, dest: Path, fps: int
) -> None:
    """Timestamp-align an encoded RGB/fisheye stream to the canonical head clock."""
    source_times = timestamps(ep / spec["timestamp_source"])
    selected = nearest(source_times, target)
    wanted: dict[int, int] = {}
    for source_index in selected:
        wanted[int(source_index)] = wanted.get(int(source_index), 0) + 1
    height, width, _ = spec["output_shape"]
    crop = spec.get("crop")
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(ep / spec["source"])) as reader, av.open(str(dest), mode="w") as writer:
        stream = writer.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for source_index, frame in enumerate(reader.decode(video=0)):
            copies = wanted.get(source_index, 0)
            if not copies:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if crop:
                x, y, w, h = (crop[k] for k in ("x", "y", "width", "height"))
                rgb = rgb[y : y + h, x : x + w]
            for _ in range(copies):
                out_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
                    width=width, height=height, format="yuv420p"
                )
                for packet in stream.encode(out_frame):
                    writer.mux(packet)
                written += 1
        for packet in stream.encode():
            writer.mux(packet)
    if written != len(target):
        raise ValueError(
            f'{ep.name}: {spec["source"]} decoded {written}/{len(target)} aligned frames'
        )


def encode_h265_aligned(
    ep: Path,
    spec: dict,
    target: np.ndarray,
    dest: Path,
    fps: int,
    preferred_backends: tuple[str, ...],
) -> str:
    """Use FFmpeg hardware decode/encode; retain PyAV only for duplicate-frame compatibility."""
    source_times = timestamps(ep / spec["timestamp_source"])
    try:
        backend = encode_h265_ffmpeg(
            ep / spec["source"],
            source_times,
            target,
            dest,
            fps,
            spec.get("crop"),
            preferred_backends,
        )
        print(f"{ep.name} {spec['source']} backend={backend}")
        return backend
    except ValueError as error:
        if "duplicate timestamp-aligned" not in str(error):
            raise
        encode_h265_aligned_pyav(ep, spec, target, dest, fps)
        return "pyav"


def encode_wrist(ep: Path, camera: str, target: np.ndarray, dest: Path, fps: int) -> str:
    files = sorted((ep / "camera" / camera / "color").glob("*.jpg"))
    source_ts = np.asarray([int(path.stem) for path in files], dtype=np.int64)
    selected = [files[index] for index in nearest(source_ts, target)]
    backend = encode_jpeg_sequence(selected, dest, fps)
    print(f"{ep.name} {camera} backend={backend}")
    return backend


def convert_camera(
    args: tuple[Path, int, Path, str, dict, np.ndarray, int, tuple[str, ...]]
) -> dict:
    ep, index, root, camera, spec, target, fps, backends = args
    destination = root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
    try:
        if spec["kind"] == "jpg_timestamp_aligned":
            source_camera = "hand_left" if "left_wrist" in camera else "hand_right"
            backend = encode_wrist(ep, source_camera, target, destination, fps)
        elif spec["kind"] == "h265_timestamp_aligned":
            backend = encode_h265_aligned(ep, spec, target, destination, fps, backends)
        else:
            raise ValueError(f'unknown camera kind: {spec["kind"]}')
        frames = video_frame_count(destination)
        error = "" if frames == len(target) else f"video_frames:{camera}:{frames}!={len(target)}"
    except Exception as exc:
        backend = backends[0] if backends else "unknown"
        error = f"{camera}:{type(exc).__name__}:{exc}"
    return {
        "root": str(root),
        "episode_index": index,
        "camera": camera,
        "backend": backend,
        "error": error,
    }


def validate_episode_output(root: Path, index: int, expected: int, cameras: list[str]) -> list[str]:
    errors = []
    data = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
    table = pq.read_table(data)
    if table.num_rows != expected:
        errors.append(f"parquet_rows:{table.num_rows}!={expected}")
    frames = np.asarray(table["frame_index"].to_pylist())
    if len(frames) and (frames[0] != 0 or frames[-1] != expected - 1):
        errors.append("frame_index_boundary_error")
    for camera in cameras:
        path = root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
        if not path.exists():
            errors.append(f"missing_video:{camera}")
        elif video_frame_count(path) != expected:
            errors.append(f"video_frames:{camera}")
    return errors


def stats_for(values: np.ndarray) -> dict:
    """Return the v2.1 per-episode numeric statistics required by the v3 migrator."""
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


def float_column(values: np.ndarray, shape: list[int]) -> pa.Array:
    """Write the exact float32 scalar/list Arrow type promised by meta/info.json."""
    array = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    if int(np.prod(shape)) == 1:
        return pa.array(array[:, 0], type=pa.float32())
    return pa.array(array.tolist(), type=pa.list_(pa.float32()))


def rgb_range_stats(count: int) -> dict:
    """Conservative normalized-RGB bounds required by LeRobot visual normalizers.

    We intentionally do not claim an image mean/std without decoding every frame;
    ImageNet normalization is selected by the training config for visual inputs.
    """
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [count],
    }


def prepare_episode(
    args: tuple[Path, int, int, Path, dict, dict, dict, str, bool, tuple[str, ...], object]
) -> dict:
    (
        ep,
        index,
        offset,
        root,
        schema,
        task,
        raw_adapter,
        gripper_convention,
        allow_state_action_fallback,
        qpos_names,
        _lane_payload,
    ) = args
    meta = read_json(ep / "meta_info.json")
    target = timestamps(ep / "camera/head_color/head_color.txt")
    fps = int(schema["fps"])
    numeric = {}
    with h5py.File(ep / "record/raw_joints.h5", "r") as h5:
        # Source device timestamps align heterogeneous H5 streams, but encoded
        # MP4 is CFR at schema fps. LeRobot requires its parquet timestamps to
        # use the same exact CFR clock for deterministic frame lookup.
        columns = {
            "timestamp": pa.array(
                (np.arange(len(target), dtype=np.float32) / fps), type=pa.float32()
            ),
            "frame_index": pa.array(np.arange(len(target), dtype="int64")),
            "episode_index": pa.array(np.full(len(target), index, dtype="int64")),
            "index": pa.array(np.arange(len(target), dtype="int64") + offset),
            "task_index": pa.array(np.zeros(len(target), dtype="int64")),
        }
        assembly = assemble_g1_qpos(
            read_g1_qpos_inputs(h5, target, raw_adapter),
            gripper_convention=gripper_convention,
            allow_state_action_fallback=allow_state_action_fallback,
        )
        state, action = assembly.state, assembly.action
        numeric["observations.state.qpos"] = state
        numeric["action.qpos"] = action
        columns["observations.state.qpos"] = float_column(state, [20])
        columns["action.qpos"] = float_column(action, [20])
        features = {
            "observations.state.qpos": {
                "dtype": "float32",
                "shape": [20],
                "names": list(qpos_names),
            },
            "action.qpos": {"dtype": "float32", "shape": [20], "names": list(qpos_names)},
        }
    numeric.update(
        {
            "timestamp": np.arange(len(target), dtype=np.float32) / fps,
            "frame_index": np.arange(len(target), dtype="int64"),
            "episode_index": np.full(len(target), index, dtype="int64"),
            "index": np.arange(len(target), dtype="int64") + offset,
            "task_index": np.zeros(len(target), dtype="int64"),
        }
    )
    for key, spec in schema["cameras"].items():
        # v3 stores video frame timing/ranges in meta/episodes, not as a data
        # parquet column. Retaining video timestamps here breaks the official
        # v2.1->v3 loader because it deliberately excludes video columns.
        features[key] = {
            "dtype": "video",
            "shape": spec["output_shape"],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": spec["output_shape"][0],
                "video.width": spec["output_shape"][1],
                "video.channels": 3,
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    data = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), data, compression="snappy")
    errors = []
    episode_stats = {key: stats_for(value) for key, value in numeric.items()}
    episode_stats.update({key: rgb_range_stats(len(target)) for key in schema["cameras"]})
    return {
        "episode_index": index,
        "tasks": [task["name"]],
        "length": len(target),
        "dataset_from_index": offset,
        "dataset_to_index": offset + len(target),
        "original_episode_id": ep.name,
        "source_task_id": meta["task_id"],
        "features": features,
        "stats": episode_stats,
        "action_fallbacks": list(assembly.action_fallbacks),
        "validation_errors": errors,
    }


def main() -> None:
    started = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="原始 dataset/618data；也兼容辅助预检产生的分组输出")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--schema", type=Path, default=REPO_ROOT / "config/618_lerobot_schema.json")
    p.add_argument("--task-map", type=Path, default=REPO_ROOT / "config/task_id_map.json")
    p.add_argument("--source", type=Path, default=REPO_ROOT / "config/sources/g1.json")
    p.add_argument(
        "--contract", type=Path, default=REPO_ROOT / "config/contracts/eva_replay_g1_qpos20_v1.json"
    )
    p.add_argument(
        "--gripper-convention",
        choices=("raw_encoder_0_120", "opening_0_1", "closed_0_1"),
        help="Override the source-config convention only when converting a different G1 recorder.",
    )
    p.add_argument("--allow-state-action-fallback", action="store_true")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--cuda-workers", type=int, default=DEFAULT_CUDA_WORKERS)
    p.add_argument("--qsv-workers", type=int, default=DEFAULT_QSV_WORKERS)
    p.add_argument(
        "--min-duration-s",
        type=float,
        default=0.0,
        help="Optional quality threshold; 0 retains every readable duration.",
    )
    p.add_argument("--allow-unmapped-empty-task", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    schema = read_json(a.schema)
    source_config = read_json(a.source)
    contract = load_qpos_contract(a.contract)
    mapping = read_json(a.task_map)["tasks"]
    raw_adapter = source_config["raw_h5_adapter"]
    gripper_convention = a.gripper_convention or source_config["gripper_convention"]
    expected_cameras = {
        key
        for key, spec in read_json(a.contract)["required_features"].items()
        if spec.get("dtype") == "video"
    }
    missing_cameras = expected_cameras - set(schema["cameras"])
    if missing_cameras:
        raise ValueError(f"schema is missing contract cameras: {sorted(missing_cameras)}")
    schema["cameras"] = {key: schema["cameras"][key] for key in expected_cameras}
    raw_layout = any(
        (child / "meta_info.json").is_file() for child in a.input.iterdir() if child.is_dir()
    )
    if raw_layout and a.allow_unmapped_empty_task:
        add_empty_text_task_mappings(a.input, mapping)
    raw_groups, rejected = (
        collect_raw_episodes(
            a.input,
            mapping,
            a.min_duration_s,
            raw_adapter,
            a.allow_state_action_fallback,
            require_fisheyes=False,
        )
        if raw_layout
        else ({}, [])
    )
    if a.dry_run:
        accepted = sum(len(group) for group in raw_groups.values())
        print(
            json.dumps(
                {
                    "contract": contract.contract_id,
                    "accepted_episodes": accepted,
                    "rejected_episodes": len(rejected),
                    "rejections": rejected[:20],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if a.output.exists():
        raise SystemExit(f"输出已存在: {a.output}")
    a.output.mkdir(parents=True)
    if raw_layout:
        with (a.output / "conversion_rejected.csv").open("w", newline="", encoding="utf8") as file:
            csv.writer(file).writerows([["episode_uuid", "task_id", "rejection"], *rejected])
    plans = []
    prepare_jobs = []
    h265_jobs = []
    jpg_jobs = []
    for tid, spec in sorted(mapping.items(), key=lambda x: x[1]["order"]):
        source = a.input / spec["directory"]
        eps = (
            raw_groups[spec["directory"]]
            if raw_layout
            else (sorted(x for x in source.iterdir() if x.is_dir()) if source.exists() else [])
        )
        root = a.output / spec["directory"]
        (root / "meta").mkdir(parents=True)
        targets = [timestamps(ep / "camera/head_color/head_color.txt") for ep in eps]
        lengths = [len(target) for target in targets]
        offsets = np.cumsum([0, *lengths[:-1]]).tolist()
        jobs = [
            (
                ep,
                i,
                offset,
                root,
                schema,
                spec,
                raw_adapter,
                gripper_convention,
                a.allow_state_action_fallback,
                contract.qpos_names,
            )
            for i, (ep, offset) in enumerate(zip(eps, offsets))
        ]
        plans.append(
            {
                "spec": spec,
                "root": root,
                "lengths": lengths,
                "start": len(prepare_jobs),
                "count": len(jobs),
            }
        )
        prepare_jobs.extend(jobs)
        for index, (ep, target) in enumerate(zip(eps, targets)):
            for camera, camera_spec in schema["cameras"].items():
                job = (ep, index, root, camera, camera_spec, target, int(schema["fps"]))
                if camera_spec["kind"] == "h265_timestamp_aligned":
                    h265_jobs.append(job)
                elif camera_spec["kind"] == "jpg_timestamp_aligned":
                    jpg_jobs.append(job)
                else:
                    raise ValueError(f'unknown camera kind: {camera_spec["kind"]}')
    prepare_workers = max(1, min(a.workers, len(prepare_jobs) or 1))
    prepare_lanes = [ProcessLane("prepare", prepare_workers, None)]
    all_results = run_routed_process_jobs(prepare_episode, prepare_jobs, prepare_lanes)
    if a.workers == 1:
        h265_lanes = video_process_lanes(
            1, min(a.cuda_workers, 1), 0 if a.cuda_workers else min(a.qsv_workers, 1)
        )
        h265_results = run_routed_process_jobs(convert_camera, h265_jobs, h265_lanes)
        jpg_results = run_routed_process_jobs(
            convert_camera, jpg_jobs, [ProcessLane("jpg_cpu", 1, ("cpu", "qsv", "cuda"))]
        )
    else:
        h265_workers = max(
            1,
            min(
                a.workers - 1,
                a.cuda_workers + a.qsv_workers if a.cuda_workers + a.qsv_workers else a.workers - 1,
            ),
        )
        jpg_workers = max(1, a.workers - h265_workers)
        h265_lanes = video_process_lanes(
            h265_workers,
            min(a.cuda_workers, h265_workers),
            min(a.qsv_workers, max(0, h265_workers - min(a.cuda_workers, h265_workers))),
        )
        jpg_lanes = [ProcessLane("jpg_cpu", jpg_workers, ("cpu", "qsv", "cuda"))]
        h265_results, jpg_results = run_process_job_groups(
            [
                (convert_camera, h265_jobs, h265_lanes),
                (convert_camera, jpg_jobs, jpg_lanes),
            ]
        )
    print(
        "camera_process_lanes",
        [(lane.name, lane.workers) for lane in h265_lanes],
        ("jpg_cpu", 1 if a.workers == 1 else jpg_workers),
        flush=True,
    )
    camera_errors = {}
    for camera_result in [*h265_results, *jpg_results]:
        if camera_result["error"]:
            key = (camera_result["root"], camera_result["episode_index"])
            camera_errors.setdefault(key, []).append(camera_result["error"])
    for plan in plans:
        spec = plan["spec"]
        root = plan["root"]
        lengths = plan["lengths"]
        start = plan["start"]
        result = all_results[start : start + plan["count"]]
        for episode_result in result:
            key = (str(root), episode_result["episode_index"])
            validation = validate_episode_output(
                root,
                episode_result["episode_index"],
                episode_result["length"],
                list(schema["cameras"]),
            )
            episode_result["validation_errors"] = [*camera_errors.get(key, []), *validation]
        features = {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        if result:
            features.update(result[0]["features"])
        info = {
            "codebase_version": "v2.1",
            "robot_type": "G1",
            "fps": schema["fps"],
            "total_episodes": len(result),
            "total_frames": sum(lengths),
            "total_tasks": 1,
            "total_videos": len(schema["cameras"]) * len(result),
            "total_chunks": 1,
            "chunks_size": 1000,
            "splits": {"train": f"0:{len(result)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "conversion_provenance": {
                "contract": contract.contract_id,
                "gripper_convention": gripper_convention,
                "gripper_semantic": "closed_fraction",
                "allow_state_action_fallback": a.allow_state_action_fallback,
            },
        }
        (root / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf8")
        (root / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        k: v
                        for k, v in x.items()
                        if k not in ("features", "validation_errors", "stats")
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for x in result
            ),
            encoding="utf8",
        )
        (root / "meta/episodes_stats.jsonl").write_text(
            "".join(
                json.dumps(
                    {"episode_index": x["episode_index"], "stats": x["stats"]}, ensure_ascii=False
                )
                + "\n"
                for x in result
            ),
            encoding="utf8",
        )
        (root / "meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": spec["name"]}, ensure_ascii=False) + "\n",
            encoding="utf8",
        )
        (root / "meta/stats.json").write_text("{}", encoding="utf8")
        with (root / "conversion_validation.csv").open("w", newline="", encoding="utf8") as file:
            csv.writer(file).writerows(
                [
                    [
                        "episode_index",
                        "original_episode_id",
                        "frames",
                        "action_fallbacks",
                        "validation",
                    ],
                    *[
                        [
                            x["episode_index"],
                            x["original_episode_id"],
                            x["length"],
                            ";".join(x["action_fallbacks"]),
                            " ;".join(x["validation_errors"]) or "ok",
                        ]
                        for x in result
                    ],
                ]
            )
        print(spec["directory"], len(result), "episodes")
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}", flush=True)


if __name__ == "__main__":
    main()

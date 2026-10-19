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
from PIL import Image

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
from packages.camera_geometry import validate_no_crop_camera_schema
from packages.eva_qpos import assemble_g1_qpos
from packages.body_order import detect_body_order
from packages.raw_preflight import (

    add_empty_text_task_mappings,
    collect_raw_episodes,
    h5_stream_present,
    read_json,
    timestamps,
)

_DETECTED_BODY_ORDERS: list = []


# Throughput tuning defaults. They are implementation policy rather than data
# semantics, so callers normally should not need to repeat them.
DEFAULT_WORKERS = 8
DEFAULT_CUDA_WORKERS = 4
DEFAULT_QSV_WORKERS = 0


def nearest(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """为每个目标时间戳寻找时间上最近的源样本下标。"""
    # searchsorted 找到目标值应该插入的位置；候选帧只需比较插入点左右两帧。
    right = np.searchsorted(values, target, side="left")
    left = np.clip(right - 1, 0, len(values) - 1)
    right = np.clip(right, 0, len(values) - 1)
    # 返回距离更小的那一帧，供 H5 状态、动作和 JPG 图片共同使用。
    return np.where(np.abs(values[right] - target) < np.abs(values[left] - target), right, left)


def h5_series_required(
    h5: h5py.File, data_key: str, target: np.ndarray, width: int, required: bool = True
) -> np.ndarray | None:
    """读取一个 G1 H5 信号，并对齐到统一目标时间轴。

    缺失的必需字段直接拒绝，不用伪造的零值掩盖原始数据问题。
    """
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
    # 原始 H5 信号可能与头部相机采样频率不同；按时间戳取最近样本，
    # 让状态/动作和视频最终拥有同一帧数。
    result = np.asarray(values[nearest(times, target)], dtype=np.float32).reshape(len(target), -1)
    if result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{data_key}: expected finite width {width}, got {result.shape}")
    return result


def read_g1_qpos_inputs(
    h5: h5py.File, target: np.ndarray, raw_adapter: dict
) -> dict[str, np.ndarray]:
    """读取并拼装 G1 状态/动作所需的各个 H5 分量。"""
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
            # 每个字段都通过同一个时间对齐函数映射到 target 时间轴。
            values = h5_series_required(
                h5,
                data_key,
                target,
                widths[name],
                required=name not in {"action.head_position", "action.waist_position"},
            )
            if values is not None:
                if name.endswith("waist_position"):
                    group_attrs = h5[data_key.rsplit("/", 1)[0]].attrs
                    names = [
                        item.decode() if isinstance(item, bytes) else str(item)
                        for item in group_attrs.get("name", [])
                    ]
                    expected = ["joint_lift_body", "joint_body_pitch"]
                    if names:
                        if set(names) != set(expected):
                            raise ValueError(f"unsupported waist names: {names}")
                        values = values[:, [names.index(key) for key in expected]]
                columns[name] = values
    if raw_adapter.get("joint_boundary_policy") == "leading_state_trailing_hold":
        key = raw_adapter["action"]["action.joint_position"]
        times = np.asarray(h5[key.rsplit("/", 1)[0] + "/timestamp"])
        leading = target < times[0]
        columns["action.joint_position"][leading] = columns[
            "observation.state.joint_position"
        ][leading]
        # nearest() already holds the last recorded command after its timestamp.
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
    with av.open(str(ep / spec["source"])) as probe:
        stream = probe.streams.video[0]
        source_shape = [int(stream.height), int(stream.width), 3]
    expected_shape = list(spec["output_shape"])
    if source_shape != expected_shape:
        raise ValueError(
            f'{ep.name}: refusing to crop/resize {spec["source"]} from '
            f"{source_shape} to {expected_shape}; conversion must preserve source resolution"
        )
    source_times = timestamps(ep / spec["timestamp_source"])
    try:
        backend = encode_h265_ffmpeg(
            ep / spec["source"],
            source_times,
            target,
            dest,
            fps,
            None,
            preferred_backends,
        )
        print(f"{ep.name} {spec['source']} backend={backend}")
        return backend
    except ValueError as error:
        if "duplicate timestamp-aligned" not in str(error):
            raise
        encode_h265_aligned_pyav(ep, spec, target, dest, fps)
        return "pyav"


def encode_wrist(
    ep: Path, camera: str, spec: dict, target: np.ndarray, dest: Path, fps: int
) -> str:
    """将腕部相机的 JPG 图片按统一时间戳编码成 MP4。

    这里负责准备输入图片和时间对齐；具体的 CPU/QSV/NVENC 编码由
    packages.ffmpeg_io.encode_jpeg_sequence() 通过 FFmpeg 完成。
    """
    # 腕部相机的原始数据是逐张 JPG，而不是一个连续的视频文件。
    files = sorted((ep / "camera" / camera / "color").glob("*.jpg"))
    if not files:
        raise ValueError(f"{ep.name}: no JPG frames found for {camera}")
    with Image.open(files[0]) as image:
        source_shape = [image.height, image.width, 3]
    expected_shape = list(spec["output_shape"])
    if source_shape != expected_shape:
        raise ValueError(
            f'{ep.name}: refusing to resize JPG camera {camera} from '
            f"{source_shape} to {expected_shape}; conversion must preserve source resolution"
        )
    # 文件名本身是采集时间戳；nearest() 会为每个统一目标时间戳选择最近的一张图。
    source_ts = np.asarray([int(path.stem) for path in files], dtype=np.int64)
    selected = [files[index] for index in nearest(source_ts, target)]
    # 注意：当前函数使用 encode_jpeg_sequence 的默认后端顺序 cpu -> qsv -> cuda。
    backend = encode_jpeg_sequence(selected, dest, fps)
    print(f"{ep.name} {camera} backend={backend}")
    return backend


def convert_camera(
    args: tuple[Path, int, Path, str, dict, np.ndarray, int, tuple[str, ...]],
) -> dict:
    """转换一个 episode 的一条相机流，并返回可审计的后端和错误信息。"""
    ep, index, root, camera, spec, target, fps, backends = args
    # 每个相机最终对应一个独立 MP4；Parquet 中只保存视频字段元数据。
    destination = root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
    try:
        if spec["kind"] == "jpg_timestamp_aligned":
            # 腕部相机的原始输入是逐张 JPG，内部会按 target 时间戳选帧再编码。
            source_camera = "hand_left" if "left_wrist" in camera else "hand_right"
            backend = encode_wrist(ep, source_camera, spec, target, destination, fps)
        elif spec["kind"] == "h265_timestamp_aligned":
            # 头部/鱼眼相机的原始输入是 H.265，使用调度器传入的后端回退顺序。
            backend = encode_h265_aligned(ep, spec, target, destination, fps, backends)
        else:
            raise ValueError(f'unknown camera kind: {spec["kind"]}')
        # 相机编码函数内部已经校验一次，这里再从转换器层统一记录帧数结果。
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
    """验证一个 episode 的 Parquet 行数、索引边界和所有 MP4 帧数。"""
    errors = []
    data = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
    table = pq.read_table(data)
    # 数值表必须与统一视频时间轴一帧一行。
    if table.num_rows != expected:
        errors.append(f"parquet_rows:{table.num_rows}!={expected}")
    frames = np.asarray(table["frame_index"].to_pylist())
    if len(frames) and (frames[0] != 0 or frames[-1] != expected - 1):
        errors.append("frame_index_boundary_error")
    for camera in cameras:
        # 每个视频字段都必须有对应 MP4，且帧数与 Parquet 行数完全一致。
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


def combine_episode_stats(parts: list[dict]) -> dict:
    """Combine per-episode population statistics without decoding videos again."""
    count = sum(int(part["count"][0]) for part in parts)
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    stds = [np.asarray(part["std"], dtype=np.float64) for part in parts]
    weights = [int(part["count"][0]) for part in parts]
    mean = sum(value * weight for value, weight in zip(means, weights)) / count
    variance = (
        sum(
            (std**2 + (value - mean) ** 2) * weight
            for value, std, weight in zip(means, stds, weights)
        )
        / count
    )
    return {
        "min": np.minimum.reduce(
            [np.asarray(part["min"], dtype=np.float64) for part in parts]
        ).tolist(),
        "max": np.maximum.reduce(
            [np.asarray(part["max"], dtype=np.float64) for part in parts]
        ).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [count],
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
    args: tuple[Path, int, int, Path, dict, dict, dict, str, str, bool, tuple[str, ...], object],
) -> dict:
    """准备一个 episode 的数值 Parquet 和视频特征元数据。"""
    (
        ep,
        index,
        offset,
        root,
        schema,
        task,
        raw_adapter,
        gripper_convention,
        action_gripper_convention,
        allow_state_action_fallback,
        qpos_names,
        _lane_payload,
    ) = args
    # 头部相机时间轴作为本 episode 的统一目标时钟。
    meta = read_json(ep / "meta_info.json")
    target = timestamps(ep / "camera/head_color/head_color.txt")
    fps = int(schema["fps"])
    numeric = {}
    with h5py.File(ep / "record/raw_joints.h5", "r") as h5:
        # Source device timestamps align heterogeneous H5 streams, but encoded
        # MP4 is CFR at schema fps. LeRobot requires its parquet timestamps to
        # use the same exact CFR clock for deterministic frame lookup.
        # 先创建 LeRobot v2.1 的基础列；之后把状态和动作写入同一张 Parquet 表。
        columns = {
            "timestamp": pa.array(
                (np.arange(len(target), dtype=np.float32) / fps), type=pa.float32()
            ),
            "frame_index": pa.array(np.arange(len(target), dtype="int64")),
            "episode_index": pa.array(np.full(len(target), index, dtype="int64")),
            "index": pa.array(np.arange(len(target), dtype="int64") + offset),
            "task_index": pa.array(np.zeros(len(target), dtype="int64")),
        }
        # H5 状态/动作通过 target 时间戳对齐，并转换为最终 20 维 qpos。
        assembly = assemble_g1_qpos(
            read_g1_qpos_inputs(h5, target, raw_adapter),
            gripper_convention=gripper_convention,
            action_gripper_convention=action_gripper_convention,
            allow_state_action_fallback=allow_state_action_fallback,
        )
        state, action = assembly.state, assembly.action
        # EVA-CLIENT 契约正典：18/19 = [waist_pitch, lift]。
        # 原始录制流的腰部顺序随批次不同（1008 = lift-first，1015/1019 = waist-first），
        # 因此做“实测原始顺序 -> 契约顺序”的映射，而不是无条件/按契约名换位。
        contract_tail = tuple(qpos_names[-2:])
        if contract_tail not in (("waist_pitch", "lift"), ("lift", "waist_pitch")):
            raise ValueError(f"unsupported body order: {qpos_names[-2:]}")
        want = "waist_first" if contract_tail == ("waist_pitch", "lift") else "lift_first"
        raw_order = detect_body_order(state)
        if raw_order is not None and raw_order != want:
            state = state.copy()
            action = action.copy()
            state[:, [18, 19]] = state[:, [19, 18]]
            action[:, [18, 19]] = action[:, [19, 18]]
            body_order_note = f"reordered raw {raw_order} -> contract {want}"
        else:
            body_order_note = f"kept raw {raw_order or 'ambiguous'} (contract {want})"
        body_order_meta = {"contract_body_order": list(contract_tail), "raw_body_order_detected": raw_order, "action": body_order_note}
        _DETECTED_BODY_ORDERS.append(raw_order)
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
        # 视频不直接写入 Parquet；这里只写入 shape、fps、codec 等特征描述。
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
    # 物化数值数据；真正的 MP4 会在后续的 convert_camera 任务中生成。
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
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="原始 dataset/618data；也兼容辅助预检产生的分组输出",
    )
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
    p.add_argument(
        "--action-gripper-convention",
        choices=("raw_encoder_0_120", "opening_0_1", "closed_0_1"),
        help="Override the source-config action convention when converting a different recorder.",
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
    validate_no_crop_camera_schema(schema)
    source_config = read_json(a.source)
    contract = load_qpos_contract(a.contract)
    mapping = read_json(a.task_map)["tasks"]
    raw_adapter = source_config["raw_h5_adapter"]
    gripper_convention = a.gripper_convention or source_config["gripper_convention"]
    action_gripper_convention = (
        a.action_gripper_convention
        or source_config.get("action_gripper_convention")
        or gripper_convention
    )
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
    # 先只规划任务，不立即执行：数值 Parquet、H.265 视频和 JPG 视频分别进入不同队列。
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
        # 原始采集模式只物化本批次实际通过预检的任务；旧 task map 中的其他任务不应
        # 变成“meta 存在但 0 episode”的空 LeRobot 数据集。
        if raw_layout and not eps:
            continue
        root = a.output / spec["directory"]
        (root / "meta").mkdir(parents=True)
        # 每个 episode 使用头部相机时间戳作为统一目标时钟。
        targets = [timestamps(ep / "camera/head_color/head_color.txt") for ep in eps]
        lengths = [len(target) for target in targets]
        offsets = np.cumsum([0, *lengths[:-1]]).tolist()
        # prepare_episode 负责 H5 -> Parquet；它与视频编码在后面分阶段执行。
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
                action_gripper_convention,
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
        # 根据 schema 把每条相机流分到 H.265 或 JPG 队列。
        for index, (ep, target) in enumerate(zip(eps, targets)):
            for camera, camera_spec in schema["cameras"].items():
                job = (ep, index, root, camera, camera_spec, target, int(schema["fps"]))
                if camera_spec["kind"] == "h265_timestamp_aligned":
                    h265_jobs.append(job)
                elif camera_spec["kind"] == "jpg_timestamp_aligned":
                    jpg_jobs.append(job)
                else:
                    raise ValueError(f'unknown camera kind: {camera_spec["kind"]}')
    # 第一阶段：先并行物化数值 Parquet，确保状态/动作数据先有明确的帧轴。
    prepare_workers = max(1, min(a.workers, len(prepare_jobs) or 1))
    prepare_lanes = [ProcessLane("prepare", prepare_workers, None)]
    all_results = run_routed_process_jobs(prepare_episode, prepare_jobs, prepare_lanes)
    if a.workers == 1:
        # workers=1 时，所有任务都限制为单进程，便于调试和复现。
        h265_lanes = video_process_lanes(
            1, min(a.cuda_workers, 1), 0 if a.cuda_workers else min(a.qsv_workers, 1)
        )
        # H.265 任务按 CUDA/QSV/CPU 通道执行；JPG 任务单独使用 jpg_cpu 通道。
        h265_results = run_routed_process_jobs(convert_camera, h265_jobs, h265_lanes)
        jpg_results = run_routed_process_jobs(
            convert_camera, jpg_jobs, [ProcessLane("jpg_cpu", 1, ("cpu", "qsv", "cuda"))]
        )
    else:
        # workers>1 时，为 H.265 硬件编码和 JPG 编码分别保留进程预算。
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
        # 两个任务组可以并行协调，但每个组内部仍由自己的进程池限制并发。
        h265_results, jpg_results = run_process_job_groups(
            [
                (convert_camera, h265_jobs, h265_lanes),
                (convert_camera, jpg_jobs, jpg_lanes),
            ]
        )
    # 输出本次实际使用的进程通道，便于确认 CUDA/JPG 并发配置是否生效。
    print(
        "camera_process_lanes",
        [(lane.name, lane.workers) for lane in h265_lanes],
        ("jpg_cpu", 1 if a.workers == 1 else jpg_workers),
        flush=True,
    )
    # 汇总每条视频任务的错误；后面会合并到对应 episode 的 validation_errors。
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
            # 同时检查 Parquet 和每一路 MP4，避免只看到“文件生成”就误认为转换成功。
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
                "body_order": {
                    "contract_body_order": list(contract.qpos_names[-2:]),
                    "raw_body_order_detected": sorted({o for o in _DETECTED_BODY_ORDERS if o}),
                },
                "gripper_convention": gripper_convention,
                "state_gripper_convention": gripper_convention,
                "action_gripper_convention": action_gripper_convention,
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
        aggregate_stats = (
            {
                key: combine_episode_stats([episode["stats"][key] for episode in result])
                for key in result[0]["stats"]
            }
            if result
            else {}
        )
        (root / "meta/stats.json").write_text(
            json.dumps(aggregate_stats, ensure_ascii=False, indent=2), encoding="utf8"
        )
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

"""Generic range-plan application for task-scoped LeRobot v2.1 datasets."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from packages.ffmpeg_io import clip_video_frames, rgb_video_stats, video_frame_count
from packages.lerobot_v21 import episode_parquet, episodes, task_roots, video_files
from packages.range_plan import validate_plan


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()
    ]


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(len(array))],
    }


def _combine(parts: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(part["count"][0]) for part in parts)
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    stds = [np.asarray(part["std"], dtype=np.float64) for part in parts]
    sizes = [int(part["count"][0]) for part in parts]
    mean = sum(value * size for value, size in zip(means, sizes)) / count
    variance = (
        sum(
            (std * std + (value - mean) ** 2) * size for value, std, size in zip(means, stds, sizes)
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


def _replacement(field: pa.Field, values: np.ndarray) -> pa.Array:
    array = np.asarray(values)
    if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
        return pa.array(array.reshape(len(array), -1).tolist(), type=field.type)
    return pa.array(array.reshape(-1), type=field.type)


def _source_video(root: Path, episode_index: int, camera: str) -> Path:
    matches = [path for path in video_files(root, episode_index) if path.parent.name == camera]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{root.name} ep={episode_index}: expected one video for {camera!r}"
        )
    return matches[0]


def _split_job(
    job: tuple[Path, Path, int, int, int, int, int, float, tuple[str, ...], tuple[str, ...]]
) -> dict[str, Any]:
    (
        source_root,
        destination_root,
        source_episode,
        start,
        end,
        output_episode,
        global_offset,
        chunk_size,
        fps,
        numeric,
        cameras,
    ) = job
    source = pq.read_table(episode_parquet(source_root, source_episode))
    clipped = source.slice(start, end - start + 1)
    length = clipped.num_rows
    replacements: dict[str, np.ndarray] = {
        "timestamp": np.arange(length, dtype=np.float32) / fps,
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, output_episode, dtype=np.int64),
        "index": np.arange(length, dtype=np.int64) + global_offset,
    }
    for name, values in replacements.items():
        if name in clipped.column_names:
            position = clipped.schema.get_field_index(name)
            clipped = clipped.set_column(
                position, name, _replacement(clipped.schema.field(position), values)
            )
    target = (
        destination_root
        / "data"
        / f"chunk-{output_episode // chunk_size:03d}"
        / f"episode_{output_episode:06d}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(clipped, target, compression="snappy")
    for camera in cameras:
        video_target = (
            destination_root
            / "videos"
            / f"chunk-{output_episode // chunk_size:03d}"
            / camera
            / f"episode_{output_episode:06d}.mp4"
        )
        clip_video_frames(
            _source_video(source_root, source_episode, camera), video_target, start, end, fps
        )
    statistics = {
        name: _stats(np.asarray(clipped[name].to_pylist()))
        for name in numeric
        if name in clipped.column_names
    }
    for camera in cameras:
        video_target = (
            destination_root
            / "videos"
            / f"chunk-{output_episode // chunk_size:03d}"
            / camera
            / f"episode_{output_episode:06d}.mp4"
        )
        statistics[camera] = rgb_video_stats(video_target)
    return {
        "episode_index": output_episode,
        "length": length,
        "dataset_from_index": global_offset,
        "dataset_to_index": global_offset + length,
        "source_episode_index": source_episode,
        "source_range": [start, end],
        "stats": statistics,
    }


def _prepare_metadata(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite split output: {destination}")
    destination.mkdir(parents=True)
    (destination / "meta").mkdir()
    for name in ("info.json", "tasks.jsonl"):
        candidate = source / "meta" / name
        if candidate.exists():
            shutil.copy2(candidate, destination / "meta" / name)
    return json.loads((destination / "meta/info.json").read_text(encoding="utf8"))


def preflight_split(input_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Read-only gate: prove planned source slices are readable before writing."""
    validate_plan(plan)
    source_by_name = {root.name: root for root in task_roots(input_root)}
    report: dict[str, Any] = {"schema": "split-preflight/v1", "tasks": []}
    for entry in plan["episodes"]:
        task = entry["task"]
        root = source_by_name.get(task)
        if root is None:
            raise ValueError(f"range plan references missing task directory {task!r}")
        episode = int(entry["episode_index"])
        source_rows = {int(row["episode_index"]): row for row in episodes(root)}
        if episode not in source_rows:
            raise ValueError(f"{task}: range plan references missing episode {episode}")
        expected = int(entry["source_frames"])
        if int(source_rows[episode]["length"]) != expected:
            raise ValueError(f"{task} ep={episode}: plan frame count differs from metadata")
        table = pq.read_table(episode_parquet(root, episode))
        if table.num_rows != expected:
            raise ValueError(f"{task} ep={episode}: source parquet rows mismatch")
        info = json.loads((root / "meta/info.json").read_text(encoding="utf8"))
        cameras = tuple(
            name for name, value in info["features"].items() if value.get("dtype") == "video"
        )
        for camera in cameras:
            video = _source_video(root, episode, camera)
            if video_frame_count(video) != expected:
                raise ValueError(
                    f"{task} ep={episode}: source video frame count mismatch for {camera}"
                )
        report["tasks"].append(
            {
                "task_directory": task,
                "episode_index": episode,
                "source_frames": expected,
                "kept_frames": sum(end - start + 1 for start, end in entry["keep_ranges"]),
            }
        )
    if not report["tasks"]:
        raise ValueError("range plan has no selected episodes")
    return report


def split_dataset(
    input_root: Path, output_root: Path, plan: dict[str, Any], workers: int = 4
) -> dict[str, Any]:
    """Apply a final range plan without mutating the input dataset."""
    preflight_split(input_root, plan)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    output_root.mkdir(parents=True)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for item in plan["episodes"]:
        by_task.setdefault(item["task"], []).append(item)
    report: dict[str, Any] = {"schema": "split-report/v1", "tasks": []}
    for source in task_roots(input_root):
        entries = sorted(by_task.get(source.name, []), key=lambda item: int(item["episode_index"]))
        if not entries:
            continue
        destination = output_root / source.name
        info = _prepare_metadata(source, destination)
        fps = float(info["fps"])
        chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
        features = info["features"]
        cameras = tuple(name for name, value in features.items() if value.get("dtype") == "video")
        numeric = tuple(name for name, value in features.items() if value.get("dtype") != "video")
        source_rows = {int(row["episode_index"]): row for row in episodes(source)}
        jobs = []
        output_episode = 0
        global_offset = 0
        for entry in entries:
            source_episode = int(entry["episode_index"])
            if source_episode not in source_rows:
                raise ValueError(
                    f"{source.name}: range plan references missing episode {source_episode}"
                )
            if int(source_rows[source_episode]["length"]) != int(entry["source_frames"]):
                raise ValueError(
                    f"{source.name} ep={source_episode}: plan frame count differs from metadata"
                )
            for start, end in entry["keep_ranges"]:
                length = end - start + 1
                jobs.append(
                    (
                        source,
                        destination,
                        source_episode,
                        start,
                        end,
                        output_episode,
                        global_offset,
                        chunk_size,
                        fps,
                        numeric,
                        cameras,
                    )
                )
                output_episode += 1
                global_offset += length
        if not jobs:
            raise ValueError(f"{source.name}: selected plan entries contain no frames")
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as pool:
            futures = [pool.submit(_split_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: int(item["episode_index"]))
        task_name = (
            _jsonl(source / "meta/tasks.jsonl")[0]["task"]
            if (source / "meta/tasks.jsonl").exists()
            else ""
        )
        episode_rows = [
            {
                "episode_index": item["episode_index"],
                "tasks": [task_name],
                "length": item["length"],
                "dataset_from_index": item["dataset_from_index"],
                "dataset_to_index": item["dataset_to_index"],
                "source_episode_index": item["source_episode_index"],
                "source_range": item["source_range"],
            }
            for item in results
        ]
        (destination / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
            encoding="utf8",
        )
        (destination / "meta/episodes_stats.jsonl").write_text(
            "".join(
                json.dumps(
                    {"episode_index": item["episode_index"], "stats": item["stats"]},
                    ensure_ascii=False,
                )
                + "\n"
                for item in results
            ),
            encoding="utf8",
        )
        info["total_episodes"] = len(results)
        info["total_frames"] = global_offset
        info["total_videos"] = len(results) * len(cameras)
        (destination / "meta/info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
        aggregate = {
            name: _combine([item["stats"][name] for item in results])
            for name in [*numeric, *cameras]
            if all(name in item["stats"] for item in results)
        }
        (destination / "meta/stats.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
        validate_split_task(destination)
        report["tasks"].append(
            {"task_directory": source.name, "episodes": len(results), "frames": global_offset}
        )
    if not report["tasks"]:
        raise ValueError("range plan did not select any task in the input dataset")
    (output_root / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    return report


def validate_split_task(root: Path) -> None:
    """Full read, index and video frame-count validation for one output task."""
    info = json.loads((root / "meta/info.json").read_text(encoding="utf8"))
    cameras = tuple(
        name for name, value in info["features"].items() if value.get("dtype") == "video"
    )
    rows = _jsonl(root / "meta/episodes.jsonl")
    global_index = 0
    for row in rows:
        episode = int(row["episode_index"])
        expected = int(row["length"])
        table = pq.read_table(episode_parquet(root, episode))
        if table.num_rows != expected:
            raise ValueError(f"{root.name} ep={episode}: parquet rows mismatch")
        if table["frame_index"].to_pylist() != list(range(expected)):
            raise ValueError(f"{root.name} ep={episode}: frame_index is not contiguous")
        if table["index"].to_pylist() != list(range(global_index, global_index + expected)):
            raise ValueError(f"{root.name} ep={episode}: dataset index is not contiguous")
        for camera in cameras:
            videos = [path for path in video_files(root, episode) if path.parent.name == camera]
            if len(videos) != 1 or video_frame_count(videos[0]) != expected:
                raise ValueError(f"{root.name} ep={episode}: video validation failed for {camera}")
        global_index += expected
    if int(info["total_frames"]) != global_index:
        raise ValueError(f"{root.name}: total_frames mismatch")

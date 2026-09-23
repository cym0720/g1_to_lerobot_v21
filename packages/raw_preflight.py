"""Read-only validation of raw G1 episode directories.

The converter owns output creation and encoding.  This module deliberately
owns only source inspection, so another G1 importer can reuse the exact same
rejection policy without importing converter implementation details.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# These two action streams are absent in the known 618 recorder.  A caller may
# explicitly opt in to using the matching state streams; every other action
# stream remains mandatory.
STATE_ACTION_FALLBACK_KEYS = frozenset({"action.head_position", "action.waist_position"})
FISHEYE_CAMERAS = ("head_center_fisheye", "head_left_fisheye", "head_right_fisheye")


def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON configuration or episode metadata file."""
    return json.loads(path.read_text(encoding="utf8"))


def timestamps(path: Path) -> np.ndarray:
    """Read nanosecond device timestamps from the first column of a text file."""
    return np.asarray(
        [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.int64,
    )


def _h5_stream_present(h5: h5py.File, data_key: str) -> bool:
    base = data_key.rsplit("/", 1)[0]
    timestamp_key = base + "/timestamp"
    return (
        data_key in h5
        and timestamp_key in h5
        and len(h5[data_key]) > 0
        and len(h5[timestamp_key]) > 0
    )


def source_errors(
    episode: Path,
    min_duration_s: float,
    raw_adapter: dict,
    allow_state_action_fallback: bool,
    require_fisheyes: bool = True,
) -> tuple[dict | None, list[str]]:
    """Return parsed metadata and all defects that make an episode unusable.

    This is intentionally read-only.  The later full LeRobot validation is
    still required after conversion because it validates generated Parquet and
    video artifacts rather than only source prerequisites.
    """
    meta_path = episode / "meta_info.json"
    errors: list[str] = []
    if not meta_path.is_file():
        return None, ["missing_meta_info.json"]
    try:
        meta = read_json(meta_path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid_meta_info.json:{exc}"]

    for relative in (
        "record/raw_joints.h5",
        "camera/head_color/head_color.h265",
        "camera/head_color/head_color.txt",
    ):
        if not (episode / relative).is_file():
            errors.append(f"missing_{relative}")
    for wrist in ("hand_left", "hand_right"):
        if not list((episode / "camera" / wrist / "color").glob("*.jpg")):
            errors.append(f"missing_{wrist}_jpg")
    if require_fisheyes:
        for fisheye in FISHEYE_CAMERAS:
            for suffix in (".h265", ".txt"):
                if not (episode / "camera" / fisheye / f"{fisheye}{suffix}").is_file():
                    errors.append(f"missing_{fisheye}{suffix}")

    h5_path = episode / "record/raw_joints.h5"
    if h5_path.is_file():
        try:
            with h5py.File(h5_path, "r") as h5:
                for group in ("state", "action"):
                    for output, data_key in raw_adapter[group].items():
                        fallback_allowed = (
                            allow_state_action_fallback and output in STATE_ACTION_FALLBACK_KEYS
                        )
                        if not _h5_stream_present(h5, data_key) and not fallback_allowed:
                            errors.append(f"missing_or_empty_{data_key}")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid_raw_joints.h5:{exc}")

    head_timestamps = episode / "camera/head_color/head_color.txt"
    if head_timestamps.is_file():
        try:
            values = timestamps(head_timestamps)
            duration = (values[-1] - values[0]) / 1e9 if len(values) >= 2 else 0.0
            if duration < min_duration_s:
                errors.append(f"too_short_head_duration_s:{duration:.3f}<{min_duration_s:g}")
        except (ValueError, OSError) as exc:
            errors.append(f"invalid_head_timestamps:{exc}")
    if require_fisheyes:
        for fisheye in FISHEYE_CAMERAS:
            stamp = episode / "camera" / fisheye / f"{fisheye}.txt"
            if stamp.is_file():
                try:
                    if len(timestamps(stamp)) == 0:
                        errors.append(f"empty_{fisheye}_timestamps")
                except (ValueError, OSError) as exc:
                    errors.append(f"invalid_{fisheye}_timestamps:{exc}")
    return meta, errors


def collect_raw_episodes(
    source: Path,
    mapping: dict,
    min_duration_s: float,
    raw_adapter: dict,
    allow_state_action_fallback: bool,
    require_fisheyes: bool = True,
) -> tuple[dict[str, list[Path]], list[list[str]]]:
    """Group valid episodes by mapped task and return every rejection reason."""
    grouped = {spec["directory"]: [] for spec in mapping.values()}
    rejected: list[list[str]] = []
    for episode in sorted(item for item in source.iterdir() if item.is_dir()):
        meta, errors = source_errors(
            episode, min_duration_s, raw_adapter, allow_state_action_fallback, require_fisheyes
        )
        task_id = "" if meta is None else str(meta.get("task_id", ""))
        spec = mapping.get(task_id)
        if spec is None and not errors:
            errors.append(f"unmapped_task_id:{task_id}")
        if errors:
            rejected.append([episode.name, task_id, ";".join(errors)])
            continue
        grouped[spec["directory"]].append(episode)
    return grouped, rejected


def add_empty_text_task_mappings(source: Path, mapping: dict) -> None:
    """Preserve mapped raw episodes without text only when caller opted in."""
    for episode in source.iterdir():
        meta_path = episode / "meta_info.json"
        if not episode.is_dir() or not meta_path.is_file():
            continue
        try:
            task_id = str(read_json(meta_path).get("task_id", ""))
        except (OSError, json.JSONDecodeError):
            continue
        if task_id and task_id not in mapping:
            mapping[task_id] = {
                "order": 1000 + int(task_id) if task_id.isdigit() else 9999,
                "name": "",
                "directory": f"empty_text_task_{task_id}",
            }

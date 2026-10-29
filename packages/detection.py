"""Reusable signal detectors. They only produce evidence and frame ranges."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from packages.range_plan import normalize_ranges


def gripper_candidate_ranges(
    qpos: np.ndarray,
    gripper_indices: tuple[int, int],
    jump_threshold: float,
    guard_frames: int = 0,
    min_segment_frames: int = 1,
) -> tuple[list[list[int]], list[int]]:
    """Split an episode at left/right closed-fraction gripper jumps.

    Returns inclusive candidate ranges and the first frame after every detected
    discontinuity. This function never writes a dataset.
    """
    values = np.asarray(qpos, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("qpos must have shape [frames, dimensions]")
    if any(index < 0 or index >= values.shape[1] for index in gripper_indices):
        raise ValueError("gripper index outside qpos dimensions")
    delta = np.abs(np.diff(values[:, list(gripper_indices)], axis=0))
    jumps = (np.flatnonzero(np.max(delta, axis=1) >= jump_threshold) + 1).tolist()
    cursor = 0
    ranges: list[list[int]] = []
    for jump in jumps:
        end = jump - guard_frames - 1
        if end >= cursor:
            ranges.append([cursor, end])
        cursor = max(cursor, jump + guard_frames)
    if cursor < len(values):
        ranges.append([cursor, len(values) - 1])
    return normalize_ranges(ranges, len(values), min_segment_frames, merge_adjacent=False), jumps


def pca_motion_scores(qpos: np.ndarray, feature_indices: Iterable[int]) -> np.ndarray:
    """Return a robust z-score of first-component frame-to-frame motion."""
    values = np.asarray(qpos, dtype=np.float64)
    indices = list(feature_indices)
    if values.ndim != 2 or len(values) < 2 or not indices:
        raise ValueError("need at least two qpos frames and one feature index")
    signal = values[:, indices]
    centered = signal - signal.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-8, scale, 1.0)
    _, _, components = np.linalg.svd(standardized, full_matrices=False)
    projection = standardized @ components[0]
    speed = np.abs(np.diff(projection, prepend=projection[0]))
    median = np.median(speed)
    mad = np.median(np.abs(speed - median))
    return (speed - median) / max(1.4826 * mad, 1e-8)


def fuse_pingpong_hits(
    candidate_ranges: Iterable[Iterable[int]],
    ball_frames: Iterable[int],
    pca_hit_frames: Iterable[int],
    max_frame_delta: int,
) -> tuple[list[list[int]], list[dict[str, int]]]:
    """Keep only candidate ranges supported by temporally aligned ball and PCA evidence."""
    pca = sorted(set(int(frame) for frame in pca_hit_frames))
    hits: list[dict[str, int]] = []
    for ball in sorted(set(int(frame) for frame in ball_frames)):
        if not pca:
            break
        nearest = min(pca, key=lambda frame: abs(frame - ball))
        if abs(nearest - ball) <= max_frame_delta:
            hits.append({"ball_frame": ball, "pca_frame": nearest})
    selected: list[list[int]] = []
    for pair in candidate_ranges:
        start, end = list(pair)
        if any(start <= hit["ball_frame"] <= end for hit in hits):
            selected.append([start, end])
    return selected, hits

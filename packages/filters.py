"""Causal startup filters for teleoperation action trajectories."""
from __future__ import annotations

import numpy as np


def smooth_startup_action(values: np.ndarray, fps: float, motion_duration_s: float = 0.5, cutoff_hz: float = 2.5, pre_roll_frames: int = 5, post_roll_frames: int = 5) -> tuple[np.ndarray, dict[str, int]]:
    """Low-pass the preserved pre-roll, startup motion, and post-roll transition.

    The final post-roll frames crossfade back to raw action, making the filtered
    segment and untouched remainder continuous at the boundary.
    """
    if fps <= 0 or motion_duration_s <= 0 or cutoff_hz <= 0:
        raise ValueError("fps, motion_duration_s, and cutoff_hz must be positive")
    pre_roll = max(0, pre_roll_frames)
    motion_frames = max(1, round(fps * motion_duration_s))
    post_roll = max(0, post_roll_frames)
    window_frames = min(len(values), pre_roll + motion_frames + post_roll)
    if window_frames < 3:
        return values.copy(), {"pre_roll_frames": min(pre_roll, len(values)), "motion_frames": max(0, window_frames - pre_roll), "post_roll_frames": 0, "window_frames": window_frames}
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps)
    filtered = values[:window_frames].copy()
    for index in range(1, window_frames):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    actual_post = min(post_roll, window_frames)
    if actual_post:
        weights = np.linspace(0.0, 1.0, actual_post, dtype=np.float32).reshape(actual_post, *([1] * (values.ndim - 1)))
        filtered[-actual_post:] = (1.0 - weights) * filtered[-actual_post:] + weights * values[window_frames - actual_post:window_frames]
    output = values.copy()
    output[:window_frames] = filtered
    return output, {"pre_roll_frames": min(pre_roll, window_frames), "motion_frames": max(0, window_frames - min(pre_roll, window_frames) - actual_post), "post_roll_frames": actual_post, "window_frames": window_frames}

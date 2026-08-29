"""Causal startup filters for teleoperation action trajectories."""
from __future__ import annotations

import numpy as np


def smooth_startup_action(values: np.ndarray, fps: float, duration_s: float = 0.5, cutoff_hz: float = 2.5, blend_frames: int = 10) -> tuple[np.ndarray, int]:
    """Smooth only the first duration_s with a causal first-order low-pass filter."""
    if fps <= 0 or duration_s <= 0 or cutoff_hz <= 0:
        raise ValueError("fps, duration_s, and cutoff_hz must be positive")
    n = min(len(values), max(1, round(fps * duration_s)))
    if n < 3:
        return values.copy(), n
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps)
    filtered = values[:n].copy()
    for index in range(1, n):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    blend = min(max(0, blend_frames), n)
    if blend:
        weights = np.linspace(0.0, 1.0, blend, dtype=np.float32).reshape(blend, *([1] * (values.ndim - 1)))
        filtered[-blend:] = (1.0 - weights) * filtered[-blend:] + weights * values[n - blend:n]
    output = values.copy()
    output[:n] = filtered
    return output, n

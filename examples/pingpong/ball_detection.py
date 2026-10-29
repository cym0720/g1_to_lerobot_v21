"""Orange-ball detection and temporal tracking for G1 ping-pong videos."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from packages.lerobot_v21 import video_files


@dataclass(frozen=True)
class BallDetectorConfig:
    """Appearance and motion thresholds for the existing OpenCV detector."""

    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    min_area: float
    max_area_fraction: float
    min_radius: float
    max_radius_fraction: float
    min_circularity: float
    cameras: tuple[str, ...]
    max_track_jump_fraction: float
    min_move_pixels: float
    motion_window: int
    min_moving_steps: int
    min_roi_area_ratio: float
    max_roi_area_ratio: float

    @classmethod
    def load(cls, path: Path) -> "BallDetectorConfig":
        raw = json.loads(path.read_text(encoding="utf8"))
        config = cls(
            hsv_low=tuple(raw["hsv_low"]),
            hsv_high=tuple(raw["hsv_high"]),
            min_area=float(raw["min_area"]),
            max_area_fraction=float(raw["max_area_fraction"]),
            min_radius=float(raw["min_radius"]),
            max_radius_fraction=float(raw["max_radius_fraction"]),
            min_circularity=float(raw["min_circularity"]),
            cameras=tuple(raw["cameras"]),
            max_track_jump_fraction=float(raw["max_track_jump_fraction"]),
            min_move_pixels=float(raw["min_move_pixels"]),
            motion_window=int(raw["motion_window"]),
            min_moving_steps=int(raw["min_moving_steps"]),
            min_roi_area_ratio=float(raw["min_roi_area_ratio"]),
            max_roi_area_ratio=float(raw["max_roi_area_ratio"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.hsv_low) != 3 or len(self.hsv_high) != 3:
            raise ValueError("HSV bounds must each contain H, S, and V")
        if any(value < 0 or value > 255 for value in (*self.hsv_low, *self.hsv_high)):
            raise ValueError("HSV channel bounds must be in 0..255")
        if any(low > high for low, high in zip(self.hsv_low, self.hsv_high)):
            raise ValueError("each hsv_low channel must be <= hsv_high")
        if (
            not self.cameras
            or self.motion_window < 1
            or not 1 <= self.min_moving_steps <= self.motion_window
        ):
            raise ValueError("invalid camera or motion-window configuration")
        if min(self.min_area, self.min_radius, self.max_track_jump_fraction) <= 0:
            raise ValueError("area, radius, and track-jump thresholds must be positive")
        if not 0 < self.max_area_fraction <= 1 or not 0 < self.max_radius_fraction <= 1:
            raise ValueError("maximum area/radius fractions must be in (0, 1]")
        if not 0 < self.min_roi_area_ratio <= self.max_roi_area_ratio:
            raise ValueError("ROI area ratios must be positive and ordered")


@dataclass
class BallCandidate:
    center: np.ndarray
    box: tuple[int, int, int, int]
    roi_area: float
    color: np.ndarray
    score: float


def find_candidates(frame: np.ndarray, config: BallDetectorConfig) -> list[BallCandidate]:
    """Find orange, roughly circular candidates in one BGR image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray(config.hsv_low, dtype=np.uint8),
        np.asarray(config.hsv_high, dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = frame.shape[0] * frame.shape[1] * config.max_area_fraction
    max_radius = min(frame.shape[:2]) * config.max_radius_fraction
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < config.min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < config.min_radius or radius > max_radius:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        pixels = hsv[contour_mask > 0]
        color = pixels.mean(axis=0) if len(pixels) else np.zeros(3, dtype=np.float32)
        candidates.append(
            BallCandidate(
                center=np.asarray([cx, cy], dtype=np.float32),
                box=(x, y, width, height),
                roi_area=float(width * height),
                color=color.astype(np.float32),
                score=float(area * circularity),
            )
        )
    return candidates


class BallTrack:
    """Track a candidate using frame-to-frame position, color, and size continuity."""

    def __init__(self, frame_shape: tuple[int, ...], config: BallDetectorConfig):
        self.config = config
        self.max_jump = max(frame_shape[:2]) * config.max_track_jump_fraction
        self.center: np.ndarray | None = None
        self.color: np.ndarray | None = None
        self.roi_area: float | None = None
        self.history: list[np.ndarray] = []

    def update(self, candidates: list[BallCandidate]) -> BallCandidate | None:
        if not candidates:
            self.reset()
            return None
        if self.center is None:
            selected = max(candidates, key=lambda candidate: candidate.score)
        else:
            nearby = [
                candidate
                for candidate in candidates
                if np.linalg.norm(candidate.center - self.center) <= self.max_jump
                and self._area_matches(candidate)
            ]
            if not nearby:
                self.reset()
                return None

            def cost(candidate: BallCandidate) -> float:
                distance = np.linalg.norm(candidate.center - self.center) / max(
                    self.max_jump, 1.0
                )
                color_distance = (
                    np.linalg.norm(candidate.color - self.color) / 100.0
                    if self.color is not None
                    else 0.0
                )
                return distance + color_distance - min(candidate.score / 1000.0, 1.0) * 0.05

            selected = min(nearby, key=cost)

        self.center = selected.center
        self.color = selected.color
        self.roi_area = selected.roi_area
        self.history.append(selected.center.copy())
        self.history = self.history[-(self.config.motion_window + 1) :]
        return selected

    def reset(self) -> None:
        self.center = None
        self.color = None
        self.roi_area = None
        self.history.clear()

    def _area_matches(self, candidate: BallCandidate) -> bool:
        if self.roi_area is None:
            return True
        ratio = candidate.roi_area / max(self.roi_area, 1.0)
        return self.config.min_roi_area_ratio <= ratio <= self.config.max_roi_area_ratio

    def is_moving(self) -> bool:
        if len(self.history) < self.config.motion_window + 1:
            return False
        distances = [
            float(np.linalg.norm(current - previous))
            for previous, current in zip(self.history, self.history[1:])
        ]
        return (
            sum(distance >= self.config.min_move_pixels for distance in distances)
            >= self.config.min_moving_steps
        )


def _camera_videos(
    task_root: Path, episode_index: int, cameras: tuple[str, ...]
) -> dict[str, Path]:
    available = video_files(task_root, episode_index)
    selected: dict[str, Path] = {}
    for camera in cameras:
        matches = [path for path in available if camera in path.parent.name]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one video for {camera} in {task_root} episode {episode_index}; "
                f"found {len(matches)}"
            )
        selected[camera] = matches[0]
    return selected


def scan_video(path: Path, expected_frames: int, config: BallDetectorConfig) -> list[int]:
    """Return detected ball frame indices; fail if video/qpos indexing can diverge."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open ping-pong camera video: {path}")
    track: BallTrack | None = None
    ball_frames: list[int] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if track is None:
                track = BallTrack(frame.shape, config)
            candidate = track.update(find_candidates(frame, config))
            if candidate is not None and track.is_moving():
                ball_frames.append(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if frame_index != expected_frames:
        raise ValueError(
            f"frame count mismatch for {path}: video={frame_index}, "
            f"qpos={expected_frames}; ball/PCA frame alignment is unsafe"
        )
    return ball_frames


def detect_episode_ball_frames(
    task_root: Path,
    episode_index: int,
    expected_frames: int,
    config: BallDetectorConfig,
) -> dict[str, list[int]]:
    """Track configured camera views and return per-camera positive frame indices."""
    paths = _camera_videos(task_root, episode_index, config.cameras)
    return {
        camera: scan_video(path, expected_frames, config)
        for camera, path in paths.items()
    }

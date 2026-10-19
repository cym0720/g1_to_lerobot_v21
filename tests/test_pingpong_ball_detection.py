from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import examples.pingpong.detect_pingpong_hit as hit_step
from examples.pingpong.ball_detection import (
    BallDetectorConfig,
    BallTrack,
    find_candidates,
    scan_video,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_orange_ball_candidates_are_detected() -> None:
    config = BallDetectorConfig.load(REPO_ROOT / "config/detectors/pingpong_ball.json")
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(frame, (40, 50), 5, (65, 106, 183), thickness=-1)

    candidates = find_candidates(frame, config)

    assert len(candidates) == 1
    assert np.allclose(candidates[0].center, (40, 50), atol=1)


def test_ball_track_requires_continuous_motion() -> None:
    config = BallDetectorConfig.load(REPO_ROOT / "config/detectors/pingpong_ball.json")
    track = BallTrack((100, 120, 3), config)
    detections = []

    for x in (30, 34, 38, 42):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.circle(frame, (x, 50), 5, (65, 106, 183), thickness=-1)
        candidate = track.update(find_candidates(frame, config))
        detections.append(candidate is not None and track.is_moving())

    assert detections == [False, False, False, True]


def test_video_scan_returns_frame_indices_and_checks_alignment() -> None:
    config = BallDetectorConfig.load(REPO_ROOT / "config/detectors/pingpong_ball.json")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "moving_ball.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 100)
        )
        if not writer.isOpened():
            pytest.skip("OpenCV MJPG video writer is unavailable")
        for x in (30, 34, 38, 42, 46, 50):
            frame = np.zeros((100, 120, 3), dtype=np.uint8)
            cv2.circle(frame, (x, 50), 5, (65, 106, 183), thickness=-1)
            writer.write(frame)
        writer.release()

        assert scan_video(path, expected_frames=6, config=config) == [3, 4, 5]
        with pytest.raises(ValueError, match="frame count mismatch"):
            scan_video(path, expected_frames=5, config=config)


def test_step_two_runs_ball_detection_and_fuses_with_pca(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "task_a"
    metadata = task_root / "meta"
    parquet_path = task_root / "data" / "chunk-000" / "episode_000000.parquet"
    metadata.mkdir(parents=True)
    parquet_path.parent.mkdir(parents=True)
    (metadata / "info.json").write_text('{"chunks_size": 1000}', encoding="utf8")
    qpos = np.zeros((10, 20), dtype=np.float32)
    pq.write_table(
        pa.table(
            {
                "observations.state.qpos": pa.array(
                    qpos.tolist(), type=pa.list_(pa.float32(), 20)
                )
            }
        ),
        parquet_path,
    )

    monkeypatch.setattr(
        hit_step,
        "detect_episode_ball_frames",
        lambda *_args: {"cam_high": [5], "cam_left_wrist": []},
    )
    monkeypatch.setattr(
        hit_step,
        "pca_motion_scores",
        lambda *_args: np.array([0, 0, 0, 0, 0, 4, 0, 0, 0, 0], dtype=np.float32),
    )
    ball_config = BallDetectorConfig.load(REPO_ROOT / "config/detectors/pingpong_ball.json")

    episode = hit_step._process_episode(
        (
            task_root,
            {
                "task": "task_a",
                "episode_index": 0,
                "source_frames": 10,
                "keep_ranges": [[0, 9]],
            },
            "observations.state.qpos",
            (8, 9, 10, 11, 12, 13, 14),
            3.0,
            2,
            None,
            ball_config,
        )
    )

    assert episode["keep_ranges"] == [[0, 9]]
    assert episode["evidence"]["aligned_hits"] == [{"ball_frame": 5, "pca_frame": 5}]
    assert episode["evidence"]["ball_frames_by_camera"]["cam_high"] == [5]

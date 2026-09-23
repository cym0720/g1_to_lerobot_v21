"""契约、乒乓球检测和通用切分的无数据集回归测试。

这些测试只构造微型 HDF5、Parquet 和视频 fixture，因此能在不读取真实录制数据的情况下，
验证输出 qpos 顺序、区间语义以及切分后元数据和视频的一致性。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess

import numpy as np
import h5py
import pyarrow as pa
import pyarrow.parquet as pq

from packages.contracts import load_qpos_contract
from packages.detection import fuse_pingpong_hits, gripper_candidate_ranges, pca_motion_scores
from packages.eva_qpos import assemble_g1_qpos, to_closed_fraction
from packages.range_plan import make_plan
from packages.lerobot_v21 import episode_parquet
from packages.ball_events import (
    BallEventsError,
    index_ball_events,
    load_ball_events,
    make_ball_events,
)
from packages.dataset_split import preflight_split, split_dataset, validate_split_task
from packages.ffmpeg_io import video_frame_count
from packages.raw_preflight import source_errors


REPO = Path(__file__).resolve().parents[1]


class ContractAndDetectionTests(unittest.TestCase):
    """保护跨脚本共享的数据契约与不可回退的清洗门禁。"""

    def test_eva_g1_contract_has_expected_qpos_layout(self) -> None:
        """EVA replay 的 G1 state/action 必须保持 20 维固定顺序。"""
        contract = load_qpos_contract(REPO / "config/contracts/eva_replay_g1_qpos20_v1.json")
        self.assertEqual(contract.robot, "g1")
        self.assertEqual(len(contract.qpos_names), 20)
        self.assertEqual(contract.gripper_indices, (7, 15))
        self.assertEqual(
            contract.qpos_names[16:], ("head_yaw", "head_pitch", "lift", "waist_pitch")
        )

    def test_gripper_ranges_are_inclusive_and_filtered(self) -> None:
        """夹爪候选范围使用闭区间，并丢弃短于阈值的片段。"""
        qpos = np.zeros((12, 20), dtype=np.float32)
        qpos[5:, 7] = 1.0
        ranges, jumps = gripper_candidate_ranges(qpos, (7, 15), 0.5, min_segment_frames=3)
        self.assertEqual(jumps, [5])
        self.assertEqual(ranges, [[0, 4], [5, 11]])

    def test_pingpong_fusion_requires_ball_and_pca_evidence(self) -> None:
        """乒乓球最终裁剪必须同时有球事件和 PCA 运动证据。"""
        selected, hits = fuse_pingpong_hits([[0, 9], [10, 19]], [4, 14], [5, 40], 2)
        self.assertEqual(selected, [[0, 9]])
        self.assertEqual(hits, [{"ball_frame": 4, "pca_frame": 5}])

    def test_pca_scores_match_frame_count(self) -> None:
        """PCA 分数始终与原始帧数对齐，方便后续融合。"""
        qpos = np.zeros((16, 20), dtype=np.float32)
        qpos[8:, 8] = 4.0
        self.assertEqual(len(pca_motion_scores(qpos, range(8, 15))), 16)

    def test_range_plan_normalizes_adjacent_ranges(self) -> None:
        """相邻的保留范围写入契约前必须归一为一个范围。"""
        plan = make_plan(
            contract="eva-replay-g1-qpos20-v1",
            source_dataset="fixture",
            stage="gripper-candidates",
            episodes=[
                {
                    "task": "task_a",
                    "episode_index": 0,
                    "source_frames": 8,
                    "keep_ranges": [[0, 2], [3, 4]],
                }
            ],
        )
        self.assertEqual(plan["episodes"][0]["keep_ranges"], [[0, 4]])

    def test_g1_qpos_uses_contract_order_and_closed_grippers(self) -> None:
        """原始 0=开、120=闭编码应转换为契约内的 closed fraction。"""
        frames = 2
        columns = {
            "observation.state.joint_position": np.arange(frames * 14, dtype=np.float32).reshape(
                frames, 14
            ),
            "observation.state.left_effector_position": np.array(
                [[0.0], [120.0]], dtype=np.float32
            ),
            "observation.state.right_effector_position": np.array(
                [[120.0], [0.0]], dtype=np.float32
            ),
            "observation.state.head_position": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "observation.state.waist_position": np.array(
                [[5.0, 6.0], [7.0, 8.0]], dtype=np.float32
            ),
            "action.joint_position": np.arange(frames * 14, dtype=np.float32).reshape(frames, 14),
            "action.left_effector_position": np.array([[0.0], [120.0]], dtype=np.float32),
            "action.right_effector_position": np.array([[120.0], [0.0]], dtype=np.float32),
            "action.head_position": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "action.waist_position": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        }
        assembly = assemble_g1_qpos(columns, gripper_convention="raw_encoder_0_120")
        self.assertEqual(assembly.state.shape, (2, 20))
        self.assertEqual(assembly.state[0, 7], 0.0)
        self.assertEqual(assembly.state[0, 15], 1.0)
        self.assertEqual(assembly.state[0, 16:].tolist(), [1.0, 2.0, 5.0, 6.0])

    def test_opening_input_is_inverted_and_action_fallback_is_explicit(self) -> None:
        self.assertEqual(
            to_closed_fraction(np.array([[0.25]], dtype=np.float32), "opening_0_1")[0, 0], 0.75
        )
        columns = {
            "observation.state.joint_position": np.zeros((1, 14), dtype=np.float32),
            "observation.state.left_effector_position": np.zeros((1, 1), dtype=np.float32),
            "observation.state.right_effector_position": np.zeros((1, 1), dtype=np.float32),
            "observation.state.head_position": np.zeros((1, 2), dtype=np.float32),
            "observation.state.waist_position": np.zeros((1, 2), dtype=np.float32),
            "action.joint_position": np.zeros((1, 14), dtype=np.float32),
            "action.left_effector_position": np.zeros((1, 1), dtype=np.float32),
            "action.right_effector_position": np.zeros((1, 1), dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            assemble_g1_qpos(columns, gripper_convention="closed_0_1")
        assembly = assemble_g1_qpos(
            columns, gripper_convention="closed_0_1", allow_state_action_fallback=True
        )
        self.assertEqual(
            assembly.action_fallbacks,
            ("observation.state.head_position", "observation.state.waist_position"),
        )

    def test_raw_preflight_requires_explicit_head_waist_action_fallback(self) -> None:
        """Only documented missing action streams may use the compatibility path."""
        source = json.loads((REPO / "config/sources/g1.json").read_text(encoding="utf8"))
        adapter = source["raw_h5_adapter"]
        with TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode"
            (episode / "record").mkdir(parents=True)
            (episode / "camera/head_color").mkdir(parents=True)
            for wrist in ("hand_left", "hand_right"):
                wrist_dir = episode / "camera" / wrist / "color"
                wrist_dir.mkdir(parents=True)
                (wrist_dir / "frame.jpg").touch()
            (episode / "meta_info.json").write_text('{"task_id": "4"}', encoding="utf8")
            (episode / "camera/head_color/head_color.h265").touch()
            (episode / "camera/head_color/head_color.txt").write_text(
                "0\n100000000\n", encoding="utf8"
            )
            with h5py.File(episode / "record/raw_joints.h5", "w") as h5:
                for group in ("state", "action"):
                    for output, key in adapter[group].items():
                        if output in {"action.head_position", "action.waist_position"}:
                            continue
                        h5.create_dataset(key, data=np.zeros((2, 1), dtype=np.float32))
                        h5.create_dataset(
                            key.rsplit("/", 1)[0] + "/timestamp",
                            data=np.array([0, 100000000], dtype=np.int64),
                        )
            _, strict_errors = source_errors(
                episode, 0.0, adapter, allow_state_action_fallback=False, require_fisheyes=False
            )
            self.assertIn("missing_or_empty_action/head/position", strict_errors)
            self.assertIn("missing_or_empty_action/waist/position", strict_errors)
            _, fallback_errors = source_errors(
                episode, 0.0, adapter, allow_state_action_fallback=True, require_fisheyes=False
            )
            self.assertEqual(fallback_errors, [])

    def test_episode_parquet_uses_declared_chunk_size(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text('{"chunks_size": 2}', encoding="utf8")
            expected = root / "data/chunk-001/episode_000003.parquet"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(episode_parquet(root, 3), expected)

    def test_ball_events_require_matching_contract_and_unique_episode(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ball_events.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ball-events/v1",
                        "contract": "eva-replay-g1-qpos20-v1",
                        "episodes": [
                            {"task": "task_a", "episode_index": 0, "frame_indices": [8, 2, 8]}
                        ],
                    }
                ),
                encoding="utf8",
            )
            self.assertEqual(
                load_ball_events(path, "eva-replay-g1-qpos20-v1"), {("task_a", 0): [2, 8]}
            )
            with self.assertRaises(BallEventsError):
                load_ball_events(path, "other-contract")

    def test_ball_events_builder_deduplicates_detector_rows(self) -> None:
        payload = make_ball_events(
            "eva-replay-g1-qpos20-v1",
            [("task_b", 1, 8), ("task_b", 1, 2), ("task_b", 1, 8)],
        )
        self.assertEqual(payload["episodes"][0]["frame_indices"], [2, 8])
        self.assertEqual(
            index_ball_events(payload, "eva-replay-g1-qpos20-v1"), {("task_b", 1): [2, 8]}
        )

    def test_generic_split_rewrites_dataset_and_videos(self) -> None:
        """A material split must prove Parquet, indexes, metadata and video agree."""
        with TemporaryDirectory() as temporary:
            parent = Path(temporary) / "input"
            root = parent / "task_a"
            output = Path(temporary) / "output"
            (root / "meta").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            video = root / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
            video.parent.mkdir(parents=True)
            features = {
                "observations.state.qpos": {"dtype": "float32", "shape": [20]},
                "action.qpos": {"dtype": "float32", "shape": [20]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "observation.images.cam_high": {"dtype": "video", "shape": [8, 8, 3]},
            }
            info = {
                "codebase_version": "v2.1",
                "fps": 5,
                "chunks_size": 1000,
                "features": features,
                "total_episodes": 1,
                "total_frames": 6,
                "total_videos": 1,
            }
            (root / "meta/info.json").write_text(json.dumps(info), encoding="utf8")
            (root / "meta/tasks.jsonl").write_text(
                '{"task_index": 0, "task": "fixture task"}\n', encoding="utf8"
            )
            (root / "meta/episodes.jsonl").write_text(
                '{"episode_index": 0, "tasks": ["fixture task"], "length": 6, "dataset_from_index": 0, "dataset_to_index": 6}\n',
                encoding="utf8",
            )
            vector_type = pa.list_(pa.float32(), 20)
            vectors = np.arange(120, dtype=np.float32).reshape(6, 20).tolist()
            table = pa.table(
                {
                    "observations.state.qpos": pa.array(vectors, type=vector_type),
                    "action.qpos": pa.array(vectors, type=vector_type),
                    "timestamp": pa.array(np.arange(6, dtype=np.float32) / 5),
                    "frame_index": pa.array(range(6), type=pa.int64()),
                    "episode_index": pa.array([0] * 6, type=pa.int64()),
                    "index": pa.array(range(6), type=pa.int64()),
                    "task_index": pa.array([0] * 6, type=pa.int64()),
                }
            )
            pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=8x8:rate=5",
                    "-frames:v",
                    "6",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            plan = make_plan(
                contract="eva-replay-g1-qpos20-v1",
                source_dataset="fixture",
                stage="pingpong-final-crop",
                episodes=[
                    {
                        "task": "task_a",
                        "episode_index": 0,
                        "source_frames": 6,
                        "keep_ranges": [[1, 3], [5, 5]],
                    }
                ],
            )

            preflight = preflight_split(parent, plan)
            self.assertEqual(preflight["tasks"][0]["kept_frames"], 4)
            report = split_dataset(parent, output, plan, workers=1)
            destination = output / "task_a"
            self.assertEqual(
                report["tasks"], [{"task_directory": "task_a", "episodes": 2, "frames": 4}]
            )
            validate_split_task(destination)
            rows = [
                json.loads(line)
                for line in (destination / "meta/episodes.jsonl")
                .read_text(encoding="utf8")
                .splitlines()
            ]
            self.assertEqual([row["length"] for row in rows], [3, 1])
            first = pq.read_table(episode_parquet(destination, 0))
            second = pq.read_table(episode_parquet(destination, 1))
            self.assertEqual(first["observations.state.qpos"].to_pylist()[0], vectors[1])
            self.assertEqual(second["observations.state.qpos"].to_pylist()[0], vectors[5])
            self.assertEqual(first["index"].to_pylist() + second["index"].to_pylist(), [0, 1, 2, 3])
            self.assertEqual(
                video_frame_count(
                    destination / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
                ),
                3,
            )
            self.assertEqual(
                video_frame_count(
                    destination / "videos/chunk-000/observation.images.cam_high/episode_000001.mp4"
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()

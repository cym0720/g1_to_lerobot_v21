from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from packages.contracts import load_qpos_contract
from packages.detection import fuse_pingpong_hits, gripper_candidate_ranges, pca_motion_scores
from packages.range_plan import make_plan


REPO = Path(__file__).resolve().parents[1]


class ContractAndDetectionTests(unittest.TestCase):
    def test_eva_g1_contract_has_expected_qpos_layout(self) -> None:
        contract = load_qpos_contract(REPO / "config/contracts/eva_replay_g1_qpos20_v1.json")
        self.assertEqual(contract.robot, "g1")
        self.assertEqual(len(contract.qpos_names), 20)
        self.assertEqual(contract.gripper_indices, (7, 15))
        self.assertEqual(contract.qpos_names[16:], ("head_yaw", "head_pitch", "lift", "waist_pitch"))

    def test_gripper_ranges_are_inclusive_and_filtered(self) -> None:
        qpos = np.zeros((12, 20), dtype=np.float32)
        qpos[5:, 7] = 1.0
        ranges, jumps = gripper_candidate_ranges(qpos, (7, 15), 0.5, min_segment_frames=3)
        self.assertEqual(jumps, [5])
        self.assertEqual(ranges, [[0, 4], [5, 11]])

    def test_pingpong_fusion_requires_ball_and_pca_evidence(self) -> None:
        selected, hits = fuse_pingpong_hits([[0, 9], [10, 19]], [4, 14], [5, 40], 2)
        self.assertEqual(selected, [[0, 9]])
        self.assertEqual(hits, [{"ball_frame": 4, "pca_frame": 5}])

    def test_pca_scores_match_frame_count(self) -> None:
        qpos = np.zeros((16, 20), dtype=np.float32)
        qpos[8:, 8] = 4.0
        self.assertEqual(len(pca_motion_scores(qpos, range(8, 15))), 16)

    def test_range_plan_normalizes_adjacent_ranges(self) -> None:
        plan = make_plan(
            contract="eva-replay-g1-qpos20-v1",
            source_dataset="fixture",
            stage="gripper-candidates",
            episodes=[{"task": "task_a", "episode_index": 0, "source_frames": 8, "keep_ranges": [[0, 2], [3, 4]]}],
        )
        self.assertEqual(plan["episodes"][0]["keep_ranges"], [[0, 4]])


if __name__ == "__main__":
    unittest.main()

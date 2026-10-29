#!/usr/bin/env python3
"""Export one LeRobot episode's ground-truth actions in deploy_g1 replay JSON format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEPLOY_NAMES = [
    *(f"left_arm_joint{i}" for i in range(1, 8)), "left_gripper",
    *(f"right_arm_joint{i}" for i in range(1, 8)), "right_gripper",
    "head_yaw", "head_pitch", "lift", "waist_pitch",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--episode-parquet", type=Path, required=True)
    parser.add_argument("--gt-arrays", type=Path, required=True)
    parser.add_argument("--gt-report", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads(args.dataset_info.read_text(encoding="utf-8"))
    report = json.loads(args.gt_report.read_text(encoding="utf-8"))
    episode_info = report["episodes"][str(args.episode)]
    source_names = info["features"]["action"]["names"]
    if len(source_names) != 20 or set(source_names) != set(DEPLOY_NAMES):
        raise ValueError("Dataset action names do not match the deploy_g1 qpos20 contract")

    table = pq.read_table(args.episode_parquet, columns=["action", "frame_index", "timestamp"])
    parquet_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    with np.load(args.gt_arrays) as arrays:
        gt_actions = np.asarray(arrays["gt_action"], dtype=np.float32)

    if parquet_actions.shape != (len(frame_indices), 20):
        raise ValueError(f"Expected source action shape [N,20], got {parquet_actions.shape}")
    if gt_actions.shape != parquet_actions.shape:
        raise ValueError(f"GT-analysis/source shape mismatch: {gt_actions.shape} vs {parquet_actions.shape}")
    if not np.isfinite(parquet_actions).all() or not np.isfinite(gt_actions).all():
        raise ValueError("Source or GT action contains non-finite values")
    if not np.array_equal(gt_actions, parquet_actions):
        max_diff = float(np.max(np.abs(gt_actions - parquet_actions)))
        raise ValueError(f"GT analysis actions differ from dataset actions (max abs diff {max_diff})")
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError("Episode frame indices are not a contiguous zero-based sequence")
    for name in ("left_gripper", "right_gripper"):
        column = source_names.index(name)
        if np.any(parquet_actions[:, column] < 0) or np.any(parquet_actions[:, column] > 1):
            raise ValueError(f"{name} values fall outside [0,1]")

    # The dataset's names are authoritative: its last two dimensions are
    # waist_pitch, lift; deploy_g1 expects lift, waist_pitch.
    permutation = [source_names.index(name) for name in DEPLOY_NAMES]
    actions = parquet_actions[:, permutation]
    source_meta = {
        "kind": "dataset_ground_truth_action",
        "control_scope": "right_arm_right_gripper",
        "dataset": report["dataset"],
        "split": episode_info["split"],
        "episode_index": args.episode,
        "source_episode_index": episode_info["source_episode_index"],
        "source_episode_id": episode_info["original_episode_id"],
        "source_dataset": episode_info["source_dataset"],
        "source_group": episode_info["source_group"],
        "action_source": "LeRobot episode Parquet column 'action'; verified elementwise against GT analysis gt_action",
        "gt_analysis_checkpoint": report["checkpoint"],
        "gt_analysis_checkpoint_sha256": report["checkpoint_sha256"],
        "source_action_names": source_names,
        "deploy_action_names": DEPLOY_NAMES,
        "model_to_deploy_index": permutation,
        "mapping_rule": "name-based permutation from dataset meta/info.json to deploy_g1 qpos20 order",
        "action_fields": {
            "normalization": "raw physical dataset action; no model prediction or inverse normalization",
            "arm_joint_unit": "radian",
            "gripper_unit": "closed_fraction_0_open_1_closed",
            "control_scope": "right_arm_right_gripper; replayer holds other dimensions at fresh feedback",
            "dataset_tail_order": ["waist_pitch", "lift"],
            "deploy_tail_order": ["lift", "waist_pitch"],
        },
        # Required by the current replay reader; these are framewise GT actions,
        # not model-predicted chunks. The executor groups them in blocks of 8.
        "selected_action_steps_per_chunk": 8,
        "sequence_kind": "framewise_ground_truth_actions",
    }
    payload = {
        "schema_version": "g1-eva-qpos20-v1",
        "state_names": DEPLOY_NAMES,
        "actions": actions.astype(float).tolist(),
        "fps": float(info["fps"]),
        "source_frame_indices": frame_indices,
        "source_timestamps_sec": timestamps,
        "source": source_meta,
        "execution_block_size": 8,
        "execution_blocking": "Framewise GT sequence; AGI replayer applies actions in blocks of 8 with its existing interpolation and safety checks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(json.dumps({
        "status": "exported",
        "output": str(args.output.resolve()),
        "frames": len(actions),
        "action_shape": list(actions.shape),
        "fps": payload["fps"],
        "source_gt_exact_match": True,
        "source_action_names": source_names,
        "deploy_action_names": DEPLOY_NAMES,
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "right_gripper_minmax": [
            float(actions[:, DEPLOY_NAMES.index("right_gripper")].min()),
            float(actions[:, DEPLOY_NAMES.index("right_gripper")].max()),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

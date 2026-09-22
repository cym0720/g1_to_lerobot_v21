"""Step 1 of the ping-pong pipeline: qpos gripper events -> candidate ranges."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.contracts import load_qpos_contract
from packages.detection import gripper_candidate_ranges
from packages.lerobot_v21 import episode_parquet, episodes, task_roots
from packages.range_plan import make_plan, write_plan


def _detect_episode(job: tuple[Path, dict, str, tuple[int, int], float, int, int]) -> dict:
    root, episode, qpos_key, gripper_indices, threshold, guard_frames, min_segment_frames = job
    episode_index = int(episode["episode_index"])
    table = pq.read_table(episode_parquet(root, episode_index), columns=[qpos_key])
    qpos = np.asarray(table[qpos_key].to_pylist(), dtype=np.float32)
    ranges, jumps = gripper_candidate_ranges(
        qpos, gripper_indices, threshold, guard_frames, min_segment_frames
    )
    return {
        "task": root.name,
        "episode_index": episode_index,
        "source_frames": int(len(qpos)),
        "keep_ranges": ranges,
        "reason": ["gripper_jump"] if jumps else ["no_gripper_jump"],
        "detectors": ["gripper-segments-v1"],
        "evidence": {"gripper_jump_frames": jumps},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "config/contracts/eva_replay_g1_qpos20_v1.json",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=REPO_ROOT / "config/detectors/gripper_segments.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input does not exist: {args.input}")
    contract = load_qpos_contract(args.contract)
    config = json.loads(args.detector_config.read_text(encoding="utf8"))
    if tuple(config["gripper_names"]) != tuple(
        contract.qpos_names[index] for index in contract.gripper_indices
    ):
        raise SystemExit("detector gripper names do not match the selected contract")
    jobs = []
    for root in task_roots(args.input):
        for episode in episodes(root):
            jobs.append(
                (
                    root,
                    episode,
                    contract.qpos_key,
                    contract.gripper_indices,
                    float(config["jump_threshold"]),
                    int(config["guard_frames"]),
                    int(config["min_segment_frames"]),
                )
            )
    if not jobs:
        raise SystemExit("input contains no LeRobot v2.1 episodes")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        futures = [pool.submit(_detect_episode, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            results.append(entry)
            if args.debug:
                print(f"{entry['task']} ep={entry['episode_index']} ranges={entry['keep_ranges']}")
            print(f"gripper_plan_progress={completed}/{len(jobs)}", flush=True)
    plan = make_plan(
        contract=contract.contract_id,
        source_dataset=str(args.input),
        stage="gripper-candidates",
        episodes=sorted(results, key=lambda item: (item["task"], item["episode_index"])),
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        write_plan(args.output, plan)
        print(f"candidate_plan={args.output}")


if __name__ == "__main__":
    main()

"""Step 2: fuse ball-events/v1 with right-arm PCA evidence into final crop ranges."""
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

from packages.ball_events import load_ball_events
from packages.contracts import load_qpos_contract
from packages.detection import fuse_pingpong_hits, pca_motion_scores
from packages.lerobot_v21 import episode_parquet, task_roots
from packages.range_plan import load_plan, make_plan, write_plan


def _process_episode(
    job: tuple[Path, dict, str, tuple[int, ...], float, int, list[int]]
) -> dict:
    root, candidate, qpos_key, feature_indices, score_threshold, max_delta, ball_frames = job
    episode_index = int(candidate["episode_index"])
    table = pq.read_table(episode_parquet(root, episode_index), columns=[qpos_key])
    qpos = np.asarray(table[qpos_key].to_pylist(), dtype=np.float32)
    candidate_ranges = candidate["keep_ranges"]
    pca_frames: list[int] = []
    for start, end in candidate_ranges:
        if end - start + 1 < 2:
            continue
        scores = pca_motion_scores(qpos[start : end + 1], feature_indices)
        pca_frames.extend((start + np.flatnonzero(scores >= score_threshold)).tolist())
    kept, hits = fuse_pingpong_hits(candidate_ranges, ball_frames, pca_frames, max_delta)
    return {
        "task": candidate["task"],
        "episode_index": episode_index,
        "source_frames": int(candidate["source_frames"]),
        "keep_ranges": kept,
        "reason": ["ball_and_pca_hit"] if kept else ["no_aligned_ball_pca_hit"],
        "detectors": ["pingpong-hit-v1"],
        "evidence": {
            "ball_frames": ball_frames,
            "pca_hit_frames": sorted(set(int(frame) for frame in pca_frames)),
            "aligned_hits": hits,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--ball-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "config/contracts/eva_replay_g1_qpos20_v1.json",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=REPO_ROOT / "config/detectors/pingpong_hit.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    contract = load_qpos_contract(args.contract)
    candidates = load_plan(args.candidate_plan)
    if candidates["contract"] != contract.contract_id or candidates["stage"] != "gripper-candidates":
        raise SystemExit("candidate plan must be a gripper-candidates plan for the selected contract")
    ball_events = load_ball_events(args.ball_events, contract.contract_id)
    config = json.loads(args.detector_config.read_text(encoding="utf8"))
    feature_indices = tuple(contract.index(name) for name in config["pca_feature_names"])
    roots = {root.name: root for root in task_roots(args.input)}
    jobs = []
    for candidate in candidates["episodes"]:
        task = candidate["task"]
        if task not in roots:
            raise SystemExit(f"candidate task {task!r} is absent from --input")
        jobs.append((
            roots[task], candidate, contract.qpos_key, feature_indices,
            float(config["pca_score_z_threshold"]), int(config["ball_pca_max_frame_delta"]),
            ball_events.get((task, int(candidate["episode_index"])), []),
        ))
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs) or 1))) as pool:
        futures = [pool.submit(_process_episode, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            results.append(entry)
            if args.debug:
                print(f"{entry['task']} ep={entry['episode_index']} final={entry['keep_ranges']}")
            print(f"pingpong_hit_progress={completed}/{len(jobs)}", flush=True)
    final_plan = make_plan(
        contract=contract.contract_id,
        source_dataset=str(args.input),
        stage="pingpong-final-crop",
        episodes=sorted(results, key=lambda item: (item["task"], item["episode_index"])),
    )
    if args.dry_run:
        print(json.dumps(final_plan, ensure_ascii=False, indent=2))
    else:
        write_plan(args.output, final_plan)
        print(f"final_crop_plan={args.output}")


if __name__ == "__main__":
    main()

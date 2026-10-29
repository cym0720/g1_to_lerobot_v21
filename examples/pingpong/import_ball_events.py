"""Import any ball detector's frame-level CSV output as `ball-events/v1`."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.ball_events import BallEventsError, make_ball_events
from packages.contracts import load_qpos_contract
from packages.range_plan import load_plan


REQUIRED_COLUMNS = ("task", "episode_index", "frame_index")


def read_csv_events(path: Path) -> list[tuple[str, int, int]]:
    """Read detector rows without imposing an OpenCV/model-specific schema."""
    with path.open("r", encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise BallEventsError(f"{path}: CSV needs columns {', '.join(REQUIRED_COLUMNS)}")
        events = []
        for line, row in enumerate(reader, start=2):
            try:
                events.append(
                    (str(row["task"]), int(row["episode_index"]), int(row["frame_index"]))
                )
            except (TypeError, ValueError) as error:
                raise BallEventsError(
                    f"{path}:{line}: invalid task/episode_index/frame_index"
                ) from error
    return events


def validate_against_candidates(events: list[tuple[str, int, int]], plan: dict) -> None:
    """Reject events outside the source episode bounds before hit fusion runs."""
    bounds = {
        (item["task"], int(item["episode_index"])): int(item["source_frames"])
        for item in plan["episodes"]
    }
    for task, episode, frame in events:
        source_frames = bounds.get((task, episode))
        if source_frames is None:
            raise BallEventsError(f"{task}/{episode}: absent from candidate range plan")
        if frame >= source_frames:
            raise BallEventsError(f"{task}/{episode}: frame {frame} outside 0..{source_frames - 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "config/contracts/eva_replay_g1_qpos20_v1.json",
    )
    parser.add_argument(
        "--candidate-plan",
        type=Path,
        help="Optional gripper-candidates range plan for task/episode/frame-bound validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    contract = load_qpos_contract(args.contract)
    events = read_csv_events(args.events_csv)
    if args.candidate_plan:
        candidates = load_plan(args.candidate_plan)
        if (
            candidates["contract"] != contract.contract_id
            or candidates["stage"] != "gripper-candidates"
        ):
            raise SystemExit(
                "--candidate-plan must be gripper-candidates for the selected contract"
            )
        validate_against_candidates(events, candidates)
    payload = make_ball_events(contract.contract_id, events)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(f"ball_events={args.output} episodes={len(payload['episodes'])}")


if __name__ == "__main__":
    main()

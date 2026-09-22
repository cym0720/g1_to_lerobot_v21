"""Apply a final inclusive range-plan/v1 to a LeRobot v2.1 dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.dataset_split import preflight_split, split_dataset
from packages.range_plan import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.dry_run:
        report = preflight_split(args.input, plan)
        print(json.dumps({"input": str(args.input), "output": str(args.output), "preflight": report}, ensure_ascii=False, indent=2))
        return
    report = split_dataset(args.input, args.output, plan, args.workers)
    if args.debug:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"split_report={args.output / 'split_report.json'}")


if __name__ == "__main__":
    main()

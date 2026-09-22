# Purpose: verify global CUDA/QSV/CPU process-lane routing and stable result order.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.execution import run_routed_process_jobs, video_process_lanes


def routed_worker(args: tuple[int, tuple[str, ...]]) -> tuple[int, str]:
    value, backends = args
    return value * value, backends[0]


def main() -> None:
    lanes = video_process_lanes(total_workers=4, cuda_workers=2, qsv_workers=1)
    results = run_routed_process_jobs(routed_worker, [(value,) for value in range(8)], lanes)
    assert [value for value, _ in results] == [value * value for value in range(8)]
    assert [backend for _, backend in results] == [
        "cuda",
        "cuda",
        "qsv",
        "cpu",
        "cuda",
        "cuda",
        "qsv",
        "cpu",
    ]
    print("process_routing_ok", results)


if __name__ == "__main__":
    main()

"""Reusable process-pool routing for CPU and hardware-accelerated workloads."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ProcessLane:
    """A bounded process pool with a payload passed to every routed job."""

    name: str
    workers: int
    payload: Any


def video_process_lanes(total_workers: int, cuda_workers: int, qsv_workers: int) -> list[ProcessLane]:
    """Build bounded CUDA/QSV/CPU lanes whose worker counts sum to total_workers."""
    if total_workers < 1 or cuda_workers < 0 or qsv_workers < 0:
        raise ValueError("worker counts must be non-negative and total_workers must be positive")
    cuda = min(cuda_workers, total_workers)
    qsv = min(qsv_workers, total_workers - cuda)
    cpu = total_workers - cuda - qsv
    lanes = []
    if cuda:
        lanes.append(ProcessLane("cuda", cuda, ("cuda", "qsv", "cpu")))
    if qsv:
        lanes.append(ProcessLane("qsv", qsv, ("qsv", "cpu")))
    if cpu:
        lanes.append(ProcessLane("cpu", cpu, ("cpu",)))
    return lanes


def run_routed_process_jobs(
    worker: Callable[[tuple[Any, ...]], Any],
    jobs: Sequence[tuple[Any, ...]],
    lanes: Iterable[ProcessLane],
) -> list[Any]:
    """Run jobs globally while strictly bounding concurrency for each backend lane.

    Results preserve input order. Each worker receives the original job tuple with
    the selected lane payload appended as its final element.
    """
    lane_list = [lane for lane in lanes if lane.workers > 0]
    if not lane_list:
        raise ValueError("at least one process lane is required")
    weighted_lanes = [lane for lane in lane_list for _ in range(lane.workers)]
    results: list[Any] = [None] * len(jobs)
    with ExitStack() as stack:
        pools = {
            lane.name: stack.enter_context(ProcessPoolExecutor(max_workers=lane.workers))
            for lane in lane_list
        }
        futures: dict[Future[Any], int] = {}
        for index, job in enumerate(jobs):
            lane = weighted_lanes[index % len(weighted_lanes)]
            futures[pools[lane.name].submit(worker, (*job, lane.payload))] = index
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def run_process_job_groups(
    groups: Sequence[tuple[Callable[[tuple[Any, ...]], Any], Sequence[tuple[Any, ...]], Sequence[ProcessLane]]],
) -> list[list[Any]]:
    """Run independent routed process-pool groups concurrently."""
    if not groups:
        return []
    with ThreadPoolExecutor(max_workers=len(groups)) as coordinator:
        futures = [
            coordinator.submit(run_routed_process_jobs, worker, jobs, lanes)
            for worker, jobs, lanes in groups
        ]
        return [future.result() for future in futures]

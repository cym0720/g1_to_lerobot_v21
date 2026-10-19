"""Reusable process-pool routing for CPU and hardware-accelerated workloads."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ProcessLane:
    """一个有并发上限的进程通道。

    name 用来标识后端通道，workers 是该通道允许同时运行的进程数，
    payload 会在任务提交时追加到 worker 参数末尾，例如 CUDA 通道会携带
    ("cuda", "qsv", "cpu") 作为后端回退顺序。
    """

    name: str
    workers: int
    payload: Any


def video_process_lanes(
    total_workers: int, cuda_workers: int, qsv_workers: int
) -> list[ProcessLane]:
    """构造 CUDA/QSV/CPU 三类视频处理通道。

    这里不直接调用 CUDA，而是控制有多少个 FFmpeg 子进程可以并行尝试
    CUDA、QSV 或 CPU 后端；所有通道的 workers 总和不超过 total_workers。
    """
    if total_workers < 1 or cuda_workers < 0 or qsv_workers < 0:
        raise ValueError("worker counts must be non-negative and total_workers must be positive")
    # 先分配 CUDA，再分配 QSV，剩余名额留给 CPU。
    cuda = min(cuda_workers, total_workers)
    qsv = min(qsv_workers, total_workers - cuda)
    cpu = total_workers - cuda - qsv
    lanes = []
    if cuda:
        # CUDA 通道优先尝试 CUDA，失败后回退到 QSV，再回退到 CPU。
        lanes.append(ProcessLane("cuda", cuda, ("cuda", "qsv", "cpu")))
    if qsv:
        # QSV 通道不再尝试 CUDA，避免把明确分配给 Intel 后端的任务改走 NVIDIA。
        lanes.append(ProcessLane("qsv", qsv, ("qsv", "cpu")))
    if cpu:
        # CPU 通道只使用 CPU，适合不依赖硬件编码的任务。
        lanes.append(ProcessLane("cpu", cpu, ("cpu",)))
    return lanes


def run_routed_process_jobs(
    worker: Callable[[tuple[Any, ...]], Any],
    jobs: Sequence[tuple[Any, ...]],
    lanes: Iterable[ProcessLane],
) -> list[Any]:
    """把任务分配到受控的进程池并保持结果顺序。

    结果按输入 jobs 的顺序返回。每个 worker 收到原始任务参数，最后再追加
    所属通道的 payload；因此 convert_camera 能知道应该优先尝试哪些编码后端。
    """
    lane_list = [lane for lane in lanes if lane.workers > 0]
    if not lane_list:
        raise ValueError("at least one process lane is required")
    # 将通道按 workers 展开，例如 CUDA=2、QSV=1、CPU=1 会变成四个槽位；
    # 后续按轮询方式把任务均匀放入这些槽位。
    weighted_lanes = [lane for lane in lane_list for _ in range(lane.workers)]
    results: list[Any] = [None] * len(jobs)
    with ExitStack() as stack:
        # 每个通道拥有独立的 ProcessPoolExecutor，避免 CUDA 并发数失控。
        pools = {
            lane.name: stack.enter_context(ProcessPoolExecutor(max_workers=lane.workers))
            for lane in lane_list
        }
        futures: dict[Future[Any], int] = {}
        for index, job in enumerate(jobs):
            lane = weighted_lanes[index % len(weighted_lanes)]
            # 把 lane.payload 追加到任务参数末尾；worker 本身不需要了解调度器。
            futures[pools[lane.name].submit(worker, (*job, lane.payload))] = index
        for future in as_completed(futures):
            # future 完成顺序可能乱序，因此使用提交时保存的 index 写回原位置。
            results[futures[future]] = future.result()
    return results


def run_process_job_groups(
    groups: Sequence[
        tuple[Callable[[tuple[Any, ...]], Any], Sequence[tuple[Any, ...]], Sequence[ProcessLane]]
    ],
) -> list[list[Any]]:
    """并行运行互不依赖的任务组，例如 H.265 组和 JPG 组。"""
    if not groups:
        return []
    # 这里的线程只负责协调多个进程池；真正的转换工作仍在子进程中执行。
    with ThreadPoolExecutor(max_workers=len(groups)) as coordinator:
        futures = [
            coordinator.submit(run_routed_process_jobs, worker, jobs, lanes)
            for worker, jobs, lanes in groups
        ]
        return [future.result() for future in futures]

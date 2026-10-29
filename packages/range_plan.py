"""Contract-driven, inclusive frame-range plans shared by all splitters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class RangePlanError(ValueError):
    pass


def normalize_ranges(
    ranges: Iterable[Iterable[int]],
    source_frames: int,
    min_length: int = 1,
    *,
    merge_adjacent: bool = True,
) -> list[list[int]]:
    """Validate, sort and merge inclusive ranges for one episode."""
    if source_frames < 1:
        raise RangePlanError("source_frames must be positive")
    normalized: list[list[int]] = []
    for pair in ranges:
        values = list(pair)
        if len(values) != 2 or not all(isinstance(value, int) for value in values):
            raise RangePlanError(f"range must be two integer frame indices: {values!r}")
        start, end = values
        if start < 0 or end < start or end >= source_frames:
            raise RangePlanError(f"range [{start}, {end}] outside 0..{source_frames - 1}")
        if end - start + 1 >= min_length:
            normalized.append([start, end])
    merged: list[list[int]] = []
    for start, end in sorted(normalized):
        overlaps = merged and start <= merged[-1][1]
        adjacent = merged and merge_adjacent and start == merged[-1][1] + 1
        if overlaps or adjacent:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def make_plan(
    *, contract: str, source_dataset: str, stage: str, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    plan = {
        "schema": "range-plan/v1",
        "contract": contract,
        "source_dataset": source_dataset,
        "stage": stage,
        "range_semantics": "inclusive",
        "episodes": episodes,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != "range-plan/v1" or plan.get("range_semantics") != "inclusive":
        raise RangePlanError("only inclusive range-plan/v1 is supported")
    if not isinstance(plan.get("contract"), str) or not plan["contract"]:
        raise RangePlanError("range plan must identify its source contract")
    seen: set[tuple[str, int]] = set()
    for entry in plan.get("episodes", []):
        task = entry.get("task")
        episode = entry.get("episode_index")
        source_frames = entry.get("source_frames")
        if not isinstance(task, str) or not isinstance(episode, int):
            raise RangePlanError("every plan episode needs string task and integer episode_index")
        if (task, episode) in seen:
            raise RangePlanError(f"duplicate plan entry for {task}/{episode}")
        seen.add((task, episode))
        entry["keep_ranges"] = normalize_ranges(entry.get("keep_ranges", []), source_frames)


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    validate_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def load_plan(path: Path) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    validate_plan(plan)
    return plan

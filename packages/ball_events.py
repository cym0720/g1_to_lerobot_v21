"""Validation and indexing for ball-detector evidence consumed by ping-pong fusion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class BallEventsError(ValueError):
    pass


def index_ball_events(payload: dict, contract: str) -> dict[tuple[str, int], list[int]]:
    """Validate an in-memory `ball-events/v1` payload and index its frames."""
    if payload.get("schema") != "ball-events/v1":
        raise BallEventsError("expected ball-events/v1")
    if payload.get("contract") != contract:
        raise BallEventsError("ball events contract does not match qpos contract")
    indexed: dict[tuple[str, int], list[int]] = {}
    for item in payload.get("episodes", []):
        task = item.get("task")
        episode = item.get("episode_index")
        frames = item.get("frame_indices")
        if not isinstance(task, str) or not isinstance(episode, int):
            raise BallEventsError("every ball-event episode needs task and episode_index")
        if not isinstance(frames, list) or not all(
            isinstance(frame, int) and frame >= 0 for frame in frames
        ):
            raise BallEventsError(
                f"{task}/{episode}: frame_indices must contain non-negative integers"
            )
        key = (task, episode)
        if key in indexed:
            raise BallEventsError(f"duplicate ball-event episode: {task}/{episode}")
        indexed[key] = sorted(set(frames))
    return indexed


def make_ball_events(contract: str, events: Iterable[tuple[str, int, int]]) -> dict:
    """Build canonical detector evidence from `(task, episode_index, frame_index)` rows."""
    indexed: dict[tuple[str, int], set[int]] = {}
    for task, episode, frame in events:
        if not isinstance(task, str) or not task or not isinstance(episode, int):
            raise BallEventsError("event rows need non-empty task and integer episode_index")
        if not isinstance(frame, int) or frame < 0:
            raise BallEventsError("event frame_index must be a non-negative integer")
        indexed.setdefault((task, episode), set()).add(frame)
    payload = {
        "schema": "ball-events/v1",
        "contract": contract,
        "episodes": [
            {"task": task, "episode_index": episode, "frame_indices": sorted(frames)}
            for (task, episode), frames in sorted(indexed.items())
        ],
    }
    index_ball_events(payload, contract)
    return payload


def load_ball_events(path: Path, contract: str) -> dict[tuple[str, int], list[int]]:
    """Load `ball-events/v1` into task/episode keyed frame lists."""
    payload = json.loads(path.read_text(encoding="utf8"))
    return index_ball_events(payload, contract)

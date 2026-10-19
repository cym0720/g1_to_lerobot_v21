"""G1 named-field adapter for the EVA replay qpos20 contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GripperConvention = Literal["raw_encoder_0_120", "opening_0_1", "closed_0_1"]


@dataclass(frozen=True)
class QposAssembly:
    state: np.ndarray
    action: np.ndarray
    action_fallbacks: tuple[str, ...]


def to_closed_fraction(values: np.ndarray, convention: GripperConvention) -> np.ndarray:
    """Convert one scalar gripper stream to EVA's 0=open, 1=closed scale."""
    source = np.asarray(values, dtype=np.float32)
    if convention == "raw_encoder_0_120":
        result = np.clip(source, 0.0, 120.0) / 120.0
    elif convention == "opening_0_1":
        result = 1.0 - np.clip(source, 0.0, 1.0)
    elif convention == "closed_0_1":
        result = np.clip(source, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported gripper convention: {convention}")
    return result.astype(np.float32, copy=False)


def _matrix(columns: dict[str, np.ndarray], name: str, frames: int, width: int) -> np.ndarray:
    if name not in columns:
        raise ValueError(f"missing required G1 field {name!r}")
    values = np.asarray(columns[name], dtype=np.float32).reshape(frames, -1)
    if values.shape[1] != width:
        raise ValueError(f"{name}: expected width {width}, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains NaN or infinity")
    return values


def _first_matrix(
    columns: dict[str, np.ndarray], names: tuple[str, ...], frames: int, width: int
) -> tuple[np.ndarray, str]:
    for name in names:
        if name in columns:
            return _matrix(columns, name, frames, width), name
    raise ValueError(f"missing every permitted field: {names}")


def assemble_g1_qpos(
    columns: dict[str, np.ndarray],
    *,
    gripper_convention: GripperConvention,
    action_gripper_convention: GripperConvention | None = None,
    allow_state_action_fallback: bool = False,
) -> QposAssembly:
    """Assemble [left arm, grip, right arm, grip, head, body] named fields.

    The function deliberately never inspects a packed legacy vector: packing
    order is a contract concern, so each component must be named first.
    """
    if not columns:
        raise ValueError("no source columns supplied")
    first = next(iter(columns.values()))
    frames = int(np.asarray(first).shape[0])
    if frames < 1:
        raise ValueError("source columns must contain at least one frame")

    state_arms = _matrix(columns, "observation.state.joint_position", frames, 14)
    state_left = to_closed_fraction(
        _matrix(columns, "observation.state.left_effector_position", frames, 1), gripper_convention
    )
    state_right = to_closed_fraction(
        _matrix(columns, "observation.state.right_effector_position", frames, 1), gripper_convention
    )
    state_head = _matrix(columns, "observation.state.head_position", frames, 2)
    state_body = _matrix(columns, "observation.state.waist_position", frames, 2)

    action_arms = _matrix(columns, "action.joint_position", frames, 14)
    # Some recorders use encoder ticks for state and normalized commands for action.
    action_convention = action_gripper_convention or gripper_convention
    action_left = to_closed_fraction(
        _matrix(columns, "action.left_effector_position", frames, 1), action_convention
    )
    action_right = to_closed_fraction(
        _matrix(columns, "action.right_effector_position", frames, 1), action_convention
    )
    action_head_names = ("action.head_position", "observation.state.head_position")
    action_body_names = ("action.waist_position", "observation.state.waist_position")
    action_head, action_head_source = _first_matrix(columns, action_head_names, frames, 2)
    action_body, action_body_source = _first_matrix(columns, action_body_names, frames, 2)
    fallbacks = tuple(
        source
        for source in (action_head_source, action_body_source)
        if source.startswith("observation.state.")
    )
    if fallbacks and not allow_state_action_fallback:
        raise ValueError(
            "action head/body fields are missing; rerun only with explicit "
            "allow_state_action_fallback after recording the provenance"
        )

    def join(
        arms: np.ndarray, left: np.ndarray, right: np.ndarray, head: np.ndarray, body: np.ndarray
    ) -> np.ndarray:
        # Preserve the source body columns here. The converter maps their order
        # to the output contract after assembly; raw batches can differ.
        return np.concatenate(
            (arms[:, :7], left, arms[:, 7:], right, head, body), axis=1
        ).astype(np.float32)

    state = join(state_arms, state_left, state_right, state_head, state_body)
    action = join(action_arms, action_left, action_right, action_head, action_body)
    if state.shape != (frames, 20) or action.shape != (frames, 20):
        raise AssertionError(f"internal G1 qpos shape error: {state.shape}/{action.shape}")
    return QposAssembly(state=state, action=action, action_fallbacks=fallbacks)

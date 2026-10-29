"""Load and validate robot-output contracts used by converters and analyzers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """The declared dataset contract is malformed or incompatible."""


@dataclass(frozen=True)
class QposContract:
    contract_id: str
    robot: str
    qpos_key: str
    action_key: str
    qpos_names: tuple[str, ...]
    gripper_indices: tuple[int, int]
    gripper_semantic: str

    def index(self, name: str) -> int:
        try:
            return self.qpos_names.index(name)
        except ValueError as exc:
            raise ContractError(f"{self.contract_id}: unknown qpos feature {name!r}") from exc


def load_qpos_contract(path: Path) -> QposContract:
    """Return the validated EVA qpos contract declared in a JSON file."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    required = raw.get("required_features", {})
    qpos_key = "observations.state.qpos"
    action_key = "action.qpos"
    qpos = required.get(qpos_key)
    action = required.get(action_key)
    if not isinstance(qpos, dict) or not isinstance(action, dict):
        raise ContractError(f"{path}: missing {qpos_key!r} or {action_key!r}")
    names = tuple(qpos.get("names", ()))
    shape = qpos.get("shape")
    if not names or shape != [len(names)] or action.get("shape") != [len(names)]:
        raise ContractError(f"{path}: qpos/action dimension or names are inconsistent")
    if len(set(names)) != len(names):
        raise ContractError(f"{path}: qpos feature names must be unique")
    try:
        grippers = (names.index("left_gripper"), names.index("right_gripper"))
    except ValueError as exc:
        raise ContractError(f"{path}: both grippers must be named in qpos") from exc
    gripper = raw.get("gripper", {})
    if gripper.get("semantic") != "closed_fraction" or gripper.get("range") != [0.0, 1.0]:
        raise ContractError(f"{path}: EVA G1 requires closed_fraction grippers in [0, 1]")
    return QposContract(
        contract_id=str(raw.get("contract_id", "")),
        robot=str(raw.get("robot", "")),
        qpos_key=qpos_key,
        action_key=action_key,
        qpos_names=names,
        gripper_indices=grippers,
        gripper_semantic="closed_fraction",
    )

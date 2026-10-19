#!/usr/bin/env python3
"""Patch train_starvla.py DDP startup verification: generalize world_size/global_batch.

Upstream hardcoded world_size==2, global_batch==64 (the 1015 DP 2x32 contract).
OFT 20261016 runs 4 GPUs x bs8 = global 32. Derive from accelerator instead of
hardcoding; keep all other verification semantics (per-rank batch, gradient
sync checksum spread, parameter update > 0). Documented task-local patch.
"""
from pathlib import Path
import py_compile

F = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local/starVLA/training/train_starvla.py")
txt = F.read_text()

OLD_VALID = '''                if self.completed_steps + 1 == 2:
                    valid = (
                        len(records) == 2
                        and all(int(row[0]) == expected_batch for row in records)
                        and audit["global_batch"] == 64
                        and audit["gradient_checksum_spread"] <= 1e-7
                        and all(row[3] > 0 for row in records)
                    )'''
NEW_VALID = '''                if self.completed_steps + 1 == 2:
                    _world = int(self.accelerator.num_processes)
                    _global_expected = expected_batch * _world
                    valid = (
                        len(records) == _world
                        and all(int(row[0]) == expected_batch for row in records)
                        and audit["global_batch"] == _global_expected
                        and audit["gradient_checksum_spread"] <= 1e-7
                        and all(row[3] > 0 for row in records)
                    )'''

OLD_RECEIPT = '''                    receipt = {
                        "status": "PASS",
                        "verified_optimizer_steps": 2,
                        "ddp_world_size": 2,
                        "per_rank_batch": 32,
                        "gradient_accumulation": 1,
                        "global_batch": 64,'''
NEW_RECEIPT = '''                    receipt = {
                        "status": "PASS",
                        "verified_optimizer_steps": 2,
                        "ddp_world_size": _world,
                        "per_rank_batch": expected_batch,
                        "gradient_accumulation": 1,
                        "global_batch": _global_expected,'''

OLD_SHAPES = '''                    "item_shapes": {
                        "action": [16, 20],
                        "state": [20],
                        "cameras": [3, 224, 224, 3],
                    },'''
NEW_SHAPES = '''                    "item_shapes": {
                        "action": [int(self.config.framework.get("action_model", {}).get("action_horizon", self.config.framework.get("chunk_size", self.config.framework.get("horizon", 16)))), 20],
                        "state": [20],
                        "cameras": [3, 224, 224, 3],
                    },'''

changed = False
for old, new, label in ((OLD_VALID, NEW_VALID, "valid"), (OLD_RECEIPT, NEW_RECEIPT, "receipt"), (OLD_SHAPES, NEW_SHAPES, "shapes")):
    if old in txt:
        txt = txt.replace(old, new)
        print("REPLACED", label)
        changed = True
    elif new in txt:
        print("ALREADY", label)
    else:
        raise SystemExit(f"anchor missing for {label}")

if changed:
    F.write_text(txt)
    print("PATCHED", F)

py_compile.compile(str(F), doraise=True)
print("COMPILE_OK")

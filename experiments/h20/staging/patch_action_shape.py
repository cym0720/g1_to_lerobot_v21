#!/usr/bin/env python3
"""Patch train_starvla.py _train_step action-shape contract: derive chunk from config.

Upstream hardcoded (16, 20) (the 1008 h16 contract). OFT 20261016 runs
horizon 32; read action_model.action_horizon when present, else chunk_size/horizon.
Documented task-local patch.
"""
from pathlib import Path
import py_compile

F = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local/starVLA/training/train_starvla.py")
txt = F.read_text()

OLD = '''        first = batch_vla[0]
        if np.asarray(first["action"]).shape != (16, 20):
            raise RuntimeError(f"unexpected action batch item shape {np.asarray(first['action']).shape}")'''
NEW = '''        first = batch_vla[0]
        _expected_chunk = int(self.config.framework.get("chunk_size", self.config.framework.get("horizon", 16)))
        _action_model = self.config.framework.get("action_model", None)
        if _action_model is not None and _action_model.get("action_horizon", None) is not None:
            _expected_chunk = int(_action_model.action_horizon)
        if np.asarray(first["action"]).shape != (_expected_chunk, 20):
            raise RuntimeError(f"unexpected action batch item shape {np.asarray(first['action']).shape}, expected ({_expected_chunk}, 20)")'''

if "_expected_chunk" in txt:
    print("PATCH_ALREADY_APPLIED")
else:
    if txt.count(OLD) != 1:
        raise SystemExit(f"anchor not found exactly once (count={txt.count(OLD)})")
    F.write_text(txt.replace(OLD, NEW))
    print("PATCHED", F)

py_compile.compile(str(F), doraise=True)
print("COMPILE_OK")

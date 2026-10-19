#!/usr/bin/env python3
"""Register 1015/20261016 data mixes + symlinks (idempotent)."""
import os
import sys
from pathlib import Path

R = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014")
DC = R / "baselines/starvla_local/examples/realRobots/PickAppleACT/train_files/data_registry/data_config.py"
overlay = R / "baselines/data_overlay"
overlay.mkdir(parents=True, exist_ok=True)

links = {
    "train36": R / "data/training_split_1015_letterbox224_v1/train",
    "train16": R / "baselines/starvla_local/playground/Datasets/pick_apple_20261016_split_v1/train",
}
for name, tgt in links.items():
    lp = overlay / name
    if lp.is_symlink() or lp.exists():
        lp.unlink()
    os.symlink(tgt, lp)
    print("LINK", name, "->", tgt, "ok" if lp.exists() else "BROKEN")

txt = DC.read_text()
added = []
if "g1_pick_apple_1015_train36_c16" not in txt:
    txt = txt.rstrip() + '\n\nDATASET_NAMED_MIXTURES["g1_pick_apple_1015_train36_c16"] = [("train36", 1.0, "g1_pick_apple")]\n'
    added.append("g1_pick_apple_1015_train36_c16")
if "g1_pick_apple_1016_train16_c16" not in txt:
    txt = txt.rstrip() + '\nDATASET_NAMED_MIXTURES["g1_pick_apple_1016_train16_c16"] = [("train16", 1.0, "g1_pick_apple")]\n'
    added.append("g1_pick_apple_1016_train16_c16")
if added:
    DC.write_text(txt)
    print("PATCHED data_config.py, added:", added)
else:
    print("data_config.py already had both mixtures")

import importlib.util
spec = importlib.util.spec_from_file_location("_reg_check", DC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
M = mod.DATASET_NAMED_MIXTURES
for k in ("g1_pick_apple_1015_train36_c16", "g1_pick_apple_1016_train16_c16"):
    print("VERIFY", k, "present" if k in M else "MISSING", M.get(k))

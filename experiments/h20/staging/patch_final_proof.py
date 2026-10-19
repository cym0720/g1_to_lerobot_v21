#!/usr/bin/env python3
"""Patch final_inference_proof.py: accept 1015 (4 val episodes, source_group=1015).

Documented task-local patch; the 1008 contract (1 val episode, group 1008)
remains valid via the same branch logic.
"""
from pathlib import Path
import py_compile

F = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/experiment/final_inference_proof.py")
txt = F.read_text()

OLD_EP = '''    if len(episode_metas) != 1:
        raise RuntimeError(f"formal 1008-only val must contain exactly one episode, got {len(episode_metas)}")'''
NEW_EP = '''    if len(episode_metas) not in (1, 4):
        raise RuntimeError(f"formal val episode count must be 1 (1008) or 4 (1015), got {len(episode_metas)}")
    _allowed_groups = {"1008", "1015"} if len(episode_metas) == 4 else {"1008"}'''

OLD_GROUP = '''        if episode["source_group"] != "1008":
            raise RuntimeError(f"final proof encountered non-1008 validation episode: {episode['source_group']}")'''
NEW_GROUP = '''        if episode["source_group"] not in _allowed_groups:
            raise RuntimeError(f"final proof encountered unexpected validation source_group: {episode['source_group']}")'''

changed = False
if "1 (1008) or 4 (1015)" not in txt:
    if txt.count(OLD_EP) != 1:
        raise SystemExit("episode-count anchor not found exactly once")
    txt = txt.replace(OLD_EP, NEW_EP)
    changed = True
if "_allowed_groups" not in txt or "unexpected validation source_group" not in txt:
    if txt.count(OLD_GROUP) != 1:
        raise SystemExit("source-group anchor not found exactly once")
    txt = txt.replace(OLD_GROUP, NEW_GROUP)
    changed = True

if changed:
    backup = F.with_suffix(".py.bak_1015")
    if not backup.exists():
        backup.write_text(F.read_text())
        print("BACKUP", backup)
    F.write_text(txt)
    print("PATCHED", F)
else:
    print("PATCH_ALREADY_APPLIED")

py_compile.compile(str(F), doraise=True)
print("COMPILE_OK")

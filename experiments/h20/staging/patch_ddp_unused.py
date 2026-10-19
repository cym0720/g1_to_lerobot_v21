#!/usr/bin/env python3
"""Patch train_starvla.py: DDP find_unused_parameters=True when DeepSpeed disabled.

DP/OFT frameworks have parameters unused in some forwards (e.g. co-training VLM
branches); strict DDP reduction then aborts at step 2. Documented task-local patch.
"""
from pathlib import Path

F = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local/starVLA/training/train_starvla.py")
txt = F.read_text()

OLD = 'accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)'
NEW = '''from accelerate.utils import DistributedDataParallelKwargs as _DDPKwargs
_ddp_kwargs = None if deepspeed_plugin is not None else _DDPKwargs(find_unused_parameters=True)
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, kwargs_handlers=([_ddp_kwargs] if _ddp_kwargs else None))'''

if "_DDPKwargs(find_unused_parameters=True)" in txt:
    print("PATCH_ALREADY_APPLIED")
else:
    count = txt.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly 1 occurrence of target line, found {count}")
    F.write_text(txt.replace(OLD, NEW))
    print("PATCHED", F)

import py_compile
py_compile.compile(str(F), doraise=True)
print("COMPILE_OK")

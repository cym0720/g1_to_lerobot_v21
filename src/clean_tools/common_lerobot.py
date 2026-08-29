# 用途：提供 LeRobot v2.1 任务级数据集的公共读写、复制和路径辅助函数。
"""Small, dependency-light helpers for the task-scoped LeRobot v2.1 files."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import pyarrow.parquet as pq

def tasks(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / 'meta/info.json').exists())

def episodes(root: Path):
    return [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text(encoding='utf8').splitlines() if x]

def write_episodes(root: Path, rows):
    (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf8')

def copy_dataset_skeleton(src: Path, dst: Path):
    if dst.exists(): raise FileExistsError(f'输出已存在: {dst}')
    shutil.copytree(src, dst)

def episode_parquet(root: Path, ep: int):
    return root / 'data/chunk-000' / f'episode_{ep:06d}.parquet'

def video_files(root: Path, ep: int):
    return list((root/'videos/chunk-000').glob(f'*/episode_{ep:06d}.mp4'))

def depth_dir(root: Path, ep: int): return root/'depth_raw'/f'episode_{ep:06d}'

def table(root: Path, ep: int): return pq.read_table(episode_parquet(root, ep))

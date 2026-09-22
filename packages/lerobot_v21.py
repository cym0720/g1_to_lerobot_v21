# 用途：提供 LeRobot v2.1 任务级数据集的公共读写、复制和路径辅助函数。
"""Small, dependency-light helpers for the task-scoped LeRobot v2.1 files."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import pyarrow.parquet as pq

def tasks(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / 'meta/info.json').exists())

def task_roots(root: Path):
    """Accept either a parent containing tasks or one task-scoped v2.1 root."""
    return [root] if (root / 'meta/info.json').exists() else tasks(root)

def episodes(root: Path):
    return [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text(encoding='utf8').splitlines() if x]

def write_episodes(root: Path, rows):
    (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf8')

def copy_dataset_skeleton(src: Path, dst: Path):
    if dst.exists(): raise FileExistsError(f'输出已存在: {dst}')
    shutil.copytree(src, dst)

def _chunk_size(root: Path) -> int:
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf8'))
    return int(info.get('chunks_size', info.get('chunk_size', 1000)))

def episode_parquet(root: Path, ep: int):
    """Locate an episode without assuming every dataset fits chunk-000."""
    expected = root / 'data' / f'chunk-{ep // _chunk_size(root):03d}' / f'episode_{ep:06d}.parquet'
    if expected.exists():
        return expected
    matches = list((root / 'data').glob(f'chunk-*/episode_{ep:06d}.parquet'))
    return matches[0] if len(matches) == 1 else expected

def video_files(root: Path, ep: int):
    expected = root / 'videos' / f'chunk-{ep // _chunk_size(root):03d}'
    if expected.exists():
        return list(expected.glob(f'*/episode_{ep:06d}.mp4'))
    return list((root / 'videos').glob(f'chunk-*/episode_{ep:06d}.mp4'))

def depth_dir(root: Path, ep: int): return root/'depth_raw'/f'episode_{ep:06d}'

def table(root: Path, ep: int): return pq.read_table(episode_parquet(root, ep))

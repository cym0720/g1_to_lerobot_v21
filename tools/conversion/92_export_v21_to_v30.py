# 用途（可选）：复制已经完成清洗的 LeRobot v2.1 任务集，导出一份 LeRobot v3 兼容副本供新版 ACT 验证；正式数据集与 00→04 流程始终以 v2.1 为准，绝不改动输入目录。
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from datasets import Dataset
from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as v30_converter


def task_roots(root: Path) -> list[Path]:
    candidates=sorted(path for path in root.iterdir() if path.is_dir() and (path / 'meta' / 'info.json').exists())
    # 01 creates mapped task directories even when this particular run has no
    # episodes for them; the upstream converter cannot create empty v3 datasets.
    return [path for path in candidates if any((path/'data').glob('*/*.parquet'))]


def convert_episodes_metadata_windows_safe(root: Path, new_root: Path, episodes_metadata: list[dict], episodes_videos_metadata: list[dict] | None = None) -> None:
    """Avoid Dataset.from_generator, which deadlocks on this Windows setup."""
    legacy_metadata = v30_converter.legacy_load_episodes(root)
    episode_stats = v30_converter.legacy_load_episodes_stats(root)
    for stats in episode_stats.values():
        for feature_stats in stats.values():
            for name in ("min", "max", "mean", "std"):
                value = feature_stats.get(name)
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    feature_stats[name] = value.reshape(1)
    expected = {len(legacy_metadata), len(episodes_metadata)}
    if episodes_videos_metadata is not None:
        expected.add(len(episodes_videos_metadata))
    if len(expected) != 1:
        raise ValueError(f"Episode metadata length mismatch: {expected}")
    rows = list(v30_converter.generate_episode_metadata_dict(
        legacy_metadata, episodes_metadata, episode_stats, episodes_videos_metadata
    ))
    v30_converter.write_episodes(Dataset.from_list(rows), new_root)
    v30_converter.write_stats(v30_converter.aggregate_stats(list(episode_stats.values())), new_root)


def main() -> None:
    parser=argparse.ArgumentParser(description='Optional copy-only LeRobot v2.1 -> v3.0 export for newer ACT.')
    parser.add_argument('--input',type=Path,required=True,help='02/03 清洗阶段的任务根目录')
    parser.add_argument('--output',type=Path,required=True,help='新的 v3 输出根目录')
    parser.add_argument('--data-file-size-mb',type=int,default=64,help='v3 每个 Parquet 最大目标大小（默认 64MB）')
    parser.add_argument('--video-file-size-mb',type=int,default=256,help='v3 每个视频分片最大目标大小（默认 256MB）')
    parser.add_argument('--tasks',nargs='*',help='Only export the specified v2.1 task directories.')
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit(f'输出已存在: {args.output}')
    roots=task_roots(args.input)
    if args.tasks:
        selected=set(args.tasks)
        roots=[root for root in roots if root.name in selected]
        missing=selected-{root.name for root in roots}
        if missing: raise SystemExit(f'Unknown or empty task directories: {sorted(missing)}')
    if not roots: raise SystemExit(f'未找到 v2.1 任务集: {args.input}')
    args.output.mkdir(parents=True)
    # PyAV used by the upstream v3 converter cannot concatenate videos from a
    # Windows path containing Chinese characters. Stage only its temporary copy
    # under the ASCII system temp path; the final v3 dataset is moved back below.
    staging=Path(tempfile.mkdtemp(prefix='lerobot_v30_'))
    report=[]
    try:
        for source in roots:
            repo_id=f'local/{source.name}'
            copied=staging/'local'/source.name
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,copied)
            original_converter = v30_converter.convert_episodes_metadata
            try:
                v30_converter.convert_episodes_metadata = convert_episodes_metadata_windows_safe
                v30_converter.convert_dataset(
                    repo_id=repo_id, root=staging, push_to_hub=False,
                    data_file_size_in_mb=args.data_file_size_mb,
                    video_file_size_in_mb=args.video_file_size_mb,
                )
            finally:
                v30_converter.convert_episodes_metadata = original_converter
            converted=staging/'local'/source.name
            info=json.loads((converted/'meta/info.json').read_text(encoding='utf8'))
            if info.get('codebase_version') != 'v3.0': raise RuntimeError(f'官方转换没有生成 v3: {source.name}')
            shutil.move(str(converted),str(args.output/source.name))
            report.append({'task_directory':source.name,'codebase_version':info['codebase_version'],
                           'episodes':info['total_episodes'],'frames':info['total_frames'],
                           'data_file_size_mb':args.data_file_size_mb,'video_file_size_mb':args.video_file_size_mb})
    finally:
        # Staging and *_old contain only copies created by this script, never user input.
        if staging.exists(): shutil.rmtree(staging)
    (args.output/'conversion_v30_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()

# 用途（主流程第 05 步）：在裁剪/删除后重新读取 LeRobot v2.1 的全部数值与 RGB 视频，重建逐轨迹 episodes_stats.jsonl 和全局 stats.json。
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import av
import av
import numpy as np
import pyarrow.parquet as pq

from packages.ffmpeg_io import rgb_video_stats
from packages.lerobot_v21 import episode_parquet, task_roots, video_files


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def array_stats(values: np.ndarray, image: bool = False) -> dict:
    data=np.asarray(values,dtype=np.float64)
    if image:
        # RGB uses LeRobot's channel-first (C,1,1) statistics convention.
        return {'min':data.min(0).reshape(3,1,1).tolist(), 'max':data.max(0).reshape(3,1,1).tolist(),
                'mean':data.mean(0).reshape(3,1,1).tolist(), 'std':data.std(0).reshape(3,1,1).tolist(),
                'count':[int(len(data))]}
    if data.ndim==1: data=data[:,None]
    return {'min':data.min(0).tolist(),'max':data.max(0).tolist(),
            'mean':data.mean(0).tolist(),'std':data.std(0).tolist(),'count':[int(len(data))]}


def video_stats_pyav(path: Path) -> dict:
    lo=np.full(3,np.inf); hi=np.full(3,-np.inf); total=np.zeros(3); square=np.zeros(3); pixels=0; frames=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        for frame in container.decode(stream):
            rgb=frame.to_ndarray(format='rgb24').reshape(-1,3).astype(np.float64)/255.0
            lo=np.minimum(lo,rgb.min(0)); hi=np.maximum(hi,rgb.max(0)); total+=rgb.sum(0); square+=(rgb*rgb).sum(0)
            pixels+=len(rgb); frames+=1
    if not frames: raise ValueError(f'empty video: {path}')
    mean=total/pixels; var=np.maximum(square/pixels-mean*mean,0.0)
    return {'min':lo.reshape(3,1,1).tolist(),'max':hi.reshape(3,1,1).tolist(),
            'mean':mean.reshape(3,1,1).tolist(),'std':np.sqrt(var).reshape(3,1,1).tolist(),'count':[frames]}


def video_stats(path: Path, compute_rgb_stats: bool = False) -> dict:
    """Use FFmpeg streaming RGB statistics only when explicitly requested."""
    return rgb_video_stats(path) if compute_rgb_stats else video_stats_pyav(path)
def combine(parts: list[dict]) -> dict:
    """Combine population mean/std using episode frame counts."""
    n=0; mean=None; m2=None; lo=None; hi=None
    for item in parts:
        count=int(item['count'][0]); current_mean=np.asarray(item['mean'],dtype=np.float64); current_std=np.asarray(item['std'],dtype=np.float64)
        current_lo=np.asarray(item['min'],dtype=np.float64); current_hi=np.asarray(item['max'],dtype=np.float64)
        if n==0:
            n=count; mean=current_mean; m2=current_std**2*count; lo=current_lo; hi=current_hi; continue
        delta=current_mean-mean; total=n+count
        mean=mean+delta*count/total; m2=m2+current_std**2*count+delta**2*n*count/total
        lo=np.minimum(lo,current_lo); hi=np.maximum(hi,current_hi); n=total
    return {'min':lo.tolist(),'max':hi.tolist(),'mean':mean.tolist(),'std':np.sqrt(m2/n).tolist(),'count':[n]}


def episode_stats(root: Path, episode: dict, numeric: list[str], cameras: list[str], compute_rgb_stats: bool = False) -> tuple[int,dict]:
    index=int(episode['episode_index']); table=pq.read_table(episode_parquet(root,index))
    stats={}
    for key in numeric:
        stats[key]=array_stats(np.asarray(table[key].to_pylist()))
    for camera in cameras:
        paths=[path for path in video_files(root,index) if path.parent.name==camera]
        if len(paths)!=1: raise ValueError(f'{root.name} ep={index}: expected one video for {camera}')
        stats[camera]=video_stats(paths[0], compute_rgb_stats)
    return index,stats


def episode_stats_job(job: tuple[Path, dict, list[str], list[str], bool]) -> tuple[int, dict]:
    return episode_stats(*job)


def main() -> None:
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description='Recompute LeRobot v2.1 metadata statistics in place.')
    parser.add_argument('--input',type=Path,required=True,help='03/04 后的 LeRobot v2.1 任务根目录')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--compute-rgb-stats',action='store_true',help='Decode RGB through FFmpeg and calculate true per-channel image statistics.')
    parser.add_argument('--tasks',nargs='*',help='仅重算指定任务目录，例如 02_task_05 03_task_06')
    args=parser.parse_args()
    selected=set(args.tasks or [])
    roots=[root for root in task_roots(args.input) if not selected or root.name in selected]
    missing=selected-{p.name for p in roots}
    if missing: raise ValueError(f'unknown task directories: {sorted(missing)}')
    for root in roots:
        info=json.loads((root/'meta/info.json').read_text(encoding='utf8'))
        if info.get('codebase_version')!='v2.1': raise ValueError(f'{root} is not v2.1')
        features=info['features']; cameras=[key for key,value in features.items() if value['dtype']=='video']
        numeric=[key for key,value in features.items() if value['dtype']!='video']
        episodes=jsonl(root/'meta/episodes.jsonl')
        jobs=[(root, episode, numeric, cameras, args.compute_rgb_stats) for episode in episodes]
        with ProcessPoolExecutor(max_workers=max(1,args.workers)) as pool:
            computed=list(pool.map(episode_stats_job,jobs))
        computed.sort(key=lambda x:x[0])
        per_episode=[{'episode_index':index,'stats':stats} for index,stats in computed]
        (root/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in per_episode),encoding='utf8')
        aggregate={key:combine([stats[key] for _,stats in computed]) for key in [*numeric,*cameras]}
        (root/'meta/stats.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2),encoding='utf8')
        print(f'{root.name}: episodes={len(computed)} features={len(aggregate)}')
    print(f'elapsed_seconds={time.perf_counter()-started:.3f}', flush=True)


if __name__=='__main__': main()

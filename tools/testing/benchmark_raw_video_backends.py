# Purpose: read-only benchmark of raw 618 video conversion backends.
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from packages.ffmpeg_io import encode_h265_aligned, encode_jpeg_sequence, nearest_indices, video_frame_count

def stamps(path): return np.asarray([int(x.split()[0]) for x in path.read_text().splitlines() if x.strip()],dtype=np.int64)
p=argparse.ArgumentParser(); p.add_argument('--episode',type=Path,required=True); p.add_argument('--frames',type=int,default=120); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
ep=a.episode; target=stamps(ep/'camera/head_color/head_color.txt')[:a.frames]; results=[]
for backend in ('cuda','qsv','cpu'):
    dest=a.output/('head_'+backend+'.mp4'); t=time.perf_counter()
    try:
        used=encode_h265_aligned(ep/'camera/head_color/head_color.h265',target,target,dest,30,{'x':0,'y':40,'width':1280,'height':720},(backend,)); results.append({'stream':'head_h265','backend':used,'seconds':time.perf_counter()-t,'frames':video_frame_count(dest),'status':'ok'})
    except Exception as exc: results.append({'stream':'head_h265','backend':backend,'status':'failed','error':str(exc)})
files=sorted((ep/'camera/hand_left/color').glob('*.jpg')); source_times=np.asarray([int(path.stem) for path in files],dtype=np.int64); selected=[files[index] for index in nearest_indices(source_times,target)]
for backend in ('cuda','qsv','cpu'):
    dest=a.output/('left_wrist_'+backend+'.mp4'); t=time.perf_counter()
    try:
        used=encode_jpeg_sequence(selected,dest,30,(backend,)); results.append({'stream':'left_wrist_jpg','backend':used,'seconds':time.perf_counter()-t,'frames':video_frame_count(dest),'status':'ok'})
    except Exception as exc: results.append({'stream':'left_wrist_jpg','backend':backend,'status':'failed','error':str(exc)})
a.output.mkdir(parents=True,exist_ok=True); (a.output/'raw_video_backend_benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf8'); print(json.dumps(results,indent=2))

"""Read-only all-episode 1004 source integrity and boundary coverage audit."""
from __future__ import annotations
import csv,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import av,h5py,numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from packages.raw_preflight import source_errors,timestamps,h5_stream_present
OUT=REPO/'output/1004_pick_apple_20261005'

def check(ep):
    source=json.loads((REPO/'config/sources/g1.json').read_text())
    meta,errors=source_errors(ep,0,source['raw_h5_adapter'],True,False)
    row=dict(episode=ep.name,task_id=meta.get('task_id') if meta else '',head_frames=0,overlap_frames=0,leading_action_gap_s=0,trailing_action_gap_s=0)
    if errors:row['reasons']=';'.join(errors);return row
    target=timestamps(ep/'camera/head_color/head_color.txt');row['head_frames']=len(target)
    if len(target)<1 or np.any(np.diff(target)<=0):errors.append('invalid_head_clock')
    streams=[]
    with h5py.File(ep/'record/raw_joints.h5') as h:
        for group in source['raw_h5_adapter'].values():
            for k in group.values():
                if not h5_stream_present(h,k):continue
                t=np.asarray(h[k.rsplit('/',1)[0]+'/timestamp']);v=np.asarray(h[k])
                if len(t)!=len(v) or np.any(np.diff(t)<0) or not np.isfinite(v).all():errors.append('invalid_stream:'+k)
                streams.append(t)
        action=np.asarray(h['action/joint/timestamp'])
        row['leading_action_gap_s']=max(0,float((action[0]-target[0])/1e9))
        row['trailing_action_gap_s']=max(0,float((target[-1]-action[-1])/1e9))
    lo=max(t[0] for t in streams);hi=min(t[-1] for t in streams)
    row['overlap_frames']=int(((target>=lo)&(target<=hi)).sum())
    try:
        with av.open(str(ep/'camera/head_color/head_color.h265')) as c:
            frames=sum(1 for _ in c.decode(video=0))
        if frames!=len(target):errors.append(f'head_frame_count:{frames}!={len(target)}')
    except Exception as exc:errors.append('corrupt_head:'+str(exc))
    for name in ('hand_left','hand_right'):
        files=sorted((ep/'camera'/name/'color').glob('*.jpg'))
        ts=np.asarray([int(p.stem) for p in files])
        if np.any(np.diff(ts)<=0):errors.append('invalid_clock:'+name)
        for f in files:
            try:
                with Image.open(f) as im:im.verify()
            except Exception as exc:errors.append('corrupt_jpg:'+str(f)+':'+str(exc))
    row['reasons']=';'.join(errors)
    return row

def main():
    eps=sorted(p for p in Path('E:/1004/record').iterdir() if p.is_dir())
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,eps))
    with (OUT/'preflight.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    accepted=[r for r in rows if not r['reasons']]
    result=dict(raw_episodes=len(rows),accepted=len(accepted),rejected=len(rows)-len(accepted),full_frames=sum(r['head_frames'] for r in accepted),overlap_frames=sum(r['overlap_frames'] for r in accepted))
    (OUT/'preflight_summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()

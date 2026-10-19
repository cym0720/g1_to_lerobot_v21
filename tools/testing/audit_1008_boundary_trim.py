"""只读比较首尾静止段裁剪候选，不修改任何数据或视频。"""
from pathlib import Path
import csv
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('output/1008_pick_apple_20261008/training_split_v1')
OUT=Path('output/1008_pick_apple_20261008/boundary_trim_test_v1')
ARMS=list(range(7))+list(range(8,15))
RULES=[('旧默认',.001,5,False),('仅增缓冲',.001,30,False),('保守候选',.0005,30,True),('更敏感候选',.0002,30,True)]

def bounds(s,a,threshold,padding,guard):
    n=len(s)
    motion=np.r_[False,np.max(np.abs(np.diff(s[:,ARMS],axis=0)),axis=1)>threshold]
    valid=np.convolve(motion.astype(int),np.ones(3,dtype=int),'same')>=3
    if guard:
        # 保留任何夹爪开合事件，不要求连续三帧，防止二值指令被忽略。
        grip=np.r_[False,np.any(np.abs(np.diff(s[:,[7,15]],axis=0))>.01,axis=1)|np.any(np.abs(np.diff(a[:,[7,15]],axis=0))>.01,axis=1)]
        head=np.r_[False,np.max(np.abs(np.diff(s[:,[16,17,19]],axis=0)),axis=1)>threshold]
        lift=np.r_[False,np.abs(np.diff(s[:,18]))>.0001]
        valid|=grip|head|lift
    if not valid.any():return 0,n-1
    ix=np.flatnonzero(valid); start=max(0,int(ix[0])-padding); end=min(n-1,int(ix[-1])+padding)
    if guard:
        # 慢动作累计变化达到0.01弧度时仍保留，不仅依据单帧差分。
        def safe(part):
            return len(part)==0 or (np.max(np.ptp(part[:,ARMS+[16,17,19]],axis=0))<=.01 and np.ptp(part[:,18])<=.002)
        if start<60 or not safe(s[:start+1]):start=0
        if n-1-end<60 or not safe(s[end:]):end=n-1
    return start,end

def work(job):
    split,ep=job
    path=ROOT/split/f'data/chunk-000/episode_{ep["episode_index"]:06d}.parquet'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    t=pq.read_table(path); s=np.asarray(t['observation.state'].to_pylist()); a=np.asarray(t['action'].to_pylist()); n=len(s)
    rows=[]
    for name,thr,pad,guard in RULES:
        start,end=bounds(s,a,thr,pad,guard)
        bad_grip=0
        for x in [s[:,[7,15]],a[:,[7,15]]]:
            events=np.flatnonzero(np.any(np.abs(np.diff(x,axis=0))>.01,axis=1))+1
            bad_grip+=int(np.sum((events<start)|(events>end)))
        rows.append(dict(split=split,source=ep['source_group'],episode=ep['episode_index'],id=ep['original_episode_id'],rule=name,threshold=thr,padding_frames=pad,frames=n,start=start,end=end,removed_head=start,removed_tail=n-1-end,removed_frames=start+n-1-end,removed_seconds=(start+n-1-end)/30,discarded_gripper_events=bad_grip,diagnostic_only=True))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for split in ['train','val']:
        eps=[json.loads(l) for l in (ROOT/split/'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if l]
        jobs.extend((split,e) for e in eps)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[r for part in pool.map(work,jobs) for r in part]
    with (OUT/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for split in ['train','val']:
        for name,*_ in RULES:
            selected=[r for r in rows if r['split']==split and r['rule']==name]
            summary.append(dict(split=split,rule=name,episodes=len(selected),candidate_episodes=sum(r['removed_frames']>0 for r in selected),removed_frames=sum(r['removed_frames'] for r in selected),removed_seconds=sum(r['removed_seconds'] for r in selected),discarded_gripper_events=sum(r['discarded_gripper_events'] for r in selected)))
    (OUT/'summary.json').write_text(json.dumps(dict(diagnostic_only=True,dataset_modified=False,ACT_smoke='skipped: diagnostic only',results=summary),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in summary:print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()

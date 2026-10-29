import argparse, csv, itertools
from pathlib import Path
import h5py
import numpy as np

NAMES=['1027','0923_g1_pick_apple','0923_g1_simple_data','1004','1015_pick_apple','1019_pick_apple','1020','1026']

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('dataset/raw/g1')); p.add_argument('--out',type=Path,default=Path('output/diagnostics/initial_pose')); p.add_argument('--tolerance-deg',type=float,default=5); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 rows=[]; errors=[]; counts={}
 for name in NAMES:
  root=a.root/name; files=sorted(root.rglob('raw_joints.h5')); metas=list(root.rglob('meta_info.json')); counts[name]=(len(metas),len(files))
  for f in files:
   try:
    with h5py.File(f,'r') as h:
     x=np.asarray(h['state/joint/position'][0],float)
     if x.shape!=(14,) or not np.isfinite(x).all(): raise ValueError('invalid 14-dimensional state')
     extra=[]
     for key in ('state/head/position','state/waist/position','state/left_effector/position','state/right_effector/position'):
      try: extra.extend(np.asarray(h[key][0],float).tolist())
      except (KeyError,IndexError,TypeError): extra.extend([float('nan')]*(2 if 'head' in key or 'waist' in key else 1))
     rows.append((name,str(f.relative_to(root)),x,extra))
   except Exception as e: errors.append((name,str(f),str(e)))
 if not rows: raise SystemExit('No usable data')
 with (a.out/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f); w.writerow(['dataset','path']+[f'arm_{i+1}_rad' for i in range(14)]+['head_1','head_2','waist_1','waist_2','left_gripper','right_gripper']); w.writerows((n,path,*x,*ex) for n,path,x,ex in rows)
 with (a.out/'errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f); w.writerow(['dataset','path','error']); w.writerows(errors)
 groups={n:np.array([x for d,_,x,_ in rows if d==n]) for n in NAMES}; tol=np.deg2rad(a.tolerance_deg)
 lines=['# 初始姿态统计','',f'口径：每条 raw_joints.h5 的 state/joint/position 首帧，14 维双臂关节；每个关节差值均不超过 {a.tolerance_deg:g}° 定义为相近。只比较双臂，头、腰、夹爪不参与相近判定；腰字段的原始顺序不作假设。各流首帧时间略有差异。', '', '| 数据集 | meta记录数 | H5数 | 成功 | 首帧各关节中位数（度，原始14维顺序） |','|---|---:|---:|---:|---|']
 for n in NAMES:
  x=groups[n]; med=', '.join(f'{v:.1f}' for v in np.rad2deg(np.median(x,axis=0))) if len(x) else '无'; lines.append(f'| {n} | {counts[n][0]} | {counts[n][1]} | {len(x)} | {med} |')
 lines+=['','## 跨数据集相近情况','','匹配率为该数据集的轨迹中，能在对方数据集找到至少一条满足逐关节容差的比例；双向比例可以不同。','', '| 数据集 A | 数据集 B | A匹配率 | B匹配率 | 最近对最大关节差（度） |','|---|---|---:|---:|---:|']
 for n,m in itertools.combinations(NAMES,2):
  x,y=groups[n],groups[m]
  if not len(x) or not len(y): continue
  d=np.max(np.abs(x[:,None,:]-y[None,:,:]),axis=2); lines.append(f'| {n} | {m} | {np.mean(np.min(d,axis=1)<=tol):.1%} | {np.mean(np.min(d,axis=0)<=tol):.1%} | {np.rad2deg(np.min(d)):.2f} |')
 lines+=['','## 日期内分布','','各关节给出 P10、中位数、P90，单位度；可用于判断是否存在多个起始姿态。','','详细首帧和其他状态见 episodes.csv，读取失败见 errors.csv。meta 数与 H5 数不一致时，缺失记录未计入姿态比较。']
 for n,x in groups.items():
  if not len(x): continue
  lines+=['',f'### {n}','','| 原始关节索引 | P10 | 中位数 | P90 |','|---:|---:|---:|---:|']
  for i,vals in enumerate(np.percentile(np.rad2deg(x),[10,50,90],axis=0).T): lines.append(f'| {i+1} | {vals[0]:.1f} | {vals[1]:.1f} | {vals[2]:.1f} |')
 (a.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines[:45])); print('errors',len(errors))
if __name__=='__main__': main()

import argparse, csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('output/diagnostics/initial_pose/episodes.csv')); p.add_argument('--output',type=Path,default=Path('output/diagnostics/initial_pose')); p.add_argument('--tolerance-deg',type=float,default=5); a=p.parse_args()
 with a.input.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if not rows: raise ValueError('No episodes')
 x=np.rad2deg(np.array([[float(r[f'arm_{i}_rad']) for i in range(1,15)] for r in rows])); dist=pdist(x,metric='chebyshev'); labels=fcluster(linkage(dist,method='complete'),t=a.tolerance_deg,criterion='distance') if len(rows)>1 else [1]
 clusters=[np.flatnonzero(np.asarray(labels)==i) for i in set(labels)]; clusters.sort(key=lambda c:(-len(c),str(rows[c[0]]['dataset']),str(rows[c[0]]['path'])))
 names=sorted(set(r['dataset'] for r in rows)); lines=['# 双臂初始姿态分组','',f'口径：每条轨迹 state/joint/position 首帧 14 维双臂关节；采用最大单关节角度差及完全链接聚类，保证组内任意两条轨迹每个关节的差均不超过 {a.tolerance_deg:g}°。分组受阈值及聚类方法影响；头、腰、夹爪不参与。代表姿态为组内实际轨迹中到其他成员最大关节差总和最小的一条。','','| 组 | 轨迹数 | '+' | '.join(names)+' | 组内最大差（度） |','|---|---:|'+'---:|'*len(names)+'---:|']
 output=[]
 for k,ids in enumerate(clusters,1):
  d=np.max(np.abs(x[ids,None,:]-x[None,ids,:]),axis=2); rep=ids[np.argmin(d.sum(axis=1))]; count=Counter(rows[i]['dataset'] for i in ids); lines.append(f'| P{k:02} | {len(ids)} | '+' | '.join(str(count[n]) for n in names)+f' | {d.max():.2f} |')
  for i in ids: output.append({'group':f'P{k:02}',**rows[i],'max_diff_to_representative_deg':f'{np.max(np.abs(x[i]-x[rep])):.4f}','representative':int(i==rep)})
 lines+=['','## 代表姿态','','角度单位为度，顺序为原始 state/joint/position 的 14 个索引。']
 for k,ids in enumerate(clusters,1):
  d=np.max(np.abs(x[ids,None,:]-x[None,ids,:]),axis=2); rep=ids[np.argmin(d.sum(axis=1))]; lines+=['',f'### P{k:02}（{len(ids)} 条）','',f"代表轨迹：{rows[rep]['dataset']} / {rows[rep]['path']}", '', ', '.join(f'{v:.2f}' for v in x[rep])]
 lines+=['','每条轨迹的分组和与代表姿态的差值见 pose_groups.csv；单条组仅表示在此阈值下未与其他记录合并。']
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'pose_groups.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 with (a.output/'pose_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
 print('\n'.join(lines[:8+len(clusters)])); print('episodes',len(rows),'groups',len(clusters))
if __name__=='__main__': main()

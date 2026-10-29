"""Full read and true RGB statistics gate for the fisheye data standard."""
from __future__ import annotations
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import av,cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq
def worker_init():
 cv2.setNumThreads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
CAMS=['observation.images.head_color','observation.images.head_left_fisheye','observation.images.head_right_fisheye']
NAMES=[*[f'left_arm_joint{i}' for i in range(1,8)],*[f'right_arm_joint{i}' for i in range(1,8)],'left_gripper','right_gripper','head_yaw','head_pitch','waist_pitch','lift']
def numeric_stats(a):
 a=np.asarray(a,dtype=np.float64)
 if a.ndim==1:a=a[:,None]
 return {'min':a.min(0).tolist(),'max':a.max(0).tolist(),'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),'count':[len(a)]}
def scan_episode(task):
 worker_init();started=time.perf_counter();root,path,cams=task;root=Path(root);path=Path(path);t=pq.read_table(path,use_threads=False);n=len(t);index=int(t['episode_index'][0].as_py());stats={}
 assert n>0
 assert np.array_equal(t['frame_index'].to_numpy(),np.arange(n)),str(path)
 assert np.all(t['episode_index'].to_numpy()==index)
 for key in ['observation.state','action']:
  a=np.asarray(t[key].to_pylist());assert a.shape==(n,20) and np.isfinite(a).all(),(key,a.shape);stats[key]=numeric_stats(a)
 for key in ['timestamp','frame_index','episode_index','index','task_index']:
  if key in t.column_names:stats[key]=numeric_stats(t[key].to_numpy())
 assert np.allclose(t['timestamp'].to_numpy(),np.arange(n)/30,atol=2e-5)
 video=[]
 for key in cams:
  p=root/'videos'/path.parent.name/key/(path.stem+'.mp4');assert p.is_file() and not p.is_symlink() and p.stat().st_size>0,str(p)
  total=np.zeros(3,np.float64);sq=total.copy();lo=np.full(3,255);hi=np.zeros(3);pixels=0;frames=0;shape=None;hist=np.zeros((3,256),np.float64)
  with av.open(str(p)) as r:
   r.streams.video[0].codec_context.thread_count=1
   for f in r.decode(video=0):
    a=f.to_ndarray(format='rgb24');shape=list(a.shape);pixels+=a.shape[0]*a.shape[1];hist+=np.stack([cv2.calcHist([a],[c],None,[256],[0,256])[:,0] for c in range(3)]);frames+=1
  assert frames==n,(str(p),frames,n)
  bins=np.arange(256,dtype=np.float64);mean=hist@bins/pixels;std=np.sqrt(np.maximum(0,hist@(bins*bins)/pixels-mean**2));lo=np.array([np.flatnonzero(h)[0] for h in hist]);hi=np.array([np.flatnonzero(h)[-1] for h in hist])
  stats[key]={'min':(lo/255).reshape(3,1,1).tolist(),'max':(hi/255).reshape(3,1,1).tolist(),'mean':(mean/255).reshape(3,1,1).tolist(),'std':(std/255).reshape(3,1,1).tolist(),'count':[frames]}
  video.append({'camera':key,'frames':frames,'shape':shape,'pixels':pixels})
 return {'episode_index':index,'frames':n,'start_index':int(t['index'][0].as_py()),'stats':stats,'videos':video,'seconds':time.perf_counter()-started}
def combine(parts):
 out={}
 for k in parts[0]['stats']:
  ps=[p['stats'][k] for p in parts];weights=np.array([p['count'][0] for p in ps],float);count=int(weights.sum());means=np.array([p['mean'] for p in ps]);stds=np.array([p['std'] for p in ps]);w=weights.reshape((-1,)+(1,)*(means.ndim-1));mean=(means*w).sum(0)/count;variance=((stds**2+(means-mean)**2)*w).sum(0)/count
  out[k]={'min':np.min([p['min'] for p in ps],0).tolist(),'max':np.max([p['max'] for p in ps],0).tolist(),'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),'count':[count]}
 return out
def validate(root,report):
 root=Path(root);report=Path(report);start=time.perf_counter();info=json.loads((root/'meta/info.json').read_text(encoding='utf8'));cams=[k for k,v in info['features'].items() if v['dtype']=='video'];assert cams==CAMS,cams
 for k in ['observation.state','action']:assert info['features'][k]['names']==NAMES,(k,info['features'][k]['names'])
 paths=sorted((root/'data').rglob('*.parquet'));assert paths
 parts=[];workers=int(os.environ.get('FISHEYE_VALIDATION_WORKERS','8'))
 with ProcessPoolExecutor(max_workers=workers,initializer=worker_init) as pool:
  futures=[pool.submit(scan_episode,(str(root),str(p),cams)) for p in paths]
  for future in as_completed(futures):
   part=future.result();parts.append(part);elapsed=time.perf_counter()-start;eta=elapsed/len(parts)*(len(paths)-len(parts));episode=part['episode_index'];print(f'validate {root.name} {len(parts)}/{len(paths)} episode={episode} seconds={elapsed:.1f} eta={eta:.1f}',flush=True)
 parts.sort(key=lambda p:p['episode_index'])
 offset=0
 for i,p in enumerate(parts):assert p['episode_index']==i and p['start_index']==offset,(i,p['start_index'],offset);offset+=p['frames']
 assert offset==info['total_frames'] and len(parts)==info['total_episodes']

 for camera in cams:
  shapes={tuple(v['shape']) for part in parts for v in part['videos'] if v['camera']==camera}
  assert shapes=={tuple(info['features'][camera]['shape'])},(camera,shapes)
 standard=json.loads((root/'meta/data_standard.json').read_text(encoding='utf8'))
 assert standard.get('standard_id') and standard.get('standard_version'), 'missing data standard version'
 stats=combine(parts)
 report.parent.mkdir(parents=True,exist_ok=True)
 (report.parent/(report.stem+'.computed_stats.json')).write_text(json.dumps(stats,indent=2))
 (report.parent/(report.stem+'.computed_episodes_stats.jsonl')).write_text(''.join(json.dumps({'episode_index':p['episode_index'],'stats':p['stats']})+'\n' for p in parts))
 result={'input':str(root.resolve()),'status':'OK','full_read':True,'true_rgb_all_frames':True,'episodes':len(parts),'frames':offset,'video_streams':len(parts)*3,'seconds':time.perf_counter()-start,'details':[{k:v for k,v in p.items() if k!='stats'} for p in parts]};report.write_text(json.dumps(result,indent=2),encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(validate(a.input,a.output))

"""Preserve both sources; merge canonical qpos20 with verified 224 RGB videos."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'tools/testing'))
from audit_1004_dataset import load_converter,video_stats

def hash_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def episode_job(job):
    source,ep,i,offset,out=job;source=Path(source);out=Path(out);c=load_converter()
    info=json.loads((source/'meta/info.json').read_text());oldi=ep['episode_index'];n=ep['length']
    path=source/f'data/chunk-000/episode_{oldi:06d}.parquet';before=hash_file(path);t=pq.read_table(path)
    sk='observation.state' if 'observation.state' in t.column_names else 'observations.state.qpos'
    ak='action' if 'action' in t.column_names else 'action.qpos'
    s=np.asarray(t[sk].to_pylist(),np.float32);a=np.asarray(t[ak].to_pylist(),np.float32)
    assert s.shape==a.shape==(n,20) and np.isfinite(s).all() and np.isfinite(a).all()
    assert np.all((a[:,15]>=0)&(a[:,15]<=1))
    # Numeric evidence separates lift (~0.22) from pitch (~0.349) in this task.
    assert .20<float(np.median(s[:,18]))<.24 and .33<float(np.median(s[:,19]))<.37
    vals={'timestamp':np.arange(n,dtype=np.float32)/30,'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,i,np.int64),'index':np.arange(offset,offset+n,dtype=np.int64),'task_index':np.zeros(n,np.int64),'observation.state':s,'action':a}
    cols={k:(c.float_column(v,[20]) if v.ndim==2 else pa.array(v)) for k,v in vals.items()}
    dest=out/f'data/chunk-000/episode_{i:06d}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(cols),dest)
    written=pq.read_table(dest)
    for key,arr in [('observation.state',s),('action',a)]:np.testing.assert_array_equal(np.asarray(written[key].to_pylist(),np.float32),arr)
    stats={k:c.stats_for(v) for k,v in vals.items()}
    for key,feature in info['features'].items():
        if feature['dtype']!='video':continue
        src=source/f'videos/chunk-000/{key}/episode_{oldi:06d}.mp4';dst=out/f'videos/chunk-000/{key}/episode_{i:06d}.mp4';dst.parent.mkdir(parents=True,exist_ok=True)
        h,w=feature['shape'][:2];scale=224/max(h,w);nh,nw=round(h*scale),round(w*scale);x,y=(224-nw)//2,(224-nh)//2
        vf=f'scale={nw}:{nh}:flags=bilinear,pad=224:224:{x}:{y}:color=black,setsar=1'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(src),'-vf',vf,'-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-an',str(dst)],check=True)
        _,stats[key]=video_stats((dst,n))
    assert hash_file(path)==before,'source parquet modified'
    print('MERGED',i,source.name,n,flush=True)
    return {'episode_index':i,'length':n,'tasks':['pick the apple to the plate'],'original_episode_id':ep['original_episode_id'],'source_dataset':str(source),'source_episode_index':oldi,'dataset_from_index':offset,'dataset_to_index':offset+n,'stats':stats,'source_parquet_sha256':before,'waist_state_min':s[:,18:].min(0).tolist(),'waist_state_max':s[:,18:].max(0).tolist(),'right_gripper_action_min':float(a[:,15].min()),'right_gripper_action_max':float(a[:,15].max()),'right_gripper_transition_count':int(np.count_nonzero(np.diff(a[:,15]))),'exact_numeric_values_preserved':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();start=time.monotonic();out=args.output
    out.mkdir(parents=True,exist_ok=False);(out/'meta').mkdir()
    sources=[Path('D:/go work/deploy_g1/examples/dataset/empty_text_task_1'),REPO/'output/1004_pick_apple_20261005/final/pick_apple']
    jobs=[];seen=set();offset=0;names=None
    for source in sources:
        info=json.loads((source/'meta/info.json').read_text());sk='observation.state' if 'observation.state' in info['features'] else 'observations.state.qpos'
        nn=info['features'][sk]['names'];assert nn[-2:]==['lift','waist_pitch'];assert names is None or names==nn;names=nn
        for ep in [json.loads(l) for l in (source/'meta/episodes.jsonl').read_text().splitlines()]:
            assert ep['original_episode_id'] not in seen;seen.add(ep['original_episode_id']);jobs.append((source,ep,len(jobs),offset,out));offset+=ep['length']
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(episode_job,jobs))
    c=load_converter();features={k:{'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None} for k in ['timestamp','frame_index','episode_index','index','task_index']}
    for k in ['observation.state','action']:features[k]={'dtype':'float32','shape':[20],'names':names}
    for k in results[0]['stats']:
        if not k.startswith('observation.images.'):continue
        features[k]={'dtype':'video','shape':[224,224,3],'names':['height','width','channels'],'info':{'video.height':224,'video.width':224,'video.channels':3,'video.fps':30.,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
    info={'codebase_version':'v2.1','robot_type':'G1','fps':30,'total_episodes':len(results),'total_frames':offset,'total_tasks':1,'total_videos':len(results)*3,'total_chunks':1,'chunks_size':1000,'splits':{'train':f'0:{len(results)}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features,'merge_provenance':{'sources':[str(s) for s in sources],'body_order':['lift','waist_pitch'],'image_transform':'224 RGB aspect-preserving bilinear black letterbox','numeric_values_unchanged':True}}
    (out/'meta/info.json').write_text(json.dumps(info,indent=2))
    (out/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'pick the apple to the plate'})+'\n')
    (out/'meta/episodes.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='stats'})+'\n' for r in results))
    (out/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps({'episode_index':r['episode_index'],'stats':r['stats']})+'\n' for r in results))
    stats={k:c.combine_episode_stats([r['stats'][k] for r in results]) for k in results[0]['stats']}
    (out/'meta/stats.json').write_text(json.dumps(stats,indent=2))
    (out/'meta/merge_validation.json').write_text(json.dumps({'status':'PASS','episodes':len(results),'frames':offset,'elapsed_seconds':time.monotonic()-start,'episodes_checked':[{k:v for k,v in r.items() if k!='stats'} for r in results]},indent=2))
    print('MERGE_PASS',len(results),offset,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()

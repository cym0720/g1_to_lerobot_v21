"""Uniform future-state ACT labels and episode-level old/new held-out split."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/testing'))
from audit_1004_dataset import load_converter

def build(source,out,selected):
    c=load_converter();out.mkdir(parents=True,exist_ok=False);(out/'meta').mkdir()
    info=json.loads((source/'meta/info.json').read_text());eps=[json.loads(l) for l in (source/'meta/episodes.jsonl').read_text().splitlines()];stats=[json.loads(l)['stats'] for l in (source/'meta/episodes_stats.jsonl').read_text().splitlines()]
    cameras=[k for k,v in info['features'].items() if v['dtype']=='video'];offset=0;results=[];checks=[]
    for i,oldi in enumerate(selected):
        ep=eps[oldi];n=ep['length'];t=pq.read_table(source/f'data/chunk-000/episode_{oldi:06d}.parquet');s=np.asarray(t['observation.state'].to_pylist(),np.float32);oldaction=np.asarray(t['action'].to_pylist(),np.float32);a=s[np.minimum(np.arange(n)+3,n-1)].copy();a[:,[7,15]]=(oldaction[:,[7,15]]>=.5).astype(np.float32)
        vals={'timestamp':np.arange(n,dtype=np.float32)/30,'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,i,np.int64),'index':np.arange(offset,offset+n,dtype=np.int64),'task_index':np.zeros(n,np.int64),'observation.state':s,'action':a}
        cols={k:c.float_column(v,[20]) if v.ndim==2 else pa.array(v) for k,v in vals.items()};p=out/f'data/chunk-000/episode_{i:06d}.parquet';p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(cols),p)
        st={k:c.stats_for(v) for k,v in vals.items()}
        for k in cameras:
            src=source/f'videos/chunk-000/{k}/episode_{oldi:06d}.mp4';dst=out/f'videos/chunk-000/{k}/episode_{i:06d}.mp4';dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);st[k]=stats[oldi][k]
        for dim in [*range(7),*range(8,15),16,17,18,19]:np.testing.assert_array_equal(a[:,dim],s[np.minimum(np.arange(n)+3,n-1),dim])
        assert set(np.unique(a[:,15])).issubset({0.,1.})
        row={k:v for k,v in ep.items() if k not in ['waist_state_min','waist_state_max','right_gripper_action_min','right_gripper_action_max','right_gripper_transition_count','exact_numeric_values_preserved']};row.update(episode_index=i,dataset_from_index=offset,dataset_to_index=offset+n,merged_episode_index=oldi);results.append({'ep':row,'stats':st});checks.append({'episode':i,'merged_episode':oldi,'frames':n,'future3_exact':True,'binary_gripper_same_frame':True});offset+=n
    info.update(total_episodes=len(results),total_frames=offset,total_videos=len(results)*3,splits={'train':f'0:{len(results)}'});info['training_label_provenance']={'continuous_action':'state[t+3] clamped inside episode','gripper_action':'same-frame original action >=0.5','body_order':['lift','waist_pitch']}
    (out/'meta/info.json').write_text(json.dumps(info,indent=2));(out/'meta/tasks.jsonl').write_bytes((source/'meta/tasks.jsonl').read_bytes());(out/'meta/episodes.jsonl').write_text(''.join(json.dumps(r['ep'])+'\n' for r in results));(out/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps({'episode_index':r['ep']['episode_index'],'stats':r['stats']})+'\n' for r in results));(out/'meta/stats.json').write_text(json.dumps({k:c.combine_episode_stats([r['stats'][k] for r in results]) for k in results[0]['stats']},indent=2))
    modality={'state':{'robot':{'start':0,'end':20,'original_key':'observation.state','dtype':'float32','absolute':True}},'action':{'robot':{'start':0,'end':20,'original_key':'action','dtype':'float32','absolute':True}},'video':{alias:{'original_key':key} for alias,key in [('cam_high_rgb',cameras[cameras.index('observation.images.cam_high')]),('cam_left_wrist_rgb','observation.images.cam_left_wrist'),('cam_right_wrist_rgb','observation.images.cam_right_wrist')]},'annotation':{'human.action.task_description':{'original_key':'task_index'}}}
    (out/'meta/modality.json').write_text(json.dumps(modality,indent=2));(out/'meta/label_validation.json').write_text(json.dumps({'episodes':checks,'all_passed':True},indent=2));print(out.name,len(results),offset,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    # Fixed, whole-episode holdout: 8 old and 2 new. No temporal leakage.
    val=[5,17,29,41,53,65,77,89,94,102];train=[i for i in range(106) if i not in val]
    build(a.input,a.output/'train96',train);build(a.input,a.output/'val10',val)
    (a.output/'split.json').write_text(json.dumps({'train_merged_episode_ids':train,'val_merged_episode_ids':val,'old_train':83,'new_train':13,'old_val':8,'new_val':2,'elapsed_seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':main()

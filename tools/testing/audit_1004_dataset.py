"""Full raw-value, clock, RGB statistics and native-column acceptance for 1004."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import av
import h5py
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from packages.raw_preflight import timestamps

def load_converter():
    spec = importlib.util.spec_from_file_location('converter_1004', REPO/'tools/conversion/00_convert_618_to_lerobot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def video_stats(job):
    path, expected = job
    total = np.zeros(3, np.float64)
    squared = np.zeros(3, np.float64)
    minimum = np.full(3, 255.)
    maximum = np.zeros(3)
    pixels = count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            a = frame.to_ndarray(format='rgb24').reshape(-1,3)
            total += a.sum(axis=0, dtype=np.float64)
            squared += np.einsum('ij,ij->j', a.astype(np.float64), a.astype(np.float64))
            minimum = np.minimum(minimum, a.min(axis=0))
            maximum = np.maximum(maximum, a.max(axis=0))
            pixels += len(a)
            count += 1
    assert count == expected, (path,count,expected)
    mean = total/pixels
    std = np.sqrt(np.maximum(0,squared/pixels-mean**2))
    shape = lambda a: (a/255.).reshape(3,1,1).tolist()
    return str(path), dict(min=shape(minimum),max=shape(maximum),mean=shape(mean),std=shape(std),count=[count])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=Path('E:/1004/record'))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--recompute-rgb',action='store_true')
    p.add_argument('--native',action='store_true')
    p.add_argument('--joint-boundary-fill',action='store_true')
    p.add_argument('--repair-waist',action='store_true')
    args=p.parse_args()
    root=args.input
    c=load_converter()
    info=json.loads((root/'meta/info.json').read_text())
    eps=[json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines()]
    stats_rows=[json.loads(s) for s in (root/'meta/episodes_stats.jsonl').read_text().splitlines()]
    source=json.loads((REPO/'config/sources/g1.json').read_text())
    cameras=[k for k,v in info['features'].items() if v['dtype']=='video']
    rows=[];jobs=[];offset=0;provenance=[]
    for ep in eps:
        i=ep['episode_index'];n=ep['length'];raw=args.raw/ep['original_episode_id']
        target=timestamps(raw/'camera/head_color/head_color.txt')
        assert len(target)==n and np.all(np.diff(target)>0), raw
        path=root/f'data/chunk-000/episode_{i:06d}.parquet'
        table=pq.read_table(path)
        sk='observation.state' if 'observation.state' in table.column_names else 'observations.state.qpos'
        ak='action' if 'action' in table.column_names else 'action.qpos'
        state=np.asarray(table[sk].to_pylist(),np.float32)
        action=np.asarray(table[ak].to_pylist(),np.float32)
        if args.repair_waist:
            assert not info.get('body_order_verified_from_h5_attributes'), 'refusing a second waist swap'
            state[:,[18,19]]=state[:,[19,18]]
            action[:,[18,19]]=action[:,[19,18]]
            table=table.set_column(table.column_names.index(sk),sk,c.float_column(state,[20]))
            table=table.set_column(table.column_names.index(ak),ak,c.float_column(action,[20]))
            pq.write_table(table,path,compression='snappy')
            stats_rows[i]['stats'][sk]=c.stats_for(state)
            stats_rows[i]['stats'][ak]=c.stats_for(action)
        assert state.shape==action.shape==(n,20)
        np.testing.assert_array_equal(table['frame_index'].to_numpy(),np.arange(n))
        np.testing.assert_array_equal(table['index'].to_numpy(),np.arange(offset,offset+n))
        np.testing.assert_allclose(table['timestamp'].to_numpy(),np.arange(n)/30,atol=1e-4,rtol=0)
        with h5py.File(raw/'record/raw_joints.h5','r') as h:
            inputs=c.read_g1_qpos_inputs(h,target,source['raw_h5_adapter'])
            command_times=np.asarray(h['action/joint/timestamp'])
            leading=target<command_times[0]
            trailing=target>command_times[-1]
            if args.joint_boundary_fill:
                inputs['action.joint_position'][leading]=inputs['observation.state.joint_position'][leading]
            # Independently assemble the model's exact order, without EVA's swap.
            def expected(group):
                pre='observation.state' if group=='state' else 'action'
                arms=inputs[pre+'.joint_position']
                denom=120 if group=='state' else 1
                left=np.clip(inputs[pre+'.left_effector_position'],0,denom)/denom
                right=np.clip(inputs[pre+'.right_effector_position'],0,denom)/denom
                head=inputs.get(pre+'.head_position',inputs['observation.state.head_position'])
                waist=inputs.get(pre+'.waist_position',inputs['observation.state.waist_position'])
                return np.concatenate([arms[:,:7],left,arms[:,7:],right,head,waist],axis=1).astype(np.float32)
            np.testing.assert_array_equal(state,expected('state'))
            np.testing.assert_array_equal(action,expected('action'))
            np.testing.assert_array_equal(state[:,18:20],inputs['observation.state.waist_position'])
            raw_body=np.asarray(h['state/waist/position'])
            body_times=np.asarray(h['state/waist/timestamp'])
            body_names=list(h['state/waist'].attrs['name'])
            order=[body_names.index('joint_lift_body'),body_names.index('joint_body_pitch')]
            np.testing.assert_array_equal(state[:,18:20],raw_body[c.nearest(body_times,target)][:,order].astype(np.float32))
            np.testing.assert_array_equal(action[:,16:20],state[:,16:20])
            for frame in range(n):
                provenance.append(dict(episode_index=i,frame_index=frame,raw_timestamp_ns=int(target[frame]),dataset_timestamp_s=frame/30,joint_action_source=('state_fallback' if leading[frame] and args.joint_boundary_fill else 'last_command_hold' if trailing[frame] else 'nearest_recorded_command'),head_action_source='state_fallback',waist_action_source='state_fallback'))
            clock={}
            for group in source['raw_h5_adapter'].values():
                for key in group.values():
                    if h[key].shape is None or not h[key].size:continue
                    times=np.asarray(h[key.rsplit('/',1)[0]+'/timestamp'])
                    assert np.all(np.diff(times)>=0),key
                    assert len(times)==len(h[key]),key
                    ix=c.nearest(times,target)
                    delta=(times[ix]-target)/1e6
                    clock[key]=dict(max_abs_ms=float(np.abs(delta).max()),p95_abs_ms=float(np.percentile(np.abs(delta),95)),outside_support_frames=int(((target<times[0])|(target>times[-1])).sum()))
        for key in cameras:
            jobs.append((root/f'videos/chunk-000/{key}/episode_{i:06d}.mp4',n))
        rows.append(dict(episode_index=i,source=raw.name,frames=n,raw_exact=True,waist_exact=True,head_waist_fallback_exact=True,leading_joint_state_frames=int(leading.sum()),trailing_joint_hold_frames=int(trailing.sum()),time_alignment=clock))
        offset+=n
    assert offset==info['total_frames'] and len(eps)==info['total_episodes']
    with args.report.with_suffix('.provenance.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(provenance[0]));writer.writeheader();writer.writerows(provenance)
    if args.repair_waist:
        info['body_order_verified_from_h5_attributes']=True
        info['conversion_provenance']['joint_boundary_policy']='leading_state_trailing_hold'
        info['conversion_provenance']['raw_body_order']=['waist_pitch','lift']
        info['conversion_provenance']['output_body_order']=['lift','waist_pitch']
        (root/'meta/info.json').write_text(json.dumps(info,indent=2))
        all_keys=stats_rows[0]['stats']
        global_stats={k:c.combine_episode_stats([r['stats'][k] for r in stats_rows]) for k in all_keys}
        (root/'meta/stats.json').write_text(json.dumps(global_stats,indent=2))
        (root/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stats_rows))
    if args.recompute_rgb:
        with ProcessPoolExecutor(max_workers=4) as pool:
            for path,stats in pool.map(video_stats,jobs):
                path=Path(path);i=int(path.stem.split('_')[-1]);stats_rows[i]['stats'][path.parent.name]=stats
        all_keys=stats_rows[0]['stats']
        global_stats={k:c.combine_episode_stats([r['stats'][k] for r in stats_rows]) for k in all_keys}
        (root/'meta/stats.json').write_text(json.dumps(global_stats,indent=2))
        (root/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stats_rows))
    if args.native:
        rename={'observations.state.qpos':'observation.state','action.qpos':'action'}
        for path in (root/'data').rglob('*.parquet'):
            t=pq.read_table(path)
            pq.write_table(t.rename_columns([rename.get(k,k) for k in t.column_names]),path,compression='snappy')
        info['features']={rename.get(k,k):v for k,v in info['features'].items()}
        (root/'meta/info.json').write_text(json.dumps(info,indent=2))
        stats=json.loads((root/'meta/stats.json').read_text())
        (root/'meta/stats.json').write_text(json.dumps({rename.get(k,k):v for k,v in stats.items()},indent=2))
        for row in stats_rows:row['stats']={rename.get(k,k):v for k,v in row['stats'].items()}
        (root/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stats_rows))
    args.report.write_text(json.dumps(dict(episodes=rows,total_frames=offset,raw_value_checks_passed=True,rgb_recomputed=args.recompute_rgb,native_columns=args.native),indent=2))
    print('RAW_VALUE_ACCEPTANCE',len(rows),offset,flush=True)

if __name__=='__main__':main()

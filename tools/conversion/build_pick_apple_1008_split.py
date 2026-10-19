"""转换后构建全新训练/验证集，不裁剪，不改变旧标签和视频。"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.raw_preflight import timestamps
sys.path.insert(0, str(ROOT / 'tools/testing'))
from audit_1004_dataset import load_converter

BASE = ROOT / 'output/1008_pick_apple_20261008'
OLD = ROOT / 'output/pick_apple_multigrasp_20261007/training'
DEST = BASE / 'training_split_v1'
CAMERAS = ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    if DEST.exists():
        raise RuntimeError('拒绝覆盖已有划分目录')
    converter = load_converter()
    with (BASE / 'read_validation.csv').open(encoding='utf-8') as f:
        validation = list(csv.DictReader(f))
    good = {(r['task_directory'], int(r['episode_index'])) for r in validation if r['status'] == 'ok'}
    excluded = [r for r in validation if r['status'] != 'ok']
    entries = []
    old_val_ids = set()
    for split in ('train133', 'val6'):
        source = OLD / split
        for ep in lines(source / 'meta/episodes.jsonl'):
            if 'original_episode_id' not in ep:
                # 多次抓取来源没有 UUID，使用原来源目录与轨迹编号的稳定组合。
                ep['original_episode_id'] = ep['source_dataset'] + ':' + str(ep['source_episode_index'])
            if split == 'val6':
                old_val_ids.add(ep['original_episode_id'])
            entries.append(dict(source=source, ep=ep, is_new=False, validation=split == 'val6'))
    new_entries = []
    for source in sorted((BASE / 'converted_v2').iterdir()):
        if not (source / 'meta/info.json').is_file():
            continue
        for ep in lines(source / 'meta/episodes.jsonl'):
            if (source.name, ep['episode_index']) in good:
                new_entries.append(dict(source=source, ep=ep, is_new=True, validation=False))
    existing_ids = {e['ep']['original_episode_id'] for e in entries}
    duplicate_new = [dict(id=e['ep']['original_episode_id'],source=str(e['source']),reason='已存在于旧数据；保留原划分，避免重复及验证泄漏') for e in new_entries if e['ep']['original_episode_id'] in existing_ids]
    new_entries = [e for e in new_entries if e['ep']['original_episode_id'] not in existing_ids]
    shortest = sorted(new_entries, key=lambda e:(e['ep']['length'], e['ep']['original_episode_id']))[:2]
    assert len(shortest) == 2
    val_new_ids = {e['ep']['original_episode_id'] for e in shortest}
    for item in new_entries:
        item['validation'] = item['ep']['original_episode_id'] in val_new_ids
    entries.extend(new_entries)
    ids = [e['ep']['original_episode_id'] for e in entries]
    assert len(ids) == len(set(ids)), '不同来源存在重复录制'
    assert len(old_val_ids) == 6
    names = read_json(OLD / 'train133/meta/info.json')['features']['observation.state']['names']
    assert names[-2:] == ['lift', 'waist_pitch']
    DEST.mkdir()
    manifest = []
    raw_adapter = read_json(ROOT / 'config/sources/g1.json')['raw_h5_adapter']
    for split, want_val in [('train', False), ('val', True)]:
        out = DEST / split
        (out / 'meta').mkdir(parents=True)
        episodes = []
        offset = 0
        for item in [e for e in entries if e['validation'] == want_val]:
            source, ep = item['source'], item['ep']
            old_index, i, n = ep['episode_index'], len(episodes), ep['length']
            t = pq.read_table(source / f'data/chunk-000/episode_{old_index:06d}.parquet')
            sk, ak = ('observations.state.qpos','action.qpos') if item['is_new'] else ('observation.state','action')
            s = np.asarray(t[sk].to_pylist(), np.float32)
            a = np.asarray(t[ak].to_pylist(), np.float32)
            assert s.shape == a.shape == (n,20) and np.isfinite(s).all() and np.isfinite(a).all()
            if item['is_new']:
                # EVA 转换基线为俯仰、升降；当前模型顺序为升降、俯仰。
                s[:,[18,19]] = s[:,[19,18]]
                a[:,[18,19]] = a[:,[19,18]]
                raw = Path('E:/1008_apple/record') / ep['original_episode_id']
                target = timestamps(raw / 'camera/head_color/head_color.txt')
                assert len(target) == n
                with h5py.File(raw / 'record/raw_joints.h5') as h:
                    body_names = list(h['state/waist'].attrs['name'])
                    order = [body_names.index('joint_lift_body'),body_names.index('joint_body_pitch')]
                    raw_body = np.asarray(h['state/waist/position'])
                    body_times = np.asarray(h['state/waist/timestamp'])
                    expected = raw_body[converter.nearest(body_times,target)][:,order].astype(np.float32)
                    np.testing.assert_array_equal(s[:,18:20],expected)
                    inputs = converter.read_g1_qpos_inputs(h,target,raw_adapter)
                    for dim,key in [(7,'left'),(15,'right')]:
                        expected_grip = np.clip(inputs[f'observation.state.{key}_effector_position'],0,120).reshape(-1)/120
                        np.testing.assert_array_equal(s[:,dim],expected_grip.astype(np.float32))
                        expected_action = np.clip(inputs[f'action.{key}_effector_position'],0,1).reshape(-1)
                        np.testing.assert_array_equal(a[:,dim],expected_action.astype(np.float32))
                # 与当前 ACT/DP 数据相同：连续动作使用未来三帧状态，夹爪保留同帧指令。
                command_grip = a[:,[7,15]].copy()
                a = s[np.minimum(np.arange(n)+3,n-1)].copy()
                a[:,[7,15]] = (command_grip >= .5).astype(np.float32)
            assert np.all((s[:,[7,15]]>=0)&(s[:,[7,15]]<=1))
            assert np.all((a[:,[7,15]]>=0)&(a[:,[7,15]]<=1))
            cols = dict(timestamp=pa.array(np.arange(n,dtype=np.float32)/30), frame_index=pa.array(np.arange(n,dtype=np.int64)), episode_index=pa.array(np.full(n,i,np.int64)), index=pa.array(np.arange(offset,offset+n,dtype=np.int64)), task_index=pa.array(np.zeros(n,np.int64)))
            cols['observation.state'] = converter.float_column(s,[20])
            cols['action'] = converter.float_column(a,[20])
            dest = out / f'data/chunk-000/episode_{i:06d}.parquet'
            dest.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.table(cols), dest)
            for key in CAMERAS:
                video = source / f'videos/chunk-000/{key}/episode_{old_index:06d}.mp4'
                dest = out / f'videos/chunk-000/{key}/episode_{i:06d}.mp4'
                dest.parent.mkdir(parents=True,exist_ok=True)
                os.link(video,dest)
            row = dict(episode_index=i,length=n,tasks=['pick up the apple'],original_episode_id=ep['original_episode_id'],source_dataset=str(source),source_episode_index=old_index,source_group='1008' if item['is_new'] else ('1004' if '1004' in ep.get('source_dataset','') else ('multigrasp' if 'EVA-CLIENT' in ep.get('source_dataset','') else 'old')),dataset_from_index=offset,dataset_to_index=offset+n)
            episodes.append(row)
            manifest.append(dict(split=split,**row))
            offset += n
        features = read_json(OLD / 'train133/meta/info.json')['features']
        for key in CAMERAS:
            # 原始图像比例保留，各来源分辨率在逐条清单中记录；不重新拉伸旧视频。
            features[key]['shape'] = [-1,-1,3]
            features[key].pop('info',None)
        info = dict(codebase_version='v2.1',robot_type='G1',fps=30,total_episodes=len(episodes),total_frames=offset,total_tasks=1,total_videos=3*len(episodes),total_chunks=1,chunks_size=1000,splits={split:f'0:{len(episodes)}'},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',features=features,training_label_provenance=dict(continuous_action='state[t+3] clamped inside episode',gripper_action='same-frame command >=0.5; 0=open,1=closed',body_order=['lift','waist_pitch']),statistics_status='pending full recomputation',trim_applied=False)
        dump(out/'meta/info.json',info)
        (out/'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in episodes),encoding='utf-8')
        (out/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0,task='pick up the apple'))+'\n',encoding='utf-8')
    train_ids = {e['original_episode_id'] for e in manifest if e['split']=='train'}
    val_ids = {e['original_episode_id'] for e in manifest if e['split']=='val'}
    assert not train_ids & val_ids and len(val_ids)==8
    dump(DEST/'split_manifest.json',dict(train_episodes=len(train_ids),validation_episodes=len(val_ids),train_val_disjoint=True,selected_1008_shortest=[dict(id=e['ep']['original_episode_id'],frames=e['ep']['length'],seconds=e['ep']['length']/30) for e in shortest],excluded_conversion_defects=excluded,excluded_duplicate_recordings=duplicate_new,entries=manifest,trim_applied=False,statistics_status='pending'))
    print('SPLIT_WRITTEN',len(train_ids),len(val_ids),flush=True)

if __name__ == '__main__':
    main()

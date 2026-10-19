"""全量统计重算后，固定验证使用训练统计并验收划分。"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import shutil
import argparse
import av
import numpy as np
import pyarrow.parquet as pq

BASE = Path('output/1008_pick_apple_20261008')
ROOT = BASE / 'training_split_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT)
    parser.add_argument('--read-report',type=Path,default=BASE/'split_read_validation.csv')
    args=parser.parse_args()
    ROOT=args.input
    manifest = read(ROOT/'split_manifest.json')
    with args.read_report.open(encoding='utf-8') as f:
        checks = list(csv.DictReader(f))
    assert len(checks)==157 and all(r['status']=='ok' for r in checks)
    train_stats_path = ROOT/'train/meta/stats.json'
    train_stats = read(train_stats_path)
    val_stats_path = ROOT/'val/meta/stats.json'
    diagnosis = val_stats_path.with_name('validation_distribution_stats.json')
    if diagnosis.exists():
        # 新图像版本继承的旧诊断统计另存，保留重算前的来源证据。
        archived=diagnosis.with_name('source_validation_distribution_stats.json')
        assert not archived.exists(), '拒绝再次覆盖诊断统计'
        diagnosis.rename(archived)
    shutil.copy2(val_stats_path,diagnosis)
    shutil.copy2(train_stats_path,val_stats_path)
    geometry = []
    totals = {}
    for split in ('train','val'):
        root = ROOT/split
        info = read(root/'meta/info.json')
        eps = [json.loads(l) for l in (root/'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if l]
        parts = [pq.read_table(root/f'data/chunk-000/episode_{e["episode_index"]:06d}.parquet') for e in eps]
        for key in ('observation.state','action'):
            values = np.concatenate([np.asarray(p[key].to_pylist(),np.float64) for p in parts])
            assert np.isfinite(values).all()
            if split=='train':
                for name,expected in [('mean',values.mean(0)),('std',values.std(0)),('min',values.min(0)),('max',values.max(0))]:
                    np.testing.assert_allclose(train_stats[key][name],expected,rtol=1e-6,atol=1e-7)
                assert train_stats[key]['count'][0] == len(values)
        offset=0
        for e,t in zip(eps,parts):
            n=e['length']
            np.testing.assert_array_equal(t['index'].to_numpy(),np.arange(offset,offset+n))
            np.testing.assert_allclose(t['timestamp'].to_numpy(),np.arange(n)/30,atol=1e-4,rtol=0)
            offset+=n
            for key,v in info['features'].items():
                if v['dtype']!='video':continue
                path=root/f'videos/chunk-000/{key}/episode_{e["episode_index"]:06d}.mp4'
                with av.open(str(path)) as c:
                    stream=c.streams.video[0]
                    geometry.append(dict(split=split,episode=e['episode_index'],camera=key,width=stream.codec_context.width,height=stream.codec_context.height))
        totals[split]=dict(episodes=len(eps),frames=offset)
        info['statistics_status']='complete'
        info['normalization_statistics_source']='train only'
        info['normalization_statistics_sha256']=hashlib.sha256(train_stats_path.read_bytes()).hexdigest()
        dump(root/'meta/info.json',info)
        # 沿用已核实的当前模型模态注册，不改变相机名称。
        shutil.copy2(Path('output/pick_apple_multigrasp_20261007/training/train133/meta/modality.json'),root/'meta/modality.json')
    assert train_stats_path.read_bytes()==val_stats_path.read_bytes()
    counts=collections.Counter((e['split'],e['source_group']) for e in manifest['entries'])
    assert counts[('val','old')]==5 and counts[('val','1004')]==1 and counts[('val','1008')]==2
    train_ids={e['original_episode_id'] for e in manifest['entries'] if e['split']=='train'}
    val_ids={e['original_episode_id'] for e in manifest['entries'] if e['split']=='val'}
    assert not train_ids & val_ids
    manifest['statistics_status']='complete'
    dump(ROOT/'split_manifest.json',manifest)
    dump(ROOT/'camera_geometry.json',geometry)
    dump(ROOT/'split_acceptance.json',dict(status='PASS',totals=totals,train_val_disjoint=True,all_157_episodes_fully_decoded=True,normalization_train_only=True,train_numeric_stats_independently_verified=True,trim_applied=False,source_counts={str(k):v for k,v in counts.items()}))
    print('SPLIT_ACCEPTANCE_PASS',totals,flush=True)

if __name__=='__main__':
    main()

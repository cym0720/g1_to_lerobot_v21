"""只读核查 1008 录制的数值来源，禁止把空 H5 当有效轨迹。"""
from pathlib import Path
import json
import h5py

ROOT = Path('E:/1008_apple/record')
OUT = Path('output/1008_pick_apple_20261008')
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for p in sorted(ROOT.iterdir()):
    if not p.is_dir():
        continue
    meta = json.loads((p / 'meta_info.json').read_text(encoding='utf-8'))
    row = dict(episode=p.name, task_id=meta.get('task_id'), streams={}, errors=[])
    hp = p / 'record/raw_joints.h5'
    if not hp.exists():
        row['errors'].append('缺少 raw_joints.h5')
    else:
        try:
            with h5py.File(hp) as h:
                for k in ('state/joint/position', 'action/joint/position', 'state/head/position', 'state/waist/position', 'state/left_effector/position', 'state/right_effector/position'):
                    shape = h[k].shape if k in h else None
                    row['streams'][k] = shape
                    if shape is None or not shape or shape[0] == 0:
                        row['errors'].append('空或缺失信号:' + k)
        except Exception as exc:
            row['errors'].append(str(exc))
    rows.append(row)
result = dict(raw_episodes=len(rows), numeric_ready=sum(not r['errors'] for r in rows), rows=rows)
(OUT / 'source_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False))
for row in rows:
    print(row['episode'], row['task_id'], ';'.join(row['errors']))

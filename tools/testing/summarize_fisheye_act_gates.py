"""Collect material-stage ACT receipts; benchmark receipts are never accepted."""
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]/'output/fisheye_rebuild_audit'
    reports=[]
    for p in sorted(root.glob('*.act.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        if r.get('benchmark_only'):continue
        assert r['real_act'] and r['three_real_cameras'] and r['loss_decreased'],p
        assert r['steps']==24 and len(r['camera_keys'])==3,p
        reports.append({'stage':p.stem.removesuffix('.act'),'input':r['input'],'report':str(p),'first_window_mean':r['first_window_mean'],'last_window_mean':r['last_window_mean'],'elapsed_seconds':r['elapsed_seconds'],'loss_decreased':True})
    result={'passed_dataset_gates':len(reports),'reports':reports,'total_act_seconds':sum(r['elapsed_seconds'] for r in reports)}
    (root/'act_gate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

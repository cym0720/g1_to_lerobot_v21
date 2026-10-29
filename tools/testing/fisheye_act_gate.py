"""Run the real three-camera ACT gate only after a complete integrity read."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation',type=Path)
    p.add_argument('--benchmark-only',action='store_true')
    a=p.parse_args()
    if not a.benchmark_only:
        if not a.validation: p.error('--validation required for material stage')
        v=json.loads(a.validation.read_text(encoding='utf-8'))
        assert v.get('status')=='OK' and v.get('full_read') is True,v
        assert Path(v['input']).resolve()==a.input.resolve(), 'validation belongs to another stage'
    info=json.loads((a.input/'meta/info.json').read_text(encoding='utf-8'))
    cams=[k for k,v in info['features'].items() if v['dtype']=='video']
    assert len(cams)==3,cams
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise FileExistsError(a.output)
    start=time.perf_counter()
    cmd=[sys.executable,str(Path(__file__).with_name('act_policy_loss_smoke.py')),'--input',str(a.input),'--output',str(a.output),'--chunk-size','16']
    with a.output.with_suffix('.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'ACT failed; see {a.output.with_suffix(".log")}')
    r=json.loads(a.output.read_text()); assert r['loss_decreased'] and r['real_act']
    r.update(input=str(a.input.resolve()),camera_keys=cams,elapsed_seconds=time.perf_counter()-start,benchmark_only=a.benchmark_only,validation=str(a.validation) if a.validation else None)
    a.output.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()

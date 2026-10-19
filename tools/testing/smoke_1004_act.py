"""Real three-camera ACT forward/backward acceptance on 1004 data (CPU)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import av,cv2,numpy as np,pyarrow.parquet as pq,torch
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-index',type=int,default=0,help='episode to exercise; useful for testing appended episodes')
    p.add_argument('--chunk-size',type=int,default=8,help='烟雾测试动作块长度；当前任务使用16')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    info=json.loads((a.input/'meta/info.json').read_text(encoding='utf-8'))
    stats=json.loads((a.input/'meta/stats.json').read_text(encoding='utf-8'))
    path=a.input/'data/chunk-000'/f'episode_{a.episode_index:06d}.parquet'
    if not path.is_file():
        raise FileNotFoundError(f'episode {a.episode_index} does not exist under {a.input}')
    table=pq.read_table(path)
    sk='observation.state' if 'observation.state' in table.column_names else 'observations.state.qpos'
    ak='action' if 'action' in table.column_names else 'action.qpos'
    states=np.asarray(table[sk].to_pylist(),np.float32);actions=np.asarray(table[ak].to_pylist(),np.float32)
    if a.chunk_size < 1:
        raise ValueError('动作块长度必须为正数')
    starts=[len(states)//4,len(states)//2];chunk=min(a.chunk_size,len(states)-max(starts))
    cams=[k for k,v in info['features'].items() if v['dtype']=='video']
    features={'observation.state':PolicyFeature(FeatureType.STATE,(20,))}
    features.update({k:PolicyFeature(FeatureType.VISUAL,(3,224,224)) for k in cams})
    ds={}
    for old,new in [(sk,'observation.state'),(ak,'action'),*[(k,k) for k in cams]]:
        ds[new]={name:torch.tensor(stats[old][name],dtype=torch.float32) for name in ('mean','std','min','max')}
    config=ACTConfig(input_features=features,output_features={'action':PolicyFeature(FeatureType.ACTION,(20,))},chunk_size=chunk,n_action_steps=chunk,dim_model=64,n_heads=4,dim_feedforward=128,n_encoder_layers=1,n_decoder_layers=1,n_vae_encoder_layers=1,latent_dim=8,pretrained_backbone_weights=None,device='cpu',dropout=0.,kl_weight=.01)
    model=ACTPolicy(config,dataset_stats=ds)
    # A compact real ACT checks gradients and labels; the saved production model
    # is not modified or retrained by this overfit gate.
    for name,param in model.named_parameters():
        if 'backbone' in name:param.requires_grad_(False)
    batch={'observation.state':torch.from_numpy(states[starts]),'action':torch.from_numpy(np.stack([actions[i:i+chunk] for i in starts])),'action_is_pad':torch.zeros(2,chunk,dtype=torch.bool)}
    for key in cams:
        video=a.input/'videos'/path.parent.name/key/(path.stem+'.mp4');found={}
        with av.open(str(video)) as c:
            for i,frame in enumerate(c.decode(video=0)):
                if i in starts:
                    rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2];scale=224/max(h,w);nw,nh=round(w*scale),round(h*scale)
                    canvas=np.zeros((224,224,3),np.uint8);x,y=(224-nw)//2,(224-nh)//2
                    canvas[y:y+nh,x:x+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
                    found[i]=canvas.transpose(2,0,1).astype(np.float32)/255
                if len(found)==len(starts):break
        batch[key]=torch.from_numpy(np.stack([found[i] for i in starts]))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    model.train();losses=[]
    for step in range(24):
        optimizer.zero_grad();loss,_=model(batch);assert torch.isfinite(loss)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();losses.append(float(loss.detach()));print('step',step+1,'loss',losses[-1],flush=True)
    first=float(np.mean(losses[:6]));last=float(np.mean(losses[-6:]));ok=last<first
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(real_act=True,three_real_cameras=True,device='cpu',episode_index=a.episode_index,chunk_size=chunk,steps=len(losses),first_window_mean=first,last_window_mean=last,loss_decreased=ok,losses=losses,production_checkpoint_modified=False),indent=2))
    assert ok,(first,last)
if __name__=='__main__':main()

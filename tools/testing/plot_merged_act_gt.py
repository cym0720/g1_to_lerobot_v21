"""Plot held-out old/new recordings using the final StarVLA deployment wrapper."""
import sys
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['STARVLA_DISABLE_DEEPSPEED'] = '1'
import json
from pathlib import Path
import numpy as np
import torch
import audit_g1_act_gt as audit
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
DATASET_NAMED_MIXTURES['g1_pick_apple_merged_16'] = [('train96', 1., 'g1_20d_threecam_16step')]
import matplotlib.pyplot as plt

ROOT = Path('/mnt/d/go work/数据清洗/output/pick_apple_merged_20261006')
DATA = Path('/home/cym/datasets/g1/pick_apple_merged_20261006/val10')
CHECKPOINT = Path('/home/cym/artifacts/starvla/runs/g1_act_pick_apple_merged_20261006_c16_s20000/final_model/pytorch_model.pt')
OUT = ROOT / 'gt_analysis_20000'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42)
wrapper = audit.make_wrapper(CHECKPOINT)
info = json.loads((DATA / 'meta/info.json').read_text())
report = {'checkpoint': str(CHECKPOINT), 'dataset': str(DATA), 'chunk': 16,
          'normalization': 'saved PolicyServerWrapper statistics', 'episodes': {}}
for ep, merged, source in [(0, 5, 'deploy_g1'), (8, 94, '1004')]:
    print('LOAD', ep, source, flush=True)
    data = audit.load_episode(DATA, info, ep)
    pred = audit.infer(wrapper, data, 'fp32')
    gt, state = data['action'], data['state']
    n = len(gt)
    executed = np.concatenate([pred[s, :min(16, n-s)] for s in range(0, n, 16)])
    mae = np.abs(executed - gt).mean(axis=0)
    metrics = {'frames': n, 'merged_episode': merged, 'source': source,
               'executed_mae_per_dim': mae.tolist(), 'right_arm_mae_rad': float(mae[8:15].mean()),
               'right_gripper_mae': float(mae[15]),
               'right_gripper_gt_range': [float(gt[:,15].min()), float(gt[:,15].max())],
               'right_gripper_pred_range': [float(executed[:,15].min()), float(executed[:,15].max())],
               'single_step_mae_per_dim': np.abs(pred[:,0] - gt).mean(axis=0).tolist()}
    report['episodes'][str(ep)] = metrics
    np.savez_compressed(OUT / f'{source}_episode_{ep:06d}.npz', gt=gt, state=state, prediction_chunks=pred, executed=executed)
    for kind, values in [('chunks16', executed), ('single_step', pred[:,0])]:
        fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
        for ax, dim in zip(axes.flat, range(8,16)):
            ax.plot(gt[:,dim], color='black', linewidth=1.3, label='GT training target')
            ax.plot(values[:,dim], color='#ed7d31', linewidth=1.1, label='ACT prediction')
            ax.plot(state[:,dim], color='#4472c4', linewidth=.8, alpha=.65, label='Observed state')
            ax.set_title(f'Right joint {dim-8+1}' if dim<15 else 'Right gripper (closed fraction)')
            ax.set_ylabel('rad' if dim<15 else 'fraction')
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=9)
        for ax in axes[-1]: ax.set_xlabel('Frame')
        fig.suptitle(f'{source} held-out episode {ep} (merged {merged}) | step 20000 | {kind} | chunk=16', fontsize=15)
        fig.tight_layout(rect=[0,0,1,.97])
        fig.savefig(OUT / f'{source}_{kind}.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(14,6),sharex=True)
    for ax, dim, name in zip(axes,[18,19],['Lift [index 18]','Waist pitch [index 19]']):
        ax.plot(gt[:,dim],label='GT'); ax.plot(executed[:,dim],label='ACT'); ax.plot(state[:,dim],label='State',alpha=.6)
        ax.set_title(name); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f'{source}_waist.png',dpi=140); plt.close(fig)
    print('RESULT', json.dumps(metrics), flush=True)
    (OUT / 'report.json').write_text(json.dumps(report,indent=2))
print('COMPLETE', OUT, flush=True)

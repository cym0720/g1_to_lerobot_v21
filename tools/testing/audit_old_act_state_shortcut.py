#!/usr/bin/env python3
"""CPU-only old-checkpoint image/state interventions and episode-held-out clock test.

Writes diagnostics only. Counterfactual inputs are not robot rollouts; sensitivity
does not establish task-correct visual grounding or the cause of temporal collapse.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import audit_g1_act_gt as audit

OUTPUT = audit.DEFAULT_OUTPUT.parent / 'state_shortcut_15000'
EPISODES = (0, 18, 36, 54, 72, 90)


def clock_test(info):
    """kNN phase prediction; split entire episodes, never adjacent random frames."""
    xs, ys, ids = [], [], []
    for ep in range(info['total_episodes']):
        path = audit.DEFAULT_DATASET / info['data_path'].format(
            episode_index=ep, episode_chunk=ep // info['chunks_size'])
        table = pd.read_parquet(path)
        state = np.stack(table['observations.state.qpos']).astype(np.float32)
        frames = np.arange(0, len(state), 5)
        xs.append(state[frames, 8:16])
        ys.append(frames / max(len(state) - 1, 1))
        ids.extend([ep] * len(frames))
    x, y, ids = np.concatenate(xs), np.concatenate(ys), np.array(ids)
    test = ids % 5 == 0
    scale = np.maximum(x[~test].std(0), 0.01)
    train_x, test_x = x[~test] / scale, x[test] / scale
    pred = []
    for start in range(0, len(test_x), 64):
        dist = ((test_x[start:start+64, None] - train_x[None]) ** 2).sum(-1)
        nearest = np.argpartition(dist, 4, axis=1)[:, :5]
        pred.extend(y[~test][nearest].mean(1))
    pred = np.array(pred)
    truth = y[test]
    baseline = np.full_like(truth, y[~test].mean())
    result = dict(train_episodes=np.unique(ids[~test]).tolist(),
                  test_episodes=np.unique(ids[test]).tolist(), test_samples=len(truth),
                  dimensions='right arm seven joints plus right gripper', k=5,
                  progress_definition='frame_index / (episode_length - 1)',
                  mae_fraction=float(np.abs(pred-truth).mean()),
                  baseline_mae_fraction=float(np.abs(baseline-truth).mean()),
                  r2=float(1-((pred-truth)**2).sum()/((truth-truth.mean())**2).sum()),
                  p90_absolute_error=float(np.quantile(np.abs(pred-truth), .9)),
                  caveat='Held out for the kNN regressor only; ACT trained on all these episodes. Progress predictability is not proof ACT encodes progress.')
    fig, ax = audit.plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(truth, pred, s=5, alpha=.3)
    ax.plot([0,1], [0,1], color='black')
    ax.set(xlabel='Actual episode progress', ylabel='Predicted from right state (held-out episodes)')
    fig.savefig(OUTPUT / 'state_progress.png', dpi=150)
    audit.plt.close(fig)
    return result


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((audit.DEFAULT_DATASET / 'meta/info.json').read_text())
    report = dict(checkpoint=str(audit.RUN / 'checkpoints/steps_15000_pytorch_model.pt'),
                  dataset=str(audit.DEFAULT_DATASET), device='cpu', precision='FP32',
                  state_clock=clock_test(info), rows=[],
                  caveats=['Offline sensitivity, not deployment success.',
                           'Black images and inconsistent image/state pairs can be out of distribution.',
                           'Arm MAE in radians; gripper reported separately in normalized closure units.'])
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('CLOCK_COMPLETE', json.dumps(report['state_clock']), flush=True)
    wrapper = audit.PolicyServerWrapper(report['checkpoint'], device='cpu', use_bf16=False,
                                        config_overrides=['framework.action_model.chunk_size=16'])
    proc = wrapper.get_norm_processor()
    proc.apply_state = lambda raw: audit.normalize_modality(proc, raw, proc.state_keys, proc.state_key_dims)
    model = wrapper._framework.float().eval()
    assert next(model.parameters()).device.type == 'cpu'
    predictions, targets = [], []
    for ep_id in EPISODES:
        ep = audit.load_episode(audit.DEFAULT_DATASET, info, ep_id)
        # Stay away from padding and choose well-separated real observations.
        starts = np.unique(np.linspace(16, len(ep['state'])-48, 5, dtype=int))
        for frame in starts:
            other = int(frame + 32)
            images = [cam[frame] for cam in ep['images']]
            later_images = [cam[other] for cam in ep['images']]
            state = ep['state'][frame]
            conditions = {'original': (images, state),
                          'images_plus32': (later_images, state),
                          'black_images': ([np.zeros_like(im) for im in images], state),
                          'state_plus32': (images, ep['state'][other]),
                          'state_episode_start': (images, ep['state'][0])}
            examples = [dict(image=ims, state=proc.apply_state(st)) for ims, st in conditions.values()]
            with torch.inference_mode():
                batch = model._examples_to_act_batch(examples, require_action=False)
                batch = model.action_model.normalize_inputs(batch)
                batch['observation.images'] = [batch[k] for k in model.action_model.config.image_features]
                out, _ = model.action_model.model(batch)
            raw = np.stack([proc.unapply_actions(chunk) for chunk in out.float().numpy()])
            assert raw.shape == (5, 16, 20) and np.isfinite(raw).all()
            gt = ep['action'][frame:frame+16]
            row = dict(episode=ep_id, frame=int(frame), other_frame=other,
                       gt_mean_arm_span=float(np.ptp(gt[:,8:15], axis=0).mean()), conditions={})
            for i, name in enumerate(conditions):
                row['conditions'][name] = dict(
                    arm_mae=float(np.abs(raw[i,:,8:15]-gt[:,8:15]).mean()),
                    arm_delta_from_original=float(np.abs(raw[i,:,8:15]-raw[0,:,8:15]).mean()),
                    arm_mean_chunk_span=float(np.ptp(raw[i,:,8:15],axis=0).mean()),
                    gripper_mae=float(np.abs(raw[i,:,15]-gt[:,15]).mean()),
                    gripper_delta_from_original=float(np.abs(raw[i,:,15]-raw[0,:,15]).mean()))
            report['rows'].append(row)
            predictions.append(raw)
            targets.append(gt)
        print('EPISODE_DONE', ep_id, 'samples', len(report['rows']), flush=True)
        (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2))
    report['summary'] = {name: {metric: float(np.mean([
        r['conditions'][name][metric] for r in report['rows']]))
        for metric in report['rows'][0]['conditions'][name]} for name in conditions}
    report['summary']['gt_mean_arm_span'] = float(np.mean([r['gt_mean_arm_span'] for r in report['rows']]))
    report['phase'] = 'complete'
    np.savez_compressed(OUTPUT / 'raw_chunks.npz', predictions=np.stack(predictions), targets=np.stack(targets),
                        conditions=np.array(list(conditions)))
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('OLD_STATE_SHORTCUT_COMPLETE', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()

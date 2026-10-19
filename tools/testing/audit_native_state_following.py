#!/usr/bin/env python3
"""CPU-only baseline and matched input interventions for native ACT checkpoints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import train_g1_native_lerobot as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=baseline.OUTPUT)
    args = parser.parse_args()
    root = args.input
    completed = sorted(p for p in (root / "gt").glob("step_*") if (p / "report.json").exists())
    if not completed:
        raise RuntimeError(f"No completed checkpoint GT available: {root}")
    step = int(completed[-1].name.split("_")[-1])
    info = json.loads((root / "dataset/meta/info.json").read_text())
    dimensions = info["features"]["action"]["shape"][0]
    assert dimensions in (8, 20), dimensions
    arm = slice(0, 7) if dimensions == 8 else slice(8, 15)
    gripper = 7 if dimensions == 8 else 15
    stats = json.loads((root / "dataset/meta/stats.json").read_text())
    scale = np.asarray(stats["action"]["std"]) + 1e-8
    out = root / f"state_following/step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rows = {}
    offsets = {}
    states = {}
    offset = 0
    for path in sorted((root / "dataset/data").rglob("*.parquet")):
        table = pq.read_table(path)
        ep = int(table["episode_index"][0].as_py())
        offsets[ep] = offset
        states[ep] = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        offset += len(table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ep in baseline.EVAL_EPISODES:
        saved = np.load(root / f"gt/step_{step:06d}/episode_{ep:06d}.npz")
        starts, pred, gt = saved["starts"], saved["predictions"], saved["gt_action"]
        target, predictions, holds = [], [], []
        for f, p in zip(starts, pred):
            if f + 16 <= len(gt):
                target.append(gt[f:f+16])
                predictions.append(p)
                holds.append(np.repeat(states[ep][f:f+1], 16, axis=0))
        target, predictions, holds = map(np.stack, (target, predictions, holds))
        mae_pred = np.abs(predictions - target).mean(axis=(0, 1))
        mae_hold = np.abs(holds - target).mean(axis=(0, 1))
        distance_state = np.abs(predictions - holds).mean(axis=(0, 1))
        correction_gt = target - holds
        correction_pred = predictions - holds
        rows[str(ep)] = dict(
            prediction_mae_by_dimension=mae_pred.tolist(), state_hold_mae_by_dimension=mae_hold.tolist(),
            prediction_distance_to_current_state_by_dimension=distance_state.tolist(),
            right_arm_prediction_mae=float(mae_pred[arm].mean()),
            right_arm_state_hold_mae=float(mae_hold[arm].mean()),
            right_arm_relative_improvement=float(1-mae_pred[arm].mean()/mae_hold[arm].mean()),
            std_scaled_inference_mae_by_dimension=(mae_pred / scale).tolist(),
            right_arm_future_correction_cosine=float(np.sum(correction_gt[:,:,arm]*correction_pred[:,:,arm]) /
                max(np.linalg.norm(correction_gt[:,:,arm])*np.linalg.norm(correction_pred[:,:,arm]),1e-12)))
        hold_executed = np.concatenate([np.repeat(states[ep][f:f+1], min(16,len(gt)-f),axis=0) for f in starts])
        fig, axes = plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            d=j+arm.start
            ax.plot(gt[:,d],color="black",label="Future action GT")
            ax.plot(saved["executed"][:,d],color="tab:orange",label="Native ACT: all 16 steps")
            ax.plot(hold_executed[:,d],color="tab:blue",alpha=.7,label="Current state repeated 16 times")
            ax.set_title(f"right_arm_joint{j+1}" if j<7 else "right_gripper")
            ax.grid(alpha=.25)
            if j==0:ax.legend(fontsize=7)
        fig.savefig(out/f"episode_{ep:06d}.png",dpi=120)
        plt.close(fig)
    baseline.dump(out / "baseline_report.json", dict(episodes=rows,
        note="Same complete non-overlapping windows; radians for arm. Training-set fit only."))
    print("STATE_HOLD_BASELINE_COMPLETE", json.dumps(rows), flush=True)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    checkpoint=root/f"run/checkpoints/{step:06d}/pretrained_model"
    policy_cfg=PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device="cpu"
    policy_cfg.use_amp=False
    cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple",root=str(root/"dataset"),
        video_backend="pyav",use_imagenet_stats=True),policy=policy_cfg)
    dataset=make_dataset(cfg)
    dataset.image_transforms=baseline.Letterbox()
    policy=ACTPolicy.from_pretrained(checkpoint,config=policy_cfg,local_files_only=True).cpu().eval()
    interventions=[]
    with torch.inference_mode():
        for ep in baseline.EVAL_EPISODES:
            for f in (52,104,156):
                original=baseline.sample_batch(dataset,[offsets[ep]+f],"cpu")
                other=baseline.sample_batch(dataset,[offsets[ep]+min(f+32,len(states[ep])-1)],"cpu")
                variants={"original":original}
                blank=dict(original)
                swapped=dict(original)
                for key in dataset.meta.camera_keys:
                    blank[key]=torch.zeros_like(original[key])
                    swapped[key]=other[key]
                variants["images_black_same_state"]=blank
                variants["images_plus32_same_state"]=swapped
                changed_state=dict(original)
                changed_state["observation.state"]=other["observation.state"]
                variants["state_plus32_same_images"]=changed_state
                output={k:policy.predict_action_chunk(v)[0].numpy() for k,v in variants.items()}
                row=dict(episode=ep,frame=f,matched_source_frame=min(f+32,len(states[ep])-1))
                for key in variants:
                    if key!="original":
                        delta=np.abs(output[key]-output["original"]).mean(axis=0)
                        row[key]=dict(right_arm_mean_delta_rad=float(delta[arm].mean()),gripper_delta=float(delta[gripper]),
                                      delta_by_dimension=delta.tolist())
                interventions.append(row)
                baseline.dump(out/"input_interventions.json",dict(samples=interventions,
                    note="Fixed checkpoint; CPU FP32; image swap holds state fixed, state swap holds images fixed. Black images are out-of-distribution."))
                print("INPUT_INTERVENTION",json.dumps(row),flush=True)
    print("STATE_FOLLOWING_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":main()

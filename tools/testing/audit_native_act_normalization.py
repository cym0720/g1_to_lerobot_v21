#!/usr/bin/env python3
"""Read-only CPU audit of raw labels, checkpoint statistics and normalization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import train_g1_native_lerobot as baseline


def main():
    torch.set_num_threads(2)
    root = baseline.REPO / "output/lerobot_act_control_20260929"
    run = root / "native_train"
    out = root / "normalization_audit"
    out.mkdir(parents=True, exist_ok=True)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    checkpoint = run / "run/checkpoints/001000/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    cfg.use_amp = False
    pipeline = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple",
        root=str(run / "dataset"), video_backend="pyav", use_imagenet_stats=True), policy=cfg)
    dataset = make_dataset(pipeline)
    dataset.image_transforms = baseline.Letterbox()
    right_stats=json.loads((root / "right_arm_train/dataset/meta/stats.json").read_text())
    for key in ("observation.state","action"):
        for statistic in ("mean","std","min","max"):
            np.testing.assert_allclose(right_stats[key][statistic],
                np.asarray(dataset.meta.stats[key][statistic])[8:16],rtol=1e-6,atol=1e-7)
    policy = ACTPolicy.from_pretrained(checkpoint,config=cfg,local_files_only=True).cpu().eval()
    all_states, all_actions, offsets = [], [], {}
    total = 0
    for path in sorted((run / "dataset/data").rglob("*.parquet")):
        table = pq.read_table(path)
        ep = int(table["episode_index"][0].as_py())
        offsets[ep] = total
        source = pq.read_table(baseline.SOURCE / path.relative_to(run / "dataset"))
        right = pq.read_table(root / "right_arm_train/dataset" / path.relative_to(run / "dataset"))
        for key, original in (("observation.state","observations.state.qpos"),("action","action.qpos")):
            raw = np.asarray(table[key].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(raw,np.asarray(source[original].to_pylist(),dtype=np.float32))
            np.testing.assert_array_equal(raw[:,8:16],np.asarray(right[key].to_pylist(),dtype=np.float32))
            (all_states if key == "observation.state" else all_actions).append(raw)
        total += len(table)
    raw = {"observation.state":torch.from_numpy(np.concatenate(all_states)),
           "action":torch.from_numpy(np.concatenate(all_actions))}
    checked = {}
    for key, module in (("observation.state",policy.normalize_inputs),("action",policy.normalize_targets)):
        buffer = getattr(module,"buffer_"+key.replace(".","_"))
        mean,std = buffer["mean"].detach(),buffer["std"].detach()
        expected_mean=torch.as_tensor(dataset.meta.stats[key]["mean"],dtype=torch.float32)
        expected_std=torch.as_tensor(dataset.meta.stats[key]["std"],dtype=torch.float32)
        torch.testing.assert_close(mean,expected_mean,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(std,expected_std,rtol=1e-6,atol=1e-7)
        normalized=module({key:raw[key]})[key]
        manual=(raw[key]-mean)/(std+1e-8)
        torch.testing.assert_close(normalized,manual,rtol=0,atol=0)
        checked[key]=dict(checkpoint_statistics_match_dataset=True,
            actual_normalization_matches_single_mean_std=True,
            mean=mean.tolist(),std=std.tolist(),
            raw_min=raw[key].amin(0).tolist(),raw_max=raw[key].amax(0).tolist(),
            normalized_min=normalized.amin(0).tolist(),normalized_max=normalized.amax(0).tolist())
    norm=policy.normalize_targets({"action":raw["action"]})["action"]
    inverse=policy.unnormalize_outputs({"action":norm})["action"]
    inverse_error=float((inverse-raw["action"]).abs().max())
    torch.testing.assert_close(inverse,raw["action"],rtol=1e-5,atol=1e-6)
    samples=[]
    with torch.inference_mode():
        for ep in (0,45,90):
            for frame in (0,48,96):
                item=dataset[offsets[ep]+frame]
                torch.testing.assert_close(item["observation.state"],raw["observation.state"][offsets[ep]+frame],rtol=0,atol=0)
                torch.testing.assert_close(item["action"],raw["action"][offsets[ep]+frame:offsets[ep]+frame+16],rtol=0,atol=0)
                batch=baseline.sample_batch(dataset,[offsets[ep]+frame],"cpu")
                state_before=batch["observation.state"].clone()
                target_before=batch["action"].clone()
                expected_inputs=policy.normalize_inputs(batch)
                normalized_model_batch=dict(expected_inputs)
                normalized_model_batch["observation.images"]=[expected_inputs[k] for k in cfg.image_features]
                raw_network_output=policy.model(normalized_model_batch)[0]
                predicted=policy.predict_action_chunk(batch)
                action_stats=policy.unnormalize_outputs.buffer_action
                manual_output=raw_network_output*action_stats["std"]+action_stats["mean"]
                torch.testing.assert_close(predicted,manual_output,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(batch["observation.state"],state_before,rtol=0,atol=0)
                torch.testing.assert_close(batch["action"],target_before,rtol=0,atol=0)
                saved=np.load(run/f"gt/step_001000/episode_{ep:06d}.npz")
                expected_saved=saved["predictions"][list(saved["starts"]).index(frame)]
                difference=float(np.abs(predicted[0].numpy()-expected_saved).max())
                # Saved diagnostics use CUDA while this audit uses CPU. Keep the
                # exact affine check above separate from cross-backend rounding.
                np.testing.assert_allclose(predicted[0].numpy(),expected_saved,rtol=0,atol=1e-3)
                image_stats={}
                for key in cfg.image_features:
                    stats=getattr(policy.normalize_inputs,"buffer_"+key.replace(".","_"))
                    torch.testing.assert_close(stats["mean"].flatten(),torch.tensor([.485,.456,.406]))
                    torch.testing.assert_close(stats["std"].flatten(),torch.tensor([.229,.224,.225]))
                    image_stats[key]=dict(raw_range=[float(batch[key].min()),float(batch[key].max())],
                        shape=list(batch[key].shape),normalized_once=True)
                    # Bilinear antialias interpolation can overshoot by a few
                    # floating-point ulps (observed max=1.0000002384), not by a
                    # normalization-scale amount. Preserve the images unchanged.
                    assert -1e-6 <= float(batch[key].min()) <= float(batch[key].max()) <= 1+1e-6
                samples.append(dict(episode=ep,frame=frame,loader_state_action_exact_raw=True,
                    prediction_equals_single_inverse=True,saved_gt_max_abs_diff=difference,images=image_stats))
                baseline.dump(out/"report.json",dict(phase="sampling",frames=total,episodes=len(all_states),
                    source_vectors_exact=True,right_arm_projection_exact=True,checks=checked,
                    action_roundtrip_max_abs_error=inverse_error,samples=samples))
    report=json.loads((out/"report.json").read_text())
    report["phase"]="complete"
    report["right_arm_metadata_statistics_exact_projection"]=True
    report["saved_gt_comparison_note"]="CPU vs CUDA predictions checked within absolute 0.001; exact normalization/inverse checks are independently tested on CPU."
    report["conclusion"]="No repeated state/action standardization or missing output inverse in native ACT and saved GT. Right-arm raw vectors/statistics are sliced, not re-scaled."
    baseline.dump(out/"report.json",report)
    print("NORMALIZATION_AUDIT_COMPLETE",total,len(all_states),"roundtrip_max_error",inverse_error,
          "saved_GT_max_error",max(s["saved_gt_max_abs_diff"] for s in samples),flush=True)


if __name__=="__main__":main()

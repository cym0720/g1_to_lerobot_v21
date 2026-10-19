#!/usr/bin/env python3
"""CPU-only fixed-checkpoint temporal cross-attention causal probe.

Scale only centered cross-attention values, retaining original Q/K/weights.
This is a diagnostic intervention, NOT a training or deployment modification.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import train_g1_native_lerobot as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=baseline.OUTPUT)
    args = parser.parse_args()
    root = args.input
    paths = sorted(p for p in (root / "gt").glob("step_*") if (p / "report.json").exists())
    step = int(paths[-1].name.split("_")[-1])
    checkpoint = root / f"run/checkpoints/{step:06d}/pretrained_model"
    out = root / f"state_following/step_{step:06d}/time_query_probe.json"
    torch.set_num_threads(2)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    cfg.use_amp = False
    pipeline = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple",
        root=str(root / "dataset"), video_backend="pyav", use_imagenet_stats=True), policy=cfg)
    dataset = make_dataset(pipeline)
    dataset.image_transforms = baseline.Letterbox()
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True).cpu().eval()
    layers = policy.model.decoder.layers
    assert len(layers) == 1, "Probe requires a single decoder layer so that Q/K stay fixed across interventions"
    arm = slice(0, 7) if dataset.meta.features["action"]["shape"][0] == 8 else slice(8, 15)
    episodes = [json.loads(line) for line in (root / "dataset/meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    offsets, offset = {}, 0
    for ep in episodes:
        offsets[ep["episode_index"]] = offset
        offset += ep["length"]
    alpha = 1.0
    captured = {}

    def probe(module, inputs, kwargs, output):
        nonlocal captured
        value = kwargs["value"]
        d = module.embed_dim
        mean_value = value.mean(dim=0, keepdim=True)
        projected = F.linear(mean_value, module.in_proj_weight[2*d:], module.in_proj_bias[2*d:])
        common = F.linear(projected, module.out_proj.weight, module.out_proj.bias)
        captured = dict(memory=value.detach().clone(), attention=output[1].detach().clone(),
                        cross_output=output[0].detach().clone(), common=common.detach().clone())
        if alpha == 1.0:
            direct, per_head = F.multi_head_attention_forward(
                kwargs["query"], kwargs["key"], value, d, module.num_heads,
                module.in_proj_weight, module.in_proj_bias, module.bias_k, module.bias_v,
                module.add_zero_attn, module.dropout, module.out_proj.weight, module.out_proj.bias,
                training=False, key_padding_mask=kwargs.get("key_padding_mask"),
                need_weights=True, attn_mask=kwargs.get("attn_mask"), average_attn_weights=False)
            torch.testing.assert_close(direct, output[0], rtol=1e-5, atol=1e-6)
            captured["per_head_attention"] = per_head.detach().clone()
            return output
        return (common + alpha * (output[0] - common), output[1])

    handle = layers[0].multihead_attn.register_forward_hook(probe, with_kwargs=True)
    rows = []
    with torch.inference_mode():
        for ep in baseline.EVAL_EPISODES:
            for frame in (52, 104, 156):
                batch = baseline.sample_batch(dataset, [offsets[ep] + frame], "cpu")
                predictions, saved = {}, {}
                for value_scale in (1.0, 0.0, 10.0):
                    alpha = value_scale
                    predictions[str(value_scale)] = policy.predict_action_chunk(batch)[0].numpy()
                    if alpha == 1.0:
                        saved = captured
                attention = saved["attention"][0]  # Mean across heads; each time query has a distribution over encoder tokens.
                entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1) / np.log(attention.shape[-1])
                per_head = saved["per_head_attention"][0]
                per_head_entropy = -(per_head*per_head.clamp_min(1e-12).log()).sum(-1)/np.log(per_head.shape[-1])
                centered_memory = saved["memory"] - saved["memory"].mean(dim=0, keepdim=True)
                cross = saved["cross_output"]
                row = dict(episode=ep, frame=frame,
                    encoder_token_centered_rms=float(centered_memory.square().mean().sqrt()),
                    cross_output_time_centered_rms=float((cross-cross.mean(dim=0,keepdim=True)).square().mean().sqrt()),
                    averaged_attention_normalized_entropy=float(entropy.mean()),
                    per_head_normalized_entropy=per_head_entropy.mean(-1).tolist(),
                    per_head_attention_time_max_pair_l1=[float(torch.cdist(w,w,p=1).max()) for w in per_head],
                    attention_time_max_pair_l1=float(torch.cdist(attention, attention,p=1).max()),
                    right_arm_mean_chunk_span_by_value_scale={k:float(np.ptp(p[:,arm],axis=0).mean()) for k,p in predictions.items()},
                    right_arm_mean_prediction_shift_alpha10=float(np.abs(predictions["10.0"][:,arm]-predictions["1.0"][:,arm]).mean()))
                rows.append(row)
                baseline.dump(out, dict(step=step,samples=rows,
                    intervention="Only centered values changed; original query, key, attention weights and model weights retained",
                    caution="Alpha10 changes representation scale and can be out of distribution. This localizes temporal-signal attenuation, not a safe fix or proof of its historical cause."))
                print("TIME_QUERY_PROBE",json.dumps(row),flush=True)
    handle.remove()
    print("TIME_QUERY_PROBE_COMPLETE",flush=True)


if __name__ == "__main__":
    main()

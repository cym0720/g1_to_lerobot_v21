#!/usr/bin/env python3
"""Read-only layer tracing and fixed-weight interventions for flat ACT chunks.

No dataset, original checkpoint or training source is changed. Intervention
outputs are causal probes, NOT validated deployment predictions or fixes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import audit_g1_act_gt as audit

FRAMES = (52, 104, 156)
CHECKPOINT_STEPS = (1000, 3000, 6000, 10000, 15000)
OUTPUT = audit.DEFAULT_OUTPUT.parent / "temporal_cause"


def token_stats(tensor):
    """Sequence-first token variation, invariant to an overall scalar gain."""
    x = tensor.detach().float()
    mean = x.mean(0, keepdim=True)
    centered = x - mean
    cos = F.cosine_similarity(x[:, 0], mean[0, 0][None, :], dim=-1)
    out = dict(shape=list(x.shape), token_norm_rms=float(x.square().mean().sqrt()),
               centered_rms=float(centered.square().mean().sqrt()),
               relative_spread=float(centered.norm() / x.norm().clamp_min(1e-12)),
               cosine_to_mean_median=float(cos.median()))
    if x.shape[0] == 149:
        image = x[2:]
        residual = image - image.mean(0, keepdim=True)
        out["image_tokens_relative_spread"] = float(residual.norm() / image.norm().clamp_min(1e-12))
        means = [image[i * 49:(i + 1) * 49].mean(0) for i in range(3)]
        out["camera_mean_relative_spread"] = float(torch.stack(means).std(0, correction=0).norm() / torch.stack(means).square().mean(0).sqrt().norm().clamp_min(1e-12))
    return out


class Trace:
    def __init__(self, core):
        self.core = core
        self.handles = []
        self.stages = {}
        self.attention = {}
        self.capture = {}
        self.value_scale = 1.0
        self.query_scale = 1.0
        self.bypass_encoder = False

        def backbone_post(module, args, result):
            feature = result["feature_map"].detach().float()
            tokens = feature.flatten(2).permute(2, 0, 1)
            row = token_stats(tokens)
            row["exact_zero_fraction"] = float((feature == 0).float().mean())
            row["active_channels"] = int((feature.abs().amax((0, 2, 3)) > 1e-8).sum())
            row["camera_call"] = len([k for k in self.stages if k.startswith("backbone.")])
            self.stages[f"backbone.camera{row['camera_call']}"] = row
        self.handles.append(core.backbone.register_forward_hook(backbone_post))

        def image_projection_post(module, args, result):
            value = result.detach().float()
            bias = module.bias.detach().float()[None, :, None, None]
            row = token_stats(value.flatten(2).permute(2, 0, 1))
            row["signal_without_bias_rms"] = float((value - bias).square().mean().sqrt())
            row["bias_rms"] = float(bias.square().mean().sqrt())
            self.stages[f"image_projection.camera{len([k for k in self.stages if k.startswith('image_projection.')])}"] = row
        self.handles.append(core.encoder_img_feat_input_proj.register_forward_hook(image_projection_post))

        def record(name):
            def callback(module, args, result):
                self.stages[name] = token_stats(result)
            return callback

        def enc_pre(module, args, kwargs):
            self.stages["encoder.input"] = token_stats(args[0])
            self.capture["encoder_input"] = args[0].detach()
        self.handles.append(core.encoder.register_forward_pre_hook(enc_pre, with_kwargs=True))
        for i, layer in enumerate(core.encoder.layers):
            for attr in ("norm1", "norm2"):
                self.handles.append(getattr(layer, attr).register_forward_hook(record(f"encoder.layer{i}.{attr}")))
            self.handles.append(layer.register_forward_hook(record(f"encoder.layer{i}.output")))

        def enc_post(module, args, result):
            if self.bypass_encoder:
                result = args[0]
            self.stages["encoder.output"] = token_stats(result)
            self.capture["encoder_output"] = result.detach()
            return result
        self.handles.append(core.encoder.register_forward_hook(enc_post))

        def decoder_pre(module, args, kwargs):
            if self.query_scale != 1:
                kwargs = dict(kwargs)
                kwargs["decoder_pos_embed"] = kwargs["decoder_pos_embed"] * self.query_scale
                return args, kwargs
        self.handles.append(core.decoder.register_forward_pre_hook(decoder_pre, with_kwargs=True))

        for i, layer in enumerate(core.decoder.layers):
            def value_pre(module, args, kwargs):
                kwargs = dict(kwargs)
                # Head-averaged weights can hide head-specific selectivity.
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                if self.value_scale != 1:
                    value = kwargs["value"]
                    mean = value.mean(0, keepdim=True)
                    kwargs["value"] = mean + self.value_scale * (value - mean)
                return args, kwargs
            self.handles.append(layer.multihead_attn.register_forward_pre_hook(value_pre, with_kwargs=True))

            def attention_post(module, args, result, index=i):
                result_tensor, weights = result
                self.stages[f"decoder.layer{index}.cross_attn"] = token_stats(result_tensor)
                w = weights.detach().float()  # B, head, time-query, source-token.
                entropy = -(w.clamp_min(1e-12).log() * w).sum(-1) / math.log(w.shape[-1])
                self.attention[str(index)] = dict(normalized_entropy=float(entropy.mean()),
                    between_time_weight_rms=float((w - w.mean(-2, keepdim=True)).square().mean().sqrt()),
                    mean_mass_latent=float(w[..., 0].mean()), mean_mass_state=float(w[..., 1].mean()),
                    max_weight=float(w.max()), per_head_normalized_entropy=entropy.mean((0, 2)).tolist())
                self.capture[f"attention_{index}"] = w.cpu().numpy()
            self.handles.append(layer.multihead_attn.register_forward_hook(attention_post))
            self.handles.append(layer.register_forward_hook(record(f"decoder.layer{i}.output")))
        self.handles.append(core.decoder.register_forward_hook(record("decoder.output")))

    def clear(self):
        self.stages, self.attention, self.capture = {}, {}, {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_batch(wrapper, episode, frame, blank_images=False, state_frame=None):
    proc = wrapper.get_norm_processor()
    state = episode["state"][frame if state_frame is None else state_frame]
    images = [cam[frame] for cam in episode["images"]]
    if blank_images:
        images = [np.zeros_like(image) for image in images]
    example = dict(image=images, state=proc.apply_state(state))
    model = wrapper._framework
    batch = model._examples_to_act_batch([example], require_action=False)
    batch = model.action_model.normalize_inputs(batch)
    batch["observation.images"] = [batch[key] for key in model.action_model.config.image_features]
    return batch


def predict(core, proc, batch, trace, target):
    trace.clear()
    with torch.inference_mode():
        actions, _ = core(batch)
    raw = proc.unapply_actions(actions[0].float().cpu().numpy())
    metrics = dict(right_arm_span=np.ptp(raw[:, 8:15], axis=0).tolist(),
                   right_arm_mae=float(np.abs(raw[:, 8:15] - target[:, 8:15]).mean()),
                   right_arm_step0=raw[0, 8:15].tolist(),
                   stages=trace.stages, cross_attention=trace.attention)
    return raw, metrics


def fixed_encoder_outputs(core, trace, proc, batch, target):
    """Hold cross-attention keys unchanged, change ONLY values' token spread."""
    results = {}
    raw_baseline = None
    for name, value_scale, query_scale, bypass in [
        ("baseline", 1, 1, False),
        ("constant_values", 0, 1, False),
        ("values_spread_x10", 10, 1, False),
        ("values_spread_x100", 100, 1, False),
        ("query_zero", 1, 0, False),
        ("query_x10", 1, 10, False),
        ("encoder_bypass", 1, 1, True),
    ]:
        trace.value_scale, trace.query_scale, trace.bypass_encoder = value_scale, query_scale, bypass
        raw, metrics = predict(core, proc, batch, trace, target)
        if name == "baseline":
            raw_baseline = raw.copy()
        metrics["difference_from_baseline_mae_rad"] = float(np.abs(raw[:, 8:15] - raw_baseline[:, 8:15]).mean())
        results[name] = metrics
    trace.value_scale, trace.query_scale, trace.bypass_encoder = 1, 1, False
    return results


def plots(report, output):
    plt = audit.plt
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for label in ["random", "1000", "3000", "6000", "10000", "15000"]:
        entries = report["checkpoints"][label]["frames"]
        stages = [name for name in entries[0]["stages"] if name.startswith("encoder.")]
        y = [np.mean([r["stages"][name]["relative_spread"] for r in entries]) for name in stages]
        ax.plot(stages, y, marker=".", label=label)
    ax.set_yscale("log")
    ax.set_ylabel("token residual norm / total norm (3 anchors averaged)")
    ax.set_title("Where temporal-query source values lose token diversity")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.25)
    ax.legend(title="checkpoint step")
    fig.savefig(output / "encoder_stage_spread.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    x = []
    pred, gt = [], []
    for label in ["random", "1000", "3000", "6000", "10000", "15000"]:
        rows = report["checkpoints"][label]["frames"]
        x.append(label)
        pred.append(np.mean([r["right_arm_span"] for r in rows]))
        gt.append(np.mean([r["gt_right_arm_span"] for r in rows]))
    ax.plot(x, gt, color="black", label="GT 16-step span")
    ax.plot(x, pred, marker="o", label="model mean 7-joint span")
    ax.set_yscale("log")
    ax.set_ylabel("rad")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "chunk_span_training_evolution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = json.loads((audit.DEFAULT_DATASET / "meta/info.json").read_text())
    episode = audit.load_episode(audit.DEFAULT_DATASET, info, 0)
    checkpoint = audit.RUN / "checkpoints/steps_6000_pytorch_model.pt"
    wrapper = audit.make_wrapper(checkpoint)
    framework = wrapper._framework.float().eval()
    core, proc = framework.action_model.model, wrapper.get_norm_processor()
    trace = Trace(core)
    report = dict(scope="read-only checkpoint tracing and in-memory interventions", frames=FRAMES,
                  dataset=str(audit.DEFAULT_DATASET), checkpoints={}, interventions={}, input_ablations={},
                  decoder_layers=len(core.decoder.layers), encoder_layers=len(core.encoder.layers),
                  caveat="Intervention outputs are OOD causal probes, not recommended model repairs. No optimizer in this script.")
    original = {k: v.detach().cpu().clone() for k, v in framework.state_dict().items()}
    # Independent random architecture, exact training config, not a mutated checkpoint.
    from lerobot.policies.act.modeling_act import ACT
    torch.manual_seed(42)
    random_core = ACT(core.config).cuda().eval()
    random_trace = Trace(random_core)
    random_rows = []
    for frame in FRAMES:
        target = episode["action"][frame:frame + 16]
        _, row = predict(random_core, proc, build_batch(wrapper, episode, frame), random_trace, target)
        row.update(frame=frame, gt_right_arm_span=np.ptp(target[:, 8:15], axis=0).tolist())
        random_rows.append(row)
    report["checkpoints"]["random"] = dict(frames=random_rows)
    random_trace.close()
    del random_core, random_trace
    torch.cuda.empty_cache()
    for step in CHECKPOINT_STEPS:
        path = audit.RUN / f"checkpoints/steps_{step}_pytorch_model.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        framework.load_state_dict(state, strict=True)
        del state
        rows = []
        for frame in FRAMES:
            target = episode["action"][frame:frame + 16]
            _, row = predict(core, proc, build_batch(wrapper, episode, frame), trace, target)
            row.update(frame=frame, gt_right_arm_span=np.ptp(target[:, 8:15], axis=0).tolist())
            rows.append(row)
        report["checkpoints"][str(step)] = dict(path=str(path), frames=rows)
        print("CHECKPOINT", step, "span", np.mean([r["right_arm_span"] for r in rows]),
              "memory_spread", np.mean([r["stages"]["encoder.output"]["relative_spread"] for r in rows]), flush=True)
        if step in (6000, 15000):
            report["interventions"][str(step)] = {}
            report["input_ablations"][str(step)] = {}
            for frame in FRAMES:
                target = episode["action"][frame:frame + 16]
                batch = build_batch(wrapper, episode, frame)
                report["interventions"][str(step)][str(frame)] = fixed_encoder_outputs(core, trace, proc, batch, target)
                baseline, _ = predict(core, proc, batch, trace, target)
                input_rows = {}
                for name, blank_images, state_frame in [("blank_images", True, None), ("state_from_30_frames_later", False, frame + 30)]:
                    raw, row = predict(core, proc, build_batch(wrapper, episode, frame, blank_images, state_frame), trace, target)
                    row["baseline_prediction_difference_rad"] = float(np.abs(raw[:, 8:15] - baseline[:, 8:15]).mean())
                    input_rows[name] = row
                report["input_ablations"][str(step)][str(frame)] = input_rows
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    # Restore original memory weights to ensure even the diagnostic object is clean.
    framework.load_state_dict(original)
    trace.close()
    plots(report, args.output)
    print("TEMPORAL_CAUSE_AUDIT_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

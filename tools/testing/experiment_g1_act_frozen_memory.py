#!/usr/bin/env python3
"""Causal time-path test: identical frozen encoder memories and GT/schedule.

Train only decoder/query/head, with dropout OFF and prior z=0 in both branches.
The sole branch difference is whether time query also enters decoder content.
This isolates the time-routing limitation from images/data/KL/padding changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import audit_g1_act_gt as audit
from audit_g1_act_temporal_cause import build_batch

OUTPUT = audit.DEFAULT_OUTPUT.parent / "frozen_memory_experiment"
BRANCHES = ("original_attention_time_path", "direct_query_content")


def decode(core, memories, positions, ids, direct):
    memory = memories[ids].permute(1, 0, 2)
    q = core.decoder_pos_embed.weight[:, None]
    x = torch.zeros((16, len(ids), 512), device="cuda")
    if direct:
        x = x + q
    hidden = core.decoder(x, memory, encoder_pos_embed=positions, decoder_pos_embed=q)
    return core.action_head(hidden.transpose(0, 1))


def evaluation(core, memories, positions, direct, processor, episode):
    preds = []
    with torch.inference_mode():
        for start in range(0, len(memories), 16):
            ids = torch.arange(start, min(start + 16, len(memories)), device="cuda")
            norm = decode(core, memories, positions, ids, direct).cpu().numpy()
            preds.extend(processor.unapply_actions(row) for row in norm)
    pred = np.stack(preds)
    metrics = audit.compare(episode, pred)
    h = 16
    starts = np.arange(len(pred) - h + 1)
    g = episode["action"][starts[:, None] + np.arange(h)[None, :], 8:15]
    dp, dg = np.diff(pred[starts, :, 8:15], axis=1), np.diff(g, axis=1)
    metrics["within_chunk_velocity_correlation"] = [float(np.corrcoef(dp[..., j].ravel(), dg[..., j].ravel())[0, 1]) for j in range(7)]
    metrics["within_chunk_velocity_mae_rad"] = np.mean(np.abs(dp - dg), axis=(0, 1)).tolist()
    return pred, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    info = json.loads((audit.DEFAULT_DATASET / "meta/info.json").read_text())
    episode = audit.load_episode(audit.DEFAULT_DATASET, info, 0)
    checkpoint = audit.RUN / "checkpoints/steps_6000_pytorch_model.pt"
    wrapper = audit.make_wrapper(checkpoint)
    framework = wrapper._framework.float().eval()
    core, proc = framework.action_model.model, wrapper.get_norm_processor()
    captured = {}
    def cache_decoder(module, pargs, kwargs):
        captured["memory"] = pargs[1].detach()
        captured["positions"] = kwargs["encoder_pos_embed"].detach()
    hook = core.decoder.register_forward_pre_hook(cache_decoder, with_kwargs=True)
    memories = []
    with torch.no_grad():
        for frame in range(len(episode["state"])):
            core(build_batch(wrapper, episode, frame))
            memories.append(captured["memory"][:, 0].clone())
    positions = captured["positions"].clone()
    memories = torch.stack(memories)
    hook.remove()
    original = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    for p in core.parameters():
        p.requires_grad_(False)
    parameters = list(core.decoder.parameters()) + list(core.decoder_pos_embed.parameters()) + list(core.action_head.parameters())
    for p in parameters:
        p.requires_grad_(True)
    target_norm = audit.normalize_action(proc, episode["action"])
    target_norm = torch.from_numpy(target_norm).cuda()
    schedule = np.random.default_rng(20260929).integers(0, len(memories) - 16 + 1, size=(args.steps, 8))
    weights = torch.full((20,), 0.1, device="cuda")
    weights[8:16] = 8.0
    report = dict(checkpoint=str(checkpoint), episode=0, steps=args.steps, batch_size=8,
                  lr=1e-4, precision="FP32", encoder="frozen real memories, identical across branches",
                  dropout="OFF in both branches", latent="zero prior, no VAE or KL",
                  targets="unaltered action.qpos; complete 16-step windows only",
                  caveat="one-episode causal diagnostic, not a deployment/generalization validation", branches={})
    all_predictions = {}
    for branch in BRANCHES:
        core.load_state_dict(original)
        core.eval()
        direct = branch == "direct_query_content"
        initial, initial_metrics = evaluation(core, memories, positions, direct, proc, episode)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-8)
        history = []
        for step, selected in enumerate(schedule, 1):
            ids = torch.tensor(selected, device="cuda")
            gt = target_norm[ids[:, None] + torch.arange(16, device="cuda")[None, :]]
            optimizer.zero_grad(set_to_none=True)
            pred = decode(core, memories, positions, ids, direct)
            loss = (F.l1_loss(pred, gt, reduction="none") * weights).sum() / (8 * 16 * weights.sum())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            history.append(dict(step=step, l1=float(loss)))
            if step == 1 or step % 100 == 0:
                print(branch, "STEP", step, "L1", float(loss), flush=True)
        final, metrics = evaluation(core, memories, positions, direct, proc, episode)
        all_predictions[branch] = final
        np.savez_compressed(args.output / f"{branch}_predictions.npz", initial=initial, final=final, gt_action=episode["action"])
        report["branches"][branch] = dict(initial_metrics=initial_metrics, final_metrics=metrics, history=history)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print("FROZEN_BRANCH_COMPLETE", branch, json.dumps(metrics), flush=True)
        del optimizer
    plt = audit.plt
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    gt, n = episode["action"], len(memories)
    for j, ax in enumerate(axes.flat):
        dim = j + 8
        ax.plot(gt[:, dim], color="black", label="GT", linewidth=1.5)
        for branch, pred in all_predictions.items():
            executed = np.concatenate([pred[s, :min(16, n - s), dim] for s in range(0, n, 16)])
            ax.plot(executed, label=branch, linewidth=0.9)
        ax.set_title(f"right_arm_joint{j + 1}" if j < 7 else "right_gripper")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Same frozen observations/labels: isolate time-query content path (dropout/KL excluded)")
    fig.savefig(args.output / "frozen_memory_gt_comparison.png", dpi=130)
    plt.close(fig)
    core.load_state_dict(original)
    print("FROZEN_MEMORY_EXPERIMENT_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated one-episode ACT time-query ablation, never modifying original train.

Both branches start from the same checkpoint and use exactly the same sample
schedule, FP32, LR, weights and stable KL calculation. The second branch adds
the existing time query embeddings to decoder input content, not only Q/K.
This is an overfit diagnostic, not a validated replacement/deployment model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import audit_g1_act_gt as audit

STEPS = 160
TRAIN_BATCH_SIZE = 8
EPISODE = 0
BRANCHES = ("fp32_original_queries", "fp32_queries_as_content")
DEFAULT_OUTPUT = audit.DEFAULT_OUTPUT.parent / "time_query_experiment"


def train_loss(model, examples):
    batch = model._examples_to_act_batch(examples)
    batch = model.action_model.normalize_inputs(batch)
    batch["observation.images"] = [batch[k] for k in model.action_model.config.image_features]
    batch = model.action_model.normalize_targets(batch)
    predicted, (mu, logvar) = model.action_model.model(batch)
    w = torch.full((20,), 0.1, device="cuda")
    w[8:16] = 8.0
    mask = ~batch["action_is_pad"].unsqueeze(-1)
    l1 = (F.l1_loss(batch["action"].float(), predicted.float(), reduction="none") * w * mask).sum() / (w * mask).sum()
    # Never compute this subtractive, near-zero expression in BF16.
    kl = (-0.5 * (1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())).sum(-1).mean()
    return l1 + model.action_model.config.kl_weight * kl, l1, kl


def query_content_hook(module, args, kwargs):
    queries = kwargs["decoder_pos_embed"]
    return (args[0] + queries, *args[1:]), kwargs


def plot(output, episode, all_predictions, histories):
    plt = audit.plt
    gt = episode["action"]
    n = len(gt)
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        dim = 8 + j
        ax.plot(gt[:, dim], color="black", label="GT", linewidth=1.5)
        for branch, pred in all_predictions.items():
            h = pred.shape[1]
            executed = np.concatenate([pred[s, :min(h, n - s), dim] for s in range(0, n, h)])
            ax.plot(executed, label=branch, linewidth=1.0)
        ax.set_title(f"right_arm_joint{j + 1}" if j < 7 else "right_gripper")
        ax.set_xlabel("frame; non-overlapping 16-step chunks")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Ep 0 overfit diagnostic: identical labels/schedule; FP32 baseline vs explicit time-query content")
    fig.savefig(output / "right_arm_gt_ablation.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for branch, rows in histories.items():
        axes[0].plot([r["step"] for r in rows], [r["l1"] for r in rows], label=branch, linewidth=1)
        axes[1].plot([r["step"] for r in rows], [r["kl"] for r in rows], label=branch, linewidth=1)
    axes[0].set_title("Weighted normalized action L1 (separate from KL)")
    axes[1].set_title("KL calculated in FP32")
    for ax in axes:
        ax.set_xlabel("local optimizer step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(output / "loss_components.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = audit.RUN / "checkpoints/steps_6000_pytorch_model.pt"
    info = json.loads((audit.DEFAULT_DATASET / "meta/info.json").read_text())
    episode = audit.load_episode(audit.DEFAULT_DATASET, info, EPISODE)
    horizon = 16
    schedule = np.random.default_rng(20260929).integers(0, len(episode["state"]) - horizon + 1,
                                                       size=(args.steps, TRAIN_BATCH_SIZE))
    report = dict(checkpoint=str(checkpoint), dataset=str(audit.DEFAULT_DATASET), episode=EPISODE,
                  steps=args.steps, lr=1e-4, precision="FP32", batch_size=TRAIN_BATCH_SIZE,
                  label_source="unchanged original action.qpos", branches={},
                  warning="ONE TRAINING EPISODE ONLY. No held-out generalization claim. Query-content branch requires this script's hook; not a drop-in deployment checkpoint.")
    all_predictions, histories = {}, {}
    for branch in BRANCHES:
        torch.manual_seed(42)
        wrapper = audit.make_wrapper(checkpoint)
        model = wrapper._framework.float()
        processor = wrapper.get_norm_processor()
        handle = None
        if branch == "fp32_queries_as_content":
            handle = model.action_model.model.decoder.register_forward_pre_hook(query_content_hook, with_kwargs=True)
        initial = audit.infer(wrapper, episode, "fp32")
        norm_state = processor.apply_state(episode["state"]).astype(np.float16)
        norm_action = audit.normalize_action(processor, episode["action"]).astype(np.float16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-8)
        rows = []
        start_time = time.monotonic()
        for step, frames in enumerate(schedule, 1):
            model.train()
            examples = [dict(image=[cam[s] for cam in episode["images"]], state=norm_state[s],
                             action=norm_action[s:s + horizon]) for s in frames]
            optimizer.zero_grad(set_to_none=True)
            loss, l1, kl = train_loss(model, examples)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {branch}, step {step}")
            loss.backward()
            query_grad = model.action_model.model.decoder_pos_embed.weight.grad
            query_grad_norm = float(query_grad.norm()) if query_grad is not None else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            row = dict(step=step, loss=float(loss), l1=float(l1), kl=float(kl), query_grad_norm=query_grad_norm)
            rows.append(row)
            if step == 1 or step % 20 == 0:
                print(branch, json.dumps(row), flush=True)
        final = audit.infer(wrapper, episode, "fp32")
        all_predictions[branch] = final
        histories[branch] = rows
        torch.save(model.state_dict(), args.output / f"{branch}_experimental_weights.pt")
        np.savez_compressed(args.output / f"{branch}_predictions.npz", initial=initial, final=final, gt_action=episode["action"])
        report["branches"][branch] = dict(initial_metrics=audit.compare(episode, initial), final_metrics=audit.compare(episode, final),
                                          loss_decreased=bool(np.mean([r["l1"] for r in rows[-20:]]) < np.mean([r["l1"] for r in rows[:20]])),
                                          history=rows, seconds=time.monotonic() - start_time)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print("BRANCH_COMPLETE", branch, json.dumps(report["branches"][branch]["final_metrics"]), flush=True)
        if handle is not None:
            handle.remove()
        del optimizer, model, wrapper, processor, loss, l1, kl
        torch.cuda.empty_cache()
    plot(args.output, episode, all_predictions, histories)
    print("EXPERIMENT_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated dropout/precision controls on identical frozen ACT memories.

Original architecture only. This is a single-episode diagnostic, not retraining
the deployment model. Encoder dropout, VAE/KL and padded windows are excluded.
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
from experiment_g1_act_frozen_memory import decode, evaluation

CASES = (
    ("fp32_dropout_on", True, False),
    ("bf16_dropout_off", False, True),
    ("bf16_dropout_on", True, True),
)
OUTPUT = audit.DEFAULT_OUTPUT.parent / "time_training_conditions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = json.loads((audit.DEFAULT_DATASET / "meta/info.json").read_text())
    episode = audit.load_episode(audit.DEFAULT_DATASET, info, 0)
    wrapper = audit.make_wrapper(audit.RUN / "checkpoints/steps_6000_pytorch_model.pt")
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
    torch.save(dict(memories=memories.cpu(), positions=positions.cpu()), args.output / "frozen_memories.pt")
    for p in core.parameters():
        p.requires_grad_(False)
    parameters = list(core.decoder.parameters()) + list(core.decoder_pos_embed.parameters()) + list(core.action_head.parameters())
    for p in parameters:
        p.requires_grad_(True)
    targets = torch.from_numpy(audit.normalize_action(proc, episode["action"])).cuda()
    schedule = np.random.default_rng(20260929).integers(0, len(memories) - 16 + 1, size=(args.steps, 8))
    weights = torch.full((20,), 0.1, device="cuda")
    weights[8:16] = 8.0
    report = dict(steps=args.steps, episode=0, checkpoint="steps_6000", cases={},
                  scope="Only decoder dropout and autocast varied; original zero-content architecture",
                  excluded="encoder dropout, VAE/KL, terminal padding; no original files changed",
                  baseline="../frozen_memory_experiment/report.json: original_attention_time_path",
                  caveat="Single-episode causal controls, not held-out generalization or full-training reproduction")
    for name, dropout, bf16 in CASES:
        core.load_state_dict(original)
        core.eval()
        core.decoder.train(dropout)
        torch.manual_seed(20260929)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-8)
        history = []
        for step, selected in enumerate(schedule, 1):
            ids = torch.tensor(selected, device="cuda")
            gt = targets[ids[:, None] + torch.arange(16, device="cuda")[None, :]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                pred = decode(core, memories, positions, ids, False)
                loss = (F.l1_loss(pred, gt, reduction="none") * weights).sum() / (8 * 16 * weights.sum())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            history.append(dict(step=step, l1=float(loss)))
            if step == 1 or step % 100 == 0:
                print(name, "STEP", step, "L1", float(loss), flush=True)
        core.eval()
        pred, metrics = evaluation(core, memories, positions, False, proc, episode)
        np.savez_compressed(args.output / f"{name}_predictions.npz", final=pred, gt_action=episode["action"])
        report["cases"][name] = dict(dropout=dropout, autocast_bf16=bf16, final_metrics=metrics, history=history)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print("TIME_CONDITION_COMPLETE", name, json.dumps(metrics), flush=True)
        del optimizer
    core.load_state_dict(original)
    print("TIME_TRAINING_CONDITIONS_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

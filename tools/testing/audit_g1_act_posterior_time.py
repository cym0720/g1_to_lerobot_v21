#!/usr/bin/env python3
"""Check whether flat time predictions are specific to inference's zero latent.

Read-only checkpoint probe. Dropout is disabled in both inference and posterior
branches to isolate latent routing, and every window contains genuine labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import audit_g1_act_gt as audit
from audit_g1_act_temporal_cause import build_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=audit.DEFAULT_OUTPUT.parent / "posterior_time")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = json.loads((audit.DEFAULT_DATASET / "meta/info.json").read_text())
    ep = audit.load_episode(audit.DEFAULT_DATASET, info, 0)
    wrapper = audit.make_wrapper(audit.RUN / "checkpoints/steps_15000_pytorch_model.pt")
    core = wrapper._framework.float().eval().action_model.model
    proc = wrapper.get_norm_processor()
    captured = {}
    mode = "prior_zero"

    def posterior(module, inputs, output):
        mu, logvar = output.detach().float().chunk(2, -1)
        captured["mu"] = mu
        captured["mu_rms"] = float(mu.square().mean().sqrt())
        captured["logvar_rms"] = float(logvar.square().mean().sqrt())

    def latent(module, inputs):
        if mode == "posterior_mean":
            inputs = (captured["mu"],)
        captured["z_rms"] = float(inputs[0].detach().float().square().mean().sqrt())
        return inputs

    hooks = [core.vae_encoder_latent_output_proj.register_forward_hook(posterior),
             core.encoder_latent_input_proj.register_forward_pre_hook(latent)]
    report = dict(checkpoint="steps_15000", episode=0, precision="FP32", dropout="OFF", frames={},
                  caveat="Training posterior sees future labels; this is a read-only latent-path diagnostic, not deployable inference")
    for frame in (52, 104, 156):
        batch = build_batch(wrapper, ep, frame)
        raw_gt = ep["action"][frame:frame + 16]
        norm_gt = audit.normalize_action(proc, raw_gt)
        batch["action"] = torch.from_numpy(norm_gt).cuda()[None]
        batch["action_is_pad"] = torch.zeros((1, 16), dtype=torch.bool, device="cuda")
        rows = {}
        for mode in ("prior_zero", "posterior_mean", "posterior_sample"):
            core.train(mode != "prior_zero")
            for module in core.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.dropout = 0.0
            mode_rows = []
            for seed in range(5 if mode == "posterior_sample" else 1):
                torch.manual_seed(20260929 + seed)
                captured.clear()
                with torch.no_grad():
                    action, _ = core(batch)
                raw = proc.unapply_actions(action[0].float().cpu().numpy())
                mode_rows.append(dict(seed=seed, mean_right_arm_span=float(np.ptp(raw[:, 8:15], axis=0).mean()),
                                      right_arm_mae=float(np.abs(raw[:, 8:15] - raw_gt[:, 8:15]).mean()),
                                      latent={k: v for k, v in captured.items() if k != "mu"}))
            rows[mode] = mode_rows
        report["frames"][str(frame)] = rows
        print("POSTERIOR_FRAME", frame, json.dumps(rows), flush=True)
    for hook in hooks:
        hook.remove()
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("POSTERIOR_TIME_AUDIT_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

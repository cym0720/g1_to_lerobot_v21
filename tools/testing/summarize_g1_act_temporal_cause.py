#!/usr/bin/env python3
"""Consolidate measured causal probes, with an actual within-chunk plot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "output/act_gt_audit_20260929"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def mean_metrics(m):
    return dict(mean_joint_span_ratio=float(np.mean(m["median_span_ratio_where_gt_span_gt_0_01"])),
                mean_joint_valid_horizon_mae_rad=float(np.mean(m["all_valid_steps_mae_right_arm"])),
                mean_joint_velocity_correlation=float(np.mean(m["within_chunk_velocity_correlation"]))
                if "within_chunk_velocity_correlation" in m else None)


def main():
    final = read("model_gt_15000/report.json")
    frozen = read("frozen_memory_experiment/report.json")
    conditions = read("time_training_conditions/report.json")
    causal = read("temporal_cause/report.json")
    posterior = read("posterior_time/report.json")
    summary = dict(production_15000={}, isolated_controls={}, fixed_weight_interventions={},
                   encoder_image_token_spread={}, posterior_mu_rms={},
                   scope="Confirmed computational mechanism; historical training trigger not uniquely isolated")
    for ep, row in final["episodes"].items():
        summary["production_15000"][ep] = mean_metrics(row["metrics"]["fp32"])
    for branch, row in frozen["branches"].items():
        summary["isolated_controls"][branch] = mean_metrics(row["final_metrics"])
    for branch, row in conditions["cases"].items():
        summary["isolated_controls"][branch] = mean_metrics(row["final_metrics"])
    for frame, row in causal["interventions"]["15000"].items():
        summary["fixed_weight_interventions"][frame] = {
            name: float(np.mean(item["right_arm_span"])) for name, item in row.items()
            if name in ("baseline", "constant_values", "values_spread_x10", "values_spread_x100")
        }
    for row in causal["checkpoints"]["15000"]["frames"]:
        summary["encoder_image_token_spread"][str(row["frame"])] = row["stages"]["encoder.output"]["image_tokens_relative_spread"]
    for frame, row in posterior["frames"].items():
        summary["posterior_mu_rms"][frame] = row["posterior_mean"][0]["latent"]["mu_rms"]
    out = ROOT / "temporal_cause"
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with np.load(ROOT / "model_gt_15000/episode_000000_predictions.npz") as original:
        gt = original["gt_action"].copy()
        pred = original["fp32"].copy()
    with np.load(ROOT / "frozen_memory_experiment/original_attention_time_path_predictions.npz") as control:
        learned = control["final"].copy()
        np.testing.assert_array_equal(gt, control["gt_action"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, frame in enumerate((52, 104, 156)):
        for row, dim in enumerate((8, 9)):
            ax = axes[row, col]
            ax.plot(np.arange(16), gt[frame:frame + 16, dim], "k.-", label="True future actions")
            ax.plot(np.arange(16), pred[frame, :, dim], "r.-", label="Production checkpoint 15000")
            ax.plot(np.arange(16), learned[frame, :, dim], "b.-", label="Frozen encoder, original decoder +1000")
            ax.set_title(f"Frame {frame}: right_arm_joint{dim - 7}")
            ax.set_xlabel("Offset within action chunk (0..15)")
            ax.set_ylabel("Action (rad)")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Flat inside 16-step chunks is a learned-model failure, not constant GT\nBlue: isolated one-episode training control, NOT a validated deployment fix")
    fig.savefig(out / "chunk_time_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("TEMPORAL_CAUSE_SUMMARY_COMPLETE", out / "summary.json")


if __name__ == "__main__":
    main()

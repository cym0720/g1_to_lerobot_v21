#!/usr/bin/env python
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014")
OUTDIR = ROOT / "audits/per_joint_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

def get_ep0(report):
    eps = report.get("episodes")
    if isinstance(eps, dict):
        return list(eps.values())[0]
    return eps[0]

def maybe_denorm(arr, norm):
    q01 = np.asarray(norm["action_q01"], dtype=np.float64)
    q99 = np.asarray(norm["action_q99"], dtype=np.float64)
    return q01 + (arr + 1.0) * (q99 - q01) / 2.0

def resolve_units(gt, pred, report):
    ep0 = get_ep0(report)
    target = ep0["metrics"]["nonoverlap_chunk_execution"]["right_arm_mae_rad"]
    mae_raw = float(np.mean(np.abs(pred[:, 8:15] - gt[:, 8:15])))
    if target and abs(mae_raw - target) / max(target, 1e-9) < 0.05:
        return gt, pred, "original", mae_raw, target
    norm = report.get("normalization") or {}
    if "action_q01" in norm and "action_q99" in norm:
        gt2 = maybe_denorm(gt, norm)
        pred2 = maybe_denorm(pred, norm)
        mae2 = float(np.mean(np.abs(pred2[:, 8:15] - gt2[:, 8:15])))
        if abs(mae2 - target) / max(target, 1e-9) < 0.05:
            return gt2, pred2, "denormalized", mae2, target
    raise SystemExit("UNRESOLVED_UNITS raw_mae=%.6f target=%.6f" % (mae_raw, target))

def plot_run(run, step):
    ep_dir = ROOT / "runs" / run / "gt" / ("step_%06d" % step) / "episode_000000"
    z = np.load(ep_dir / "arrays.npz")
    report = json.load(open(ep_dir.parent / "report.json"))
    gt = z["gt_action"].astype(np.float64)
    pred = z["nonoverlap_chunk_execution"].astype(np.float64)
    starts = z["chunk_starts"]
    frames = gt.shape[0]
    gt, pred, units, mae_check, mae_target = resolve_units(gt, pred, report)
    print("%s step_%d units=%s arm_mae_check=%.6f report=%.6f" % (run, step, units, mae_check, mae_target))

    dims = [(8, "Joint 0", "rad"), (9, "Joint 1", "rad"), (10, "Joint 2", "rad"),
            (11, "Joint 3", "rad"), (12, "Joint 4", "rad"), (13, "Joint 5", "rad"),
            (14, "Joint 6", "rad"), (15, "Gripper", "opening")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), sharex=True)
    x = np.arange(frames)
    for k, (col, name, unit) in enumerate(dims):
        ax = axes[k // 4][k % 4]
        g = gt[:, col]; p = pred[:, col]
        mae = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        ax.plot(x, g, color="tab:blue", lw=1.3, label="GT")
        ax.plot(x, p, color="tab:orange", lw=1.1, ls="--", label="Prediction (nonoverlap chunk exec)")
        for s in starts:
            if 0 < s < frames:
                ax.axvline(s, color="gray", lw=0.6, ls=":", alpha=0.55)
        ax.set_title("%s (%s): MAE=%.4f, RMSE=%.4f" % (name, unit, mae, rmse), fontsize=10)
        ax.grid(alpha=0.3)
        if k % 4 == 0:
            ax.set_ylabel("Value (%s)" % unit, fontsize=9)
        if k // 4 == 1:
            ax.set_xlabel("Frame index", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    ep0 = get_ep0(report)
    split = ep0.get("split", "?")
    src_ep = ep0.get("source_episode_index", "?")
    fig.suptitle("%s  step %d  |  %s episode (source ep %s)  %d frames  |  right arm joints + gripper: GT vs prediction"
                 % (run, step, split, src_ep, frames), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTDIR / ("%s_step%06d_per_joint.png" % (run, step))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)

if len(sys.argv) > 1:
    TARGETS = []
    for spec in sys.argv[1:]:
        run, step = spec.rsplit(":", 1)
        TARGETS.append((run, int(step)))
else:
    TARGETS = [
        ("DP_1008_h20_aug_h16_20k_retry1", 20000),
        ("ACT_1008_h20_aug_h16_20k_retry1", 12000),
        ("ACT_1008_h20_aug_h16_20k_retry1", 20000),
    ]
for run, step in TARGETS:
    plot_run(run, step)
print("ALL_PLOTS_DONE")

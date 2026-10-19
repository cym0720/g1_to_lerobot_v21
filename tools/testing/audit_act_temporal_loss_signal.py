#!/usr/bin/env python3
"""Analyze saved chunks: block bias versus time-varying target loss signals.

Read-only, no model/GPU work. Identical L1 signs imply identical direct output
loss derivatives, not necessarily identical parameter gradients.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "output/lerobot_act_control_20260929"
MIN_MOVING_SPAN = 0.01  # rad; excludes nearly static joint windows.


def analyze_run(root):
    result = {}
    for directory in sorted((root / "gt").glob("step_*")):
        if not (directory / "report.json").exists():
            continue
        episodes = {}
        for path in sorted(directory.glob("episode_*.npz")):
            saved = np.load(path)
            gt = saved["gt_action"]
            arm = slice(0,7) if gt.shape[-1] == 8 else slice(8,15)
            targets, predictions = [], []
            for frame, prediction in zip(saved["starts"], saved["predictions"]):
                if frame + 16 <= len(gt):
                    targets.append(gt[frame:frame+16,arm])
                    predictions.append(prediction[:,arm])
            target, prediction = np.stack(targets), np.stack(predictions)
            error = prediction-target
            moving = np.ptp(target,axis=1) > MIN_MOVING_SPAN
            uniform_sign = (error>0).all(axis=1) | (error<0).all(axis=1)
            bias = error.mean(axis=1)
            shape_error = error-bias[:,None,:]
            mse = float(np.mean(error**2))
            bias_mse = float(np.mean(bias**2))
            shape_mse = float(np.mean(shape_error**2))
            assert abs(mse-bias_mse-shape_mse) < 1e-6
            episode = path.stem.split("_")[-1]
            episodes[episode] = dict(complete_chunks=len(target),
                moving_joint_windows=int(moving.sum()),
                moving_windows_identical_l1_output_derivatives=int((moving & uniform_sign).sum()),
                fraction_moving_windows_identical_l1_output_derivatives=float(uniform_sign[moving].mean()),
                right_arm_mse_rad2=mse,chunk_bias_mse_rad2=bias_mse,time_shape_mse_rad2=shape_mse,
                fraction_error_energy_chunk_bias=bias_mse/mse if mse else None,
                centered_time_shape_mae_rad=float(np.abs(shape_error).mean()))
        result[directory.name] = episodes
    return result


def main():
    report = dict(full_20d=analyze_run(ROOT/"native_train"),right_8d=analyze_run(ROOT/"right_arm_train"),
        note="Same stored complete 16-step chunks, right arm only. Moving span >0.01 rad. Positive affine normalization preserves signs. The derivative observation is inferred at saved inference predictions, NOT a measurement of actual training/posterior gradients. MSE decomposition describes error energy, not the training objective (which is L1). Equal direct output derivatives do not prove equal model parameter gradients or a unique root cause.")
    output=ROOT/"temporal_loss_signal.json"
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":main()

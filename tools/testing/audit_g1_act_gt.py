#!/usr/bin/env python3
"""Read-only GT/precision audit for the local G1 ACT checkpoint (run in WSL).

The dataset/checkpoint are never changed. Plots keep single-step predictions
separate from non-overlapping 16-step chunks. All cameras use the model's
training-time letterbox; normalization uses PolicyNormProcessor, not new math.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STARVLA = Path("/home/cym/Projects_2026/starvla-g1-validation/third_party/starvla")
sys.path.insert(0, str(STARVLA))
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.framework.VM4A.ACT import _letterbox_image

DEFAULT_DATASET = Path("/home/cym/datasets/g1/0923_g1_pick_apple/empty_text_task_1")
RUN = Path("/home/cym/artifacts/starvla/runs/g1_act_0923_pick_apple_plate_16step")
DEFAULT_OUTPUT = Path("/mnt/d/go work/数据清洗/output/act_gt_audit_20260929/model_gt")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
RIGHT_ARM = slice(8, 15)
BATCH_SIZE = 4  # Leave space for the original training process on the 8 GB GPU.
MODES = ("fp32", "autocast_bf16", "weights_bf16")


def normalize_modality(processor, values, keys, dims):
    data = {}
    cursor = 0
    values = np.asarray(values, dtype=np.float32)
    for key in keys:
        dim = dims[key]
        # StateActionToTensor is the first training transform and requires numpy.
        data[key] = values[..., cursor:cursor + dim].copy()
        cursor += dim
    if cursor != values.shape[-1] or not np.isfinite(values).all():
        raise ValueError("Modality does not match saved normalization metadata")
    out = processor.transform(data)
    return np.concatenate([out[k].detach().cpu().numpy() for k in keys], axis=-1)


def normalize_action(processor, actions):
    return normalize_modality(processor, actions, processor.action_keys, processor.action_key_dims)


def load_episode(dataset, info, episode):
    args = dict(episode_index=episode, episode_chunk=episode // info["chunks_size"])
    frame_table = pd.read_parquet(dataset / info["data_path"].format(**args))
    state_key = "observation.state" if "observation.state" in frame_table else "observations.state.qpos"
    action_key = "action" if "action" in frame_table else "action.qpos"
    state = np.stack(frame_table[state_key].to_numpy()).astype(np.float32)
    action = np.stack(frame_table[action_key].to_numpy()).astype(np.float32)
    if state.shape != action.shape or state.shape[1] != 20:
        raise ValueError(f"ep {episode}: unexpected state/action shapes")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"ep {episode}: non-finite state/action")
    if not np.array_equal(frame_table["frame_index"].to_numpy(), np.arange(len(state))):
        raise ValueError(f"ep {episode}: non-contiguous frame_index")
    images = []
    video_metadata = {}
    for camera in CAMERAS:
        key = "observation.images." + camera
        path = dataset / info["video_path"].format(video_key=key, **args)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        video_metadata[key] = dict(path=str(path), fps=cap.get(cv2.CAP_PROP_FPS),
                                   width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = []
        try:
            for frame in range(len(state)):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"{path}: missing video frame {frame}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(np.asarray(_letterbox_image(Image.fromarray(rgb), (224, 224))).copy())
        finally:
            cap.release()
        images.append(np.stack(frames))
    return dict(episode=episode, state=state, action=action, images=images, videos=video_metadata)


def infer(wrapper, episode, mode):
    model = wrapper._framework
    model.to(torch.bfloat16 if mode == "weights_bf16" else torch.float32).eval()
    result = []
    for start in range(0, len(episode["state"]), BATCH_SIZE):
        examples = [dict(image=[cam[i] for cam in episode["images"]], state=episode["state"][i])
                    for i in range(start, min(start + BATCH_SIZE, len(episode["state"])))]
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if mode == "autocast_bf16" else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            out = wrapper.predict_action(examples)["actions"]
        result.append(np.asarray(out, dtype=np.float32))
    pred = np.concatenate(result)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite predictions in {mode}")
    return pred


def posterior_loss(wrapper, episode, mode):
    """Separate the actual training weighted L1 and KL on six fixed windows.

    This is a posterior-conditioned diagnostic, not deployment/prior loss.
    Enable only the core's VAE branch while keeping all dropout/BN in eval;
    do not mutate checkpoint buffers. The same seed is used for every mode.
    """
    model = wrapper._framework
    proc = wrapper.get_norm_processor()
    horizon = int(model.config.framework.chunk_size)
    starts = np.linspace(0, len(episode["state"]) - horizon, 6, dtype=int)
    norm_actions = normalize_action(proc, episode["action"])
    examples = [dict(image=[cam[s] for cam in episode["images"]],
                     state=proc.apply_state(episode["state"][s]),
                     action=norm_actions[s:s + horizon].astype(np.float16)) for s in starts]
    # Mirror the training packer's float16 state conversion too.
    for ex in examples:
        ex["state"] = ex["state"].astype(np.float16)
    torch.manual_seed(1234)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if mode != "fp32" else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        batch = model._examples_to_act_batch(examples)
        batch = model.action_model.normalize_inputs(batch)
        batch["observation.images"] = [batch[k] for k in model.action_model.config.image_features]
        batch = model.action_model.normalize_targets(batch)
        core = model.action_model.model
        old_training = core.training
        try:
            core.training = True  # VAE gate only; children retain eval mode.
            predicted, (mu, logvar) = core(batch)
        finally:
            core.training = old_training
        w = torch.full((20,), 0.1, device="cuda")
        w[8:16] = 8.0
        mask = ~batch["action_is_pad"].unsqueeze(-1)
        error = (batch["action"] - predicted).abs()
        l1 = (error * mask * w).sum() / (mask * w).sum()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        stable_kl = (-0.5 * (1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())).sum(-1).mean()
    return dict(starts=starts.tolist(), weighted_l1=float(l1), kl=float(kl),
                kl_computed_fp32=float(stable_kl), kl_weight=model.action_model.config.kl_weight,
                composite=float(l1 + kl * model.action_model.config.kl_weight),
                prediction_dtype=str(predicted.dtype), posterior_dtype=str(mu.dtype),
                note="six posterior windows with dropout/BN in eval; not the full training log or prior inference loss")


def compare(episode, predictions):
    gt, state = episode["action"], episode["state"]
    n = len(gt)
    horizon = predictions.shape[1]
    index = np.arange(n)[:, None] + np.arange(horizon)[None, :]
    valid = index < n
    chunk_gt = gt[np.minimum(index, n - 1)]
    error = np.abs(predictions - chunk_gt)
    starts = np.arange(0, n - horizon + 1, horizon)
    gt_span = np.ptp(chunk_gt[starts, :, RIGHT_ARM], axis=1)
    pred_span = np.ptp(predictions[starts, :, RIGHT_ARM], axis=1)
    chunk_repeat = np.mean(np.all(np.diff(predictions[starts, :, RIGHT_ARM], axis=1) == 0, axis=2))
    ratio = np.divide(pred_span, gt_span, out=np.full_like(pred_span, np.nan), where=gt_span > 0.01)
    held = np.broadcast_to(state[:, None, :], predictions.shape)
    return dict(
        first_step_mae_right_arm=np.mean(error[:, 0, RIGHT_ARM], axis=0).tolist(),
        all_valid_steps_mae_right_arm=np.mean(error[valid][:, RIGHT_ARM], axis=0).tolist(),
        first_step_right_gripper_mae=float(np.mean(error[:, 0, 15])),
        first_step_right_arm_span=np.ptp(predictions[:, 0, RIGHT_ARM], axis=0).tolist(),
        gt_right_arm_span=np.ptp(gt[:, RIGHT_ARM], axis=0).tolist(),
        median_chunk_span_gt=np.median(gt_span, axis=0).tolist(),
        median_chunk_span_prediction=np.median(pred_span, axis=0).tolist(),
        median_span_ratio_where_gt_span_gt_0_01=np.nanmedian(ratio, axis=0).tolist(),
        exact_adjacent_repeat_inside_right_arm_chunk=float(chunk_repeat),
        first_step_adjacent_repeat_right_arm=float(np.mean(np.all(np.diff(predictions[:, 0, RIGHT_ARM], axis=0) == 0, axis=1))),
        zero_order_state_hold_valid_steps_mae_right_arm=np.mean(np.abs(held - chunk_gt)[valid][:, RIGHT_ARM], axis=0).tolist(),
    )


def plot_episode(output, episode, predictions):
    ep, gt, n = episode["episode"], episode["action"], len(episode["action"])
    colors = dict(fp32="tab:red", autocast_bf16="tab:blue", weights_bf16="tab:orange")
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        dim = 8 + j
        ax.plot(gt[:, dim], color="black", label="GT action", linewidth=1.6)
        for mode, pred in predictions.items():
            ax.plot(pred[:, 0, dim], color=colors[mode], label=mode + " step 0", linewidth=0.9, alpha=0.85)
        ax.set_title("right_arm_joint" + str(j + 1) if j < 7 else "right_gripper")
        ax.set_xlabel("source frame (one prediction per frame; no overlapping chunks)")
        ax.set_ylabel("rad" if j < 7 else "closed fraction")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Episode {ep}: same checkpoint, same letterboxed inputs, GT vs single-step predictions")
    fig.savefig(output / f"episode_{ep:06d}_right_arm_gt.png", dpi=130)
    plt.close(fig)

    h = predictions["fp32"].shape[1]
    # No shared/repeated GT timestamps: execute only chunks starting at 0, h, 2h...
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        dim = 8 + j
        ax.plot(gt[:, dim], color="black", label="GT action", linewidth=1.6)
        for mode, pred in predictions.items():
            executed = np.concatenate([pred[s, :min(h, n - s), dim] for s in range(0, n, h)])
            ax.plot(executed, color=colors[mode], label=mode + " non-overlap chunks", linewidth=0.9)
        for s in range(0, n, h):
            ax.axvline(s, color="grey", linewidth=0.4, alpha=0.2)
        ax.set_title("right_arm_joint" + str(j + 1) if j < 7 else "right_gripper")
        ax.set_xlabel("source frame; faint vertical lines = 16-step chunk boundaries")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Episode {ep}: non-overlapping ACT chunks (deployment-style, without temporal ensemble)")
    fig.savefig(output / f"episode_{ep:06d}_executed_chunks.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    starts = np.arange(0, n - h + 1, h)
    index = starts[:, None] + np.arange(h)[None, :]
    for j, ax in enumerate(axes.flat):
        if j == 7:
            ax.axis("off")
            continue
        dim = 8 + j
        ax.plot(starts, np.ptp(gt[index, dim], axis=1), color="black", label="GT span over 16 frames")
        for mode, pred in predictions.items():
            ax.plot(starts, np.ptp(pred[starts, :, dim], axis=1), color=colors[mode], label=mode + " chunk span")
        ax.set_title(f"right_arm_joint{j + 1}")
        ax.set_ylabel("max - min (rad)")
        ax.set_xlabel("chunk start frame")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Episode {ep}: is the model predicting movement WITHIN its 16-step output?")
    fig.savefig(output / f"episode_{ep:06d}_chunk_motion.png", dpi=130)
    plt.close(fig)


def plot_loss(output):
    logs = Path("/home/cym/artifacts/starvla/runs/logs")
    names = ["g1_act_0923_pick_apple_plate_16step_launcher.log",
             "g1_act_0923_pick_apple_plate_16step_resume_20260929.log"]
    series = []
    for name in names:
        path = logs / name
        if not path.exists():
            continue
        text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
        matches = re.findall(r"Step (\d+), Loss:.*?'action_dit_loss':\s*([0-9.eE+-]+)", text, re.S)
        points = np.array([(int(s), float(v)) for s, v in matches])
        if len(points):
            series.append((name, points))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    records = []
    for name, points in series:
        smooth = pd.Series(points[:, 1]).rolling(200, min_periods=20).mean()
        ax.plot(points[:, 0], points[:, 1], alpha=0.12, linewidth=0.5)
        ax.plot(points[:, 0], smooth, label=name.replace("g1_act_0923_pick_apple_plate_16step", "run"))
        blocks = []
        for start in range(0, int(points[-1, 0]) + 1, 500):
            selected = points[(points[:, 0] >= start) & (points[:, 0] < start + 500), 1]
            if len(selected):
                blocks.append(dict(start=start, count=len(selected), mean=float(selected.mean()), median=float(np.median(selected))))
        records.append(dict(log=str(logs / name), last_step=int(points[-1, 0]), last_loss=float(points[-1, 1]), blocks=blocks))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("logged weighted L1 + 10 * KL (not pure action MAE)")
    ax.set_title("Original training and resumed training; lines = 200-step rolling mean")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.25)
    if series:
        ax.legend()
    fig.savefig(output / "training_loss_curve.png", dpi=150)
    plt.close(fig)
    return records


def make_wrapper(checkpoint):
    """Load unchanged weights with two isolated wrapper compatibility aliases."""
    from omegaconf import OmegaConf
    saved_cfg = OmegaConf.load(checkpoint.parents[1] / "config.yaml")
    horizon = int(saved_cfg.framework.chunk_size)
    wrapper = PolicyServerWrapper(str(checkpoint), use_bf16=False,
                                  config_overrides=[f"framework.action_model.chunk_size={horizon}"])
    processor = wrapper.get_norm_processor()
    processor.apply_state = lambda raw: normalize_modality(processor, raw, processor.state_keys, processor.state_key_dims)
    return wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "checkpoints/steps_6000_pytorch_model.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 45, 90])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    print("LOAD_CHECKPOINT", args.checkpoint, flush=True)
    # ACT saves its horizon at framework.chunk_size; the generic deployment
    # wrapper also expects framework.action_model. Supply an equivalent,
    # in-memory metadata alias without editing the checkpoint configuration.
    wrapper = make_wrapper(args.checkpoint)
    # A BF16 cast discards weight bits; casting back to FP32 cannot recover them.
    # Restore original FP32 weights before EVERY episode's precision comparison.
    original_weights = {k: v.detach().cpu().clone() for k, v in wrapper._framework.state_dict().items()}
    processor = wrapper.get_norm_processor()
    # The generic wrapper's apply_state passes a tensor into its numpy-only
    # first transform. Fix the input TYPE in this diagnostic instance only;
    # retain the exact saved stats and training transform pipeline.
    report = dict(dataset=str(args.dataset), checkpoint=str(args.checkpoint), episodes={},
                  camera_order=CAMERAS, image_mode="model training-time letterbox 224x224", modes=MODES,
                  action_keys=processor.action_keys, state_keys=processor.state_keys,
                  note="Read-only. Terminal padded targets excluded. No repeated-time GT curves. Generic wrapper horizon alias and numpy input-type compatibility supplied in memory only.")
    for ep in args.episodes:
        wrapper._framework.float().load_state_dict(original_weights)
        episode = load_episode(args.dataset, info, ep)
        predictions, metrics, losses = {}, {}, {}
        for mode in MODES:
            print("INFER", ep, mode, "frames", len(episode["state"]), flush=True)
            predictions[mode] = infer(wrapper, episode, mode)
            metrics[mode] = compare(episode, predictions[mode])
            losses[mode] = posterior_loss(wrapper, episode, mode)
            print("RESULT", ep, mode, json.dumps(metrics[mode]), "LOSS", json.dumps(losses[mode]), flush=True)
        np.savez_compressed(args.output / f"episode_{ep:06d}_predictions.npz",
                            state=episode["state"], gt_action=episode["action"], **predictions)
        plot_episode(args.output, episode, predictions)
        report["episodes"][str(ep)] = dict(frames=len(episode["state"]), videos=episode["videos"],
                                             metrics=metrics, posterior_losses=losses,
                                             fp32_vs_autocast_bf16_mae_right_arm=np.mean(np.abs(predictions["fp32"][:, :, RIGHT_ARM] - predictions["autocast_bf16"][:, :, RIGHT_ARM]), axis=(0, 1)).tolist())
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    report["training_logs"] = plot_loss(args.output)
    report["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("AUDIT_COMPLETE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()

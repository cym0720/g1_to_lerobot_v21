# Purpose: run a real LeRobot v2.1 ACT forward/backward/optimizer smoke test on local v2.1 Parquet + MP4 data and require loss to decrease.
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V21_SITE = REPO_ROOT / ".venv_lerobot_v21" / "Lib" / "site-packages"

# Windows venv launchers in this workspace detach their base interpreter. Re-exec
# the base Python directly so that the test has reliable process ownership/logging.
if os.environ.get("LEROBOT_V21_NATIVE") != "1":
    env = os.environ | {"PYTHONPATH": str(V21_SITE), "LEROBOT_V21_NATIVE": "1", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    base_python = Path(getattr(sys, "_base_executable", sys.executable))
    raise SystemExit(subprocess.call([str(base_python), "-u", *sys.argv], env=env))

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def stats_to_tensors(meta_stats: dict, key: str) -> dict[str, torch.Tensor]:
    if key not in meta_stats:
        # Some legacy v2.1 exports intentionally omitted decoded RGB statistics.
        # These are the same conservative normalized-RGB defaults used by conversion.
        return {
            "min": torch.zeros((3, 1, 1)), "max": torch.ones((3, 1, 1)),
            "mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25),
        }
    return {name: torch.tensor(value, dtype=torch.float32) for name, value in meta_stats[key].items() if name in {"min", "max", "mean", "std"}}


def load_frame(video: Path, frame_index: int, size: int) -> torch.Tensor:
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float().div_(255)
                return F.interpolate(rgb.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    raise IndexError(f"{video}: missing frame {frame_index}")


def load_sample(root: Path, episode: dict, start: int, chunk: int, image_size: int, cameras: tuple[str, ...]) -> dict[str, torch.Tensor]:
    index = int(episode["episode_index"])
    table = pq.read_table(root / "data/chunk-000" / f"episode_{index:06d}.parquet", columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    end = min(len(action), start + chunk)
    action_chunk = np.zeros((chunk, action.shape[1]), dtype=np.float32)
    action_chunk[: end - start] = action[start:end]
    is_pad = np.zeros(chunk, dtype=bool); is_pad[end - start :] = True
    batch = {
        "observation.state": torch.from_numpy(state[start]).unsqueeze(0),
        "action": torch.from_numpy(action_chunk).unsqueeze(0),
        "action_is_pad": torch.from_numpy(is_pad).unsqueeze(0),
    }
    for camera in cameras:
        video = root / "videos/chunk-000" / camera / f"episode_{index:06d}.mp4"
        batch[camera] = load_frame(video, start, image_size).unsqueeze(0)
    return batch


def predict(policy: ACTPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the v2.1 ACT inference path, excluding GT-only training keys."""
    inputs = {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
    return policy.predict_action_chunk(inputs)


def save_gt_plot(path: Path, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (1500, 1050), "white"); draw = ImageDraw.Draw(image)
    try: title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 23); label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError: title_font = label_font = ImageFont.load_default()
    groups = [("left arm (0-6)", 0, 7), ("right arm (7-13)", 7, 14), ("waist/head/grippers (14-19)", 14, 20)]
    colors = ["#2878b5", "#398c4f", "#7d5ba6", "#d0702b", "#8d4f65", "#1896a5", "#99953b"]
    draw.text((35, 15), "LeRobot v2.1 ACT: predicted action chunk vs held-out ground truth", fill="#111111", font=title_font)
    for panel, (name, start, end) in enumerate(groups):
        left, top, right, bottom = 80, 85 + panel * 310, 1450, 340 + panel * 310
        values = np.concatenate((ground_truth[:, start:end].ravel(), prediction[:, start:end].ravel()))
        lo, hi = float(values.min()), float(values.max()); margin = max((hi - lo) * .08, 1e-9); lo -= margin; hi += margin
        for tick in range(5): draw.line((left, top + tick * (bottom-top)/4, right, top + tick*(bottom-top)/4), fill="#e4e4e4")
        for channel in range(start, end):
            color = colors[(channel-start) % len(colors)]
            for data, width in ((ground_truth[:, channel], 3), (prediction[:, channel], 1)):
                points = [(left + i * (right-left)/max(1, len(data)-1), bottom - (float(v)-lo)*(bottom-top)/(hi-lo)) for i,v in enumerate(data)]
                draw.line(points, fill=color, width=width)
        draw.rectangle((left, top, right, bottom), outline="#555555", width=2); draw.text((left, top-25), name + " (thick=GT, thin=prediction)", fill="#222222", font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Native LeRobot v2.1 ACT smoke training with one resized head camera.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Single LeRobot v2.1 task root.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report; must not exist.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--three-cameras", action="store_true", help="Use head, left wrist, and right wrist simultaneously after resize.")
    parser.add_argument("--fixed-sample", action="store_true", help="Repeat one real action chunk; recommended for a deterministic loss-decrease smoke test.")
    parser.add_argument("--gt-analysis-dir", type=Path, help="Optional held-out GT evaluation directory; reserves the final 20%% episodes for evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 10:
        raise SystemExit("--steps must be at least 10")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf8"))
    if info.get("codebase_version") != "v2.1":
        raise SystemExit("dataset is not LeRobot v2.1")
    if info["features"]["observation.state"]["shape"] != [20] or info["features"]["action"]["shape"] != [20]:
        raise SystemExit("expected native G1 20D observation.state/action")
    episodes = [json.loads(line) for line in (args.dataset_root / "meta/episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    meta_stats = json.loads((args.dataset_root / "meta/stats.json").read_text(encoding="utf8"))
    cameras = ("observation.images.cam_high_rgb",)
    if args.three_cameras:
        cameras = (
            "observation.images.cam_high_rgb",
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_right_wrist_rgb",
        )
    if any(camera not in info["features"] for camera in cameras):
        raise SystemExit(f"dataset lacks requested camera fields: {cameras}")
    config = ACTConfig(
        device="cpu", chunk_size=args.chunk_size, n_action_steps=1,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
        dim_model=128, dim_feedforward=256, n_heads=4, n_encoder_layers=2, n_decoder_layers=1,
        use_vae=False, dropout=0.0,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{camera: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)) for camera in cameras},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
    )
    policy = ACTPolicy(config, {"observation.state": stats_to_tensors(meta_stats, "observation.state"), "action": stats_to_tensors(meta_stats, "action"), **{camera: stats_to_tensors(meta_stats, camera) for camera in cameras}})
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr)
    losses: list[float] = []
    usable = [episode for episode in episodes if int(episode["length"]) >= args.chunk_size]
    split = max(1, int(len(usable) * .8))
    train_episodes = usable[:split] if args.gt_analysis_dir else usable
    eval_episodes = usable[split:] if args.gt_analysis_dir else []
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        episode = train_episodes[0] if args.fixed_sample else train_episodes[int(rng.integers(len(train_episodes)))]
        start = 0 if args.fixed_sample else int(rng.integers(int(episode["length"]) - args.chunk_size + 1))
        batch = load_sample(args.dataset_root, episode, start, args.chunk_size, args.image_size, cameras)
        loss, _ = policy(batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        value = float(loss.detach()); losses.append(value)
        print(f"step={step + 1}/{args.steps} loss={value:.7f}", flush=True)
    width = max(3, len(losses) // 4)
    first, last = float(np.mean(losses[:width])), float(np.mean(losses[-width:]))
    report = {"runtime": "LeRobot v2.1 native ACTPolicy", "state_dim": 20, "action_dim": 20, "cameras": list(cameras), "image_size": args.image_size, "steps": args.steps, "fixed_sample": args.fixed_sample, "losses": losses, "first_window_mean": first, "last_window_mean": last, "loss_decreased": last < first}
    if args.gt_analysis_dir:
        policy.eval(); all_gt=[]; all_pred=[]
        with torch.no_grad():
            for episode in eval_episodes:
                start = max(0, (int(episode["length"]) - args.chunk_size) // 2)
                batch = load_sample(args.dataset_root, episode, start, args.chunk_size, args.image_size, cameras)
                all_gt.append(batch["action"][0].numpy()); all_pred.append(predict(policy, batch)[0].numpy())
        ground_truth = np.concatenate(all_gt); prediction = np.concatenate(all_pred); error = prediction - ground_truth
        groups = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "waist_head": slice(14, 18), "grippers": slice(18, 20), "all": slice(0, 20)}
        metrics = {name: {"mae": float(np.abs(error[:, section]).mean()), "rmse": float(np.sqrt(np.square(error[:, section]).mean()))} for name, section in groups.items()}
        plot_path = args.gt_analysis_dir / "prediction_vs_gt.png"; save_gt_plot(plot_path, ground_truth[:args.chunk_size], prediction[:args.chunk_size])
        report["gt_analysis"] = {"train_episodes": len(train_episodes), "held_out_episodes": len(eval_episodes), "held_out_action_frames": int(len(ground_truth)), "metrics": metrics, "plot": str(plot_path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["loss_decreased"]:
        raise SystemExit("ACT ran but loss did not decrease")


if __name__ == "__main__":
    main()

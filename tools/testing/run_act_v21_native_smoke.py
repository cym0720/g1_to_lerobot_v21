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


def load_sample(root: Path, episode: dict, start: int, chunk: int, image_size: int) -> dict[str, torch.Tensor]:
    index = int(episode["episode_index"])
    table = pq.read_table(root / "data/chunk-000" / f"episode_{index:06d}.parquet", columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    end = min(len(action), start + chunk)
    action_chunk = np.zeros((chunk, action.shape[1]), dtype=np.float32)
    action_chunk[: end - start] = action[start:end]
    is_pad = np.zeros(chunk, dtype=bool); is_pad[end - start :] = True
    video = root / "videos/chunk-000/observation.images.cam_high_rgb" / f"episode_{index:06d}.mp4"
    return {
        "observation.state": torch.from_numpy(state[start]).unsqueeze(0),
        "observation.images.cam_high_rgb": load_frame(video, start, image_size).unsqueeze(0),
        "action": torch.from_numpy(action_chunk).unsqueeze(0),
        "action_is_pad": torch.from_numpy(is_pad).unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Native LeRobot v2.1 ACT smoke training with one resized head camera.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Single LeRobot v2.1 task root.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report; must not exist.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    image_key = "observation.images.cam_high_rgb"
    config = ACTConfig(
        device="cpu", chunk_size=args.chunk_size, n_action_steps=1,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
        dim_model=128, dim_feedforward=256, n_heads=4, n_encoder_layers=2, n_decoder_layers=1,
        use_vae=False, dropout=0.0,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            image_key: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
    )
    policy = ACTPolicy(config, {"observation.state": stats_to_tensors(meta_stats, "observation.state"), "action": stats_to_tensors(meta_stats, "action"), image_key: stats_to_tensors(meta_stats, image_key)})
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr)
    losses: list[float] = []
    usable = [episode for episode in episodes if int(episode["length"]) >= args.chunk_size]
    for step in range(args.steps):
        episode = usable[step % len(usable)]
        start = (step * 7) % (int(episode["length"]) - args.chunk_size + 1)
        batch = load_sample(args.dataset_root, episode, start, args.chunk_size, args.image_size)
        loss, _ = policy(batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        value = float(loss.detach()); losses.append(value)
        print(f"step={step + 1}/{args.steps} loss={value:.7f}", flush=True)
    width = max(3, len(losses) // 4)
    first, last = float(np.mean(losses[:width])), float(np.mean(losses[-width:]))
    report = {"runtime": "LeRobot v2.1 native ACTPolicy", "state_dim": 20, "action_dim": 20, "camera": image_key, "image_size": args.image_size, "steps": args.steps, "losses": losses, "first_window_mean": first, "last_window_mean": last, "loss_decreased": last < first}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["loss_decreased"]:
        raise SystemExit("ACT ran but loss did not decrease")


if __name__ == "__main__":
    main()

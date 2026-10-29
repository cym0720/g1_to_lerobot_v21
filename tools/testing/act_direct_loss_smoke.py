"""Small, real ACT optimizer smoke test without the v3 dataset loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    # EVA keeps qpos names on disk. LeRobot 0.3.3 ACT internally still looks
    # up literal `observation.state` and `action`, so this gate adapts only
    # its in-memory batch rather than contaminating the stored contract.
    table = pq.read_table(args.parquet, columns=["observations.state.qpos", "action.qpos"])
    states = np.asarray(table["observations.state.qpos"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action.qpos"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"expected rank-2 state/action, got {states.shape}/{actions.shape}")
    task_root = args.parquet.parents[2]
    stats = json.loads((task_root / "meta/stats.json").read_text(encoding="utf8"))
    dataset_stats = {
        policy_key: {
            name: np.asarray(stats[source_key][name], dtype=np.float32)
            for name in ("mean", "std", "min", "max")
        }
        for policy_key, source_key in (
            ("observation.state", "observations.state.qpos"),
            ("action", "action.qpos"),
        )
    }
    chunk = min(8, len(actions))
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, tuple(states.shape[1:])),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (1,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, tuple(actions.shape[1:]))},
        chunk_size=chunk,
        n_action_steps=chunk,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        use_vae=True,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    torch.manual_seed(42)
    model = ACTPolicy(config, dataset_stats=dataset_stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    starts = np.linspace(0, len(actions) - chunk, num=min(4, len(actions) - chunk + 1), dtype=int)
    losses: list[float] = []
    model.train()
    for step in range(args.steps):
        selected = starts[(np.arange(len(starts)) + step) % len(starts)]
        batch = {
            "observation.state": torch.from_numpy(states[selected]),
            "observation.environment_state": torch.zeros((len(selected), 1), dtype=torch.float32),
            "action": torch.from_numpy(np.stack([actions[i : i + chunk] for i in selected])),
            "action_is_pad": torch.zeros((len(selected), chunk), dtype=torch.bool),
        }
        loss, _ = model.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step={step + 1} loss={losses[-1]:.8f}", flush=True)
    width = max(3, len(losses) // 4)
    first = float(np.mean(losses[:width]))
    last = float(np.mean(losses[-width:]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"steps={args.steps}\nfirst_window_mean={first:.8g}\nlast_window_mean={last:.8g}\nloss_decreased={last < first}\n",
        encoding="utf8",
    )
    if last >= first:
        raise SystemExit("ACT loss did not decrease")


if __name__ == "__main__":
    main()

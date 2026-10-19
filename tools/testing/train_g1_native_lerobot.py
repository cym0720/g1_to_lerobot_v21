#!/usr/bin/env python3
"""Native LeRobot ACT control: unchanged labels, canonical names, letterbox.

No StarVLA adapter, custom decoder or weighted reconstruction loss. The only
trainer hooks add letterbox, metrics and read-only checkpoint GT diagnostics.
Rarely changed training settings deliberately live below rather than in CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "output/rebuild_20260924_g1_retasked/final/0923_g1_pick_apple/empty_text_task_1"
WSL_CACHE = Path("/home/cym/datasets/g1/0923_g1_pick_apple/empty_text_task_1")
OUTPUT = REPO / "output/lerobot_act_control_20260929/native_train"
RENAME = {"observations.state.qpos": "observation.state", "action.qpos": "action"}
STEPS, BATCH_SIZE, SAVE_FREQ = 15000, 8, 1000
IMAGE_SIZE, CHUNK_SIZE = 224, 16
EVAL_EPISODES = (0, 45, 90)
RIGHT_ARM_DIMS = tuple(range(8, 15))
RIGHT_GRIPPER_DIM = 15


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def renamed(row):
    return {RENAME.get(k, k): v for k, v in row.items()}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare(source, destination):
    """Rename columns, not their values; verify cached videos before linking."""
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "meta").mkdir()
    info = json.loads((source / "meta/info.json").read_text())
    info["features"] = renamed(info["features"])
    dump(destination / "meta/info.json", info)
    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy2(source / "meta" / name, destination / "meta" / name)
    stats = source / "meta/stats.json"
    if stats.exists():
        dump(destination / "meta/stats.json", renamed(json.loads(stats.read_text())))
    episodes_stats = [json.loads(l) for l in (source / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    for row in episodes_stats:
        row["stats"] = renamed(row["stats"])
    (destination / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes_stats))
    checked, frames, videos = [], 0, 0
    global_offset = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        n = len(table)
        assert np.array_equal(table["frame_index"].to_numpy(), np.arange(n)), path
        assert np.array_equal(table["index"].to_numpy(), np.arange(global_offset, global_offset + n)), path
        assert np.max(np.abs(table["timestamp"].to_numpy() - np.arange(n) / info["fps"])) < 1e-4, path
        for key in RENAME:
            a = np.asarray(table[key].to_pylist(), dtype=np.float32)
            assert a.shape == (n, 20) and np.isfinite(a).all(), (path, key)
        out = destination / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.rename_columns([RENAME.get(k, k) for k in table.column_names]), out)
        written = pq.read_table(out)
        for old, new in RENAME.items():
            np.testing.assert_array_equal(np.asarray(table[old].to_pylist()), np.asarray(written[new].to_pylist()))
        checked.append(dict(episode=int(table["episode_index"][0].as_py()), frames=n, exact_same_state_action=True))
        frames += n
        global_offset += n
    for path in sorted((source / "videos").rglob("*.mp4")):
        cached = WSL_CACHE / path.relative_to(source)
        target = path
        if cached.exists():
            assert sha256(path) == sha256(cached), f"Cached video differs: {path}"
            target = cached
        out = destination / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.symlink_to(target.resolve())
        videos += 1
    assert frames == info["total_frames"] and len(checked) == info["total_episodes"]
    assert videos == info["total_videos"]
    dump(destination.parent / "adapter_validation.json", dict(episodes=checked, frames=frames, videos=videos,
         column_renaming_only=True, video_values_unchanged=True, cached_videos_sha256_verified=True))
    print("ADAPTER_VALIDATED", len(checked), frames, videos, flush=True)


class Letterbox:
    def __call__(self, image):
        h, w = image.shape[-2:]
        ratio = IMAGE_SIZE / max(h, w)
        nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
        x = F.interpolate(image[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
        y, xpad = (IMAGE_SIZE - nh) // 2, (IMAGE_SIZE - nw) // 2
        return F.pad(x, (xpad, IMAGE_SIZE - nw - xpad, y, IMAGE_SIZE - nh - y), value=0)


def sample_batch(dataset, indices, device):
    items = [dataset[i] for i in indices]
    keys = [k for k, v in items[0].items() if isinstance(v, torch.Tensor)]
    return {k: torch.stack([s[k] for s in items]).to(device) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate native dataset/chunks without training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(args.output / "cache/torch"))
    torch.set_num_threads(4)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy
    from lerobot.scripts import train as native
    from lerobot.utils.utils import init_logging
    import audit_g1_act_gt as audit

    dataset_root = args.output / "dataset"
    prepare(args.input, dataset_root)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple", root=str(dataset_root),
                              video_backend="pyav", use_imagenet_stats=True),
        policy=ACTConfig(chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE, device="cuda", use_amp=False,
                         push_to_hub=False),
        output_dir=args.output / "run", job_name="g1_0923_native_act_16step", seed=1000,
        num_workers=4, batch_size=BATCH_SIZE, steps=STEPS, eval_freq=0, log_freq=100, save_freq=SAVE_FREQ)
    dataset = make_dataset(cfg)
    dataset.image_transforms = Letterbox()
    tables = {}
    offsets = {}
    running = 0
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        ep = int(table["episode_index"][0].as_py())
        tables[ep] = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        offsets[ep] = running
        running += len(table)
    tests = []
    for ep in EVAL_EPISODES:
        for f in (0, 52, 104, len(tables[ep]) - 1):
            item = dataset[offsets[ep] + f]
            index = np.minimum(f + np.arange(CHUNK_SIZE), len(tables[ep]) - 1)
            np.testing.assert_array_equal(item["action"].numpy(), tables[ep][index])
            np.testing.assert_array_equal(item["action_is_pad"].numpy(), f + np.arange(CHUNK_SIZE) >= len(tables[ep]))
            for key in dataset.meta.camera_keys:
                assert item[key].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
            tests.append(dict(ep=ep, frame=f, exact_future_chunk=True, correct_padding=True))
    dump(args.output / "native_loader_validation.json", dict(tests=tests, action_offsets=list(range(16)),
         image_preprocessing="letterbox, bilinear antialias, black padding", columns=RENAME))
    print("NATIVE_LOADER_VALIDATED", len(tests), flush=True)
    dump(args.output / "requested_train_config.json", cfg.to_dict())
    if args.dry_run:
        print("NATIVE_DRY_RUN_COMPLETE", flush=True)
        return
    torch.manual_seed(cfg.seed)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.train()
    fixed = sample_batch(dataset, [52, 104, 156, 220], "cuda")
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.policy.optimizer_lr, weight_decay=cfg.policy.optimizer_weight_decay)
    losses = []
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy(fixed)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
        optimizer.step()
        losses.append(float(loss.detach()))
    decreased = float(np.mean(losses[-5:])) < float(np.mean(losses[:5]))
    dump(args.output / "native_act_smoke.json", dict(losses=losses, loss_decreased=decreased,
         initial_window=float(np.mean(losses[:5])), final_window=float(np.mean(losses[-5:])),
         note="Temporary smoke model is discarded; formal native trainer creates a fresh model"))
    if not decreased:
        raise RuntimeError("Native ACT smoke loss did not decrease")
    del optimizer, policy, fixed
    torch.cuda.empty_cache()
    print("NATIVE_ACT_SMOKE_OK", flush=True)

    native.make_dataset = lambda current_cfg: dataset
    old_update = native.update_policy
    update_count = 0

    def logged_update(*a, **kw):
        nonlocal update_count
        result = old_update(*a, **kw)
        update_count += 1
        if update_count == 1 or update_count % 100 == 0:
            metric = dict(step=update_count, elapsed_seconds=time.monotonic() - started, **result[1])
            with (args.output / "loss_history.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
            dump(args.output / "status.json", dict(phase="training", **metric))
            print("NATIVE_LOSS", json.dumps(metric), flush=True)
        return result

    old_save = native.save_checkpoint

    def save_with_gt(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None):
        old_save(checkpoint_dir, step, cfg, policy, optimizer, scheduler)
        out = args.output / f"gt/step_{step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        was_training = policy.training
        policy.eval()
        rows = {}
        with torch.inference_mode():
            for ep in EVAL_EPISODES:
                gt = tables[ep]
                starts = list(range(0, len(gt), CHUNK_SIZE))
                pred = []
                for f in starts:
                    batch = sample_batch(dataset, [offsets[ep] + f], "cuda")
                    pred.append(policy.predict_action_chunk(batch)[0].float().cpu().numpy())
                pred = np.stack(pred)
                executed = np.concatenate([p[:min(CHUNK_SIZE, len(gt) - f)] for f, p in zip(starts, pred)])
                ratios = []
                for dim in RIGHT_ARM_DIMS:
                    pairs = [(np.ptp(p[:, dim]), np.ptp(gt[f:f + CHUNK_SIZE, dim])) for f, p in zip(starts, pred)
                             if f + CHUNK_SIZE <= len(gt) and np.ptp(gt[f:f + CHUNK_SIZE, dim]) > .01]
                    ratios.append(float(np.median([a / b for a, b in pairs])) if pairs else None)
                rows[str(ep)] = dict(span_ratios=ratios, executed_mae_right_arm=float(np.abs(executed[:, RIGHT_ARM_DIMS] - gt[:, RIGHT_ARM_DIMS]).mean()),
                                     note="Non-overlapping chunks; incomplete last window excluded from span ratios")
                np.savez_compressed(out / f"episode_{ep:06d}.npz", starts=np.array(starts), predictions=pred, executed=executed, gt_action=gt)
                fig, axes = audit.plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
                for j, ax in enumerate(axes.flat):
                    dim = (*RIGHT_ARM_DIMS, RIGHT_GRIPPER_DIM)[j]
                    ax.plot(gt[:, dim], color="black", label="GT")
                    ax.plot(executed[:, dim], color="tab:orange", label="Native ACT, each of 16 steps consumed")
                    ax.set_title(f"right_arm_joint{j + 1}" if j < 7 else "right_gripper")
                    ax.grid(alpha=.25)
                    if j == 0: ax.legend(fontsize=8)
                fig.savefig(out / f"episode_{ep:06d}.png", dpi=120)
                audit.plt.close(fig)
        dump(out / "report.json", dict(step=step, episodes=rows, assessment="training-set fit, not held-out robot success"))
        policy.train(was_training)
        print("NATIVE_CHECKPOINT_GT_COMPLETE", step, json.dumps(rows), flush=True)

    native.update_policy = logged_update
    native.save_checkpoint = save_with_gt
    init_logging()
    started = time.monotonic()
    native.train(cfg)
    dump(args.output / "status.json", dict(phase="complete", step=STEPS, elapsed_seconds=time.monotonic() - started))
    print("NATIVE_TRAINING_COMPLETE", args.output / "run/checkpoints/last/pretrained_model", flush=True)


if __name__ == "__main__":
    main()

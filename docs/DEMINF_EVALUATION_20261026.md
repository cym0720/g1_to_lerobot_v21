# Pick Apple DemInf evaluation method (2026-10-26)

> 历史实验方法记录。对应的一次性实现脚本已于 2026-10-29 从 `tools/curation_deminf/` 清理；下方命令和实现描述仅用于解释当时流程，不能直接运行。

This folder is an optional assessment tool. It reads an already materialized LeRobot dataset and writes derived files only beneath the requested `--out` directory. It does not import or call the cleaning pipeline, alter source parquet/video/metadata, or replace the training dataset.

## First-stage scoring

The scorer takes episode-level state and action vectors (`observations.state.qpos` / `action.qpos`, or `observation.state` / `action`), trains separate small VAEs for state, absolute action, and relative action (`action[t] - state[t]`), then estimates per-episode dependence with a KSG-1 k-nearest-neighbor MI estimator. It ranks training episodes by relative-action MI and records the absolute-action score as an ablation. The resulting subset is only a candidate.

This source has train and val folders. To keep the historically selected short-episode val set out of model selection, the scorer reserves a fixed 10% of the 145 source train episodes as a test set (seed 17), ranks the remaining 90% (130 episodes), and retains 80% of that ranking. All 16 original val episodes stay as the validation set. Only the 130 training episodes fit VAEs and rank for retention. The split manifest uses keys like `train:12` and `validation:3`; reuse it unchanged in all policy comparisons.

```powershell
uv run python tools/curation_deminf/score_pick_apple.py `
  --dataset output/merged/fisheye/0923+1004+1015+1019+1020_pick_apple_20261022 `
  --out experiments/deminF_20261026/full
```

The HTML report includes score plots, per-episode split/decision, and middle-frame previews from all source cameras when OpenCV can decode the files. `episode_split_manifest.json` records train, validation, test, and selected training episode IDs. No selected dataset is copied automatically.

Smoke mode: add `--limit-episodes 12 --max-frames-per-episode 24 --vae-epochs 2`; this is a plumbing check and must not be interpreted as a scientific score.

## Policy-effectiveness evaluation

The MI score is a proxy, not a policy-quality measurement. Train the *same policy architecture* under identical input/output contract, preprocessing, seeds, validation, optimization budget, and checkpoint selection for:

1. All 130 training episodes.
2. Uniform random subset at the exact retained episode count (104 episodes; repeat with at least 3 fixed seeds).
3. DemInf-ranked subset (same retained count).

For a fast pilot, use a fixed 15,000 optimizer-step budget for all three arms, one seed, and report the warm-start/resume details if any. Then repeat with at least 3 fixed seeds before making a strong conclusion. Report training steps/wall time and GPU, validation action-chunk loss, per-right-arm-joint MAE, gripper timing/state error, and performance on the fixed episode test set. Give mean and standard deviation across seeds. The primary offline claim is supported only if DemInf improves held-out imitation metrics over random at the same data fraction without materially degrading task-stage/gripper metrics. A result that only beats a lower-data random baseline does not establish superiority to full data.

Offline losses cannot establish closed-loop success. If a suitable simulator exists, run identical reset conditions, object/plate placements, success criteria, and a preregistered number of trials for all three variants; report pick, transport, release, overall success, drops, timeouts, and interventions with confidence intervals. Real-robot testing is a separate, explicitly supervised step and is not part of this offline pipeline.

## Remote training and local report

Remote GPU training should start only after a read-only host check confirms identity, GPU/VRAM/load, free storage, dataset availability/hash, and active-job state. The selected local source has a matching H20 letterbox224 copy: train/val episode IDs and lengths matched 145/145 and 16/16 in the October 26, 2026 check. Use an isolated new run root on H20; do not reuse or replace existing checkpoints. Keep the HTML report, manifests, fetched training logs, checkpoint metadata, predictions, and metrics under `experiments/deminF_20261026/`.

Current source: `output/merged/fisheye/0923+1004+1015+1019+1020_pick_apple_20261022` (train 145 episodes / 59,539 frames; val 16 episodes / 3,664 frames; head RGB plus left/right fisheye). It has `observation.state` and `action` float32[20] with named joints. Source files remain read-only.

# Output directory and retention rules

Follow `robot-data-cleaning-gated` for dataset stage locations. This document records the repository-specific layout and retention handling; the skill is authoritative if details change.

## Dataset stages

- `dataset/` contains only `raw/` and `converted/` stage roots.
- Raw recordings stay under `dataset/raw/<source>/<batch>/` and are never modified or deleted as part of output cleanup.
- Converted baselines for one date stay under `dataset/converted/<data-standard>/<camera-standard>/<date-or-batch>/`.
- Cleaned datasets stay under `output/cleaned/<data-standard>/<camera-standard>/<batch-or-dataset>/`.
- Cross-batch datasets stay under `output/merged/<data-standard>/<camera-standard>/<merged-name>/`.
- Keep fish-eye and wrist-camera streams in separate camera-standard directories. Do not mix EVA qpos20, RollFlow 20D, 1008 legacy ordering, or other joint layouts under the same data-standard directory.
- Do not create `output/converted/`, `output/cleaning/`, `dataset/cleaned/`, or `dataset/merged/` for new work. Preserve existing legacy paths until they have been verified and archived; do not delete or relocate them solely because their path is legacy.
- The old `output/cleaning/` workspace is preserved at `artifacts/datasets/legacy_cleaning_20261029/`. Its stages remain historical; audit camera standard and current acceptance requirements before using or promoting any dataset from that archive.
- `output/current/` is reserved for disposable intermediate results from the active pipeline run. Clear it before a new run only after confirming no process is using it.

## Other outputs

- Keep validation, preflight, statistics, and diagnostic reports under `output/diagnostics/` or `output/preflight/`, organized by task and date. Keep the minimal reports needed to support accepted data and model decisions.
- Keep model-versus-GT evaluations, replay plots, training experiments, and their logs/code under `experiments/<experiment>/` (or `experiments/gt_analysis/<experiment>/` for standalone GT analyses), not `output/diagnostics/`.
- Keep model checkpoints, deployment packages, and transferable archives outside the dataset stage directories. Record their source run and checksum before cleanup.
- Keep delivery bundles and deployable assets under `artifacts/<kind>/`; keep experiment-specific prompts, configs, reports, plots, and scripts under `experiments/<experiment>/`.
- Put scratch data under `output/.scratch/` and temporary staging under `output/.staging/`. Remove only disposable test fixtures and caches after confirming they are not the sole copy of a report or deliverable.
- Do not store generated outputs in the repository root. Do not add datasets, checkpoints, caches, or generated reports to Git.

## Cleanup rules

- Python bytecode, `__pycache__`, pytest cache, and confirmed disposable test fixtures may be deleted.
- A LeRobot directory containing `meta/info.json` is a dataset artifact: verify its standard, provenance, and duplicate/backup status before moving or deleting it.
- A ZIP or archive that may be the only copy must be retained or copied to the designated archive, then checked by hash before deleting the source.
- Logs, smoke videos, and temporary plans may be removed only after their needed findings have been recorded in a durable report.
- Never remove a checkpoint, dataset, delivery package, or historical report based only on a nonconforming directory name.

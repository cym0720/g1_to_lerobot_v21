---
name: model-bundle-transfer
description: Package model checkpoints and deployment dependencies, split the archive, and transfer chunks using parallel SCP processes with resumable SHA256 verification. Use when this user asks to package, download, or deliver trained models from H20 or another SSH host.
---

# Model bundle transfer

For this user's model deliveries, use the requested sequence: **package → split → parallel process transfer → ordered reassembly → verification**. Do not substitute one SCP process for the complete archive, even when it seems convenient. This requirement applies to model delivery, not ordinary small-file inspection or GT preview downloads.

## Package

- Resolve the requested run, exact optimizer step, checkpoint, destination, and deployment loading contract from the conversation and saved configuration. Do not guess the latest checkpoint from filenames alone.
- Include the checkpoint, matching configuration and normalization statistics, required loading adapter/source patch, concise deployment instructions, validation receipt, and relevant GT comparison. Preserve exact weights/statistics unless changes were requested.
- A sidecar configuration does not implement a runtime override. When delivering changed sampling behavior such as DP `clip_sample_range=2.0`, supply and verify a loader or source patch that sets it on the policy actually used for inference, including EMA. Record when a plain unmodified loader would ignore the setting.
- Use a new, descriptive export directory/archive. Generate a SHA256 manifest for all payload files and a separate full-archive SHA256. Validate loading in proportion to the change. If a GPU check is needed, inspect occupancy before starting it.

## Split

- Defaults: **128 MiB per chunk; six parallel transfers**. Keep these as central constants rather than adding numerous CLI options. Respect explicit overrides from the user or demonstrated host limitations.
- Finish the archive before splitting it. Create chunks in a dedicated adjacent directory, with deterministic zero-padded numeric names, e.g. `part_0000`, `part_0001`.
- Save a manifest containing archive basename, total bytes, archive SHA256, and the ordered list of chunk names, byte sizes and SHA256 values. Bind resumptions to this manifest, not just to a similarly named archive.
- Use a scoped Python helper or native Linux `split`/hash tools over SSH. Keep remote paths shell-quoted; do not let PowerShell expand remote shell variables. Never embed credentials or copy authentication files.

## Parallel process transfer

- Download the small manifest first, then launch up to **six independent SCP processes**, one chunk per process. A Python thread pool orchestrating six `subprocess` SCP workers is acceptable: the transfers themselves must be separate processes. A single sequential SCP loop is not parallel transfer.
- Use the configured SSH alias (for example `H20`) and `ClearAllForwardings=yes`. Limit each worker to three retries with finite connection/inactivity timeouts. Report persistent errors rather than retrying forever.
- Write each in-progress chunk to a `.part` file. Verify its exact size and SHA256 before renaming it complete. Resume by rechecking existing completed chunks against the manifest and downloading only missing or corrupt ones; size equality alone is insufficient.
- Keep progress tied to verified bytes/chunks. For a long transfer, use durable logging and monitor process state, progress and errors until completion. If interrupted, preserve verified chunks and report the resume location.

## Reassemble and verify

- Concatenate verified chunks in manifest order into a temporary archive using streaming I/O. Do not load an entire model archive into memory.
- Verify total size and full-archive SHA256 before giving the archive its final name. Preserve an existing different destination file; use a new filename or ask before replacement. If the destination already has the expected hash, reuse it without retransmission.
- Inspect archive structure and required deployment files; validate payload hashes when extracting. Only then report delivery complete.
- Default local destination for the data-cleaning workspace is `D:\go work\数据清洗\output\deployment_packages`, unless the user selects another path.
- Keep chunks until final verification passes. Any cleanup must target only this transfer's exact generated files; do not recursively remove a general models/output directory. Avoid re-transferring a completed, verified archive merely to demonstrate the preferred transport method.

## Delivery

Return a clickable absolute local archive path, size, verification result, and the one loading detail needed to apply the requested inference settings. State separately if the archive is still only on the remote host. Packaging does not authorize deployment, training, or transfer to an unrelated machine.

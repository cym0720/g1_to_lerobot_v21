# Conversion tools

## Maintained entry points

- `00_convert_618_to_lerobot.py`: convert raw recordings using the selected source, schema, task map, and versioned data contract.
- `03_sync_task_prompts.py`: synchronize LeRobot task text and task IDs.
- `05_recompute_metadata_stats.py`: recompute metadata statistics after a data transformation.
- `build_train_val_split.py`: create episode-level train/validation splits.
- `merge_final_datasets.py`: merge datasets with matching, already-validated semantics.
- `preflight_g1_batch.py`: inspect a raw G1 batch before conversion.
- `92_export_v21_to_v30.py`: explicitly export LeRobot v2.1 data to v3.0.

New converted outputs follow `dataset/converted/<contract_id>/<camera-standard>/<batch-or-dataset>/`.
The maintained 618 pipeline derives `<contract_id>` from its selected versioned contract and currently
uses the `wrist_lr` camera standard. Its scratch data and reports are namespaced the same way under
`output/current/<contract_id>/<camera-standard>/`. Do not mix fish-eye and wrist-camera streams in one
camera directory, or datasets with different named state/action orders in one data-standard directory.

Batch-specific 1006, 1008, 1026, 1027, and fisheye construction scripts were removed on 2026-10-29 after their outputs and reports had been generated. Historical artifacts remain under `output/` and `experiments/`; they are not executable pipeline steps. Rebuild new data with the maintained entry points, a versioned contract, and a new output path.

Historical tools may use `[lift, waist_pitch]`, while the EVA contract uses `[waist_pitch, lift]`. Verify source metadata and semantic mappings before using or merging any historical dataset.

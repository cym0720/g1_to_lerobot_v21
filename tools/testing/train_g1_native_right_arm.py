#!/usr/bin/env python3
"""Matched native ACT experiment: right arm + gripper, 8D state and action.

Same images, labels, 16-step chunks, seed and optimizer as the 20D baseline.
Only new experiment copies are projected; original datasets are never edited.
CLI is inherited from the baseline (--input, --output, --dry-run).
"""
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import train_g1_native_lerobot as baseline

SELECT = list(range(8, 16))  # Right arm joints 1..7, then continuous gripper.
original_prepare = baseline.prepare


def project_stats(stats):
    for key in ("observation.state", "action"):
        for name, values in stats[key].items():
            arr = np.asarray(values)
            # count is a scalar count vector, not a per-joint statistic.
            if arr.shape == (20,):
                stats[key][name] = arr[SELECT].tolist()
    return stats


def prepare_right(source, destination):
    original_prepare(source, destination)
    info_path = destination / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        feature["shape"] = [8]
        names = feature.get("names")
        if isinstance(names, list) and len(names) == 20:
            feature["names"] = [names[i] for i in SELECT]
        elif isinstance(names, dict):
            for axis, labels in names.items():
                if len(labels) == 20:
                    names[axis] = [labels[i] for i in SELECT]
    baseline.dump(info_path, info)
    stats_path = destination / "meta/stats.json"
    if stats_path.exists():
        baseline.dump(stats_path, project_stats(json.loads(stats_path.read_text())))
    path = destination / "meta/episodes_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["stats"] = project_stats(row["stats"])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checked = 0
    for path in sorted((destination / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        expected = {}
        for key in ("observation.state", "action"):
            expected[key] = np.asarray(table[key].to_pylist(), dtype=np.float32)[:, SELECT]
            table = table.set_column(table.column_names.index(key), key,
                                     pa.array(expected[key].tolist(), type=pa.list_(pa.float32(), 8)))
        pq.write_table(table, path)
        written = pq.read_table(path)
        for key, value in expected.items():
            np.testing.assert_array_equal(np.asarray(written[key].to_pylist(), dtype=np.float32), value)
        checked += 1
    baseline.dump(destination.parent / "right_arm_projection.json", dict(
        selected_original_dimensions=SELECT, state_dimensions=8, action_dimensions=8,
        checked_episodes=checked, values_exact=True, source_unchanged=True,
        changes="Only state/action dimensions and matching metadata statistics; same three cameras"))
    print("RIGHT_ARM_PROJECTION_VALIDATED", checked, flush=True)


if __name__ == "__main__":
    baseline.OUTPUT = baseline.REPO / "output/lerobot_act_control_20260929/right_arm_train"
    baseline.RIGHT_ARM_DIMS = tuple(range(7))
    baseline.RIGHT_GRIPPER_DIM = 7
    baseline.prepare = prepare_right
    baseline.main()

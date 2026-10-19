"""CPU-only, non-mutating audit of 0923 raw commands and final qpos labels.

Writes JSON, CSV and PNG artifacts into a new output directory. Source datasets,
conversion code, and the training project are never modified. Positive offsets
compare action[t] with state[t + offset], separately within every episode.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
if len(sys.argv) < 2 or sys.argv[1] != "_render":
    import h5py
    import pyarrow.parquet as pq
    from packages.eva_qpos import assemble_g1_qpos
    from packages.raw_preflight import h5_stream_present, timestamps

DEFAULT_INPUT = REPO_ROOT / "output/rebuild_20260924_g1_retasked/final/0923_g1_pick_apple/empty_text_task_1"
DEFAULT_OUTPUT = REPO_ROOT / "output/act_gt_audit_20260929/data_audit/verified"
RAW_ROOT = REPO_ROOT / "dataset/raw/g1/0923_g1_pick_apple"
UNCUT_ROOT = REPO_ROOT / "dataset/converted/rebuild_20260924_g1_body_order/0923_g1_pick_apple/empty_text_task_1"
OFFSETS = tuple(range(-15, 16))
PLOT_EPISODES = (0, 1, 3, 9, 42, 45, 90)
RIGHT_ARM = slice(8, 15)
MOTION_THRESHOLD_RAD = 1e-3


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def matrix(table, key):
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converter_functions():
    """Execute only the actual converter's three numeric alignment functions."""
    path = REPO_ROOT / "tools/conversion/00_convert_618_to_lerobot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"nearest", "h5_series_required", "read_g1_qpos_inputs"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError("converter numeric alignment functions changed")
    namespace = {"np": np, "h5py": h5py, "h5_stream_present": h5_stream_present}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {node.name: node.lineno for node in selected}


def names_of(group):
    return [item.decode() if isinstance(item, bytes) else str(item) for item in group.attrs.get("name", [])]


def normalized_name(name):
    return re.sub(r"^idx\d+_", "", name)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {key: fn(values, axis=0).tolist() for key, fn in (
        ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std), ("ptp", np.ptp)
    )}


def motion_stats(parts):
    values = np.concatenate(parts).astype(np.float64)
    differences = np.concatenate([np.diff(part.astype(np.float64), axis=0) for part in parts if len(part) > 1])
    absolute = np.abs(differences)
    result = stats(values)
    result.update({
        "adjacent_pairs": len(differences),
        "exact_repeat_fraction": (absolute == 0).mean(axis=0).tolist(),
        "all_seven_joints_exact_repeat_fraction": float(np.all(absolute == 0, axis=1).mean()),
        "delta_abs_quantiles": {str(q): np.quantile(absolute, q, axis=0).tolist() for q in (.5, .9, .99)},
        "delta_abs_nonzero_median": [float(np.median(v[v > 0])) if np.any(v > 0) else 0.0 for v in absolute.T],
        "delta_gt_1e3_rad_fraction": (absolute > MOTION_THRESHOLD_RAD).mean(axis=0).tolist(),
        "episode_ptp_median": np.median([np.ptp(part, axis=0) for part in parts], axis=0).tolist(),
    })
    return result


def corr_columns(x, y):
    x = x.astype(np.float64) - x.mean(axis=0)
    y = y.astype(np.float64) - y.mean(axis=0)
    numerator = (x * y).sum(axis=0)
    denominator = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)


def lag_curve(parts, common_support=True):
    rows = []
    for offset in OFFSETS:
        commands, states, dcommands, dstates = [], [], [], []
        for action, state in parts:
            n = len(action)
            if common_support:
                start, end = 15, n - 15
            else:
                start, end = max(0, -offset), min(n, n - offset)
            if end <= start + 1:
                continue
            a = action[start:end].astype(np.float64)
            s = state[start + offset:end + offset].astype(np.float64)
            commands.append(a)
            states.append(s)
            # Differentiate BEFORE concatenation, to exclude episode boundaries.
            dcommands.append(np.diff(a, axis=0))
            dstates.append(np.diff(s, axis=0))
        a, s = np.concatenate(commands), np.concatenate(states)
        da, ds = np.concatenate(dcommands), np.concatenate(dstates)
        mae = np.abs(a - s).mean(axis=0)
        rmse = np.sqrt(np.square(a - s).mean(axis=0))
        correlation = corr_columns(a, s)
        velocity_corr = corr_columns(da, ds)
        rows.append({
            "offset_frames": offset,
            "pairs": len(a),
            "mae_mean_rad": float(mae.mean()),
            "rmse_mean_rad": float(rmse.mean()),
            "position_correlation_mean": float(correlation.mean()),
            "delta_correlation_mean": float(velocity_corr.mean()),
            "mae_per_joint_rad": mae.tolist(),
            "rmse_per_joint_rad": rmse.tolist(),
            "position_correlation_per_joint": correlation.tolist(),
            "delta_correlation_per_joint": velocity_corr.tolist(),
        })
    return rows


def stream_diagnostics(h5, data_key, target, nearest):
    raw_values = np.asarray(h5[data_key])
    raw_times = np.asarray(h5[data_key.rsplit("/", 1)[0] + "/timestamp"], dtype=np.int64)
    count = min(len(raw_values), len(raw_times))
    raw_times = raw_times[:count]
    indices = nearest(raw_times, target)
    differences_ms = (raw_times[indices] - target).astype(np.float64) / 1e6
    intervals_ms = np.diff(raw_times).astype(np.float64) / 1e6
    before, after = target < raw_times[0], target > raw_times[-1]
    return {
        "raw_values_shape": list(raw_values.shape), "raw_timestamp_count": len(raw_times),
        "strictly_increasing_timestamps": bool(np.all(np.diff(raw_times) > 0)),
        "duplicate_timestamps": int(np.sum(np.diff(raw_times) == 0)),
        "decreasing_timestamps": int(np.sum(np.diff(raw_times) < 0)),
        "median_sample_interval_ms": float(np.median(intervals_ms)),
        "p99_sample_interval_ms": float(np.quantile(intervals_ms, .99)),
        "max_sample_interval_ms": float(np.max(intervals_ms)),
        "selected_timestamp_delta_ms_quantiles": {
            str(q): float(np.quantile(differences_ms, q)) for q in (0, .5, .95, 1)
        },
        "in_coverage_camera_frames": int(np.sum(~(before | after))),
        "in_coverage_abs_timestamp_delta_ms_p99": float(np.quantile(np.abs(differences_ms[~(before | after)]), .99)) if np.any(~(before | after)) else None,
        "first_raw_timestamp_ns": int(raw_times[0]), "last_raw_timestamp_ns": int(raw_times[-1]),
        "first_camera_timestamp_ns": int(target[0]), "last_camera_timestamp_ns": int(target[-1]),
        "camera_frames_before_stream": int(before.sum()),
        "camera_frames_after_stream": int(after.sum()),
        "camera_frames_reusing_previous_source_sample": int(np.sum(np.diff(indices) == 0)),
    }, before | after


def command_gap_rows(h5, target, state, episode, nearest):
    raw_times = np.asarray(h5["action/joint/timestamp"], dtype=np.int64)
    raw_positions = np.asarray(h5["action/joint/position"], dtype=np.float64)[:, 7:14]
    gaps = np.diff(raw_times) / 1e6
    nearest_indices = nearest(raw_times, target)
    causal_indices = np.clip(np.searchsorted(raw_times, target, side="right") - 1, 0, len(raw_times) - 1)
    delta_state = np.r_[0., np.max(np.abs(np.diff(state[:, RIGHT_ARM], axis=0)), axis=1)]
    result = []
    for idx in np.flatnonzero(gaps > 100):
        in_gap = (target > raw_times[idx]) & (target < raw_times[idx + 1])
        switched_early = in_gap & (nearest_indices == idx + 1)
        action_step = float(np.max(np.abs(raw_positions[idx + 1] - raw_positions[idx])))
        result.append({
            "episode_index": episode, "raw_sample_before_index": int(idx),
            "raw_gap_ms": float(gaps[idx]), "target_frames_in_gap": int(in_gap.sum()),
            "target_frames_using_next_command_before_its_timestamp": int(switched_early.sum()),
            "raw_command_endpoint_max_delta_rad": action_step,
            "state_motion_frames_gt_1e3_rad_in_gap": int(np.sum(in_gap & (delta_state > MOTION_THRESHOLD_RAD))),
            "first_target_frame_in_gap": int(np.flatnonzero(in_gap)[0]) if np.any(in_gap) else None,
            "last_target_frame_in_gap": int(np.flatnonzero(in_gap)[-1]) if np.any(in_gap) else None,
            "max_future_command_selection_ms": float(np.max((raw_times[nearest_indices[switched_early]] - target[switched_early]) / 1e6)) if np.any(switched_early) else 0.,
            "max_nearest_vs_causal_command_delta_rad_in_gap": float(np.max(np.abs(raw_positions[nearest_indices[in_gap]] - raw_positions[causal_indices[in_gap]]))) if np.any(in_gap) else 0.,
        })
    return result


def write_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit artifacts: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    info = read_json(args.input / "meta/info.json")
    episodes = read_jsonl(args.input / "meta/episodes.jsonl")
    config = read_json(REPO_ROOT / "config/sources/g1.json")
    funcs, function_lines = converter_functions()
    validation_path = UNCUT_ROOT / "conversion_validation.csv"
    trim_path = args.input.parent / "trim_report.csv"
    with validation_path.open(encoding="utf-8-sig", newline="") as file:
        validation = {int(row["episode_index"]): row for row in csv.DictReader(file)}
    with trim_path.open(encoding="utf-8-sig", newline="") as file:
        trims = {int(row["episode"]): row for row in csv.DictReader(file) if row["task"] == args.input.name}
    all_parts, summary_rows, source_checks, plot_parts, gap_rows = [], [], [], {}, []
    expected_names = info["features"]["action.qpos"]["names"]
    expected_arm_names = expected_names[:7] + expected_names[8:15]
    per_episode_lags = []
    index_offset = 0
    for meta in episodes:
        ep = int(meta["episode_index"])
        table_path = args.input / "data/chunk-000" / f"episode_{ep:06d}.parquet"
        table = pq.read_table(table_path)
        action, state = matrix(table, "action.qpos"), matrix(table, "observations.state.qpos")
        n = len(action)
        if action.shape != (n, 20) or state.shape != action.shape or not np.isfinite(action).all() or not np.isfinite(state).all():
            raise RuntimeError(f"invalid vectors: episode {ep}")
        if not np.array_equal(np.asarray(table["frame_index"]), np.arange(n)):
            raise RuntimeError(f"invalid frame indices: episode {ep}")
        if not np.array_equal(np.asarray(table["index"]), np.arange(n) + index_offset):
            raise RuntimeError(f"invalid global indices: episode {ep}")
        index_offset += n
        raw_episode = RAW_ROOT / meta["original_episode_id"]
        target_all = timestamps(raw_episode / "camera/head_color/head_color.txt")
        start, end = int(meta.get("trim_source_start", 0)), int(meta.get("trim_source_end", len(target_all) - 1))
        target = target_all[start:end + 1]
        v, tr = validation[ep], trims[ep]
        source_meta = read_json(raw_episode / "meta_info.json")
        trim_matches = tr["result"] == f"{start}:{end}" or (tr["result"] == "all_static_skip" and n == len(target_all))
        lineage_ok = (
            v["validation"] == "ok" and v["original_episode_id"] == meta["original_episode_id"]
            and int(v["frames"]) == len(target_all) and int(tr["source_frames"]) == len(target_all)
            and len(target) == n == int(meta["length"]) and trim_matches
        )
        with h5py.File(raw_episode / "record/raw_joints.h5", "r") as h5:
            inputs = funcs["read_g1_qpos_inputs"](h5, target, config["raw_h5_adapter"])
            assembly = assemble_g1_qpos(inputs, gripper_convention=config["gripper_convention"],
                action_gripper_convention=config["action_gripper_convention"], allow_state_action_fallback=True)
            source_names = {key: names_of(h5[key]) for key in ("action/joint", "state/joint", "state/waist")}
            names_ok = all([normalized_name(name) for name in source_names[key]] == expected_arm_names for key in ("action/joint", "state/joint"))
            action_diag, action_outside = stream_diagnostics(h5, "action/joint/position", target, funcs["nearest"])
            state_diag, _ = stream_diagnostics(h5, "state/joint/position", target, funcs["nearest"])
            delta = np.r_[0.0, np.max(np.abs(np.diff(state[:, RIGHT_ARM], axis=0)), axis=1)]
            direct_error = np.max(np.abs(action[:, RIGHT_ARM] - assembly.action[:, RIGHT_ARM]), axis=0)
            check = {
                "episode_index": ep, "raw_episode_uuid": meta["original_episode_id"],
                "raw_h5_path": str(raw_episode / "record/raw_joints.h5"),
                "raw_meta_robot_type": source_meta.get("robot_type"),
                "raw_meta_sw_version": source_meta.get("sw_version"),
                "raw_camera_frames": len(target_all), "trim_start_inclusive": start,
                "trim_end_inclusive": end, "final_frames": n, "lineage_mapping_matches": lineage_ok,
                "raw_joint_names": source_names, "raw_and_final_arm_order_matches": names_ok,
                "raw_to_final_right_arm_action_max_abs_error_rad": direct_error.tolist(),
                "raw_to_final_right_arm_state_max_abs_error_rad": np.max(np.abs(state[:, RIGHT_ARM] - assembly.state[:, RIGHT_ARM]), axis=0).tolist(),
                "current_adapter_to_final_all_action_max_abs_error": float(np.max(np.abs(action - assembly.action))),
                "current_adapter_to_final_all_state_max_abs_error": float(np.max(np.abs(state - assembly.state))),
                "action_timestamp_diagnostics": action_diag, "state_timestamp_diagnostics": state_diag,
                "out_of_coverage_command_frames_with_state_motion_gt_1e3_rad": int(np.sum(action_outside & (delta > MOTION_THRESHOLD_RAD))),
                "state_waist_source_first_row": inputs["observation.state.waist_position"][0].tolist(),
                "final_named_lift_waist_pitch_first_row": state[0, 18:20].tolist(),
                "action_outside_coverage_frame_indices": np.flatnonzero(action_outside).tolist(),
                "right_arm_action_episode_ptp_rad": np.ptp(action[:, RIGHT_ARM], axis=0).tolist(),
                "right_arm_state_episode_ptp_rad": np.ptp(state[:, RIGHT_ARM], axis=0).tolist(),
                "source_files_sha256": {str(table_path): sha256(table_path), str(raw_episode / "record/raw_joints.h5"): sha256(raw_episode / "record/raw_joints.h5")},
            }
            gaps = command_gap_rows(h5, target, state, ep, funcs["nearest"])
            gap_rows.extend(gaps)
            check["command_timestamp_gaps_gt_100ms"] = gaps
            if ep in PLOT_EPISODES:
                uncut = pq.read_table(UNCUT_ROOT / "data/chunk-000" / f"episode_{ep:06d}.parquet")
                pre_action, pre_state = matrix(uncut, "action.qpos"), matrix(uncut, "observations.state.qpos")
                all_inputs = funcs["read_g1_qpos_inputs"](h5, target_all, config["raw_h5_adapter"])
                check["uncut_parquet_frames"] = uncut.num_rows
                check["uncut_vs_raw_right_arm_action_max_abs_error_rad"] = float(np.max(np.abs(pre_action[:, RIGHT_ARM] - all_inputs["action.joint_position"][:, 7:14])))
                check["final_vs_uncut_trim_right_arm_action_max_abs_error_rad"] = float(np.max(np.abs(action[:, RIGHT_ARM] - pre_action[start:end+1, RIGHT_ARM])))
                check["final_vs_uncut_trim_right_arm_state_max_abs_error_rad"] = float(np.max(np.abs(state[:, RIGHT_ARM] - pre_state[start:end+1, RIGHT_ARM])))
        source_checks.append(check)
        part = (action[:, RIGHT_ARM], state[:, RIGHT_ARM])
        all_parts.append(part)
        curve = lag_curve([part])
        best = min(curve, key=lambda row: row["mae_mean_rad"])
        zero = next(row for row in curve if row["offset_frames"] == 0)
        per_episode_lags.extend({"episode_index": ep, **row} for row in curve)
        summary_rows.append({
            "episode_index": ep, "raw_episode_uuid": meta["original_episode_id"], "frames": n,
            "lineage_mapping_matches": lineage_ok, "arm_order_matches": names_ok,
            "raw_to_final_action_max_error": float(np.max(direct_error)),
            "best_state_offset_frames": best["offset_frames"], "zero_lag_mae_rad": zero["mae_mean_rad"],
            "best_lag_mae_rad": best["mae_mean_rad"],
            "action_mean_episode_ptp_rad": float(np.ptp(part[0], axis=0).mean()),
            "state_mean_episode_ptp_rad": float(np.ptp(part[1], axis=0).mean()),
            "meaningful_motion_for_lag_estimation": bool(np.max(np.ptp(part[1], axis=0)) > .01),
            "action_outside_coverage_frames": action_diag["camera_frames_before_stream"] + action_diag["camera_frames_after_stream"],
            "action_outside_coverage_moving_frames": check["out_of_coverage_command_frames_with_state_motion_gt_1e3_rad"],
        })
        if ep in PLOT_EPISODES:
            plot_parts[ep] = (action, state, meta, check, best["offset_frames"])
        if ep % 20 == 0:
            print(f"audited episode {ep}: raw/final action max error={np.max(direct_error):.9g}, best future-state offset={best['offset_frames']}", flush=True)

    global_lags = lag_curve(all_parts)
    global_best = min(global_lags, key=lambda row: row["mae_mean_rad"])
    global_zero = next(row for row in global_lags if row["offset_frames"] == 0)
    lag_hist = {str(offset): sum(row["best_state_offset_frames"] == offset for row in summary_rows) for offset in OFFSETS}
    positive_best = min((row for row in global_lags if row["offset_frames"] > 0), key=lambda row: row["mae_mean_rad"])
    known_sources = [REPO_ROOT / "packages/eva_qpos.py", REPO_ROOT / "tools/conversion/00_convert_618_to_lerobot.py",
        REPO_ROOT / "config/sources/g1.json", REPO_ROOT / "src/clean_tools/02_trim_arm_static_frames.py", validation_path, trim_path]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU-only raw/final data audit; no dataset/training changes; no model inference",
        "input": str(args.input.resolve()), "output": str(args.output.resolve()), "episodes": len(episodes), "frames": index_offset,
        "right_arm_final_indices": list(range(8, 15)), "right_arm_raw_joint_indices": list(range(7, 14)),
        "offset_definition": "Compare action[t] with state[t+offset]; positive means command precedes state. No pairing or differences across episode boundaries.",
        "offset_support": "Same t in [15,episode_length-15) at every offset to remove different-support bias",
        "converter_function_lines": function_lines, "source_hashes": {str(path): sha256(path) for path in known_sources},
        "integrity": {
            "total_frames_matches_info": index_offset == info["total_frames"],
            "total_episodes_matches_info": len(episodes) == info["total_episodes"],
            "all_lineage_mappings_match": all(row["lineage_mapping_matches"] for row in source_checks),
            "all_arm_orders_match": all(row["raw_and_final_arm_order_matches"] for row in source_checks),
            "all_right_arm_actions_exactly_match_raw_nearest_command": all(not any(row["raw_to_final_right_arm_action_max_abs_error_rad"]) for row in source_checks),
            "all_right_arm_states_exactly_match_raw_nearest_state": all(not any(row["raw_to_final_right_arm_state_max_abs_error_rad"]) for row in source_checks),
            "all_actions_match_current_adapter": all(row["current_adapter_to_final_all_action_max_abs_error"] < 1e-6 for row in source_checks),
            "all_states_match_current_adapter": all(row["current_adapter_to_final_all_state_max_abs_error"] == 0 for row in source_checks),
        },
        "right_arm_command_motion": motion_stats([a for a, _ in all_parts]),
        "right_arm_state_motion": motion_stats([s for _, s in all_parts]),
        "global_offset_curve": global_lags, "zero_offset": global_zero, "best_offset": global_best,
        "per_episode_best_offset_histogram": lag_hist,
        "per_episode_best_offset_histogram_moving_only": {str(offset): sum(row["best_state_offset_frames"] == offset and row["meaningful_motion_for_lag_estimation"] for row in summary_rows) for offset in OFFSETS},
        "per_joint_best_position_mae_offsets": [min(global_lags, key=lambda row: row["mae_per_joint_rad"][j])["offset_frames"] for j in range(7)],
        "per_joint_best_delta_correlation_offsets": [max(global_lags, key=lambda row: row["delta_correlation_per_joint"][j])["offset_frames"] for j in range(7)],
        "future_state_candidate": {
            "best_positive_offset_frames": positive_best["offset_frames"],
            "best_positive_offset_seconds": positive_best["offset_frames"] / info["fps"],
            "command_vs_future_state_mae_rad": positive_best["mae_mean_rad"],
            "mae_reduction_vs_same_frame_fraction": 1 - positive_best["mae_mean_rad"] / global_zero["mae_mean_rad"],
            "interpretation": "A nonzero lag alone is consistent with actuator response delay and does not prove a label bug. Future measured state changes command semantics; compare in a new dataset/experiment and drop out-of-episode future targets rather than pairing across episodes.",
        },
        "source_checks": source_checks,
        "boundary_clamping": {
            "total_action_outside_coverage_frames": sum(row["action_outside_coverage_frames"] for row in summary_rows),
            "total_outside_coverage_moving_frames": sum(row["action_outside_coverage_moving_frames"] for row in summary_rows),
            "interpretation": "nearest() extends the first/last raw command outside its timestamp coverage; verify moving boundary frames before treating this as acceptable padding.",
        },
        "long_command_gaps": {
            "threshold_ms": 100, "gaps_count": len(gap_rows),
            "gaps_with_nontrivial_endpoint_delta_gt_1e3_rad": sum(row["raw_command_endpoint_max_delta_rad"] > 1e-3 for row in gap_rows),
            "total_target_frames_in_gaps": sum(row["target_frames_in_gap"] for row in gap_rows),
            "total_target_frames_using_next_command_early": sum(row["target_frames_using_next_command_before_its_timestamp"] for row in gap_rows),
            "interpretation": "A recording gap can legitimately hold the last target. nearest() instead changes to the next message halfway through a gap; timestamp lead and endpoint changes are quantified in raw_command_gaps.csv. This is a measured acausal alignment behavior, not proof the recorder's commands or training are wrong.",
        },
        "body_order_separate_finding": {
            "source_waist_names_variants": sorted({tuple(row["raw_joint_names"]["state/waist"]) for row in source_checks}),
            "current_adapter_swaps_source_body_columns": True,
            "source_config_claim": config["qpos_order_confirmation"]["body_source_order"],
            "final_named_order": expected_names[18:20],
            "interpretation": "If raw named attrs identify [joint_body_pitch,joint_lift_body], the current [1,0] swap contradicts those names. This is a body-index issue, independent of the right arm. Confirm raw recorder names/calibration before changing that unrelated pipeline branch.",
        },
        "gripper_provenance": {
            "final_info_conversion_provenance": info.get("conversion_provenance"),
            "current_source_config_action_gripper_convention": config["action_gripper_convention"],
            "interpretation": "final metadata may retain the pre-repair raw_encoder action convention; exact value reconstruction tests the current repaired numerics separately.",
        },
        "limitations": [
            "No recorder source code is supplied; raw named action fields and HAL command-topic logs establish available provenance but cannot certify each original message was actually executed.",
            "Global position correlation is partly influenced by similar episode-level poses; per-episode lag curves and within-episode delta correlations are also exported.",
            "This report cannot diagnose model precision, loss decomposition or prediction collapse without the main agent's checkpoint inference.",
        ],
        "ACT_smoke": "skipped: read-only diagnostic artifact; GPU prohibited by request",
    }
    (args.output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "episode_summary.csv", summary_rows)
    write_csv(args.output / "per_episode_offset_curve.csv", per_episode_lags)
    write_csv(args.output / "global_offset_curve.csv", global_lags)
    write_csv(args.output / "raw_command_gaps.csv", gap_rows)

    plot_payload = {str(ep): {"action": action.tolist(), "state": state.tolist(), "meta": meta,
        "check": check, "best_offset": best_offset} for ep, (action, state, meta, check, best_offset) in plot_parts.items()}
    (args.output / "plot_inputs.json").write_text(json.dumps(plot_payload, ensure_ascii=False), encoding="utf-8")
    # Reuse an existing compatible CPU plotting environment; never install/change dependencies.
    render_python = Path("D:/Anaconda/Anaconda/envs/x-anylabeling/python.exe")
    subprocess.run([str(render_python), str(Path(__file__).resolve()), "_render", str(args.output)], check=True)
    print(json.dumps({"output": str(args.output), "episodes": len(episodes), "frames": index_offset,
        "integrity": report["integrity"], "best_offset": global_best, "future_state_candidate": report["future_state_candidate"],
        "boundary_clamping": report["boundary_clamping"]}, indent=2), flush=True)


def render_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = read_json(output / "audit_report.json")
    payload = read_json(output / "plot_inputs.json")
    global_lags, global_best = report["global_offset_curve"], report["best_offset"]
    expected_names = [f"right_arm_joint{j}" for j in range(1, 8)] + ["right_gripper"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for j in range(7):
        axes[0].plot(OFFSETS, [row["mae_per_joint_rad"][j] for row in global_lags], label=f"joint{j+1}")
        axes[1].plot(OFFSETS, [row["delta_correlation_per_joint"][j] for row in global_lags], label=f"joint{j+1}")
    for axis in axes:
        axis.axvline(global_best["offset_frames"], color="black", linestyle="--", alpha=.5)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Command vs measured state MAE (rad)")
    axes[0].legend(ncol=4)
    axes[1].set_ylabel("Within-episode frame delta correlation")
    axes[1].set_xlabel("State offset (frames): action[t] vs state[t+offset]")
    fig.suptitle(f"All {report['episodes']} episodes / {report['frames']} frames; same support at every offset")
    fig.tight_layout()
    fig.savefig(output / "right_arm_lag_curve.png", dpi=160)
    plt.close(fig)

    for ep_string, part in payload.items():
        ep = int(ep_string)
        action, state = np.asarray(part["action"]), np.asarray(part["state"])
        meta, check, best_offset = part["meta"], part["check"], part["best_offset"]
        fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
        for axis, joint in zip(axes.flat, range(8, 16)):
            axis.plot(action[:, joint], color="tab:blue", label="Final action = raw command", linewidth=1.5)
            axis.plot(state[:, joint], color="tab:orange", label="Measured state (same frame)", linewidth=1.2)
            t = np.arange(max(0, -best_offset), min(len(action), len(action) - best_offset))
            axis.plot(t, state[t + best_offset, joint], color="tab:green", linestyle="--", label=f"Measured state at t{best_offset:+d}", linewidth=1)
            outside = np.asarray(check["action_outside_coverage_frame_indices"], dtype=int)
            if len(outside):
                cuts = np.r_[0, np.flatnonzero(np.diff(outside) > 1) + 1, len(outside)]
                for left, right in zip(cuts[:-1], cuts[1:]):
                    axis.axvspan(outside[left] - .5, outside[right-1] + .5, color="gray", alpha=.12)
            for gap in check["command_timestamp_gaps_gt_100ms"]:
                if gap["first_target_frame_in_gap"] is not None:
                    axis.axvspan(gap["first_target_frame_in_gap"] - .5, gap["last_target_frame_in_gap"] + .5, color="red", alpha=.08)
            axis.set_title(expected_names[joint - 8])
            axis.set_ylabel("rad" if joint != 15 else "closed fraction")
            axis.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel("Final frame index")
        fig.suptitle(f"Episode {ep:06d}, raw {meta['original_episode_id']}\ntrim [{check['trim_start_inclusive']},{check['trim_end_inclusive']}], command reconstruction max error={max(check['raw_to_final_right_arm_action_max_abs_error_rad']):.3g} rad")
        fig.tight_layout()
        fig.savefig(output / f"episode_{ep:06d}_right_arm_command_state.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_render":
        render_figures(Path(sys.argv[2]))
    else:
        main()

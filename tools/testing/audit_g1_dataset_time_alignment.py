"""CPU-only read audit of three G1 final datasets, labels and video clocks.

No datasets, environments, or training projects are changed. Reports go to a new
directory. ffprobe demuxes H.264 presentation timestamps without hardware decoding.
Positive lag means comparing action[t] against measured state[t + lag].
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from packages.eva_qpos import assemble_g1_qpos
from packages.raw_preflight import h5_stream_present, timestamps

DEFAULT_INPUT = REPO / "output/rebuild_20260924_g1_retasked/final"
DEFAULT_OUTPUT = REPO / "output/lerobot_act_control_20260929/dataset_audit"
DATASETS = ("0923_g1_pick_apple", "618data", "record_0904")
RIGHT = slice(8, 15)
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
GRIPPERS = (7, 15)
CHUNK = 16
LAGS = tuple(range(-32, 33))
MOTION = 1e-3
VIDEO_WORKERS = 2
PROBE_TIMEOUT_SECONDS = 45
RAW_GAP_MS = 100
PREVIOUS_0923 = REPO / "output/act_gt_audit_20260929/data_audit/verified/audit_report.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(table, key, vector=False):
    values = table[key].combine_chunks()
    return np.asarray(values.to_pylist() if vector else values.to_numpy())


def add_issue(rows, dataset, task, episode, category, severity, detail, camera=""):
    rows.append({"dataset": dataset, "task": task, "episode": episode, "camera": camera,
        "category": category, "severity": severity, "detail": detail})


def actual_converter_functions():
    path = REPO / "tools/conversion/00_convert_618_to_lerobot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    desired = {"nearest", "h5_series_required", "read_g1_qpos_inputs"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in desired]
    if {node.name for node in selected} != desired:
        raise RuntimeError("converter alignment functions cannot be located")
    namespace = {"np": np, "h5py": h5py, "h5_stream_present": h5_stream_present}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {node.name: node.lineno for node in selected}


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    denominator = np.sqrt(np.sum(x*x, axis=0) * np.sum(y*y, axis=0))
    return np.divide(np.sum(x*y, axis=0), denominator, out=np.zeros(x.shape[1]), where=denominator > 1e-20)


def lag_parts(action, state):
    """Same moving target frames for every lag; never cross episode boundaries."""
    n = len(action)
    if n <= 65:
        return []
    da = np.diff(action.astype(np.float64), axis=0, prepend=action[:1])
    ds = np.diff(state.astype(np.float64), axis=0, prepend=state[:1])
    moving = np.maximum(np.max(np.abs(da), axis=1), np.max(np.abs(ds), axis=1)) > MOTION
    t = np.arange(32, n - 32)
    t = t[moving[t]]
    if not len(t):
        return []
    return [(lag, action[t], state[t+lag], da[t], ds[t+lag]) for lag in LAGS]


def metrics(a, s, da, ds):
    a, s = np.asarray(a, dtype=np.float64), np.asarray(s, dtype=np.float64)
    mae = np.abs(a-s).mean(axis=0)
    rmse = np.sqrt(np.square(a-s).mean(axis=0))
    return {"pairs": len(a), "mae_mean_rad": float(mae.mean()), "rmse_mean_rad": float(rmse.mean()),
        "mae_per_joint_rad": mae.tolist(), "rmse_per_joint_rad": rmse.tolist(),
        "position_correlation_per_joint": correlation(a.copy(), s.copy()).tolist(),
        "delta_correlation_per_joint": correlation(np.asarray(da).copy(), np.asarray(ds).copy()).tolist()}


def chunk_stats(action):
    if len(action) < CHUNK:
        return {"valid_16_frame_windows": 0}
    windows = np.lib.stride_tricks.sliding_window_view(action, CHUNK, axis=0)
    spans = np.ptp(windows.astype(np.float64), axis=-1)
    right = spans[:, RIGHT]
    return {"valid_16_frame_windows": len(spans),
        "all_20_dimensions_exactly_constant_windows": int(np.sum(np.all(spans == 0, axis=1))),
        "right_arm_exactly_constant_windows": int(np.sum(np.all(right == 0, axis=1))),
        "right_arm_max_span_le_1e3_rad_windows": int(np.sum(np.max(right, axis=1) <= MOTION)),
        "right_arm_max_span_gt_1e3_rad_windows": int(np.sum(np.max(right, axis=1) > MOTION)),
        "per_dimension_span_median": np.median(spans, axis=0).tolist(),
        "per_dimension_span_p90": np.quantile(spans, .9, axis=0).tolist(),
        "per_dimension_span_max": np.max(spans, axis=0).tolist(),
        "right_arm_mean_joint_span_median_rad": float(np.median(np.mean(right, axis=1))),
        "right_arm_mean_joint_span_p90_rad": float(np.quantile(np.mean(right, axis=1), .9)),
        "right_arm_max_joint_span_median_rad": float(np.median(np.max(right, axis=1))),
        "window_semantics": "Every complete real action[t:t+16], no end padding and no episode crossing"}


def stream_stats(times, target, nearest):
    times = np.asarray(times, dtype=np.int64)
    if not len(times):
        return {"samples": 0}, None
    selected = nearest(times, target)
    delta = (times[selected] - target).astype(np.float64) / 1e6
    covered = (target >= times[0]) & (target <= times[-1])
    intervals = np.diff(times).astype(np.float64) / 1e6
    result = {"samples": len(times), "first_timestamp_ns": int(times[0]), "last_timestamp_ns": int(times[-1]),
        "duplicates": int(np.sum(np.diff(times) == 0)), "decreasing": int(np.sum(np.diff(times) < 0)),
        "interval_median_ms": float(np.median(intervals)) if len(intervals) else None,
        "interval_p99_ms": float(np.quantile(intervals, .99)) if len(intervals) else None,
        "interval_max_ms": float(np.max(intervals)) if len(intervals) else None,
        "before_source_frames": int(np.sum(target < times[0])), "after_source_frames": int(np.sum(target > times[-1])),
        "covered_target_frames": int(np.sum(covered)), "repeated_source_indices": int(np.sum(np.diff(selected) == 0)),
        "nearest_delta_ms_min": float(delta.min()), "nearest_delta_ms_max": float(delta.max()),
        "nearest_abs_delta_ms_p99_covered": float(np.quantile(np.abs(delta[covered]), .99)) if np.any(covered) else None,
        "gaps_gt_100ms": int(np.sum(intervals > RAW_GAP_MS))}
    return result, selected


def raw_audit(dataset, task, meta, action, state, config, funcs):
    """Reconstruct frame selection from original device timestamps, not CFR output."""
    uuid = meta.get("original_episode_id")
    path = REPO / "dataset/raw/g1" / dataset / str(uuid)
    result = {"raw_episode_uuid": uuid, "expected_raw_path": str(path), "raw_path_confirmed": False}
    h5_path, head_path = path / "record/raw_joints.h5", path / "camera/head_color/head_color.txt"
    if not uuid or not h5_path.is_file() or not head_path.is_file():
        result["missing_raw_files"] = [str(p) for p in (h5_path, head_path) if not p.is_file()]
        return result
    result["raw_path_confirmed"] = True
    target_all = timestamps(head_path)
    start = int(meta.get("trim_source_start", 0))
    end = int(meta.get("trim_source_end", len(target_all)-1))
    target = target_all[start:end+1]
    result.update({"source_head_frames": len(target_all), "trim_start": start, "trim_end_inclusive": end,
        "trim_count_matches_final": len(target) == len(action), "streams": {}})
    if len(target) != len(action):
        return result
    nearest = funcs["nearest"]
    head_diagnostics, _ = stream_stats(target_all, target, nearest)
    result["streams"]["camera/head_color"] = head_diagnostics
    result["source_head_target_is_exact_trim_slice"] = True
    result["source_head_vs_nominal_30fps_elapsed_max_error_ms"] = float(np.max(np.abs((target-target[0])/1e6 - np.arange(len(target))*1000/30)))
    for camera in ("hand_left", "hand_right"):
        files = sorted((path / "camera" / camera / "color").glob("*.jpg"))
        if files:
            camera_times = np.asarray([int(file.stem) for file in files], dtype=np.int64)
            diag, _ = stream_stats(camera_times, target, nearest)
            result["streams"]["camera/" + camera] = diag
        else:
            result["streams"]["camera/" + camera] = {"samples": 0, "missing_path": str(path / "camera" / camera / "color")}
    with h5py.File(h5_path, "r") as h5:
        for key in ("action/joint", "state/joint", "action/left_effector", "action/right_effector", "state/left_effector", "state/right_effector"):
            if key + "/timestamp" in h5:
                result["streams"][key], _ = stream_stats(h5[key + "/timestamp"], target, nearest)
        named = {}
        for key in ("action/joint", "state/joint"):
            named[key] = [v.decode() if isinstance(v, bytes) else str(v) for v in h5[key].attrs.get("name", [])]
        result["raw_joint_names"] = named
        raw_inputs = funcs["read_g1_qpos_inputs"](h5, target_all, config["raw_h5_adapter"])
        assembly = assemble_g1_qpos(raw_inputs, gripper_convention=config["gripper_convention"],
            action_gripper_convention=config["action_gripper_convention"], allow_state_action_fallback=True)
        expected_a, expected_s = assembly.action[start:end+1], assembly.state[start:end+1]
        result["same_frame_raw_to_final_right_arm_action_max_error_rad"] = float(np.max(np.abs(expected_a[:, RIGHT] - action[:, RIGHT])))
        result["same_frame_raw_to_final_right_arm_state_max_error_rad"] = float(np.max(np.abs(expected_s[:, RIGHT] - state[:, RIGHT])))
        result["same_frame_raw_to_final_all_arm_action_max_error_rad"] = float(np.max(np.abs(expected_a[:, ARM_INDICES] - action[:, ARM_INDICES])))
        result["same_frame_raw_to_final_all_arm_state_max_error_rad"] = float(np.max(np.abs(expected_s[:, ARM_INDICES] - state[:, ARM_INDICES])))
        result["same_frame_raw_to_final_all_qpos_action_max_error"] = float(np.max(np.abs(expected_a - action)))
        result["same_frame_raw_to_final_all_qpos_state_max_error"] = float(np.max(np.abs(expected_s - state)))
        t = np.arange(16, len(action)-16)
        shifted = {}
        for offset in (-16, 0, 16):
            if len(t):
                aa, ss = assembly.action[start+t+offset], assembly.state[start+t+offset]
                shifted[str(offset)] = {
                    "pairs": len(t), "right_arm_action_mae_rad": float(np.abs(action[t, RIGHT]-aa[:, RIGHT]).mean()),
                    "all_arm_action_mae_rad": float(np.abs(action[t][:, ARM_INDICES]-aa[:, ARM_INDICES]).mean()),
                    "right_arm_state_mae_rad": float(np.abs(state[t, RIGHT]-ss[:, RIGHT]).mean()),
                    "all_arm_state_mae_rad": float(np.abs(state[t][:, ARM_INDICES]-ss[:, ARM_INDICES]).mean())}
        result["raw_frame_offset_reconstruction"] = shifted
        result["all_arm_motion_is_significant"] = bool(np.max(np.ptp(state[:, ARM_INDICES], axis=0)) > .01)
    return result


def probe_video(job):
    dataset, task, ep, camera, path, fps, expected_rows, shape, executable = job
    base = {"dataset": dataset, "task": task, "episode": ep, "camera": camera,
        "path": str(path), "expected_frames": expected_rows, "expected_fps": fps}
    if not path.is_file():
        return {**base, "error": "missing_video"}
    command = [executable, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_packets", "-show_streams",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames,nb_read_packets:packet=pts,dts,duration",
        "-of", "json", str(path)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
        if process.returncode:
            return {**base, "error": "ffprobe_failed", "error_detail": process.stderr[:1200]}
        data = json.loads(process.stdout)
        streams = data.get("streams", [])
        if len(streams) != 1:
            return {**base, "error": "ffprobe_stream_count", "stream_count": len(streams)}
        stream = streams[0]
        packets = data.get("packets", [])
        timescale = Fraction(stream["time_base"])
        pts = [int(p["pts"]) for p in packets if "pts" in p]
        dts = [int(p["dts"]) for p in packets if "dts" in p]
        sorted_pts = np.asarray(sorted(pts), dtype=np.int64)
        grid = np.arange(len(sorted_pts), dtype=np.float64) / fps
        seconds = sorted_pts.astype(np.float64) * float(timescale)
        intervals = np.diff(seconds)
        tick_tolerance = max(float(timescale) * 1.1, 1e-8)
        header_frames = stream.get("nb_frames")
        header_frames = int(header_frames) if header_frames and header_frames != "N/A" else None
        count_packets = int(stream.get("nb_read_packets", len(packets)))
        missing_pts = len(packets)-len(pts)
        non_grid = int(np.sum(np.abs(seconds-grid) > tick_tolerance))
        bad_cfr = int(np.sum(np.abs(intervals - 1/fps) > tick_tolerance))
        result = {**base, "error": None, "codec": stream.get("codec_name"), "height": stream.get("height"), "width": stream.get("width"),
            "metadata_nb_frames": header_frames, "packet_count": count_packets, "packet_pts_count": len(pts),
            "unique_pts_count": len(set(pts)), "time_base": str(timescale),
            "avg_fps": float(Fraction(stream.get("avg_frame_rate", "0/1"))),
            "r_fps": float(Fraction(stream.get("r_frame_rate", "0/1"))),
            "first_presentation_pts_sec": float(seconds[0]) if len(seconds) else None,
            "last_presentation_pts_sec": float(seconds[-1]) if len(seconds) else None,
            "stream_start_time_sec": float(stream.get("start_time", 0)),
            "stream_duration_sec": float(stream.get("duration", 0)),
            "last_expected_frame_pts_sec": (expected_rows-1)/fps,
            "presentation_grid_max_error_sec": float(np.max(np.abs(seconds-grid))) if len(seconds) else None,
            "presentation_grid_bad_frames": non_grid, "cfr_bad_intervals": bad_cfr,
            "pts_missing_packets": missing_pts, "pts_duplicates": len(pts)-len(set(pts)),
            "packet_order_pts_reorders": int(np.sum(np.diff(pts) < 0)) if len(pts)>1 else 0,
            "packet_order_dts_decreases": int(np.sum(np.diff(dts) < 0)) if len(dts)>1 else 0,
            "cfr_interval_min_sec": float(intervals.min()) if len(intervals) else None,
            "cfr_interval_max_sec": float(intervals.max()) if len(intervals) else None,
            "probe_method": "stream metadata + demuxed H.264 access-unit packet count + sorted integer PTS grid; no pixel decoding"}
        errors = []
        if header_frames is not None and header_frames != expected_rows:
            errors.append("video_header_frame_count_mismatch")
        if count_packets != expected_rows or len(sorted_pts) != expected_rows:
            errors.append("video_packet_frame_count_mismatch")
        if missing_pts or result["pts_duplicates"]:
            errors.append("video_missing_or_duplicate_pts")
        if non_grid or bad_cfr:
            errors.append("video_noncanonical_pts_or_non_cfr")
        if abs(result["avg_fps"]-fps) > 1e-6 or abs(result["r_fps"]-fps) > 1e-6:
            errors.append("video_fps_mismatch")
        if shape and [result["height"], result["width"]] != shape[:2]:
            errors.append("video_shape_metadata_mismatch")
        if result["packet_order_dts_decreases"]:
            errors.append("video_dts_decreasing")
        result["error_categories"] = errors
        return result
    except Exception as exc:
        return {**base, "error": "ffprobe_exception", "error_detail": f"{type(exc).__name__}: {exc}"}


def write_csv(path, rows):
    if not rows:
        path.write_text("dataset,task,episode,category,severity,detail\n", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict,list)) else value for key,value in row.items()})


def make_figures(report_dir):
    """Reuse the existing CPU plotting environment, without changing dependencies."""
    plot_python = Path("D:/Anaconda/Anaconda/envs/x-anylabeling/python.exe")
    if not plot_python.is_file():
        return {"generated": False, "reason": "compatible existing plotting interpreter unavailable"}
    code = '''import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(sys.argv[1])
r=json.loads((p/"audit_report.json").read_text(encoding="utf-8"))
curves=list(csv.DictReader((p/"right_arm_global_lag_curves.csv").open(encoding="utf-8")))
fig,axes=plt.subplots(2,1,figsize=(11,10))
for summary in r["datasets"]:
    rows=[v for v in curves if v["dataset"]==summary["dataset"]]
    x=[int(v["offset_frames"]) for v in rows]
    y=[float(v["mae_mean_rad"]) for v in rows]
    axes[0].plot(x,y,label=summary["dataset"])
for offset in (-16,16): axes[0].axvline(offset,color="gray",linestyle=":",alpha=.6)
axes[0].axvline(3,color="black",linestyle="--",alpha=.5,label="+3 frames")
axes[0].set_xlabel("State offset: action[t] vs state[t+offset] (frames)")
axes[0].set_ylabel("Moving right-arm MAE (rad)")
axes[0].set_title("All three datasets: moving-frame support, no episode crossing")
axes[0].grid(alpha=.2)
axes[0].legend()
names=[s["dataset"] for s in r["datasets"]]
total=np.array([s["valid_real_action16_windows"] for s in r["datasets"]])
constant=np.array([s["right_arm_constant_action16_windows"] for s in r["datasets"]])/total*100
moving=np.array([s["right_arm_moving_action16_windows"] for s in r["datasets"]])/total*100
tiny=100-constant-moving
axes[1].bar(names,constant,label="Right arm exactly constant",color="gray")
axes[1].bar(names,tiny,bottom=constant,label="Nonzero span <=0.001 rad",color="orange")
axes[1].bar(names,moving,bottom=constant+tiny,label="Span >0.001 rad",color="seagreen")
axes[1].set_ylabel("Complete real 16-frame action windows (%)")
axes[1].set_ylim(0,110)
axes[1].set_title("Static right-arm windows can be legitimate in left-arm tasks")
axes[1].legend(loc="upper left",fontsize=8)
fig.tight_layout()
fig.savefig(p/"right_arm_lag_and_action16.png",dpi=160)
plt.close(fig)
'''
    subprocess.run([str(plot_python), "-c", code, str(report_dir)], check=True)
    return {"generated": True, "path": str(report_dir / "right_arm_lag_and_action16.png")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit artifacts: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required; no environment changes are performed")
    funcs, source_lines = actual_converter_functions()
    config = read_json(REPO / "config/sources/g1.json")
    input_tasks = []
    for dataset in DATASETS:
        root = args.input / dataset
        if not root.is_dir():
            raise FileNotFoundError(root)
        for info_path in sorted(root.rglob("meta/info.json")):
            task_path = info_path.parent.parent
            if (task_path / "data").is_dir():
                input_tasks.append((dataset, task_path, info_path))
    input_manifest = [{"dataset": dataset, "task": str(task_path.relative_to(args.input/dataset)), "path": str(task_path.resolve()),
        "info_sha256": digest(info_path)} for dataset, task_path, info_path in input_tasks]
    (args.output / "input_manifest.json").write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input_tasks": input_manifest, "ffprobe": ffprobe}, ensure_ascii=False), flush=True)
    all_episodes, issues, task_rows, video_jobs, lag_rows, raw_rows = [], [], [], [], [], []
    lag_buffers = {dataset: {lag: [] for lag in LAGS} for dataset in DATASETS}
    lag_hist = {dataset: Counter() for dataset in DATASETS}
    hashes = {}
    for dataset, task_path, info_path in input_tasks:
        task = str(task_path.relative_to(args.input/dataset))
        info = read_json(info_path)
        metas = read_jsonl(task_path / "meta/episodes.jsonl")
        task_records = read_jsonl(task_path / "meta/tasks.jsonl")
        valid_tasks = {int(row["task_index"]) for row in task_records}
        task_text_map = {int(row["task_index"]): row["task"] for row in task_records}
        fps = float(info["fps"])
        cameras = {key: spec for key,spec in info["features"].items() if spec.get("dtype")=="video"}
        declared_state = tuple(info["features"]["observations.state.qpos"]["shape"])
        declared_action = tuple(info["features"]["action.qpos"]["shape"])
        chunk_size = int(info.get("chunks_size", 1000))
        expected_global, total_rows = 0, 0
        initial_issue_count = len(issues)
        seen = set()
        for position, meta in enumerate(metas):
            ep = int(meta["episode_index"])
            if ep in seen or ep != position:
                add_issue(issues,dataset,task,ep,"metadata_episode_index_noncontiguous","error",f"position={position}; episode={ep}; duplicate={ep in seen}")
            seen.add(ep)
            path = task_path / info["data_path"].format(episode_chunk=ep//chunk_size, episode_index=ep)
            record = {"dataset":dataset,"task":task,"episode":ep,"parquet_path":str(path),"raw_episode_uuid":meta.get("original_episode_id"),"fps":fps}
            episode_issue_start = len(issues)
            try:
                hashes[str(path)] = digest(path)
                table = pq.read_table(path)
                n = table.num_rows
                record["frames"] = n
                if n != int(meta["length"]):
                    add_issue(issues,dataset,task,ep,"parquet_length_vs_metadata","error",f"rows={n}; metadata={meta['length']}")
                frames = array(table,"frame_index")
                episode_indices, global_indices, task_indices = (array(table,key) for key in ("episode_index","index","task_index"))
                ts = array(table,"timestamp")
                expected_ts = np.arange(n,dtype=np.float32)/np.float32(fps)
                strict_ts = bool(np.all(np.diff(ts.astype(np.float64)) > 0)) if n>1 else True
                exact_ts = bool(np.array_equal(ts,expected_ts))
                record.update({"timestamp_dtype":str(ts.dtype),"timestamp_exact_float32_frame_div_fps":exact_ts,
                    "timestamp_strictly_monotonic":strict_ts,"timestamp_expected_max_error_sec":float(np.max(np.abs(ts-expected_ts))) if n else None,
                    "timestamp_mathematical_rounding_max_error_sec":float(np.max(np.abs(ts.astype(np.float64)-np.arange(n)/fps))) if n else None,
                    "timestamp_frame_interval_max_error_sec":float(np.max(np.abs(np.diff(ts.astype(np.float64))-1/fps))) if n>1 else 0.})
                tests = {
                    "timestamp_noncanonical":exact_ts,
                    "timestamp_not_monotonic":strict_ts,
                    "frame_index_noncontiguous":np.array_equal(frames,np.arange(n)),
                    "episode_index_not_constant":np.all(episode_indices==ep),
                    "global_index_noncontiguous":np.array_equal(global_indices,np.arange(n)+expected_global),
                    "task_index_unknown":set(map(int,task_indices)).issubset(valid_tasks),
                    "metadata_global_bounds_mismatch":int(meta["dataset_from_index"])==expected_global and int(meta["dataset_to_index"])==expected_global+n,
                    "task_prompt_mismatch":all(task_text_map[int(idx)] in meta.get("tasks",[]) for idx in set(task_indices) if int(idx) in valid_tasks),
                }
                for category,ok in tests.items():
                    if not ok:
                        add_issue(issues,dataset,task,ep,category,"error",f"rows={n}, expected_global_start={expected_global}")
                state, action = array(table,"observations.state.qpos",True), array(table,"action.qpos",True)
                for key,values,declared in (("state",state,declared_state),("action",action,declared_action)):
                    record[key+"_shape"] = list(values.shape)
                    if values.shape != (n,*declared) or values.shape != (n,20):
                        add_issue(issues,dataset,task,ep,key+"_shape_invalid","error",f"actual={values.shape}, declared={declared}")
                    if not np.isfinite(values).all():
                        add_issue(issues,dataset,task,ep,key+"_nonfinite","error",f"nonfinite_values={np.sum(~np.isfinite(values))}")
                    grip = values[:,GRIPPERS]
                    record[key+"_gripper_min"] = grip.min(axis=0).tolist()
                    record[key+"_gripper_max"] = grip.max(axis=0).tolist()
                    if np.any(grip < -1e-6) or np.any(grip > 1+1e-6):
                        add_issue(issues,dataset,task,ep,key+"_gripper_out_of_0_1_range","error",f"min={grip.min(axis=0)}, max={grip.max(axis=0)}")
                record["action_16frame_windows"] = chunk_stats(action)
                a, s = action[:,RIGHT], state[:,RIGHT]
                d_a, d_s = np.diff(a,axis=0), np.diff(s,axis=0)
                effective = np.maximum(np.max(np.abs(d_a),axis=1),np.max(np.abs(d_s),axis=1)) > MOTION if n>1 else np.empty(0,dtype=bool)
                record["right_arm_effective_motion_transitions"] = int(np.sum(effective))
                record["right_arm_action_ptp_rad"] = np.ptp(a,axis=0).tolist()
                record["right_arm_state_ptp_rad"] = np.ptp(s,axis=0).tolist()
                parts = lag_parts(a,s)
                curve = []
                for lag,aa,ss,da,ds in parts:
                    row = {"dataset":dataset,"task":task,"episode":ep,"offset_frames":lag,**metrics(aa,ss,da,ds)}
                    lag_rows.append(row)
                    curve.append(row)
                    lag_buffers[dataset][lag].append((aa,ss,da,ds))
                if curve:
                    best = min(curve,key=lambda row:row["mae_mean_rad"])
                    zero = next(row for row in curve if row["offset_frames"]==0)
                    record["right_arm_lag_pairs_same_support"] = best["pairs"]
                    record["right_arm_best_state_lag_frames"] = best["offset_frames"]
                    record["right_arm_best_state_lag_mae_rad"] = best["mae_mean_rad"]
                    record["right_arm_zero_lag_mae_rad"] = zero["mae_mean_rad"]
                    record["right_arm_lag_minus16_mae_rad"] = next(row for row in curve if row["offset_frames"]==-16)["mae_mean_rad"]
                    record["right_arm_lag_plus16_mae_rad"] = next(row for row in curve if row["offset_frames"]==16)["mae_mean_rad"]
                    lag_hist[dataset][best["offset_frames"]] += 1
                    if abs(best["offset_frames"]) >= 16:
                        add_issue(issues,dataset,task,ep,"right_arm_large_lag_candidate","review",f"best offset={best['offset_frames']}; pairs={best['pairs']}; MAE={best['mae_mean_rad']}; physical delay vs indexing unresolved by correlation alone")
                else:
                    record["right_arm_lag_pairs_same_support"] = 0
                    record["right_arm_best_state_lag_frames"] = None
                    reason = "episode too short for a common +/-32-frame support" if n<=65 else "no right-arm transitions >0.001 rad on common support"
                    record["right_arm_lag_not_estimable_reason"] = reason
                    add_issue(issues,dataset,task,ep,"right_arm_lag_not_estimable","info",reason)
                try:
                    raw = raw_audit(dataset,task,meta,action,state,config,funcs)
                    raw_rows.append({"dataset":dataset,"task":task,"episode":ep,**raw})
                    if not raw["raw_path_confirmed"]:
                        add_issue(issues,dataset,task,ep,"raw_path_unconfirmed","review",json.dumps(raw,ensure_ascii=False))
                    elif not raw.get("trim_count_matches_final"):
                        add_issue(issues,dataset,task,ep,"raw_trim_length_mismatch","error",json.dumps(raw,ensure_ascii=False))
                    else:
                        for key in ("same_frame_raw_to_final_all_arm_action_max_error_rad","same_frame_raw_to_final_all_arm_state_max_error_rad"):
                            if raw[key] > 1e-6:
                                add_issue(issues,dataset,task,ep,"raw_final_arm_value_mismatch","error",f"{key}={raw[key]}")
                        for stream,diag in raw["streams"].items():
                            if diag.get("decreasing",0):
                                add_issue(issues,dataset,task,ep,"raw_stream_timestamp_decreasing","error",f"{stream}: {diag}")
                            if diag.get("samples",0)==0:
                                add_issue(issues,dataset,task,ep,"raw_stream_timestamp_missing","review",stream)
                        cmd = raw["streams"].get("action/joint",{})
                        before,after = cmd.get("before_source_frames",0),cmd.get("after_source_frames",0)
                        if before or after:
                            add_issue(issues,dataset,task,ep,"raw_command_coverage_hold","info",f"first-command extension frames={before}; last-command hold frames={after}; hold is not automatically an error")
                        if cmd.get("gaps_gt_100ms",0):
                            add_issue(issues,dataset,task,ep,"raw_command_gaps_gt_100ms","review",f"raw gaps={cmd['gaps_gt_100ms']}; max gap={cmd['interval_max_ms']} ms; nearest may use a future command, not proof of data corruption")
                except Exception as exc:
                    raw_rows.append({"dataset":dataset,"task":task,"episode":ep,"raw_path_confirmed":False,"raw_audit_error":f"{type(exc).__name__}: {exc}"})
                    add_issue(issues,dataset,task,ep,"raw_audit_failed","review",f"{type(exc).__name__}: {exc}")
                for camera,spec in cameras.items():
                    video_path = task_path / info["video_path"].format(episode_chunk=ep//chunk_size,episode_index=ep,video_key=camera)
                    video_jobs.append((dataset,task,ep,camera,video_path,fps,n,spec.get("shape"),ffprobe))
                    if video_path.is_file():
                        stat = video_path.stat()
                        hashes[str(video_path)] = [stat.st_size,stat.st_mtime_ns]
                total_rows += n
                expected_global += n
                record["issues_before_video_probe"] = [row["category"] for row in issues[episode_issue_start:]]
            except Exception as exc:
                add_issue(issues,dataset,task,ep,"parquet_audit_failed","error",f"{type(exc).__name__}: {exc}")
                record["audit_error"] = f"{type(exc).__name__}: {exc}"
            all_episodes.append(record)
            if len(all_episodes)%25==0:
                print(f"CPU audit {len(all_episodes)} episodes completed; task={dataset}/{task}; ep={ep}",flush=True)
        task_errors = []
        if total_rows != info["total_frames"]:
            task_errors.append("task_total_frames_mismatch")
        if len(metas) != info["total_episodes"]:
            task_errors.append("task_total_episodes_mismatch")
        actual_parquets = list((task_path/"data").glob("chunk-*/episode_*.parquet"))
        if len(actual_parquets)!=len(metas):
            task_errors.append("task_parquet_file_count_mismatch")
        for category in task_errors:
            add_issue(issues,dataset,task,-1,category,"error",f"actual frames={total_rows}, episodes={len(metas)}, parquet files={len(actual_parquets)}")
        task_rows.append({"dataset":dataset,"task":task,"episodes":len(metas),"frames":total_rows,
            "camera_count":len(cameras),"expected_videos":len(metas)*len(cameras),
            "info_total_frames":info["total_frames"],"issue_count_before_video_probe":len(issues)-initial_issue_count})
    write_csv(args.output/"episode_audit.csv",all_episodes)
    write_csv(args.output/"right_arm_episode_lag_curves.csv",lag_rows)
    (args.output/"raw_source_audit.json").write_text(json.dumps(raw_rows,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Numerical/raw audit complete: episodes={len(all_episodes)}, videos queued={len(video_jobs)}; CPU ffprobe workers={VIDEO_WORKERS}",flush=True)
    videos = []
    with ThreadPoolExecutor(max_workers=VIDEO_WORKERS) as pool:
        futures = [pool.submit(probe_video,job) for job in video_jobs]
        for future in as_completed(futures):
            video = future.result()
            videos.append(video)
            for category in ([video["error"]] if video.get("error") else video.get("error_categories",[])):
                add_issue(issues,video["dataset"],video["task"],video["episode"],category,"error",json.dumps(video,ensure_ascii=False),video["camera"])
            if len(videos)%100==0 or len(videos)==len(video_jobs):
                print(f"ffprobe videos={len(videos)}/{len(video_jobs)}; error videos={sum(bool(v.get('error') or v.get('error_categories')) for v in videos)}",flush=True)
    videos.sort(key=lambda row:(row["dataset"],row["task"],row["episode"],row["camera"]))
    global_curves = []
    for dataset in DATASETS:
        for lag in LAGS:
            parts = lag_buffers[dataset][lag]
            if parts:
                aa,ss,da,ds = [np.concatenate([part[idx] for part in parts]) for idx in range(4)]
                global_curves.append({"dataset":dataset,"offset_frames":lag,**metrics(aa,ss,da,ds)})
    changed = []
    for path_string,snapshot in hashes.items():
        path = Path(path_string)
        if isinstance(snapshot,list):
            stat=path.stat()
            if [stat.st_size,stat.st_mtime_ns]!=snapshot:
                changed.append(path_string)
        elif digest(path)!=snapshot:
            changed.append(path_string)
    previous = None
    if PREVIOUS_0923.is_file():
        old = read_json(PREVIOUS_0923)
        previous={"path":str(PREVIOUS_0923),"sha256":digest(PREVIOUS_0923),"episodes":old["episodes"],"frames":old["frames"],
            "prior_integrity":old["integrity"],"position_mae_best_offset":old["best_offset"]["offset_frames"],
            "boundary_hold":old["boundary_clamping"],"prior_long_gap_evidence":old.get("long_command_gaps")}
    datasets_summary=[]
    for dataset in DATASETS:
        eps=[row for row in all_episodes if row["dataset"]==dataset]
        vids=[row for row in videos if row["dataset"]==dataset]
        raws=[row for row in raw_rows if row["dataset"]==dataset]
        curves=[row for row in global_curves if row["dataset"]==dataset]
        best=min(curves,key=lambda row:row["mae_mean_rad"]) if curves else None
        comparable_raw=[row for row in raws if "raw_frame_offset_reconstruction" in row]
        moving_raw=[row for row in comparable_raw if row["all_arm_motion_is_significant"]]
        shifted16_excluded=sum(all(row["raw_frame_offset_reconstruction"].get(str(offset),{}).get("all_arm_action_mae_rad",0)>1e-4 for offset in (-16,16)) and row["same_frame_raw_to_final_all_arm_action_max_error_rad"]==0 for row in moving_raw)
        datasets_summary.append({"dataset":dataset,"tasks":sum(row["dataset"]==dataset for row in task_rows),"episodes":len(eps),
            "frames":sum(row.get("frames",0) for row in eps),"videos_probed":len(vids),
            "error_categories":dict(Counter(row["category"] for row in issues if row["dataset"]==dataset and row["severity"]=="error")),
            "review_categories":dict(Counter(row["category"] for row in issues if row["dataset"]==dataset and row["severity"]=="review")),
            "info_categories":dict(Counter(row["category"] for row in issues if row["dataset"]==dataset and row["severity"]=="info")),
            "right_arm_motion_transitions":sum(row.get("right_arm_effective_motion_transitions",0) for row in eps),
            "right_arm_lag_estimable_episodes":sum(row.get("right_arm_lag_pairs_same_support",0)>0 for row in eps),
            "right_arm_best_lag_histogram":dict(sorted(lag_hist[dataset].items())),"global_moving_right_arm_best_lag":best,
            "valid_real_action16_windows":sum(row.get("action_16frame_windows",{}).get("valid_16_frame_windows",0) for row in eps),
            "right_arm_constant_action16_windows":sum(row.get("action_16frame_windows",{}).get("right_arm_exactly_constant_windows",0) for row in eps),
            "right_arm_moving_action16_windows":sum(row.get("action_16frame_windows",{}).get("right_arm_max_span_gt_1e3_rad_windows",0) for row in eps),
            "confirmed_raw_paths":sum(row.get("raw_path_confirmed",False) for row in raws),
            "raw_final_arm_exact_match_episodes":sum(row.get("same_frame_raw_to_final_all_arm_action_max_error_rad") == 0 and row.get("same_frame_raw_to_final_all_arm_state_max_error_rad") == 0 for row in raws),
            "raw_any_arm_moving_episodes":len(moving_raw),"fixed_plusminus16_command_shift_excluded_episodes":shifted16_excluded,
            "unconfirmed_raw_paths":[{"task":row["task"],"episode":row["episode"],"expected_raw_path":row.get("expected_raw_path"),"error":row.get("raw_audit_error")} for row in raws if not row.get("raw_path_confirmed",False)]})
    report={"status":"completed","scope":"CPU read-only audit; no model training/inference or data repair",
        "input_root":str(args.input.resolve()),"output_root":str(args.output.resolve()),"elapsed_seconds":time.perf_counter()-started,
        "counts":{"datasets":len(DATASETS),"tasks":len(input_tasks),"episodes":len(all_episodes),"frames":sum(row.get("frames",0) for row in all_episodes),
            "videos_expected":len(video_jobs),"videos_probed":len(videos)},"datasets":datasets_summary,"tasks":task_rows,
        "anomaly_categories_by_severity":{level:dict(Counter(row["category"] for row in issues if row["severity"]==level)) for level in ("error","review","info")},
        "episode_anomalies":issues,"input_manifest":input_manifest,"previous_0923_source_evidence":previous,
        "offset_definition":"action[t] vs state[t+lag]; positive means measured response follows command",
        "lag_support":"moving frames with max right-arm per-frame change >0.001 rad in action or state; same t in [32,n-32) for all 65 offsets; no cross-episode pairing",
        "video_method":"ffprobe demux only; header nb_frames plus H.264 access-unit packets and sorted integer presentation timestamps. B-frame packet-order PTS reordering is legal and not flagged as a presentation clock error.",
        "fixed_16_frame_assessment":"Raw/final exact reconstruction at source frame offset 0 and comparison to offsets +/-16 directly test conversion indexing. Lag estimates alone cannot distinguish feedback dynamics from shifted data. Static arm channels cannot rule out a temporal shift.",
        "original_timestamp_rule":{"head":"raw camera/head_color/head_color.txt is the selection clock; final trim slices that original clock",
            "wrist":"nearest numeric JPG filename timestamp to selected head clock",
            "state_and_command":"nearest timestamp in each H5 named stream; final Parquet/video use new CFR frame/fps clock",
            "physical_clock_limit":"Cannot independently certify device clock calibration, recorder message execution or raw-to-encoded image-content pairing without recorder/runtime code or pixel comparison.",
            "converter_source_lines":source_lines,"converter_sha256":digest(REPO/"tools/conversion/00_convert_618_to_lerobot.py")},
        "immutability_check":{"parquet_sha256_count":sum(not isinstance(v,list) for v in hashes.values()),"video_size_mtime_count":sum(isinstance(v,list) for v in hashes.values()),"changed_input_files":changed},
        "limitations":["No full video pixel decode; packet/header/PTS checks cannot certify decoded pixel correctness or camera-content synchronization.",
            "Native LeRobot training/configuration belongs to the main thread and other agents.","Review categories report risks or non-estimable situations and do not automatically imply dataset corruption."],
        "ACT_smoke":"skipped by request: CPU read-only audit; main thread owns native LeRobot training"}
    (args.output/"audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (args.output/"video_audit.json").write_text(json.dumps(videos,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write_csv(args.output/"video_audit.csv",videos)
    write_csv(args.output/"episode_anomalies.csv",issues)
    write_csv(args.output/"task_summary.csv",task_rows)
    write_csv(args.output/"right_arm_global_lag_curves.csv",global_curves)
    make_figures(args.output)
    print(json.dumps({"counts":report["counts"],"severity_categories":report["anomaly_categories_by_severity"],
        "changed_input_files":changed,"output":str(args.output),"elapsed_seconds":report["elapsed_seconds"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

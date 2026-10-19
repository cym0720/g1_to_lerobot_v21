#!/usr/bin/env python3
"""CPU checkpoint diagnostics for the two durable native ACT experiments."""
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "output/lerobot_act_control_20260929"
RUNS = {"full_20d": (ROOT / "native_train", 1165),
        "right_8d": (ROOT / "right_arm_train", 2281238)}


def summarize():
    result = {}
    for name, (root, _) in RUNS.items():
        steps = {}
        for path in sorted((root / "state_following").glob("step_*")):
            baseline_path = path / "baseline_report.json"
            intervention_path = path / "input_interventions.json"
            if not baseline_path.exists() or not intervention_path.exists():
                continue
            rows = json.loads(baseline_path.read_text())["episodes"]
            gt_path = root / "gt" / path.name / "report.json"
            gt_rows = json.loads(gt_path.read_text())["episodes"] if gt_path.exists() else {}
            interventions = json.loads(intervention_path.read_text())["samples"]
            if len(interventions) != 9:
                continue
            steps[path.name] = {
                "episodes": {ep: {k: row[k] for k in ("right_arm_prediction_mae", "right_arm_state_hold_mae",
                    "right_arm_relative_improvement", "right_arm_future_correction_cosine")} for ep, row in rows.items()},
                "image_swap_mean_delta_rad": sum(r["images_plus32_same_state"]["right_arm_mean_delta_rad"] for r in interventions)/9,
                "state_swap_mean_delta_rad": sum(r["state_plus32_same_images"]["right_arm_mean_delta_rad"] for r in interventions)/9,
                "per_joint_chunk_span_ratios_by_episode": {ep: r["span_ratios"] for ep, r in gt_rows.items()},
            }
        result[name] = steps
    (ROOT / "state_following_comparison.json").write_text(json.dumps(result, indent=2))


def main():
    print("CHECKPOINT_DIAGNOSTIC_WATCH_STARTED", flush=True)
    attempted = set()
    while True:
        alive = False
        for name, (root, pid) in RUNS.items():
            process = Path(f"/proc/{pid}/cmdline")
            cmd = process.read_bytes() if process.exists() else b""
            alive |= b"train_g1_native_" in cmd
            ready = sorted(p for p in (root / "gt").glob("step_*") if (p / "report.json").exists())
            if not ready:
                continue
            latest = ready[-1]
            key = (name, latest.name)
            destination = root / "state_following" / latest.name
            log_path = destination / "diagnostic.log"
            if key in attempted:
                continue
            existing = destination / "input_interventions.json"
            destination.mkdir(parents=True, exist_ok=True)
            if not existing.exists() or len(json.loads(existing.read_text())["samples"]) != 9:
                with log_path.open("a") as log:
                    code = subprocess.call([sys.executable, "-u", str(REPO / "tools/testing/audit_native_state_following.py"),
                                            "--input", str(root)], stdout=log, stderr=subprocess.STDOUT)
                print("CHECKPOINT_DIAGNOSTIC_EXIT", name, latest.name, code, flush=True)
            probe_path = destination / "time_query_probe.json"
            if not probe_path.exists() or len(json.loads(probe_path.read_text())["samples"]) != 9:
                with (destination / "time_query_probe.log").open("a") as log:
                    code = subprocess.call([sys.executable, "-u", str(REPO / "tools/testing/probe_native_act_time_queries.py"),
                                            "--input", str(root)], stdout=log, stderr=subprocess.STDOUT)
                print("TIME_QUERY_DIAGNOSTIC_EXIT", name, latest.name, code, flush=True)
            attempted.add(key)
        summarize()
        if not alive:
            print("CHECKPOINT_DIAGNOSTIC_WATCH_FINISHED", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

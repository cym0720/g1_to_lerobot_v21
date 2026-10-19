# Purpose: synchronize LeRobot v2.1 task prompts from task_id_map.json into tasks.jsonl and every episode's tasks field.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.lerobot_v21 import tasks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize task language prompts for existing LeRobot v2.1 task directories."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, default=REPO_ROOT / "config/task_id_map.json")
    parser.add_argument("--apply", action="store_true", help="Write files; omit for a dry run.")
    parser.add_argument(
        "--prompt",
        help="直接指定新提示词，跳过 task map / source_task_id 解析（用于缺 source_task_id 的历史划分）",
    )
    parser.add_argument(
        "--match",
        help="只改写当前提示词恰好等于该值的任务目录；与 --prompt 连用，防止波及同级任务",
    )
    args = parser.parse_args()
    mapping = json.loads(args.task_map.read_text(encoding="utf8"))["tasks"]
    report = []
    for root in tasks(args.input):
        episodes_path = root / "meta/episodes.jsonl"
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf8").splitlines()
            if line.strip()
        ]
        if args.prompt is not None:
            current = {
                row.get("task", "")
                for row in (
                    json.loads(x)
                    for x in (root / "meta/tasks.jsonl").read_text(encoding="utf8").splitlines()
                    if x.strip()
                )
            }
            if args.match is not None and current != {args.match}:
                print(f"{root.name}: skipped current={sorted(current)!r}", flush=True)
                continue
            if len(current) != 1:
                raise ValueError(f"{root.name}: 期望单任务提示词，实际 {sorted(current)}")
            prompt, task_id = args.prompt, "forced"
        else:
            ids = {str(row.get("source_task_id", "")) for row in rows}
            if len(ids) != 1 or next(iter(ids)) not in mapping:
                raise ValueError(
                    f"{root.name}: cannot resolve exactly one mapped source_task_id: {sorted(ids)}"
                )
            task_id = next(iter(ids))
            prompt = mapping[task_id]["name"]
        report.append(
            {
                "task_directory": root.name,
                "task_id": task_id,
                "prompt": prompt,
                "episodes": len(rows),
            }
        )
        if args.apply:
            write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": prompt}])
            for row in rows:
                row["tasks"] = [prompt]
            write_jsonl(episodes_path, rows)
        print(f"{root.name}: task_id={task_id} prompt={prompt!r} episodes={len(rows)}", flush=True)
    report_path = args.input / "task_prompt_sync_report.json"
    report_path.write_text(
        json.dumps({"apply": args.apply, "tasks": report}, ensure_ascii=False, indent=2),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()

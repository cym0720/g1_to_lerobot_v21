# 用途（列名统一）：把 EVA qpos20 合约列名 observations.state.qpos / action.qpos 重命名为
# observation.state / action，与 1008 系产物及评估脚本口径一致；只改数据侧，不改任何评估脚本。
"""统一 LeRobot v2.1 数据集的 state/action 列名。

- Parquet：重命名两列，位置不变，其余列与视频不动；
- meta/info.json features、stats.json、episodes_stats.jsonl 的对应键同步重命名；
- 写 rename_provenance 记录；已重命名过的根目录会跳过（幂等）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.lerobot_v21 import episodes, task_roots  # noqa: E402

QPOS_NAMES = {"observations.state.qpos": "observation.state", "action.qpos": "action"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rename_features(features: dict) -> bool:
    changed = False
    for old, new in QPOS_NAMES.items():
        if old in features:
            rebuilt = {}
            for key, value in features.items():
                rebuilt[new if key == old else key] = value
            features.clear()
            features.update(rebuilt)
            changed = True
    return changed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="任务父目录或单任务根目录")
    p.add_argument("--workers", type=int, default=4, help="保留参数以兼容调用习惯；当前为顺序执行")
    a = p.parse_args()
    roots = task_roots(a.input)
    if not roots:
        raise SystemExit(f"未找到数据集根目录: {a.input}")
    for root in roots:
        info_path = root / "meta/info.json"
        info = read_json(info_path)
        features = info["features"]
        if not any(old in features for old in QPOS_NAMES):
            print(f"{root.name}: skipped (already renamed or unknown schema)", flush=True)
            continue
        # Parquet 列重命名
        parquets = sorted((root / "data").rglob("episode_*.parquet"))
        if not parquets:
            raise SystemExit(f"{root}: 没有 parquet")
        for path in parquets:
            table = pq.read_table(path)
            mapping = {c: QPOS_NAMES.get(c, c) for c in table.column_names}
            pq.write_table(table.rename_columns([mapping[c] for c in table.column_names]), path, compression="snappy")
        # info.json
        assert rename_features(features), "features 重命名异常"
        info["rename_provenance"] = {
            "from": dict(QPOS_NAMES),
            "reason": "统一为评估脚本与 1008 系产物口径 observation.state/action；评估脚本未做任何修改",
        }
        dump(info_path, info)
        # stats.json
        stats_path = root / "meta/stats.json"
        if stats_path.is_file():
            stats = read_json(stats_path)
            rename_features(stats)
            dump(stats_path, stats)
        # episodes_stats.jsonl
        es_path = root / "meta/episodes_stats.jsonl"
        if es_path.is_file():
            rows = [json.loads(x) for x in es_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            for row in rows:
                rename_features(row.get("stats", {}))
            es_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{root.name}: renamed {len(parquets)} parquet + meta", flush=True)


if __name__ == "__main__":
    main()

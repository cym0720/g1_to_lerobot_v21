# 用途（主流程第 04 步）：按任务聚类检测轨迹离群值，生成视频审查清单，并按人工删除列表输出保留数据。
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from packages.robot_data.lerobot_v21 import (
    tasks,
    episodes,
    table,
    video_files,
    copy_dataset_skeleton,
    episode_parquet,
    write_episodes,
)


def feature(t, d):
    keys = [
        k
        for k in (
            "observation.state.joint_position",
            "action.joint_position",
            "observation.state.joint_velocity",
            "action.joint_velocity",
        )
        if k in t.column_names
    ]
    parts = []
    for k in keys:
        x = np.asarray(t[k].to_pylist(), dtype=np.float32)[-d:]
        if len(x) < d:
            x = np.pad(x, ((d - len(x), 0),) + ((0, 0),) * (x.ndim - 1), mode="edge")
        parts += [x.mean(axis=0).ravel(), x.std(axis=0).ravel(), (x[-1] - x[0]).ravel()]
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--tail-frames", type=int, default=30)
    p.add_argument("--contamination", type=float, default=0.1)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    a.review_dir.mkdir(parents=True, exist_ok=True)
    if a.apply:
        out = a.review_dir / "cleaned_dataset"
        out.mkdir(exist_ok=False)
        for src in tasks(a.input):
            if not episodes(src):
                continue
            delete = {
                x.strip()
                for x in (a.review_dir / src.name / "delete_list.txt")
                .read_text(encoding="utf8")
                .splitlines()
                if x.strip() and not x.startswith("#")
            }
            dst = out / src.name
            copy_dataset_skeleton(src, dst)
            keep = []
            for e in episodes(dst):
                if str(e["episode_index"]) in delete or e.get("original_episode_id") in delete:
                    r = (
                        a.review_dir
                        / "rejected_manual"
                        / src.name
                        / f"episode_{e['episode_index']:06d}"
                    )
                    r.mkdir(parents=True, exist_ok=True)
                    for f in [
                        episode_parquet(dst, e["episode_index"]),
                        *video_files(dst, e["episode_index"]),
                    ]:
                        if f.exists():
                            shutil.move(str(f), r / f.name)
                else:
                    keep.append(e)
            write_episodes(dst, keep)
        return
    for src in tasks(a.input):
        es = episodes(src)
        if not es:
            continue
        X = np.stack([feature(table(src, e["episode_index"]), a.tail_frames) for e in es])
        scores = (
            np.zeros(len(es))
            if len(es) < 3
            else -IsolationForest(contamination=min(a.contamination, 0.49), random_state=0)
            .fit(StandardScaler().fit_transform(X))
            .score_samples(StandardScaler().fit_transform(X))
        )
        rd = a.review_dir / src.name
        rd.mkdir(exist_ok=True)
        rows = []
        for e, s in zip(es, scores):
            # Copy a representative camera video for human review; source data remains untouched.
            vs = video_files(src, e["episode_index"])
            preview = rd / f"episode_{e['episode_index']:06d}_score_{s:.3f}.mp4"
            if vs:
                shutil.copy2(next((v for v in vs if "head" in str(v)), vs[0]), preview)
            rows.append(
                [
                    e["episode_index"],
                    e.get("original_episode_id", ""),
                    e["length"],
                    float(s),
                    preview.name if vs else "",
                ]
            )
        rows.sort(key=lambda x: x[3], reverse=True)
        with (rd / "review_candidates.csv").open("w", newline="", encoding="utf8") as f:
            csv.writer(f).writerows(
                [
                    [
                        "episode_index",
                        "original_episode_id",
                        "frames",
                        "outlier_score",
                        "preview_video",
                    ],
                    *rows,
                ]
            )
        (rd / "delete_list.txt").write_text(
            "# 填入要删除的 episode_index 或 original_episode_id，每行一个。\n", encoding="utf8"
        )


if __name__ == "__main__":
    main()

"""Exercise split-local video metadata through the merge CLI."""
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def test_merge_counts_videos_per_split(tmp_path):
    sources = [tmp_path / "a", tmp_path / "b"]
    cameras = ["observation.images.left", "observation.images.right"]
    for source in sources:
        for split, count in [("train", 2), ("val", 1)]:
            root = source / split
            (root / "meta").mkdir(parents=True)
            (root / "meta/info.json").write_text(json.dumps({
                "total_videos": 999,
                "features": {key: {"dtype": "video"} for key in cameras},
            }), encoding="utf-8")
            data = root / "data/chunk-000"
            data.mkdir(parents=True)
            for episode in range(count):
                pq.write_table(pa.table({
                    "observation.state": [[0.0] * 20],
                    "action": [[1.0] * 20],
                    "episode_index": [episode], "index": [episode],
                }), data / f"episode_{episode:06d}.parquet")
                for camera in cameras:
                    video = root / "videos/chunk-000" / camera / f"episode_{episode:06d}.mp4"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    # Merge only links bytes; video decoding is outside this test.
                    video.write_bytes(b"video fixture")
    output = tmp_path / "merged"
    script = Path(__file__).resolve().parents[1] / "tools/conversion/merge_final_datasets.py"
    subprocess.run([sys.executable, str(script), "--a", str(sources[0]),
                    "--b", str(sources[1]), "--output", str(output)], check=True)
    for split, episodes, videos in [("train", 4, 8), ("val", 2, 4)]:
        root = output / split
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        assert info["total_episodes"] == episodes
        assert info["total_videos"] == videos == len(list((root / "videos").rglob("*.mp4")))
        table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
        assert table["action"].to_pylist() == [[1.0] * 20]
        assert table["observation.state"].to_pylist() == [[0.0] * 20]

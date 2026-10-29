"""Check per-video 224 letterbox geometry and black padding on sampled decoded frames."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

CAMERAS = ["observation.images.head_color", "observation.images.head_left_fisheye", "observation.images.head_right_fisheye"]


def sample_video(task):
    root, split, camera, episode, expected, source_geometry = task
    video = Path(root) / split / "videos/chunk-000" / camera / f"episode_{episode:06d}.mp4"
    table = pq.read_table(Path(root) / split / "data/chunk-000" / f"episode_{episode:06d}.parquet", columns=["frame_index"])
    n = table.num_rows
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(n)):
        raise RuntimeError(f"non-contiguous frame indexes: {video}")
    indices = sorted({0, n // 2, n - 1})
    h, w, _ = source_geometry
    scale = min(224 / w, 224 / h)
    content_w, content_h = max(1, round(w * scale)), max(1, round(h * scale))
    pad_left, pad_top = (224 - content_w) // 2, (224 - content_h) // 2
    pad_right, pad_bottom = 224 - pad_left - content_w, 224 - pad_top - content_h
    if (pad_top, pad_bottom) == (0, 0) and (pad_left, pad_right) == (0, 0):
        raise RuntimeError(f"source aspect is square but metadata claims padded transform: {video}")
    inspected = 0
    with av.open(str(video)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for idx, frame in enumerate(container.decode(video=0)):
            if idx not in indices:
                continue
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (224, 224, 3):
                raise RuntimeError(f"target geometry mismatch: {video} frame={idx} shape={image.shape}")
            # Ignore two edge rows/columns where yuv420 chroma can ring across
            # the content/padding boundary after H.264 encoding.
            top_end = max(0, pad_top - 2)
            bottom_start = min(224, 224 - max(0, pad_bottom - 2))
            left_end = max(0, pad_left - 2)
            right_start = min(224, 224 - max(0, pad_right - 2))
            pad_regions = []
            if top_end:
                pad_regions.append(image[:top_end])
            if bottom_start < 224:
                pad_regions.append(image[bottom_start:])
            if left_end:
                pad_regions.append(image[:, :left_end])
            if right_start < 224:
                pad_regions.append(image[:, right_start:])
            if not pad_regions or max(int(region.max(initial=0)) for region in pad_regions) > 16:
                raise RuntimeError(f"black padding check failed: {video} frame={idx}")
            content = image[pad_top:pad_top + content_h, pad_left:pad_left + content_w]
            if content.size == 0 or int(content.max(initial=0)) <= 16:
                raise RuntimeError(f"letterbox content area is empty/black: {video} frame={idx}")
            inspected += 1
    if inspected != len(indices):
        raise RuntimeError(f"sampled frame missing: {video} got={inspected} expected={len(indices)}")
    return {"split": split, "camera": camera, "episode_index": episode, "frames": n,
            "sampled_frames": indices, "target_geometry": [224, 224, 3],
            "source_geometry": list(source_geometry), "resized_geometry": [content_h, content_w, 3],
            "padding": {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right},
            "black_padding_interior_max_rgb": 16}


def validate(root: Path, output: Path) -> dict:
    tasks = []
    for split in ("train", "val"):
        info = json.loads((root / split / "meta/info.json").read_text(encoding="utf-8"))
        standard = json.loads((root / split / "meta/data_standard.json").read_text(encoding="utf-8"))
        if standard.get("image_geometry") != "letterbox224" or standard.get("image_preprocessing") != "letterbox224_bilinear_black_pad_center":
            raise RuntimeError(f"wrong image standard in {split}")
        transform = standard.get("video_transform", {})
        source_shapes = transform.get("source_geometry_by_camera", {})
        if transform.get("target_geometry") != [224, 224, 3] or transform.get("aspect_ratio_preserved") is not True or transform.get("padding_rgb") != [0, 0, 0]:
            raise RuntimeError(f"incomplete letterbox metadata in {split}")
        rows = [json.loads(line) for line in (root / split / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != info["total_episodes"]:
            raise RuntimeError(f"episode count mismatch in {split}")
        for row in rows:
            idx = int(row["episode_index"])
            for camera in CAMERAS:
                if info["features"][camera]["shape"] != [224, 224, 3]:
                    raise RuntimeError(f"info geometry mismatch {split}/{camera}")
                shape = source_shapes.get(camera)
                if not shape:
                    raise RuntimeError(f"missing source geometry metadata {split}/{camera}")
                tasks.append((str(root), split, camera, idx, int(row["length"]), shape))
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(sample_video, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    report = {"status": "PASS", "dataset": str(root.resolve()), "streams_checked": len(results),
              "sampled_frames_checked": sum(len(r["sampled_frames"]) for r in results),
              "camera_keys": CAMERAS, "target_geometry": [224, 224, 3],
              "method": "sample first, middle, last decoded frame of every episode-camera stream; test geometry and interior black padding after excluding 2 boundary rows for H.264 chroma ringing",
              "details": sorted(results, key=lambda r: (r["split"], r["episode_index"], r["camera"]))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.input, args.output), ensure_ascii=False))

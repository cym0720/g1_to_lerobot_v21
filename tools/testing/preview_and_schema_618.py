"""Preview 618data RGB sources before conversion.

This is deliberately a read-only source inspector: it never modifies dataset/618data.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def first_jpg(path: Path) -> Path:
    try:
        return sorted(path.glob("*.jpg"))[0]
    except StopIteration as exc:
        raise FileNotFoundError(f"未找到 JPG: {path}") from exc


def extract_head_frame(source: Path, output: Path, fps: int) -> None:
    """A raw H265 stream has no reliable container FPS; force the recorded 30 FPS."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-r",
            str(fps),
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="dataset/618data")
    parser.add_argument("--output", type=Path, required=True, help="Preview directory")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    episode = sorted(path for path in args.input.iterdir() if path.is_dir())[0]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    camera = episode / "camera"
    head_original = out / "head_original_1280x800.png"
    extract_head_frame(camera / "head_color" / "head_color.h265", head_original, args.fps)
    with Image.open(head_original) as image:
        crop = image.crop((0, 40, 1280, 760))
        crop.save(out / "head_center_crop_1280x720.png")
    left = first_jpg(camera / "hand_left" / "color")
    right = first_jpg(camera / "hand_right" / "color")
    shutil.copy2(left, out / "left_wrist_original_848x480.jpg")
    shutil.copy2(right, out / "right_wrist_original_848x480.jpg")
    print(f"已写入: {out.resolve()}")


if __name__ == "__main__":
    main()

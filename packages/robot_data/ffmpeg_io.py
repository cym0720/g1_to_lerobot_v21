"""FFmpeg subprocess helpers and hardware-backend capability discovery."""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *command], check=True)


def video_frame_count(path: Path) -> int:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)], capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def ffmpeg_capabilities() -> dict[str, bool]:
    """Return compile-time availability; runtime probes still decide job placement."""
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True, check=True).stdout
    return {"nvenc": "h264_nvenc" in encoders, "qsv_encoder": "h264_qsv" in encoders, "cuda_decoder": "hevc_cuvid" in decoders, "qsv_decoder": "hevc_qsv" in decoders}

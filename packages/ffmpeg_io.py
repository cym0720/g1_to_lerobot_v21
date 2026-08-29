"""FFmpeg video probing, timestamp-aligned decode, and hardware-aware H.264 encoding."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterable
import numpy as np

class FFmpegBackendError(RuntimeError):
    pass

def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *command], check=True)

def video_frame_count(path: Path) -> int:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)], capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

def ffmpeg_capabilities() -> dict[str, bool]:
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True, check=True).stdout
    return {"nvenc": "h264_nvenc" in encoders, "qsv_encoder": "h264_qsv" in encoders, "cuda_decoder": "hevc_cuvid" in decoders, "qsv_decoder": "hevc_qsv" in decoders}

def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    right = np.clip(right, 0, len(source_timestamps) - 1)
    return np.where(np.abs(source_timestamps[right] - target_timestamps) < np.abs(source_timestamps[left] - target_timestamps), right, left)

def _select_filter(indices: Iterable[int], fps: int, crop: dict | None) -> str:
    expression = "+".join("eq(n\\,{})".format(int(index)) for index in indices)
    filters = ["select='{}'".format(expression), "setpts=N/({}*TB)".format(fps)]
    if crop:
        filters.append("crop={}:{}:{}:{}".format(crop['width'], crop['height'], crop['x'], crop['y']))
    return ",".join(filters)

def _backend_command(backend: str, source: Path, video_filter: str, fps: int, destination: Path) -> list[str]:
    if backend == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "hevc_cuvid", "-threads", "1", "-i", str(source), "-vf", "hwdownload,format=nv12," + video_filter, "-r", str(fps), "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p", str(destination)]
    if backend == "qsv":
        return ["-hwaccel", "qsv", "-c:v", "hevc_qsv", "-threads", "1", "-i", str(source), "-vf", video_filter, "-r", str(fps), "-c:v", "h264_qsv", "-global_quality", "23", str(destination)]
    return ["-threads", "1", "-i", str(source), "-vf", video_filter, "-r", str(fps), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(destination)]

def encode_h265_aligned(source: Path, source_timestamps: np.ndarray, target_timestamps: np.ndarray, destination: Path, fps: int, crop: dict | None = None, preferred_backends: tuple[str, ...] = ("cuda", "qsv", "cpu")) -> str:
    selected = nearest_indices(source_timestamps, target_timestamps)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("duplicate timestamp-aligned H.265 frame selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph = _select_filter(selected, fps, crop)
    failures = []
    for backend in preferred_backends:
        try:
            destination.unlink(missing_ok=True)
            run_ffmpeg(_backend_command(backend, source, graph, fps, destination))
            count = video_frame_count(destination)
            if count != len(selected):
                raise RuntimeError("frame count {} != {}".format(count, len(selected)))
            return backend
        except (subprocess.CalledProcessError, RuntimeError) as error:
            destination.unlink(missing_ok=True)
            failures.append("{}: {}".format(backend, error))
    raise FFmpegBackendError("; ".join(failures))

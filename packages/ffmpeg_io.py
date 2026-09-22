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

def clip_video_frames(source: Path, destination: Path, start: int, end: int, fps: float) -> None:
    """Decode and re-encode an inclusive frame interval with an exact count gate."""
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive video range [{start}, {end}]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.mp4")
    temporary.unlink(missing_ok=True)
    run_ffmpeg([
        "-i", str(source),
        "-vf", f"select=between(n\\,{start}\\,{end}),setpts=N/({fps}*TB)",
        "-frames:v", str(end - start + 1),
        "-r", str(fps),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(temporary),
    ])
    frames = video_frame_count(temporary)
    if frames != end - start + 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{source}: clipped {frames} frames, expected {end - start + 1}")
    temporary.replace(destination)

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

def _video_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    width, height = result.stdout.strip().split(",")
    return int(width), int(height)


def rgb_video_stats(path: Path, chunk_frames: int = 8) -> dict:
    """Decode RGB through FFmpeg and stream channel statistics without frame caching."""
    width, height = _video_size(path)
    frame_bytes = width * height * 3
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    total = np.zeros(3, dtype=np.float64); square = np.zeros(3, dtype=np.float64)
    frames = 0
    while True:
        raw = process.stdout.read(frame_bytes * chunk_frames)
        if not raw:
            break
        if len(raw) % frame_bytes:
            process.kill()
            raise RuntimeError("FFmpeg emitted an incomplete RGB frame")
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
        lo = np.minimum(lo, data.min(axis=0)); hi = np.maximum(hi, data.max(axis=0))
        total += data.sum(axis=0); square += np.square(data).sum(axis=0)
        frames += len(raw) // frame_bytes
    stderr = process.stderr.read().decode("utf8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError("FFmpeg RGB decode failed: " + stderr[-1000:])
    if not frames:
        raise ValueError("empty video: {}".format(path))
    pixels = frames * width * height
    mean = total / pixels
    variance = np.maximum(square / pixels - np.square(mean), 0.0)
    return {"min": lo.reshape(3, 1, 1).tolist(), "max": hi.reshape(3, 1, 1).tolist(), "mean": mean.reshape(3, 1, 1).tolist(), "std": np.sqrt(variance).reshape(3, 1, 1).tolist(), "count": [frames]}

def encode_jpeg_sequence(selected_files: list[Path], destination: Path, fps: int, preferred_backends: tuple[str, ...] = ("cpu", "qsv", "cuda")) -> str:
    """Encode timestamp-aligned JPG files through NVENC, QSV, or libx264."""
    import os
    import shutil
    import tempfile
    if not selected_files:
        raise ValueError("no JPEG frames selected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory(prefix="lerobot_jpg_") as temporary:
        directory = Path(temporary)
        for index, source in enumerate(selected_files):
            target = directory / "{:06d}.jpg".format(index)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        for backend in preferred_backends:
            try:
                destination.unlink(missing_ok=True)
                base = ["-framerate", str(fps), "-i", str(directory / "%06d.jpg"), "-frames:v", str(len(selected_files)), "-r", str(fps), "-pix_fmt", "yuv420p"]
                if backend == "cuda":
                    command = [*base, "-c:v", "h264_nvenc", "-preset", "p4", str(destination)]
                elif backend == "qsv":
                    command = [*base, "-c:v", "h264_qsv", "-global_quality", "23", str(destination)]
                else:
                    command = [*base, "-c:v", "libx264", "-crf", "20", str(destination)]
                run_ffmpeg(command)
                count = video_frame_count(destination)
                if count != len(selected_files):
                    raise RuntimeError("frame count {} != {}".format(count, len(selected_files)))
                return backend
            except (subprocess.CalledProcessError, RuntimeError) as error:
                destination.unlink(missing_ok=True)
                failures.append("{}: {}".format(backend, error))
    raise FFmpegBackendError("; ".join(failures))

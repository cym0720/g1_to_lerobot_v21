"""FFmpeg video probing, timestamp-aligned decode, and hardware-aware H.264 encoding."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterable
import numpy as np


class FFmpegBackendError(RuntimeError):
    pass


def run_ffmpeg(command: list[str]) -> None:
    """执行一条 FFmpeg 命令。

    这里的 Python 代码不直接实现视频编码，而是负责拼接参数并启动
    FFmpeg 子进程。真正使用 libx264、NVENC 或 QSV 编码的是 FFmpeg。
    """
    # -y：目标文件已存在时直接覆盖；-loglevel error：只输出错误，避免日志过于冗长。
    # check=True：FFmpeg 返回非零状态时立即抛出异常，让上层尝试下一个后端。
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *command], check=True)


def video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def clip_video_frames(source: Path, destination: Path, start: int, end: int, fps: float) -> None:
    """Decode and re-encode an inclusive frame interval with an exact count gate."""
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive video range [{start}, {end}]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.mp4")
    temporary.unlink(missing_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(source),
            "-vf",
            f"select=between(n\\,{start}\\,{end}),setpts=N/({fps}*TB)",
            "-frames:v",
            str(end - start + 1),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(temporary),
        ]
    )
    frames = video_frame_count(temporary)
    if frames != end - start + 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{source}: clipped {frames} frames, expected {end - start + 1}")
    temporary.replace(destination)


def ffmpeg_capabilities() -> dict[str, bool]:
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    decoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "nvenc": "h264_nvenc" in encoders,
        "qsv_encoder": "h264_qsv" in encoders,
        "cuda_decoder": "hevc_cuvid" in decoders,
        "qsv_decoder": "hevc_qsv" in decoders,
    }


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    right = np.clip(right, 0, len(source_timestamps) - 1)
    return np.where(
        np.abs(source_timestamps[right] - target_timestamps)
        < np.abs(source_timestamps[left] - target_timestamps),
        right,
        left,
    )


def _select_filter(indices: Iterable[int], fps: int, crop: dict | None) -> str:
    """生成 FFmpeg 视频滤镜链：选帧、重置时间戳，以及可选裁剪。"""
    # eq(n, index) 按解码后的帧序号选择指定帧；多个条件用 + 连接表示“满足任意一个”。
    expression = "+".join("eq(n\\,{})".format(int(index)) for index in indices)
    # select 会丢弃帧，因此必须用 setpts 重新生成连续的输出时间戳。
    filters = ["select='{}'".format(expression), "setpts=N/({}*TB)".format(fps)]
    if crop:
        # crop 的参数顺序是 width:height:x:y。
        filters.append(
            "crop={}:{}:{}:{}".format(crop["width"], crop["height"], crop["x"], crop["y"])
        )
    return ",".join(filters)


def _backend_command(
    backend: str, source: Path, video_filter: str, fps: int, destination: Path
) -> list[str]:
    """根据后端名称生成 FFmpeg 命令参数。

    注意：函数只返回参数列表，不在这里执行命令；执行统一由 run_ffmpeg() 完成。
    """
    if backend == "cuda":
        # CUDA 路径：
        # 1) hevc_cuvid/NVDEC 在 GPU 上解码 H.265；
        # 2) hwdownload 把硬件帧取回系统内存，便于执行 select/crop 等滤镜；
        # 3) h264_nvenc/NVENC 在 GPU 上编码为 H.264。
        return [
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-c:v",
            "hevc_cuvid",
            "-threads",
            "1",
            "-i",
            str(source),
            "-vf",
            "hwdownload,format=nv12," + video_filter,
            "-r",
            str(fps),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    if backend == "qsv":
        # Intel Quick Sync 路径，作为 NVIDIA CUDA 不可用时的备用硬件后端。
        return [
            "-hwaccel",
            "qsv",
            "-c:v",
            "hevc_qsv",
            "-threads",
            "1",
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-r",
            str(fps),
            "-c:v",
            "h264_qsv",
            "-global_quality",
            "23",
            str(destination),
        ]
    # CPU 兜底路径：FFmpeg 使用 libx264 软件编码器。
    return [
        "-threads",
        "1",
        "-i",
        str(source),
        "-vf",
        video_filter,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]


def encode_h265_aligned(
    source: Path,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    destination: Path,
    fps: int,
    crop: dict | None = None,
    preferred_backends: tuple[str, ...] = ("cuda", "qsv", "cpu"),
) -> str:
    """按目标时间戳抽取 H.265 帧，再编码为帧数严格一致的 MP4。"""
    # 将源视频时间戳映射到统一的目标时间戳，得到需要保留的源帧下标。
    selected = nearest_indices(source_timestamps, target_timestamps)
    if len(np.unique(selected)) != len(selected):
        # 同一源帧被选中两次会导致时间对齐和输出帧数语义不明确，因此直接拒绝。
        raise ValueError("duplicate timestamp-aligned H.265 frame selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 多数头部相机使用自己的时间戳作为目标时钟，因此源帧通常已经是完整、连续的。
    # 这种情况下只需 setpts，不要生成几千项的 select 表达式；Windows 可能因命令行过长
    # 报 WinError 206。
    if len(selected) == len(source_timestamps) and np.array_equal(
        selected, np.arange(len(source_timestamps))
    ):
        filters = ["setpts=N/({}*TB)".format(fps)]
        if crop:
            filters.append(
                "crop={}:{}:{}:{}".format(
                    crop["width"], crop["height"], crop["x"], crop["y"]
                )
            )
        graph = ",".join(filters)
    else:
        graph = _select_filter(selected, fps, crop)
    failures = []
    for backend in preferred_backends:
        try:
            # 每次尝试前删除上一个失败后可能留下的不完整 MP4。
            destination.unlink(missing_ok=True)
            run_ffmpeg(_backend_command(backend, source, graph, fps, destination))
            count = video_frame_count(destination)
            if count != len(selected):
                # FFmpeg 没报错不等于数据正确，还必须验证输出帧数。
                raise RuntimeError("frame count {} != {}".format(count, len(selected)))
            return backend
        except (subprocess.CalledProcessError, RuntimeError) as error:
            # 当前后端失败后记录原因，继续尝试下一个后端。
            destination.unlink(missing_ok=True)
            failures.append("{}: {}".format(backend, error))
    raise FFmpegBackendError("; ".join(failures))


def _video_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = result.stdout.strip().split(",")
    return int(width), int(height)


def rgb_video_stats(path: Path, chunk_frames: int = 8) -> dict:
    """Decode RGB through FFmpeg and stream channel statistics without frame caching."""
    width, height = _video_size(path)
    frame_bytes = width * height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    total = np.zeros(3, dtype=np.float64)
    square = np.zeros(3, dtype=np.float64)
    frames = 0
    while True:
        raw = process.stdout.read(frame_bytes * chunk_frames)
        if not raw:
            break
        if len(raw) % frame_bytes:
            process.kill()
            raise RuntimeError("FFmpeg emitted an incomplete RGB frame")
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
        lo = np.minimum(lo, data.min(axis=0))
        hi = np.maximum(hi, data.max(axis=0))
        total += data.sum(axis=0)
        square += np.square(data).sum(axis=0)
        frames += len(raw) // frame_bytes
    stderr = process.stderr.read().decode("utf8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError("FFmpeg RGB decode failed: " + stderr[-1000:])
    if not frames:
        raise ValueError("empty video: {}".format(path))
    pixels = frames * width * height
    mean = total / pixels
    variance = np.maximum(square / pixels - np.square(mean), 0.0)
    return {
        "min": lo.reshape(3, 1, 1).tolist(),
        "max": hi.reshape(3, 1, 1).tolist(),
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": np.sqrt(variance).reshape(3, 1, 1).tolist(),
        "count": [frames],
    }


def encode_jpeg_sequence(
    selected_files: list[Path],
    destination: Path,
    fps: int,
    preferred_backends: tuple[str, ...] = ("cpu", "qsv", "cuda"),
) -> str:
    """将按时间戳选出的 JPG 序列编码成 MP4。

    默认后端顺序是 CPU → QSV → CUDA；因此 JPG 腕部相机在 CPU 编码成功时，
    不会继续使用 NVENC。输出的 MP4 帧数必须与输入 JPG 数量一致。
    """
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
            # FFmpeg 的图片序列输入要求文件名连续；优先使用硬链接，失败时再复制文件。
            target = directory / "{:06d}.jpg".format(index)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        for backend in preferred_backends:
            try:
                # 同一个输出路径会依次尝试不同编码后端，尝试前先清理旧文件。
                destination.unlink(missing_ok=True)
                # -framerate 是输入图片读取速度；-r 是输出视频帧率。
                # %06d.jpg 对应 000000.jpg、000001.jpg……这样的连续图片序列。
                base = [
                    "-framerate",
                    str(fps),
                    "-i",
                    str(directory / "%06d.jpg"),
                    "-frames:v",
                    str(len(selected_files)),
                    "-r",
                    str(fps),
                    "-pix_fmt",
                    "yuv420p",
                ]
                if backend == "cuda":
                    # NVENC：使用 NVIDIA 的硬件 H.264 编码器；这里主要加速编码阶段。
                    command = [*base, "-c:v", "h264_nvenc", "-preset", "p4", str(destination)]
                elif backend == "qsv":
                    # QSV：使用 Intel 硬件 H.264 编码器。
                    command = [*base, "-c:v", "h264_qsv", "-global_quality", "23", str(destination)]
                else:
                    # libx264：CPU 软件 H.264 编码器；当前默认顺序下通常先走这里。
                    command = [*base, "-c:v", "libx264", "-crf", "20", str(destination)]
                run_ffmpeg(command)
                count = video_frame_count(destination)
                if count != len(selected_files):
                    # 编码过程无报错，但帧数不一致时仍视为失败，继续尝试下一个后端。
                    raise RuntimeError("frame count {} != {}".format(count, len(selected_files)))
                return backend
            except (subprocess.CalledProcessError, RuntimeError) as error:
                # 删除不完整输出并保留错误信息，最终全部失败时一次性汇总抛出。
                destination.unlink(missing_ok=True)
                failures.append("{}: {}".format(backend, error))
    raise FFmpegBackendError("; ".join(failures))

#!/usr/bin/env python3
"""
视频分割工具：

功能：
- 读取一个视频文件路径
- 将视频切分为每段 60s（可配置），若最后一段 < 10s（可配置）则并入上一段
- 视觉：按段导出图像帧（可选指定 fps，不指定则按原始帧率导出）
- 音频：与视觉切割严格对齐，按段导出为 WAV（可配置采样率/是否转单声道）
- 返回：按顺序返回每段的元信息（起止时间、时长、帧目录、音频路径）

依赖：
- 需要本机安装 ffmpeg / ffprobe（通过子进程调用）

示例：
    python video_segmenter.py /path/to/video.mp4 \
        --segment-length 60 --min-last 10 --fps 0 --audio-rate 16000 --mono

说明：
- 如果 `--fps` 未指定或为 0，则按原视频帧率导出所有帧。
- 默认音频导出为 16-bit PCM WAV，可通过参数控制采样率与声道。
- 输出目录默认位于与输入视频同级，命名为 `<视频名>_segments`。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, List, Optional, Sequence


@dataclass
class SegmentInfo:
    index: int
    start: float
    end: float
    duration: float
    frames_dir: str
    audio_path: str
    # 若选择以无声 mp4 作为“视觉帧”输出，则提供 video_path
    video_path: Optional[str] = None


class FFmpegError(RuntimeError):
    pass


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _require_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if _which(b) is None]
    if missing:
        raise EnvironmentError(
            f"缺少依赖: {', '.join(missing)}。请安装 ffmpeg (含 ffprobe)。"
        )


def _run(cmd: List[str]) -> None:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"命令失败: {' '.join(cmd)}\nSTDOUT: {e.stdout.decode(errors='ignore')}\nSTDERR: {e.stderr.decode(errors='ignore')}"
        )


def probe_duration_sec(video_path: str) -> float:
    """使用 ffprobe 获取媒体总时长（秒）。"""
    _require_binaries()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"ffprobe 失败: {e.stderr.decode(errors='ignore')}"
        )
    info = json.loads(res.stdout.decode())
    # 优先用 format.duration；若无，再从最长的 stream 推断
    dur = None
    if "format" in info and info["format"].get("duration"):
        try:
            dur = float(info["format"]["duration"])
        except ValueError:
            dur = None
    if dur is None and "streams" in info:
        for s in info["streams"]:
            d = s.get("duration")
            if d:
                try:
                    val = float(d)
                except ValueError:
                    continue
                dur = max(dur or 0.0, val)
    if dur is None:
        raise FFmpegError("无法从 ffprobe 结果中解析到视频总时长。")
    return float(dur)


def compute_segments(total_duration: float, segment_length: float = 60.0, min_last: float = 10.0) -> List[tuple[float, float]]:
    """根据总时长计算切段边界。

    - 固定分段长度为 `segment_length`
    - 若最后一段 < `min_last`，则并入上一段
    - 返回 [(start, end), ...]
    """
    if total_duration <= 0:
        return [(0.0, 0.0)]

    segments: List[tuple[float, float]] = []
    start = 0.0
    while start + segment_length < total_duration:
        end = start + segment_length
        segments.append((start, end))
        start = end

    # 处理最后一段
    last_end = total_duration
    last_dur = last_end - start
    if segments and last_dur < min_last:
        # 并入上一段
        prev_start, _ = segments[-1]
        segments[-1] = (prev_start, last_end)
    else:
        segments.append((start, last_end))

    # 规范化到非负与递增
    norm = []
    for s, e in segments:
        s = max(0.0, float(s))
        e = max(s, float(e))
        norm.append((s, e))
    return norm


def _format_ts(t: float) -> str:
    """格式化秒为 ffmpeg 可读的秒字符串（保留毫秒）。"""
    return f"{t:.3f}"


def segment_video(
    video_path: str,
    segment_length: float = 60.0,
    min_last: float = 10.0,
    output_dir: Optional[str] = None,
    image_ext: str = "jpg",
    fps: Optional[float] = None,
    audio_rate: Optional[int] = 16000,
    mono: bool = True,
    video_output: bool = False,
) -> List[SegmentInfo]:
    """分割视频并分别导出每段帧与音频，返回段落信息列表。

    参数：
    - video_path: 输入视频路径
    - segment_length: 每段长度（秒）
    - min_last: 最后一段若小于该阈值，则并入上一段
    - output_dir: 输出根目录（默认：与视频同级 `<name>_segments`）
    - image_ext: 帧图像后缀（jpg/png...）
    - fps: 指定导出帧率；None 或 0 表示使用原始帧率（不降采样）
    - audio_rate: 音频采样率；None 表示保持原始采样率
    - mono: 是否转为单声道
    """
    _require_binaries()

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    total_dur = probe_duration_sec(video_path)
    segs = compute_segments(total_dur, segment_length, min_last)

    # 准备输出目录
    if output_dir is None:
        out_root = vpath.with_suffix("").as_posix() + "_segments"
    else:
        out_root = output_dir
    Path(out_root).mkdir(parents=True, exist_ok=True)

    results: List[SegmentInfo] = []
    for idx, (start, end) in enumerate(segs):
        dur = max(0.0, end - start)
        seg_dir = Path(out_root) / f"segment_{idx:04d}"
        frames_dir = seg_dir / "frames"
        seg_dir.mkdir(parents=True, exist_ok=True)
        audio_path = seg_dir / f"audio.wav"
        video_path_obj = seg_dir / f"video.mp4"
        if not video_output:
            frames_dir.mkdir(parents=True, exist_ok=True)

        # 导出帧（只要视频，不要音频）。为保证切割精确，-ss/-t 放在 -i 之后。
        vf_filters: List[str] = []
        if fps and fps > 0:
            vf_filters.append(f"fps={fps}")
        vf_arg: List[str] = []
        if vf_filters:
            vf_arg = ["-vf", ",".join(vf_filters)]

        if video_output:
            # 输出为无声 mp4，必要时重采样帧率
            v_args: List[str] = []
            if fps and fps > 0:
                # 使用 -vf fps=... 保持稳定的帧采样
                v_args += ["-vf", f"fps={fps}"]
            # 重新编码以保证切割精确
            cmd_video = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                vpath.as_posix(),
                "-ss",
                _format_ts(start),
                "-t",
                _format_ts(dur),
                "-an",
                *v_args,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                video_path_obj.as_posix(),
            ]
            _run(cmd_video)
        else:
            frame_pattern = (frames_dir / f"%06d.{image_ext}").as_posix()
            cmd_frames = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                vpath.as_posix(),
                "-ss",
                _format_ts(start),
                "-t",
                _format_ts(dur),
                "-an",
                *vf_arg,
                "-vsync",
                "vfr",
                frame_pattern,
            ]
            _run(cmd_frames)

        # 导出音频（只要音频，不要视频），严格对齐
        a_args: List[str] = []
        if mono:
            a_args += ["-ac", "1"]
        if audio_rate and audio_rate > 0:
            a_args += ["-ar", str(audio_rate)]
        # 使用 PCM 16-bit little endian，保证兼容
        a_args += ["-c:a", "pcm_s16le"]

        cmd_audio = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            vpath.as_posix(),
            "-ss",
            _format_ts(start),
            "-t",
            _format_ts(dur),
            "-vn",
            *a_args,
            audio_path.as_posix(),
        ]
        _run(cmd_audio)

        results.append(
            SegmentInfo(
                index=idx,
                start=start,
                end=end,
                duration=dur,
                frames_dir=frames_dir.as_posix() if not video_output else "",
                audio_path=audio_path.as_posix(),
                video_path=video_path_obj.as_posix() if video_output else None,
            )
        )

    return results


def iter_segments(
    video_path: str,
    segment_length: float = 60.0,
    min_last: float = 10.0,
    output_dir: Optional[str] = None,
    image_ext: str = "jpg",
    fps: Optional[float] = None,
    audio_rate: Optional[int] = 16000,
    mono: bool = True,
) -> Generator[SegmentInfo, None, None]:
    """分割并按顺序逐段产生 SegmentInfo。"""
    segments = segment_video(
        video_path=video_path,
        segment_length=segment_length,
        min_last=min_last,
        output_dir=output_dir,
        image_ext=image_ext,
        fps=fps,
        audio_rate=audio_rate,
        mono=mono,
    )
    for s in segments:
        yield s


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按段切分视频帧与对齐音频")
    p.add_argument("video", help="输入视频文件路径")
    p.add_argument("--segment-length", type=float, default=60.0, help="每段长度（秒）")
    p.add_argument("--min-last", type=float, default=10.0, help="最后一段小于该阈值则并入上一段（秒）")
    p.add_argument("--output-dir", type=str, default=None, help="输出根目录（默认：与视频同级 <name>_segments）")
    p.add_argument("--image-ext", type=str, default="jpg", help="帧图像后缀：jpg/png 等；若使用 --video-output 则忽略")
    p.add_argument("--fps", type=float, default=0.0, help="若>0则指定导出帧率（对图像与 mp4 均生效）；<=0 保留原始帧率")
    p.add_argument("--audio-rate", type=int, default=16000, help="导出音频采样率；<=0 则保留原始")
    p.add_argument("--mono", action="store_true", help="将音频转换为单声道")
    p.add_argument("--stereo", action="store_true", help="保持立体声（若与 --mono 同时给出，以 --stereo 为准）")
    p.add_argument("--video-output", action="store_true", help="将视觉帧输出为无声 mp4（而非图像序列）")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    mono = True
    if args.stereo:
        mono = False
    elif args.mono:
        mono = True

    fps = None if args.fps is None or args.fps <= 0 else float(args.fps)
    audio_rate = None if args.audio_rate is None or args.audio_rate <= 0 else int(args.audio_rate)

    segments = segment_video(
        video_path=args.video,
        segment_length=float(args.segment_length),
        min_last=float(args.min_last),
        output_dir=args.output_dir,
        image_ext=args.image_ext,
        fps=fps,
        audio_rate=audio_rate,
        mono=mono,
        video_output=bool(args.video_output),
    )

    # 终端打印简要结果
    print(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

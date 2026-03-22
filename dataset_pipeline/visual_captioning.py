#!/usr/bin/env python3
"""
Visual captioning for silent video segments using Qwen3-Omni Thinking (vLLM).

功能概述：
- 扫描单个视频文件或分段目录（<video>_segments/segment_0000/...）。
- 仅针对无音轨（silent）的 mp4 段执行视觉描述（visual captioning）。
- 通过统一的 Qwen 客户端调用本地 vLLM OpenAI 兼容接口。
- 内置默认的 system/user prompt；也支持通过参数覆盖（字符串或 @file）。
- 结果默认打印到标准输出；若输入为目录，可为每个段写入 caption.txt。
 - 可选：上传前用 ffmpeg 生成“无音轨”副本（不改原文件）。

依赖：
- pip install python-dotenv
- 可选：ffprobe（用于准确检测是否存在音轨）。若缺失则跳过强校验。
- 本地 vLLM 服务：vllm serve Qwen3-Omni-30B-A3B-Thinking --port 8901 ...

示例：
  1) 针对单个无声 mp4：
     python visual_captioning.py \
       --input /path/to/<video>_segments/segment_0000/video.mp4 \
       [--strip-audio]

  2) 针对整个 segments 根目录（逐段处理）：
     python visual_captioning.py \
       --input /path/to/<video>_segments \
       --out-dir /path/to/<video>_segments \
       [--require-silent] [--strip-audio]

注意：如需让 video_segmenter 产出无声 mp4，请使用其 --video-output 参数，或启用 --strip-audio 由脚本自动提取无声副本。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import tempfile
from dotenv import load_dotenv
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, QwenClientError, default_client


# ----------------------------
# Defaults: external prompts
# ----------------------------

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "visual_captioning_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "visual_captioning_user.md"


# ----------------------------
# Utils: environment and deps
# ----------------------------

def _load_env() -> None:
    # Load from local .env if present (repo keeps one under code/.env)
    # We call load_dotenv() twice to cover both repo root and script dir cases.
    load_dotenv()  # default search from CWD upward
    # Also try alongside this script explicitly
    script_dir = Path(__file__).resolve().parent
    env_candidate = script_dir / ".env"
    if env_candidate.exists():
        load_dotenv(env_candidate.as_posix(), override=False)


# ----------------------------
# ffprobe-based audio detection
# ----------------------------

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def has_audio_stream(video_path: Path) -> Optional[bool]:
    """Return True if video has audio stream, False if none, None if unknown.

    Uses ffprobe when available; if not installed or any error occurs, returns None.
    """
    if _which("ffprobe") is None:
        return None
    try:
        # If there is at least one audio stream, ffprobe will print indices.
        # Empty output => no audio.
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    out = proc.stdout.decode(errors="ignore").strip()
    return bool(out)  # non-empty => has audio


def strip_audio_copy(video_path: Path, out_dir: Path) -> Optional[Path]:
    """Create a silent copy of the given video using ffmpeg. Returns new path or None.

    Preferred: stream copy with -c copy -an. If that fails, re-encode H.264.
    """
    if _which("ffmpeg") is None:
        return None
    out_path = out_dir / f"{video_path.stem}.silent{video_path.suffix}"

    # Try stream copy first
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path.as_posix(),
                "-c",
                "copy",
                "-an",
                out_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass

    # Fallback: re-encode video only
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path.as_posix(),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                out_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        return None
    return None


# ----------------------------
# Scan for silent videos
# ----------------------------

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}


def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES


def find_segment_videos(root: Path) -> List[Path]:
    """Heuristics to find per-segment silent mp4 files.

    Priority:
      1) <root>/segment_XXXX/video.mp4 (depth=1)
      2) any video.mp4 named exactly under subdirs
      3) fallback: any *.mp4|*.webm|*.mkv|*.mov under root
    """
    results: List[Path] = []
    if root.is_file() and _is_video_file(root):
        return [root]

    if not root.is_dir():
        return []

    # 1) Typical layout from video_segmenter with --video-output
    for seg_dir in sorted(root.glob("segment_*")):
        v = seg_dir / "video.mp4"
        if v.exists() and v.is_file():
            results.append(v)
    if results:
        return results

    # 2) Any file named video.mp4 under subtree
    for v in root.rglob("video.mp4"):
        if v.is_file():
            results.append(v)
    if results:
        return results

    # 3) Any known video file type under subtree
    for v in root.rglob("*"):
        if _is_video_file(v):
            results.append(v)
    return results


# ----------------------------
# Qwen captioning
# ----------------------------

@dataclass
class CaptionJob:
    video: Path
    out_path: Optional[Path]  # where to write text; None => print only


def _read_prompt_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("@") and len(value) > 1:
        p = Path(value[1:])
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text(encoding="utf-8")
    return value


# Prefer reading default prompts from markdown files; fall back to None when missing.
def _load_default_prompt(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return text or None


# 找到这个函数并用下面的代码替换它
def upload_and_caption(
    client,
    video_path: Path,
    system_prompt: Optional[str],
    user_prompt: str,
    *,
    model: Optional[str],
) -> str:
    text_chunks: List[str] = []
    if user_prompt:
        text_chunks.append(user_prompt)
    else:
        text_chunks.append("请详细描述该视频的视觉内容。")

    return client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        video_path=video_path,
        model=model,
        temperature=0.2,
        top_p=0.9,
    )


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual captioning for silent video segments using Qwen3-Omni Thinking")
    p.add_argument("--input", required=True, help="Path to a silent mp4 or segments root/segment dir")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt text or @file (default: prompts/visual_captioning_system.md)",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt text or @file (default: prompts/visual_captioning_user.md)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="If set and input is a directory, write each caption to <out-dir>/<segment>/caption.txt",
    )
    p.add_argument(
        "--require-silent",
        action="store_true",
        help="Skip files that are detected to contain audio (requires ffprobe)",
    )
    p.add_argument(
        "--strip-audio",
        action="store_true",
        help="Before upload, create a silent copy of each video (requires ffmpeg)",
    )
    p.add_argument("--dry-run", action="store_true", help="List files to be processed without calling the API")
    return p.parse_args(argv)


def _plan_jobs(input_path: Path, out_dir: Optional[Path]) -> List[CaptionJob]:
    videos = find_segment_videos(input_path)
    jobs: List[CaptionJob] = []

    if not videos:
        return []

    # If input is a single file and no out_dir given, default to print only
    if input_path.is_file() and (out_dir is None):
        return [CaptionJob(video=videos[0], out_path=None)]

    # Directory input: compute out path per segment directory
    for v in videos:
        seg_dir = v.parent if v.parent.name.startswith("segment_") else v.parent
        out_base = out_dir if out_dir is not None else seg_dir
        out_path = Path(out_base) / seg_dir.name / "caption.txt" if out_dir is not None else seg_dir / "caption.txt"
        # Ensure parent dir path is consistent when out_dir is set
        if out_dir is not None:
            (Path(out_base) / seg_dir.name).mkdir(parents=True, exist_ok=True)
        jobs.append(CaptionJob(video=v, out_path=out_path))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)

    # Prepare prompts: prefer CLI overrides, else load markdown defaults
    default_system_prompt = _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    default_user_prompt = _load_default_prompt(DEFAULT_USER_PROMPT_PATH)
    system_prompt = _read_prompt_value(args.system) if args.system else default_system_prompt
    user_prompt = _read_prompt_value(args.user) if args.user else default_user_prompt
    effective_user_prompt = user_prompt or ""

    model = args.model or QWEN_DEFAULT_MODEL

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"未找到输入路径: {input_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else None
    jobs = _plan_jobs(input_path, out_dir)
    if not jobs:
        print("未找到可处理的视频文件（期望无声 mp4）。", file=sys.stderr)
        return 1

    # Pre-check: filter by silence if requested
    if args.require_silent:
        filtered: List[CaptionJob] = []
        for j in jobs:
            has_audio = has_audio_stream(j.video)
            if has_audio is None:
                print(f"[警告] 无法检测音轨（未安装 ffprobe），仍将处理: {j.video}")
                filtered.append(j)
            elif has_audio:
                print(f"[跳过] 检测到音轨（非无声）: {j.video}")
            else:
                filtered.append(j)
        jobs = filtered
        if not jobs:
            print("根据 --require-silent 过滤后无可处理文件。", file=sys.stderr)
            return 1

    if args.dry_run:
        print("将要处理以下文件（dry-run）:")
        for j in jobs:
            print(f"- {j.video}")
        return 0

    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    # Prepare temp workspace for optional audio stripping
    temp_root: Optional[Path] = None
    if args.strip_audio:
        temp_root = Path(tempfile.mkdtemp(prefix="vcap_"))

    # Process sequentially
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] 上传并生成描述: {job.video}")
        use_path = job.video
        created_file: Optional[Path] = None
        if args.strip_audio:
            if temp_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="vcap_"))
            silent = strip_audio_copy(job.video, temp_root)
            if silent is not None:
                use_path = silent
                created_file = silent
        try:
            text = upload_and_caption(
                client=client,
                video_path=use_path,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
                model=model,
            )
        except QwenClientError as e:
            print(f"  -> 失败: {e}", file=sys.stderr)
            continue
        finally:
            if created_file is not None and created_file != job.video:
                try:
                    created_file.unlink(missing_ok=True)
                except Exception:
                    pass

        if job.out_path is None:
            print("--- Caption Start ---")
            print(text)
            print("--- Caption End ---")
        else:
            try:
                job.out_path.parent.mkdir(parents=True, exist_ok=True)
                job.out_path.write_text(text, encoding="utf-8")
                print(f"  -> 已写入: {job.out_path}")
            except Exception as e:
                print(f"  -> 保存失败: {e}", file=sys.stderr)

    # Cleanup temp root directory
    if temp_root is not None:
        try:
            # Attempt to remove directory if empty
            for p in temp_root.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            temp_root.rmdir()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

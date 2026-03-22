#!/usr/bin/env python3
"""
Vocal captioning for per-segment videos using Qwen3-Omni with ASR context.

目标：
- 对 video_segmenter 切出的每个 segment，不再二次切分；
- 用 audio_preprocess 产出的 voice.* 作为唯一音轨，与该段 video.mp4 复合生成一段“仅人声”的 mp4；
- 运行 ASR（调用本地 Qwen3-Omni），将识别文本一并提供给同一模型做 caption（可自备 system/user prompt）。

输入：
- --input 指向某个 segment_XXXX 目录或整个 segments 根目录。

产出：
- <segment>/voice_clip.mp4 （或 --out-dir 镜像路径下）
- <segment>/voice_asr.txt （ASR 文本）
- <segment>/vocal_caption.txt （结合 ASR 的 Qwen caption）

依赖：
- ffmpeg/ffprobe
- python-dotenv
- 本地 vLLM 提供的 Qwen3-Omni-30B-A3B-Thinking 服务（或设置 QWEN_MODEL）

示例：
  python code/vocal_caption.py \
    --input /path/to/<video>_segments/segment_0000 \
    --model Qwen3-Omni-30B-A3B-Thinking \
    --system @system.txt \
    --user @user.txt

  # 针对根目录逐段处理
  python code/vocal_caption.py --input /path/to/<video>_segments --model Qwen3-Omni-30B-A3B-Thinking
"""

from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, QwenClientError, default_client
from temp_utils import prepare_temp_dir, stage_file, cleanup_temp_dir


SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------
# Env & deps
# ----------------------------

def _load_env() -> None:
    load_dotenv()
    env_candidate = SCRIPT_DIR / ".env"
    if env_candidate.exists():
        load_dotenv(env_candidate.as_posix(), override=False)


class FFmpegError(RuntimeError):
    pass


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _require_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if _which(b) is None]
    if missing:
        raise EnvironmentError(f"缺少依赖: {', '.join(missing)}。请安装 ffmpeg (含 ffprobe)。")


def _run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"命令失败: {' '.join(cmd)}\nSTDOUT: {e.stdout.decode(errors='ignore')}\nSTDERR: {e.stderr.decode(errors='ignore')}"
        )


# ----------------------------
# Probe helpers
# ----------------------------

def probe_duration_sec(media_path: Path) -> float:
    _require_binaries()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", media_path.as_posix()
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"ffprobe 失败: {e.stderr.decode(errors='ignore')}")
    import json as _json
    info = _json.loads(res.stdout.decode() or "{}")
    try:
        return float(info["format"]["duration"])  # seconds
    except Exception:
        raise FFmpegError("无法从 ffprobe 输出解析 duration")


# ----------------------------
# Locate inputs
# ----------------------------

def find_segment_dirs(root: Path) -> List[Path]:
    if root.is_dir() and root.name.startswith("segment_"):
        return [root]
    if root.is_dir():
        ds = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("segment_")]
        if ds:
            return ds
    return []


VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}


def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES


def resolve_segment_video(seg_dir: Path) -> Optional[Path]:
    v = seg_dir / "video.mp4"
    if v.exists() and v.is_file():
        return v
    for p in seg_dir.iterdir():
        if _is_video_file(p):
            return p
    return None


def find_voice_file(seg_dir: Path) -> Optional[Path]:
    # 1) <seg>/audio_sep/voice.*
    d = seg_dir / "audio_sep"
    if d.exists() and d.is_dir():
        for p in d.iterdir():
            if p.is_file() and p.name.startswith("voice."):
                return p
    # 2) nested audio_sep/voice.*
    for p in seg_dir.rglob("audio_sep/voice.*"):
        if p.is_file():
            return p
    # 3) *.lalalai/voice.*
    for d in seg_dir.rglob("*.lalalai"):
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.name.startswith("voice."):
                    return p
    return None


# ----------------------------
# Mux: video + voice (no re-seg)
# ----------------------------

def mux_video_with_voice(base_video: Path, voice_audio: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", base_video.as_posix(),
        "-i", voice_audio.as_posix(),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_path.as_posix(),
    ]
    _run(cmd)


# ----------------------------
# Gemini upload & caption
# ----------------------------
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "vocal_caption_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "vocal_caption_user.md"


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


def _load_default_prompt(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return text or None


def upload_and_caption(
    client,
    media_path: Path,
    *,
    system_prompt: Optional[str],
    user_prompt: str,
    asr_text: Optional[str],
    model: Optional[str],
    audio_path: Optional[Path] = None,
) -> str:
    text_chunks: List[str] = []
    effective_prompt = user_prompt or ""
    if effective_prompt:
        text_chunks.append(effective_prompt)
    if asr_text and asr_text.strip():
        text_chunks.append("ASR transcript (for reference):\n" + asr_text.strip())
    if not text_chunks:
        text_chunks.append("请根据提供的视频生成详尽的人声描述。")

    return client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        video_path=media_path,
        audio_path=audio_path,
        model=model,
        temperature=0.25,
        top_p=0.9,
    )


# ----------------------------
# ASR via Qwen
# ----------------------------

TRANSCRIPT_SYSTEM_PROMPT = "You are a meticulous transcription assistant. Output only the verbatim transcript."
TRANSCRIPT_USER_PROMPT = (
    "请逐字转写所附音频内容，保持时间顺序，不要添加解释或额外文本。"
)


def run_asr_capture(
    client,
    audio_path: Path,
    model: Optional[str],
) -> Optional[str]:
    try:
        text = client.generate(
            system_prompt=TRANSCRIPT_SYSTEM_PROMPT,
            text_chunks=[TRANSCRIPT_USER_PROMPT],
            audio_path=audio_path,
            model=model,
            temperature=0.0,
            top_p=0.5,
        )
    except QwenClientError as exc:
        print(f"[警告] Qwen ASR 请求失败: {exc}", file=sys.stderr)
        return None
    return text.strip() or None


# ----------------------------
# Orchestration
# ----------------------------

@dataclass
class VocalJob:
    seg_dir: Path
    video: Path
    voice: Path
    out_dir: Path  # base out dir for this segment


def plan_jobs(input_path: Path, out_root: Optional[Path]) -> List[VocalJob]:
    segs = find_segment_dirs(input_path)
    jobs: List[VocalJob] = []
    for seg in segs:
        v = resolve_segment_video(seg)
        if v is None:
            print(f"[跳过] 未找到视频: {seg}")
            continue
        voice = find_voice_file(seg)
        if voice is None:
            print(f"[跳过] 未找到 voice.*（请先运行 audio_preprocess）: {seg}")
            continue
        out_dir = (seg if out_root is None else Path(out_root) / seg.name)
        jobs.append(VocalJob(seg_dir=seg, video=v, voice=voice, out_dir=out_dir))
    return jobs


def process_segment(
    job: VocalJob,
    client,
    model: Optional[str],
    system_prompt: Optional[str],
    user_prompt: str,
    skip_asr: bool = False,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """处理单段：生成 voice_clip.mp4，运行 ASR 并写入 voice_asr.txt，调用 Qwen 生成 vocal_caption.txt。
    返回 (voice_clip, asr_txt_path, caption_txt_path)。
    """
    clip_path = job.out_dir / "voice_clip.mp4"
    asr_txt_path = job.out_dir / "voice_asr.txt"
    cap_txt_path = job.out_dir / "vocal_caption.txt"
    tmp_dir = prepare_temp_dir(job.seg_dir)
    try:
        temp_clip_path = tmp_dir / "voice_clip.mp4"

        voice_for_processing = job.voice
        try:
            voice_for_processing = stage_file(job.voice, tmp_dir)
        except Exception as exc:
            print(f"  -> 临时目录写入 voice 音频失败，改为直接使用原路径: {exc}", file=sys.stderr)

        # 1) 复合视频+人声
        try:
            mux_video_with_voice(job.video, voice_for_processing, temp_clip_path)
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            if clip_path.resolve() != temp_clip_path.resolve():
                shutil.copy2(temp_clip_path, clip_path)
        except Exception as e:
            print(f"  -> 复合失败: {clip_path} -> {e}", file=sys.stderr)
            return (None, None, None)

        # 2) ASR
        asr_text: Optional[str] = None
        if not skip_asr:
            asr_text = run_asr_capture(client, voice_for_processing, model)
            if asr_text:
                try:
                    asr_txt_path.parent.mkdir(parents=True, exist_ok=True)
                    asr_txt_path.write_text(asr_text, encoding="utf-8")
                except Exception:
                    pass
            else:
                print("  -> ASR 为空或失败，继续进行 caption（无文本上下文）")

        # 3) Gemini caption
        try:
            text = upload_and_caption(
                client,
                media_path=temp_clip_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                asr_text=asr_text,
                model=model,
                audio_path=voice_for_processing,
            )
            cap_txt_path.parent.mkdir(parents=True, exist_ok=True)
            cap_txt_path.write_text(text, encoding="utf-8")
        except QwenClientError as e:
            print(f"  -> Caption 失败: {e}", file=sys.stderr)
            return (clip_path, asr_txt_path if asr_text else None, None)

        return (clip_path, asr_txt_path if asr_text else None, cap_txt_path)
    finally:
        cleanup_temp_dir(tmp_dir)


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-segment vocal captioning with ASR context (Qwen3-Omni)")
    p.add_argument("--input", required=True, help="Path to a segment dir or segments root")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt text or @file (default: prompts/vocal_caption_system.md)",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt text or @file (default: prompts/vocal_caption_user.md)",
    )
    p.add_argument("--model", default=QWEN_DEFAULT_MODEL, help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=None, help="If set, write outputs under <out-dir>/<segment>/...")
    p.add_argument("--skip-asr", action="store_true", help="Skip ASR and caption without transcript context")
    p.add_argument("--dry-run", action="store_true", help="Plan only, do not mux/ASR/upload")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"未找到输入路径: {input_path}", file=sys.stderr)
        return 2

    default_system_prompt = _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    default_user_prompt = _load_default_prompt(DEFAULT_USER_PROMPT_PATH)
    system_prompt = _read_prompt_value(args.system) if args.system else default_system_prompt
    user_prompt = _read_prompt_value(args.user) if args.user else default_user_prompt
    effective_user_prompt = user_prompt or ""
    model = args.model or QWEN_DEFAULT_MODEL
    out_root = Path(args.out_dir) if args.out_dir else None

    jobs = plan_jobs(input_path, out_root)
    if not jobs:
        print("未找到可处理的 segment（需要 video.mp4 与 audio_sep/voice.*）。", file=sys.stderr)
        return 1

    if args.dry_run:
        print("将要处理以下 segments：")
        for j in jobs:
            print(f"- {j.seg_dir} -> voice: {j.voice.name}, out: {j.out_dir}")
        return 0

    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] 处理 segment: {job.seg_dir}")
        try:
            clip, asr_txt, cap_txt = process_segment(
                job,
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
                skip_asr=bool(args.skip_asr),
            )
        except QwenClientError as e:
            print(f"  -> 失败: {e}", file=sys.stderr)
            continue
        if cap_txt:
            print(f"  -> 已写入 caption: {cap_txt}")
        elif clip:
            print("  -> 已生成 voice_clip，但 caption 失败")

    return 0


if __name__ == "__main__":
    sys.exit(main())

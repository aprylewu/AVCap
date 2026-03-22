#!/usr/bin/env python3
"""
Joint captioning for a segment: fuse original full-audio video with expert captions
and generate a high-granularity audio-visual caption via Qwen3-Omni Thinking (vLLM).

目标：
- 针对 video_segmenter 切分出的某一段（或一组段），将“原始视频（含完整音轨）”与以下专家输出一起提供给 Qwen 模型：
  * vocal_caption（来自 vocal_caption.py 的输出）
  * bgm_caption（来自 bgm_caption.py 的输出）
  * visual_caption（来自 visual_captioning.py 的输出）
- 由 Qwen 参考上述专家输出并结合原视频，生成更高细粒度的“音视频联合 caption”。

约束：
- 所有操作严格限定在单个 segment 内，不跨越分段边界。

输入：
- --input 指向 segment_XXXX 目录或整个 segments 根目录。

产出（默认写回各 segment 目录；若提供 --out-dir，则镜像到该目录下）：
- <segment>/full_av.mp4           （复合后的全音轨视频：video.mp4 + audio.wav）
- <segment>/joint_caption.txt     （Qwen 生成的联合 caption）

依赖：
- ffmpeg/ffprobe
- python-dotenv
- 本地 vLLM 服务（Qwen3-Omni-30B-A3B-Thinking，或设置 QWEN_MODEL）

用法：
  python code/joint_caption.py \
    --input /path/to/<video>_segments/segment_0000 \
    --model Qwen3-Omni-30B-A3B-Thinking \
    --system @system.txt \
    --user @user.txt

  # 针对根目录逐段处理
  python code/joint_caption.py --input /path/to/<video>_segments --model Qwen3-Omni-30B-A3B-Thinking
"""

from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def resolve_segment_audio(seg_dir: Path) -> Optional[Path]:
    a = seg_dir / "audio.wav"
    if a.exists() and a.is_file():
        return a
    # 兜底：查找常见音频文件名
    for p in seg_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
            return p
    return None


def _compose_bgm_per_clip(seg_dir: Path) -> Optional[str]:
    """优先读取 <segment>/bgm_chunks/clip_*.txt 并按 Clip 1/2/... 组合，
    以便 Qwen 在理解 clip 顺序时使用起始时间 0 语义。若找不到则返回 None。
    """
    chunks_dir = seg_dir / "bgm_chunks"
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        return None
    txts = [p for p in chunks_dir.iterdir() if p.is_file() and p.name.startswith("clip_") and p.suffix == ".txt"]
    if not txts:
        return None

    def _key(p: Path) -> int:
        # clip_000.txt -> 0
        stem = p.stem  # clip_000
        try:
            return int(stem.split("_")[1])
        except Exception:
            return 10**9

    txts.sort(key=_key)
    parts: List[str] = []
    parts.append("BGM expert captions per clip (timestamps reset at 0 for each clip):")
    for i, p in enumerate(txts, 1):
        try:
            body = p.read_text(encoding="utf-8").strip()
        except Exception:
            body = ""
        parts.append(f"\n[Clip {i}]\n{body}" if body else f"\n[Clip {i}]\n<empty>")
    return "\n".join(parts).strip()


def collect_expert_texts(seg_dir: Path) -> Dict[str, str]:
    """读取专家输出文本：
    - vocal_caption.txt
    - bgm_chunks/clip_*.txt 并按 [Clip N] 组合（若无则回退到 bgm_captions.txt）
    - caption.txt（visual）
    - voice_asr.txt（可选）
    """
    results: Dict[str, str] = {}

    # Vocal
    vc = seg_dir / "vocal_caption.txt"
    if vc.exists():
        try:
            results["vocal_caption"] = vc.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # BGM: prefer per-clip composition
    per_clip = _compose_bgm_per_clip(seg_dir)
    if per_clip:
        results["bgm_caption"] = per_clip
    else:
        bgm = seg_dir / "bgm_captions.txt"
        if bgm.exists():
            try:
                results["bgm_caption"] = bgm.read_text(encoding="utf-8").strip()
            except Exception:
                pass

    # Visual
    visual = seg_dir / "caption.txt"
    if visual.exists():
        try:
            results["visual_caption"] = visual.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # Optional ASR
    asr_txt = seg_dir / "voice_asr.txt"
    if asr_txt.exists():
        try:
            results["voice_asr"] = asr_txt.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return results


# ----------------------------
# Assemble full A/V for the segment
# ----------------------------

def mux_full_av(silent_video: Path, audio_file: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", silent_video.as_posix(),
        "-i", audio_file.as_posix(),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_path.as_posix(),
    ]
    _run(cmd)


# ----------------------------
# Qwen captioning
# ----------------------------

PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "joint_caption_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "joint_caption_user.md"


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
    expert_texts: Dict[str, str],
    system_prompt: Optional[str],
    user_prompt: str,
    *,
    model: Optional[str],
    audio_path: Optional[Path] = None,
) -> str:
    label_order = [
        ("vocal_caption", "Vocal expert caption"),
        ("bgm_caption", "BGM expert caption"),
        ("visual_caption", "Visual expert caption"),
        ("voice_asr", "ASR transcript"),
    ]

    text_chunks: List[str] = []
    for key, label in label_order:
        val = expert_texts.get(key)
        if val and val.strip():
            text_chunks.append(f"{label}:\n{val.strip()}")
    if user_prompt:
        text_chunks.append(user_prompt)
    if not text_chunks:
        text_chunks.append("请综合所有可用信息生成详细的音视频联合描述。")

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
# Orchestration
# ----------------------------

@dataclass
class JointJob:
    seg_dir: Path
    video_silent: Path
    audio_full: Path
    out_dir: Path


def plan_jobs(input_path: Path, out_root: Optional[Path]) -> List[JointJob]:
    segs = find_segment_dirs(input_path)
    jobs: List[JointJob] = []
    for seg in segs:
        v = resolve_segment_video(seg)
        a = resolve_segment_audio(seg)
        if v is None or a is None:
            miss = []
            if v is None:
                miss.append("video.mp4")
            if a is None:
                miss.append("audio.wav")
            print(f"[跳过] 缺少 {', '.join(miss)}: {seg}")
            continue
        out_dir = seg if out_root is None else Path(out_root) / seg.name
        jobs.append(JointJob(seg_dir=seg, video_silent=v, audio_full=a, out_dir=out_dir))
    return jobs


def process_segment(
    job: JointJob,
    client,
    model: Optional[str],
    system_prompt: Optional[str],
    user_prompt: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """为该 segment 生成 full_av.mp4，并调用 Qwen 参考专家文本生成联合 caption。
    返回 (full_av_path, joint_caption_path)。
    """
    full_av = job.out_dir / "full_av.mp4"
    joint_txt = job.out_dir / "joint_caption.txt"
    tmp_dir = prepare_temp_dir(job.seg_dir)
    try:
        temp_full_av = tmp_dir / "full_av.mp4"

        staged_audio = job.audio_full
        try:
            staged_audio = stage_file(job.audio_full, tmp_dir, name=f"full_audio{job.audio_full.suffix}")
        except Exception as exc:
            print(f"  -> 临时目录写入完整音频失败，改为直接使用原路径: {exc}", file=sys.stderr)
            staged_audio = job.audio_full

        # 1) 复合原始视频（含完整音轨）
        try:
            mux_full_av(job.video_silent, staged_audio, temp_full_av)
            full_av.parent.mkdir(parents=True, exist_ok=True)
            if full_av.resolve() != temp_full_av.resolve():
                shutil.copy2(temp_full_av, full_av)
        except Exception as e:
            print(f"  -> 复合失败: {full_av} -> {e}", file=sys.stderr)
            return (None, None)

        # 2) 收集专家文本
        experts = collect_expert_texts(job.seg_dir)

        # 3) Qwen 生成联合 caption
        try:
            text = upload_and_caption(
                client=client,
                media_path=temp_full_av,
                expert_texts=experts,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                audio_path=staged_audio,
            )
            joint_txt.parent.mkdir(parents=True, exist_ok=True)
            joint_txt.write_text(text, encoding="utf-8")
        except QwenClientError as e:
            print(f"  -> Caption 失败: {e}", file=sys.stderr)
            return (full_av, None)

        return (full_av, joint_txt)
    finally:
        cleanup_temp_dir(tmp_dir)


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint audio-visual captioning with expert prompts (Qwen3-Omni Thinking)")
    p.add_argument("--input", required=True, help="Path to a segment dir or segments root")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt text or @file (default: prompts/joint_caption_system.md)",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt text or @file (default: prompts/joint_caption_user.md)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument("--out-dir", default=None, help="If set, write outputs under <out-dir>/<segment>/...")
    p.add_argument("--dry-run", action="store_true", help="Plan only, do not mux/upload")
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
        print("未找到可处理的 segment（需要 video.mp4 与 audio.wav）。", file=sys.stderr)
        return 1

    if args.dry_run:
        print("将要处理以下 segments：")
        for j in jobs:
            print(f"- {j.seg_dir} -> out: {j.out_dir}")
        return 0

    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] 处理 segment: {job.seg_dir}")
        try:
            full_av, joint_txt = process_segment(
                job,
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
            )
        except QwenClientError as e:
            print(f"  -> 失败: {e}", file=sys.stderr)
            continue
        if joint_txt:
            print(f"  -> 已写入联合 caption: {joint_txt}")
        elif full_av:
            print("  -> 已生成 full_av，但 caption 失败")

    return 0


if __name__ == "__main__":
    sys.exit(main())

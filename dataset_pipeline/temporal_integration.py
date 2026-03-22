#!/usr/bin/env python3
"""
Temporal integration of per-segment joint captions using Qwen3-Omni Thinking.

目的：
- 给定一个原始长视频路径与其 segments 根目录：
  - 若分段数 >= 2：收集每段的 joint_caption.txt，连同“完整原视频（不切分）”一起提供给 Qwen 模型，
    让其在时间维度上衔接这些段落 caption，生成整部视频的高细粒度联合描述。
  - 若分段数 == 1：直接返回该段的 joint caption（不再调用 Qwen 接口）。

限制：
- 本脚本不负责跑完整的 caption 流程，仅整合已存在的 joint_caption.txt。

输出：
- <segments_root>/temporal_caption.txt（或 --out-dir 镜像路径下）

依赖：
- ffmpeg/ffprobe（仅用于探测每段时长，可选）
- python-dotenv
- 本地 vLLM 提供的 Qwen3-Omni-30B-A3B-Thinking（当分段数>=2时需要，或设置 QWEN_MODEL）
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


# ----------------------------
# Env & deps
# ----------------------------

def _load_env() -> None:
    load_dotenv()
    script_dir = Path(__file__).resolve().parent
    env_candidate = script_dir / ".env"
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
# Helpers
# ----------------------------

def _format_ts(t: float) -> str:
    return f"{t:.3f}"


def probe_duration_sec(media_path: Path) -> Optional[float]:
    if _which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", media_path.as_posix()
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return None
    import json as _json
    try:
        info = _json.loads(res.stdout.decode() or "{}")
        return float(info["format"]["duration"])  # seconds
    except Exception:
        return None


def default_segments_root(video_path: Path) -> Path:
    return video_path.with_suffix("").parent / f"{video_path.stem}_segments"


def find_segment_dirs(segments_root: Path) -> List[Path]:
    if not segments_root.exists() or not segments_root.is_dir():
        return []
    return [p for p in sorted(segments_root.iterdir()) if p.is_dir() and p.name.startswith("segment_")]


def collect_joint_captions(seg_dirs: List[Path]) -> List[Tuple[Path, Optional[float], Optional[str]]]:
    results: List[Tuple[Path, Optional[float], Optional[str]]] = []
    for seg in seg_dirs:
        text: Optional[str] = None
        j = seg / "joint_caption.txt"
        if j.exists():
            try:
                text = j.read_text(encoding="utf-8").strip()
            except Exception:
                text = None
        # Try probe segment video duration to include approx timing
        dur = None
        v = seg / "video.mp4"
        if v.exists():
            dur = probe_duration_sec(v)
        results.append((seg, dur, text))
    return results


def compose_segments_payload(items: List[Tuple[Path, Optional[float], str]]) -> str:
    parts: List[str] = []
    parts.append("Per-segment joint captions in order (with approximate durations):")
    for idx, (seg, dur, text) in enumerate(items, 1):
        hdr = f"[Segment {idx} - {seg.name} - approx {dur:.1f}s]" if dur else f"[Segment {idx} - {seg.name}]"
        body = (text or "<missing>").strip()
        parts.append(hdr + "\n" + body)
    return "\n\n".join(parts).strip()


# ----------------------------
# Qwen
# ----------------------------

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "temporal_integration_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "temporal_integration_user.md"


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
    full_video: Path,
    payload_text: str,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    *,
    model: Optional[str],
) -> str:
    text_chunks: List[str] = [payload_text]
    if user_prompt:
        text_chunks.append(user_prompt)

    return client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        video_path=full_video,
        model=model,
        temperature=0.2,
        top_p=0.9,
    )


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal integration of per-segment joint captions using Qwen3-Omni Thinking")
    p.add_argument("--video", required=True, help="Path to the original full-length video")
    p.add_argument("--segments-root", default=None, help="Segments root dir (default: <video>_segments next to the video)")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt or @file (default: prompts/temporal_integration_system.md)",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt or @file (default: prompts/temporal_integration_user.md)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument("--out-dir", default=None, help="If set, write temporal_caption.txt under here; else under segments root")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not upload")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()

    args = _parse_args(argv)
    video = Path(args.video)
    if not video.exists():
        print(f"未找到原始视频: {video}", file=sys.stderr)
        return 2

    segments_root = Path(args.segments_root) if args.segments_root else default_segments_root(video)
    if not segments_root.exists():
        print(f"未找到 segments 根目录: {segments_root}", file=sys.stderr)
        return 2

    seg_dirs = find_segment_dirs(segments_root)
    if not seg_dirs:
        print(f"在 {segments_root} 下未找到 segment_* 目录", file=sys.stderr)
        return 1

    items = collect_joint_captions(seg_dirs)
    existing = [(s, d, t) for (s, d, t) in items if t and t.strip()]
    if not existing:
        print("未找到任何 joint_caption.txt，可先运行每段的 joint caption 流程。", file=sys.stderr)
        return 1

    # 单段：直接返回其 caption
    if len(existing) == 1:
        _, _, text = existing[0]
        out_base = Path(args.out_dir) if args.out_dir else segments_root
        out_path = out_base / "temporal_caption.txt"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text or "", encoding="utf-8")
        except Exception as e:
            print(f"写入失败: {e}", file=sys.stderr)
            return 2
        print(f"已写入: {out_path} (单段直接返回)")
        return 0

    # 多段：整合并调用 Qwen
    payload = compose_segments_payload(existing)
    if args.dry_run:
        print("将要整合以下段落：")
        for i, (s, d, _) in enumerate(existing, 1):
            dd = f" ~{d:.1f}s" if d else ""
            print(f"- [{i}] {s.name}{dd}")
        return 0

    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    default_system_prompt = _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    default_user_prompt = _load_default_prompt(DEFAULT_USER_PROMPT_PATH)
    system_prompt = _read_prompt_value(args.system) if args.system else default_system_prompt
    user_prompt = _read_prompt_value(args.user) if args.user else default_user_prompt
    effective_user_prompt = user_prompt or ""
    model = args.model or QWEN_DEFAULT_MODEL

    try:
        text = upload_and_caption(
            client=client,
            full_video=video,
            payload_text=payload,
            system_prompt=system_prompt,
            user_prompt=effective_user_prompt,
            model=model,
        )
    except QwenClientError as e:
        print(f"生成失败: {e}", file=sys.stderr)
        return 2

    out_base = Path(args.out_dir) if args.out_dir else segments_root
    out_path = out_base / "temporal_caption.txt"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"已写入: {out_path}")
    except Exception as e:
        print(f"写入失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Visual-focused review of the final temporal caption using local Qwen3-Omni Thinking.

This script samples silent segment videos one frame every N seconds (default: 2s),
passes the chronologically ordered frames together with the final temporal caption
to the local Qwen model served via vLLM, and requests a structured JSON score that
captures visual alignment quality.

Usage example:
    python visual_filtering.py \
        --segments-root /path/to/video_segments \
        --model Qwen3-Omni-30B-A3B-Thinking

Default prompts live under prompts/visual_filtering_*.md; pass --system/--user with a
string or @path to override at runtime.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, QwenClientError, default_client

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "visual_filtering_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "visual_filtering_user.md"
DEFAULT_VISUAL_FILENAME = "caption.txt"

DEFAULT_FRAME_INTERVAL = 5.0
DEFAULT_MAX_TOTAL_FRAMES = 20
DEFAULT_FRAME_WIDTH = 512


def _load_env() -> None:
    load_dotenv()
    script_dir = Path(__file__).resolve().parent
    env_candidate = script_dir / ".env"
    if env_candidate.exists():
        load_dotenv(env_candidate.as_posix(), override=False)


def _read_text(path: Path) -> Optional[str]:
    try:
        data = path.read_text(encoding="utf-8")
    except Exception:
        return None
    data = data.strip()
    return data or None


def _read_prompt_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("@") and len(value) > 1:
        path = Path(value[1:])
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _load_default_prompt(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return text or None


def discover_visual_captions(
    segments_root: Path,
    filename: str = DEFAULT_VISUAL_FILENAME,
) -> List[Tuple[Path, str]]:
    entries: List[Tuple[Path, str]] = []
    if not segments_root.exists() or not segments_root.is_dir():
        return entries
    for seg in sorted(segments_root.iterdir()):
        if not seg.is_dir() or not seg.name.startswith("segment_"):
            continue
        visual_path = seg / filename
        if not visual_path.exists():
            continue
        text = _read_text(visual_path)
        if text:
            entries.append((visual_path, text))
    return entries


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".m4v"}


def _gather_segment_videos(root: Path) -> List[Path]:
    if root.is_file() and _is_video_file(root):
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    videos: List[Path] = []
    for seg_dir in sorted(root.glob("segment_*")):
        candidate = seg_dir / "video.mp4"
        if candidate.exists() and candidate.is_file():
            videos.append(candidate)
    if videos:
        return videos
    for candidate in root.rglob("video.mp4"):
        if candidate.is_file():
            videos.append(candidate)
    if videos:
        return videos
    for candidate in root.rglob("*"):
        if candidate.is_file() and _is_video_file(candidate):
            videos.append(candidate)
    return videos


def _sample_frames(
    videos: List[Path],
    interval: float,
    max_total_frames: int,
    scale_width: Optional[int],
) -> List[Dict[str, Any]]:
    if interval <= 0:
        raise ValueError("frame sampling interval 必须大于 0")
    if max_total_frames <= 0:
        raise ValueError("max_total_frames 必须大于 0")

    collected: List[Dict[str, Any]] = []
    if not videos:
        return collected

    with tempfile.TemporaryDirectory(prefix="visual_frames_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for video in videos:
            if len(collected) >= max_total_frames:
                break
            video_tmp = tmp_root / uuid4().hex
            video_tmp.mkdir(parents=True, exist_ok=True)
            frame_pattern = video_tmp / "frame_%06d.jpg"

            filter_parts = [f"fps=1/{interval:.6f}"]
            if scale_width and scale_width > 0:
                filter_parts.append(f"scale={scale_width}:-1")
            filter_expr = ",".join(filter_parts)

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video.as_posix(),
                "-vf",
                filter_expr,
                "-vsync",
                "vfr",
                frame_pattern.as_posix(),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[Warn] 抽帧失败 {video}: {exc}", file=sys.stderr)
                continue

            frame_paths = sorted(video_tmp.glob("frame_*.jpg"))
            if not frame_paths:
                continue

            for idx, frame_path in enumerate(frame_paths, start=1):
                with frame_path.open("rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")
                timestamp = round((idx - 1) * interval, 3)
                collected.append(
                    {
                        "video": video.as_posix(),
                        "timestamp": timestamp,
                        "data_url": f"data:image/jpeg;base64,{b64}",
                    }
                )
                if len(collected) >= max_total_frames:
                    break

    return collected


def compose_payload(
    visual_entries: List[Tuple[Path, str]],
    final_caption: str,
    frame_interval: float,
    frame_count: int,
) -> str:
    parts: List[str] = []
    if visual_entries:
        parts.append("Per-segment visual captions (for reference only):")
        for idx, (path, text) in enumerate(visual_entries, 1):
            seg_name = path.parent.name
            parts.append(f"[Segment {idx} - {seg_name}]")
            parts.append(text.strip())
        parts.append("")
    parts.append("Final temporal caption under evaluation:")
    parts.append(final_caption.strip() or "<empty>")
    parts.append("")
    parts.append(
        f"Attached below are {frame_count} sampled frames (one every ~{frame_interval} seconds, chronological order)."
        " Assess whether the final temporal caption aligns with the visual evidence."
    )
    return "\n".join(parts).strip()


def _build_frame_manifest(frames: List[Dict[str, Any]]) -> Optional[str]:
    if not frames:
        return None
    lines = ["Frame manifest (chronological order):"]
    for idx, frame in enumerate(frames, 1):
        video_name = Path(frame["video"]).name
        lines.append(f"{idx}. {video_name} @ ~{frame['timestamp']:.2f}s")
    return "\n".join(lines)


def score_caption(
    client,
    model: str,
    payload_text: str,
    frames: List[Dict[str, Any]],
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    if not frames:
        raise ValueError("未能抽取任何图像帧，无法执行视觉过滤")

    manifest = _build_frame_manifest(frames)

    with tempfile.TemporaryDirectory(prefix="visual_filter_frames_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        image_paths: List[Path] = []
        for idx, frame in enumerate(frames, 1):
            data_url = frame.get("data_url")
            if not isinstance(data_url, str) or "," not in data_url:
                continue
            try:
                b64_data = data_url.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)
            except Exception:
                continue
            img_path = tmp_root / f"frame_{idx:03d}.jpg"
            try:
                img_path.write_bytes(image_bytes)
            except Exception:
                continue
            image_paths.append(img_path)

        if not image_paths:
            raise ValueError("无法解析任何抽取的帧图像")

        text_chunks: List[str] = []
        if user_prompt:
            text_chunks.append(user_prompt)
        text_chunks.append(
            "You are auditing a long-form temporal caption for visual fidelity. "
            "Use the attached frames (chronological order) to judge whether the caption aligns with the visuals. "
            "Respond strictly in JSON with keys: score (float 0-1), verdict (short string), summary (string), "
            "strengths (array of strings), issues (array of strings)."
        )
        text_chunks.append(payload_text)
        if manifest:
            text_chunks.append(manifest)

        response_text = client.generate(
            system_prompt=system_prompt,
            text_chunks=text_chunks,
            image_paths=image_paths,
            model=model,
            temperature=temperature,
            top_p=0.9,
            response_format={"type": "json_object"},
        )

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen 返回结果无法解析为 JSON: {response_text}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Qwen 返回的 JSON 非对象类型: {parsed}")

    return response_text, parsed


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual filtering review via local Qwen multimodal model")
    p.add_argument("--segments-root", required=True, help="Segments root directory containing segment_* folders or videos")
    p.add_argument(
        "--caption",
        default=None,
        help="Temporal caption file to evaluate (defaults to <segments-root>/temporal_caption.txt)",
    )
    p.add_argument(
        "--visual",
        default=None,
        help="Optional aggregated visual caption file. If omitted, gather <segment>/caption.txt when available",
    )
    p.add_argument(
        "--visual-filename",
        default=DEFAULT_VISUAL_FILENAME,
        help="Per-segment visual caption filename (default: caption.txt)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument("--system", default=None, help="System prompt string or @file")
    p.add_argument("--user", default=None, help="User prompt string or @file")
    p.add_argument("--output", default=None, help="Output JSON path for the score (default: <segments-root>/visual_filtering.json)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--frame-interval",
        type=float,
        default=DEFAULT_FRAME_INTERVAL,
        help="Seconds between sampled frames (default: 2.0)",
    )
    p.add_argument(
        "--max-total-frames",
        type=int,
        default=DEFAULT_MAX_TOTAL_FRAMES,
        help="Maximum number of frames to include in the request (default: 120)",
    )
    p.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Resize width for sampled frames before upload (default: 512)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)

    segments_root = Path(args.segments_root)
    if not segments_root.exists():
        print(f"未找到分段结果目录: {segments_root}", file=sys.stderr)
        return 2

    if args.visual:
        visual_path = Path(args.visual)
        if not visual_path.exists():
            print(f"未找到指定的视觉 caption 文件: {visual_path}", file=sys.stderr)
            return 2
        visual_text = _read_text(visual_path)
        if not visual_text:
            print(f"指定的视觉 caption 文件为空: {visual_path}", file=sys.stderr)
            return 2
        visual_entries = [(visual_path, visual_text)]
    else:
        visual_entries = discover_visual_captions(segments_root, filename=args.visual_filename)

    if args.caption:
        caption_path = Path(args.caption)
    else:
        caption_path = segments_root / "temporal_caption.txt"
    if not caption_path.exists():
        print(f"未找到待评估的 temporal caption 文件: {caption_path}", file=sys.stderr)
        return 2
    final_caption = _read_text(caption_path)
    if not final_caption:
        print(f"Temporal caption 文件为空: {caption_path}", file=sys.stderr)
        return 2

    videos = _gather_segment_videos(segments_root)
    if not videos:
        print("未找到任何可用的视频片段用于抽帧，请检查分段输出。", file=sys.stderr)
        return 2

    frames = _sample_frames(
        videos=videos,
        interval=args.frame_interval,
        max_total_frames=args.max_total_frames,
        scale_width=args.frame_width,
    )
    if not frames:
        print("抽帧结果为空，请检查 ffmpeg 是否可用以及视频内容。", file=sys.stderr)
        return 2

    payload_text = compose_payload(
        visual_entries=visual_entries,
        final_caption=final_caption,
        frame_interval=args.frame_interval,
        frame_count=len(frames),
    )

    system_prompt = _read_prompt_value(args.system) if args.system else _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    user_prompt = _read_prompt_value(args.user) if args.user else _load_default_prompt(DEFAULT_USER_PROMPT_PATH)

    print("即将调用本地 Qwen 模型进行视觉聚焦打分...")
    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    try:
        raw_text, parsed = score_caption(
            client=client,
            model=args.model,
            payload_text=payload_text,
            frames=frames,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
        )
    except QwenClientError as exc:
        print(f"调用 Qwen 失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"视觉过滤执行失败: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else segments_root / "visual_filtering.json"
    frame_meta = [{"video": f["video"], "timestamp": f["timestamp"]} for f in frames]
    output = {
        "model": args.model,
        "segments_root": segments_root.as_posix(),
        "visual_entries": [p.as_posix() for p, _ in visual_entries],
        "caption_file": caption_path.as_posix(),
        "system_prompt": bool(system_prompt),
        "user_prompt": bool(user_prompt),
        "frame_interval": args.frame_interval,
        "frame_count": len(frames),
        "frames": frame_meta,
        "score": parsed,
        "raw_response": raw_text,
    }
    try:
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"写入输出文件失败: {output_path}: {exc}", file=sys.stderr)
        return 1
    print(f"评分结果已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

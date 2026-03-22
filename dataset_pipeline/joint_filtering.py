#!/usr/bin/env python3
"""
Final joint audio-visual filtering of the temporal caption using Qwen3-Omni Thinking.

This reviewer uploads the full-length video (with audio) to the local Qwen endpoint,
provides the final integrated temporal caption as textual context, and asks the model to emit a structured
JSON score summarising alignment quality across both modalities.

Usage example:
    python joint_filtering.py \
        --video /path/to/full_video.mp4 \
        --segments-root /path/to/video_segments \
        --model Qwen3-Omni-30B-A3B-Thinking

The script mirrors the ergonomics of audio_filtering/visual_filtering:
- Prompts default to prompts/joint_filtering_*.md unless overridden with --system/--user
- Output defaults to <segments-root>/joint_filtering.json
- Requires a running vLLM instance serving Qwen3-Omni-30B-A3B-Thinking (or set QWEN_MODEL)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, QwenClientError, default_client

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "joint_filtering_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "joint_filtering_user.md"


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


def compose_payload(final_caption: str) -> str:
    parts = [
        "Final temporal caption under evaluation (audio + visual alignment):",
        final_caption.strip() or "<empty>",
    ]
    return "\n".join(parts).strip()


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



def _ensure_video_path(video_path: Path) -> None:
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"未找到输入视频文件: {video_path}")


def score_caption(
    client,
    model: str,
    video_path: Path,
    payload_text: str,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, Any]]:
    text_chunks: List[str] = []
    if user_prompt:
        text_chunks.append(user_prompt)
    text_chunks.append(
        "You are auditing the final temporal caption for combined audio-visual fidelity. "
        "Use the attached full-length video (with audio) to evaluate accuracy, completeness, "
        "and potential hallucinations. Respond strictly in JSON with keys: score (float 0-1), "
        "verdict (short string), summary (string), strengths (array of strings), issues (array of strings)."
    )
    text_chunks.append(payload_text)

    response_text = client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        video_path=video_path,
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


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint audio-visual filtering review via local Qwen model")
    p.add_argument("--video", required=True, help="Path to the full video (with audio)")
    p.add_argument("--segments-root", required=True, help="Segments root for locating temporal_caption.txt")
    p.add_argument("--caption", default=None, help="Temporal caption file (default: <segments-root>/temporal_caption.txt)")
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument("--system", default=None, help="System prompt text or @file")
    p.add_argument("--user", default=None, help="User prompt text or @file")
    p.add_argument("--output", default=None, help="Output JSON path (default: <segments-root>/joint_filtering.json)")
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)

    video_path = Path(args.video)
    try:
        _ensure_video_path(video_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    segments_root = Path(args.segments_root)
    if not segments_root.exists() or not segments_root.is_dir():
        print(f"未找到分段结果目录: {segments_root}", file=sys.stderr)
        return 2

    if args.caption:
        caption_path = Path(args.caption)
    else:
        caption_path = segments_root / "temporal_caption.txt"

    if not caption_path.exists():
        print(f"未找到 temporal caption 文件: {caption_path}", file=sys.stderr)
        return 2

    final_caption = _read_text(caption_path)
    if not final_caption:
        print(f"Temporal caption 文件为空: {caption_path}", file=sys.stderr)
        return 2

    payload_text = compose_payload(final_caption)

    system_prompt = _read_prompt_value(args.system) if args.system else _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    user_prompt = _read_prompt_value(args.user) if args.user else _load_default_prompt(DEFAULT_USER_PROMPT_PATH)

    print("即将调用本地 Qwen 模型进行音视频联合一致性评分...")
    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    try:
        raw_text, parsed = score_caption(
            client=client,
            model=args.model,
            video_path=video_path,
            payload_text=payload_text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
        )
    except QwenClientError as exc:
        print(f"调用 Qwen 失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"音视频联合过滤失败: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else segments_root / "joint_filtering.json"
    output = {
        "model": args.model,
        "video": video_path.as_posix(),
        "caption_file": caption_path.as_posix(),
        "segments_root": segments_root.as_posix(),
        "system_prompt": bool(system_prompt),
        "user_prompt": bool(user_prompt),
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

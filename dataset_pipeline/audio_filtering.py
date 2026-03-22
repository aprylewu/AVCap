#!/usr/bin/env python3
"""
Evaluate the final temporal caption against the full audio track using the local Qwen3-Omni Thinking model.

Given a full-length audio file and the temporal caption to review, this script feeds the
audio to the local vLLM endpoint, composes an audio-centric review prompt, and requests a
JSON-formatted score plus reasoning. Prompts default to markdown files in prompts/ but can
be overridden via CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, QwenClientError, default_client

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "audio_filtering_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "audio_filtering_user.md"


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
    parts: List[str] = []
    parts.append("Temporal caption under evaluation (audio alignment focus):")
    parts.append(final_caption.strip() or "<empty>")
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


def score_caption(
    client,
    model: str,
    audio_path: Path,
    payload_text: str,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, Any]]:
    text_chunks: List[str] = []
    if user_prompt:
        text_chunks.append(user_prompt)
    text_chunks.append(
        "You are auditing a temporal caption for audio fidelity. "
        "Listen to the attached audio and determine whether the caption is accurate, comprehensive, "
        "and free from hallucinations. Respond strictly in JSON with keys: score (float 0-1), "
        "verdict (short string), summary (string), strengths (array of strings), issues (array of strings)."
    )
    text_chunks.append(payload_text)

    response_text = client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        audio_path=audio_path,
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
    p = argparse.ArgumentParser(description="Score a temporal caption using local Qwen audio review")
    p.add_argument("--audio", required=True, help="Path to the full-length audio track of the video")
    p.add_argument(
        "--segments-root",
        required=True,
        help="Directory containing temporal caption outputs (used for defaults and results)",
    )
    p.add_argument(
        "--caption",
        default=None,
        help="Temporal caption file to evaluate (defaults to temporal_caption.txt)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument("--system", default=None, help="System prompt string or @file")
    p.add_argument("--user", default=None, help="User prompt string or @file")
    p.add_argument("--output", default=None, help="Output JSON path for the score")
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"未找到音频文件: {audio_path}", file=sys.stderr)
        return 2

    segments_root = Path(args.segments_root)
    if not segments_root.exists() or not segments_root.is_dir():
        print(f"未找到分段结果目录: {segments_root}", file=sys.stderr)
        return 2

    default_caption_candidates: List[Path] = []
    if not args.caption:
        default_caption_candidates.append(segments_root / "temporal_caption.txt")

    caption_path: Optional[Path]
    if args.caption:
        caption_path = Path(args.caption)
    else:
        caption_path = next((p for p in default_caption_candidates if p.exists()), None)

    if caption_path is None:
        default_list = ", ".join(p.name for p in default_caption_candidates)
        print(f"默认 caption 文件不存在: {default_list}", file=sys.stderr)
        return 2

    if not caption_path.exists():
        print(f"未找到待评估 caption 文件: {caption_path}", file=sys.stderr)
        return 2
    final_caption = _read_text(caption_path)
    if not final_caption:
        print(f"Caption 文件为空: {caption_path}", file=sys.stderr)
        return 2

    payload_text = compose_payload(final_caption)

    system_prompt = _read_prompt_value(args.system) if args.system else _load_default_prompt(DEFAULT_SYSTEM_PROMPT_PATH)
    user_prompt = _read_prompt_value(args.user) if args.user else _load_default_prompt(DEFAULT_USER_PROMPT_PATH)

    print("即将调用本地 Qwen 模型进行音频聚焦打分...")
    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    try:
        raw_text, parsed = score_caption(
            client=client,
            model=args.model,
            audio_path=audio_path,
            payload_text=payload_text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
        )
    except QwenClientError as exc:
        print(f"调用 Qwen 失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"音频过滤执行失败: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else segments_root / "audio_filtering.json"
    output = {
        "model": args.model,
        "audio": audio_path.as_posix(),
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

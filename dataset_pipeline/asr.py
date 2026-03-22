#!/usr/bin/env python3
"""
ASR for vocal tracks produced by audio_preprocess.py using a local Qwen3-Omni
deployment (vLLM OpenAI-compatible API).

Usage examples:
  # Transcribe a single segment directory (containing audio_sep/voice.*)
  python asr.py --input /path/to/<video>_segments/segment_0000

  # Transcribe all segments under a root
  python asr.py --input /path/to/<video>_segments

  # Transcribe a specific voice file
  python asr.py --input /path/to/.../audio_sep/voice.wav

Environment:
  - Reads Qwen connection settings from .env (via python-dotenv) if available.
  - Falls back to defaults defined in qwen_client (base URL from QWEN_BASE_URL or
    http://127.0.0.1:8901/v1, model from QWEN_MODEL or Qwen3-Omni-30B-A3B-Thinking).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from qwen_client import QwenClientError, default_client


TRANSCRIPT_SYSTEM_PROMPT = "You are a meticulous transcription assistant. Output only the verbatim transcript."
TRANSCRIPT_USER_PROMPT = "请逐字转写所附音频内容，保持时间顺序，不要添加解释或额外文本。"


def _is_audio_file(p: Path) -> bool:
    return p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


def find_voice_files(root: Path) -> List[Path]:
    """Locate vocal audio produced by audio_preprocess."""

    results: List[Path] = []

    if root.is_file():
        if _is_audio_file(root):
            return [root]
        return []

    if root.is_dir() and (root.name == "audio_sep" or root.suffix == ".lalalai"):
        for p in root.iterdir():
            if p.is_file() and p.name.startswith("voice.") and _is_audio_file(p):
                results.append(p)
        return results

    direct = root / "audio_sep"
    if direct.exists() and direct.is_dir():
        for p in direct.iterdir():
            if p.is_file() and p.name.startswith("voice.") and _is_audio_file(p):
                results.append(p)
        if results:
            return results

    for p in root.rglob("audio_sep/voice.*"):
        if p.is_file() and _is_audio_file(p):
            results.append(p)

    for d in root.rglob("*.lalalai"):
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.name.startswith("voice.") and _is_audio_file(p):
                    results.append(p)

    seen = set()
    unique: List[Path] = []
    for p in results:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _read_prompt_override(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("@") and len(value) > 1:
        path = Path(value[1:])
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    return value


def _transcribe_with_qwen(
    audio_path: Path,
    *,
    client,
    model: Optional[str],
    system_prompt: Optional[str],
    user_prompt: Optional[str],
) -> str:
    system = (system_prompt or TRANSCRIPT_SYSTEM_PROMPT).strip()
    user = (user_prompt or TRANSCRIPT_USER_PROMPT).strip()
    return client.generate(
        system_prompt=system,
        text_chunks=[user],
        audio_path=audio_path,
        model=model,
        temperature=0.0,
        top_p=0.5,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ASR on vocal audio using local Qwen3-Omni (vLLM)."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to a segment dir, a segments root, an audio_sep dir, or a voice.* file",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override Qwen model name (default: qwen_client config)",
    )
    p.add_argument(
        "--system",
        default=None,
        help="System prompt override or @file",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt override or @file",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    root = Path(args.input)
    voice_files = find_voice_files(root)
    if not voice_files:
        print(f"No voice.* files found under {root}", file=sys.stderr)
        return 1

    try:
        client = default_client()
    except Exception as exc:
        print(f"Failed to initialise Qwen client: {exc}", file=sys.stderr)
        return 2

    try:
        system_prompt = _read_prompt_override(args.system)
        user_prompt = _read_prompt_override(args.user)
    except Exception as exc:
        print(f"Failed to load prompt override: {exc}", file=sys.stderr)
        return 2

    for vf in voice_files:
        print(f"Transcribing: {vf}")
        try:
            text = _transcribe_with_qwen(
                vf,
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except QwenClientError as exc:
            print(f"Failed: {vf} -> {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"Failed: {vf} -> {exc}", file=sys.stderr)
            continue

        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())

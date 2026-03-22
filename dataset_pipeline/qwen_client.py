#!/usr/bin/env python3
"""
Utility wrapper around a local vLLM deployment that serves
`Qwen3-Omni-30B-A3B-Thinking` through an OpenAI-compatible API.

The goal is to centralise multimodal request handling (video, audio, images,
and text) so that the rest of the pipeline can swap out Gemini/OpenRouter code
without duplicating HTTP glue everywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import requests

DEFAULT_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8901/v1")
DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "Qwen3-Omni-30B-A3B-Thinking")
_MEDIA_TYPES = {"image", "video", "audio"}


class QwenClientError(RuntimeError):
    """Raised when the local LLM service rejects a request."""


@dataclass
class QwenClientConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 1_000_000_000  # prevent client-side aborts during long generations
    max_retries: int = 3
    retry_backoff: float = 8.0
    api_key: Optional[str] = None  # vLLM usually leaves this unset


def _as_path(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _file_url(path: Path) -> str:
    # vLLM accepts absolute file:// URLs when launched with
    # --allowed-local-media-path /
    return f"file://{path.resolve()}"


_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_think_tokens(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_PATTERN.sub("", text)
    # In case the model emits orphan tags, strip them conservatively.
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned


def _flatten_message_content(message: Dict[str, Any]) -> str:
    """
    The OpenAI chat response schema can carry text either as a plain string or
    an array of typed blocks. This helper normalises that into a single string.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, Iterable):
        return ""

    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "output_text"):
            text = block.get("text") or block.get("content")
            if text:
                parts.append(str(text))
        elif block_type == "tool_result":
            # Treat tool results as plain text for now so callers still see them.
            text = block.get("content")
            if text:
                if isinstance(text, list):
                    for item in text:
                        if isinstance(item, dict):
                            maybe_text = item.get("text")
                            if maybe_text:
                                parts.append(str(maybe_text))
                        elif item:
                            parts.append(str(item))
                else:
                    parts.append(str(text))
    return "\n".join(parts).strip()


class QwenClient:
    """
    Simple non-streaming client for the local Qwen 3 Omni deployment.

    Typical usage:
        client = QwenClient()
        text = client.generate(
            system_prompt="You are a helpful assistant",
            text_chunks=["Describe the following clip."],
            video_path="segment_0000/video.mp4",
            audio_path="segment_0000/audio.wav",
        )
    """

    def __init__(self, config: Optional[QwenClientConfig] = None, **overrides: Any) -> None:
        if config is None:
            config = QwenClientConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if self.config.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"

    # ------------------------------------------------------------------ helpers
    def _build_media_block(self, kind: str, path: Path) -> Dict[str, str]:
        if kind not in _MEDIA_TYPES:
            raise ValueError(f"Unsupported media kind '{kind}'")
        url = _file_url(path)
        if kind == "image":
            return {"type": "image_url", "image_url": {"url": url}}
        if kind == "video":
            return {"type": "video_url", "video_url": {"url": url}}
        if kind == "audio":
            return {"type": "audio_url", "audio_url": {"url": url}}
        raise ValueError(f"Unsupported media kind '{kind}'")

    def _build_user_content(
        self,
        text_chunks: Sequence[str],
        video_path: Optional[Path],
        audio_path: Optional[Path],
        image_paths: Sequence[Path],
        extra_contents: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for chunk in text_chunks:
            if chunk is None:
                continue
            text = str(chunk).strip()
            if text:
                content.append({"type": "text", "text": text})

        if video_path:
            content.append(self._build_media_block("video", video_path))
        if audio_path:
            content.append(self._build_media_block("audio", audio_path))

        for img in image_paths:
            content.append(self._build_media_block("image", img))

        if extra_contents:
            for block in extra_contents:
                if isinstance(block, dict):
                    content.append(block)
        if not content:
            raise ValueError("user content is empty; provide text, media or extra content")
        return content

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        retries = max(0, self.config.max_retries)
        backoff = max(0.0, self.config.retry_backoff)
        error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self._session.post(
                    url,
                    data=json.dumps(payload),
                    timeout=self.config.timeout,
                )
            except Exception as exc:  # pragma: no cover - network failure path
                error = exc
            else:
                if response.ok:
                    return response.json()
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                error = QwenClientError(f"HTTP {response.status_code}: {detail}")
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
        assert error is not None
        raise error

    # ---------------------------------------------------------------- interface
    def generate(
        self,
        *,
        system_prompt: Optional[str] = None,
        text_chunks: Sequence[str] = (),
        video_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
        image_paths: Sequence[Union[str, Path]] = (),
        extra_user_contents: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: Optional[float] = 1.1,
        max_output_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Send a single-turn chat completion request. Returns the model text.

        `text_chunks` becomes multiple `{"type": "text"}` blocks in the user
        message so prompts can stay logically separated.
        """
        video = _as_path(video_path)
        audio = _as_path(audio_path)
        images = [Path(p) for p in image_paths]

        user_content = self._build_user_content(
            text_chunks=text_chunks,
            video_path=video,
            audio_path=audio,
            image_paths=images,
            extra_contents=extra_user_contents,
        )

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if stop:
            payload["stop"] = list(stop)
        if response_format:
            payload["response_format"] = response_format

        data = self._request(payload)
        choices = data.get("choices") or []
        if not choices:
            raise QwenClientError(f"Empty completion response: {data}")
        message = choices[0].get("message") or {}
        text = _strip_think_tokens(_flatten_message_content(message)).strip()
        if not text:
            # Fallback to raw content to aid debugging
            raise QwenClientError(f"No textual content in response: {message}")
        return text


# Convenience singleton for scripts that prefer module-level usage.
_default_client: Optional[QwenClient] = None


def default_client() -> QwenClient:
    global _default_client
    if _default_client is None:
        api_key = os.getenv("QWEN_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL)
        model = os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        timeout_env = os.getenv("QWEN_TIMEOUT")
        max_retries_env = os.getenv("QWEN_MAX_RETRIES")
        backoff_env = os.getenv("QWEN_RETRY_BACKOFF")

        config = QwenClientConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
        if timeout_env:
            try:
                config.timeout = float(timeout_env)
            except ValueError:
                pass
        if max_retries_env:
            try:
                config.max_retries = max(0, int(max_retries_env))
            except ValueError:
                pass
        if backoff_env:
            try:
                config.retry_backoff = float(backoff_env)
            except ValueError:
                pass
        _default_client = QwenClient(config=config)
    return _default_client

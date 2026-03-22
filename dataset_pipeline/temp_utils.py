#!/usr/bin/env python3
"""
Temporary workspace helpers for staging media files that will be consumed by
the local Qwen service (via file:// URLs).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4


CODE_ROOT = Path(__file__).resolve().parent
TMP_ROOT = CODE_ROOT / "tmp"
_SAFE_EXTRA_CHARS = {"-", "_", "."}


def _safe_component(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in _SAFE_EXTRA_CHARS else "_" for c in value)
    return cleaned or "item"


def prepare_temp_dir(name: Union[str, Path]) -> Path:
    """
    Create a unique temp directory under CODE_ROOT/tmp/staging/<name>/<uuid>.
    """
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    staging_root = TMP_ROOT / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    if isinstance(name, Path):
        raw = name.as_posix()
    else:
        raw = str(name)
    raw = raw.replace(os.sep, "_")
    base = _safe_component(raw)
    seg_tmp = staging_root / base / uuid4().hex[:8]
    seg_tmp.mkdir(parents=True, exist_ok=True)
    return seg_tmp


def cleanup_temp_dir(path: Path) -> None:
    """
    Remove a temp directory and prune empty parents within the staging root.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        return

    staging_root = TMP_ROOT / "staging"
    current = path.parent
    while staging_root in current.parents or current == staging_root:
        try:
            current.rmdir()
        except OSError:
            break
        if current == staging_root:
            break
        current = current.parent


def stage_file(src: Path, tmp_dir: Path, *, name: Optional[str] = None) -> Path:
    """
    Copy `src` into `tmp_dir` (optionally overriding the basename). Returns the
    staged path. If `src` already resides under the same location, no copy is
    performed.
    """
    if not src.exists():
        raise FileNotFoundError(src)
    target_name = _safe_component(name) if name else src.name
    dst = tmp_dir / target_name
    try:
        if src.resolve() == dst.resolve():
            return dst
    except Exception:
        # Fallback when resolve() fails (e.g. permissions); still attempt copy.
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst

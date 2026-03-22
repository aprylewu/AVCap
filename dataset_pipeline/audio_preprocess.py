#!/usr/bin/env python3
"""
Audio preprocessing pipeline that:
  1) Accepts segmented audio files from video_segmenter (typically <seg_dir>/audio.wav)
  2) Uses Demucs to separate voice (vocals) and background tracks locally
  3) Splits the background track into 15-second chunks, with the last chunk rule:
     - If the final remainder is <= 5s, merge it into the previous one (allowing last up to 20s)
     - Otherwise, keep it as its own final chunk
  4) Leaves the voice track uncut

Notes:
  - Requires ffmpeg/ffprobe on PATH for chunking and duration probing
  - Requires the demucs Python package and model weights (downloaded on first use)
  - Can be pointed at a single file or a directory containing many segment folders

Usage examples:
  # Process a single segment directory (containing audio.wav)
  python audio_preprocess.py --input /path/to/<video>_segments/0000 \
      --demucs-model mdx_extra --format mp3

  # Or process an entire segments root (multiple 0000, 0001, ...)
  python audio_preprocess.py --input /path/to/<video>_segments \
      --demucs-model mdx_extra --format mp3

  # Or process a single audio file
  python audio_preprocess.py --input /path/to/some/audio.wav
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEMUCS_BATCH_SCRIPT = _REPO_ROOT / "demucs_batch-multigpu" / "separate_from_folder.py"


class FFmpegError(RuntimeError):
    pass


# -------------- Utilities --------------
def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _require_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if _which(b) is None]
    if missing:
        raise EnvironmentError(
            f"Missing dependencies: {', '.join(missing)}. Please install ffmpeg (with ffprobe)."
        )


def _run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT: {e.stdout.decode(errors='ignore')}\n"
            f"STDERR: {e.stderr.decode(errors='ignore')}"
        )


def _format_ts(t: float) -> str:
    # Format seconds as HH:MM:SS.mmm
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def probe_duration_sec(media_path: str) -> float:
    _require_binaries()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        media_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"ffprobe failed: {e.stderr.decode(errors='ignore')}")
    info = json.loads(res.stdout.decode() or "{}")
    try:
        dur = float(info["format"]["duration"])  # seconds
    except Exception:  # pragma: no cover - resilience
        raise FFmpegError("Unable to read duration from ffprobe output")
    return dur


def compute_bg_chunks(total: float, seg_len: float = 15.0, min_last: float = 5.0) -> List[Tuple[float, float]]:
    """Compute background chunks per rules:
    - Cut in segments of seg_len seconds
    - If last remainder <= min_last, merge into previous (so last can be up to seg_len+min_last)
    - Else, keep last remainder as separate chunk
    Returns list of (start, duration).
    """
    if total <= 0:
        return []

    # Number of full segments
    n_full = int(total // seg_len)
    remainder = total - n_full * seg_len

    chunks: List[Tuple[float, float]] = []
    # Add full segments
    for i in range(n_full):
        start = i * seg_len
        dur = seg_len
        chunks.append((start, dur))

    if remainder <= 1e-6:
        return chunks

    if n_full == 0:
        # Total duration shorter than one full segment: output single chunk
        chunks.append((0.0, remainder))
        return chunks

    # If remainder is small (<= min_last), merge into previous
    if remainder <= min_last:
        # Extend the last full segment's duration
        start_prev, dur_prev = chunks[-1]
        chunks[-1] = (start_prev, dur_prev + remainder)
    else:
        # Add it as the final chunk
        chunks.append((n_full * seg_len, remainder))

    return chunks

@dataclass
class SeparationResult:
    voice_path: Path
    background_path: Path


@dataclass
class BatchSeparationOutcome:
    results: Dict[Path, SeparationResult]
    errors: Dict[Path, str]


def separate_file(
    input_file: Path,
    out_dir: Path,
    model: str = "mdx_extra",
    two_stems: str = "vocals",
    export_format: str = "mp3",
    jobs: Optional[int] = None,
) -> SeparationResult:
    """Run Demucs separation for a single audio file and standardize the outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = out_dir / "_demucs_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    ext_flags = {
        "mp3": ["--mp3"],
        "flac": ["--flac"],
        "wav": [],
    }
    if export_format not in ext_flags:
        raise ValueError(f"Unsupported export format '{export_format}'")

    args = [
        "--two-stems",
        two_stems,
        "-n",
        model,
        "-o",
        str(tmp_root),
        *ext_flags[export_format],
    ]
    if jobs is not None:
        args.extend(["-j", str(jobs)])
    args.append(input_file.as_posix())

    try:
        demucs.separate.main(args)
    except SystemExit as exc:  # demucs CLI may call sys.exit
        code = exc.code or 0
        if code != 0:
            raise RuntimeError(f"Demucs separation failed with exit code {code}") from None

    model_dir = tmp_root / model
    if not model_dir.exists():
        raise RuntimeError(f"Demucs did not create expected model directory '{model_dir}'")

    track_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not track_dirs:
        raise RuntimeError(f"No Demucs output directory found under '{model_dir}'")
    track_name = input_file.stem
    track_dir = next((p for p in track_dirs if p.name == track_name), track_dirs[0])

    suffix = {
        "mp3": ".mp3",
        "wav": ".wav",
        "flac": ".flac",
    }[export_format]

    voice_src = track_dir / f"{two_stems}{suffix}"
    background_src = track_dir / f"no_{two_stems}{suffix}"
    if not voice_src.exists() or not background_src.exists():
        available = ", ".join(sorted(p.name for p in track_dir.iterdir()))
        raise RuntimeError(
            "Expected Demucs outputs not found. Looking for "
            f"'{voice_src.name}' and '{background_src.name}'. Available: {available}"
        )

    voice_dst = out_dir / f"voice{suffix}"
    background_dst = out_dir / f"background{suffix}"
    if voice_dst.exists():
        voice_dst.unlink()
    if background_dst.exists():
        background_dst.unlink()

    try:
        shutil.move(voice_src.as_posix(), voice_dst.as_posix())
        shutil.move(background_src.as_posix(), background_dst.as_posix())
        return SeparationResult(voice_path=voice_dst, background_path=background_dst)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def cut_background_chunks(background_file: Path, chunk_dir: Path, seg_len: float = 15.0, min_last: float = 5.0) -> List[Path]:
    """Cut background audio into seg_len chunks, merging last remainder if <= min_last.
    Returns list of generated chunk file paths in order.
    """
    _require_binaries()
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total = probe_duration_sec(background_file.as_posix())
    plan = compute_bg_chunks(total, seg_len=seg_len, min_last=min_last)
    if not plan:
        return []

    # Use container extension of source
    ext = background_file.suffix
    out_files: List[Path] = []
    for idx, (start, dur) in enumerate(plan):
        out_path = chunk_dir / f"bg_{idx:03d}{ext}"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _format_ts(start),
            "-i",
            background_file.as_posix(),
            "-t",
            _format_ts(dur),
            "-c",
            "copy",
            out_path.as_posix(),
        ]
        _run(cmd)
        out_files.append(out_path)

    return out_files


def _find_segment_audio_files(root: Path) -> List[Path]:
    """Find audio files to process under a root directory.
    Priority:
      1) <dir>/audio.wav if present (single segment dir)
      2) <dir>/**/audio.wav (segments root)
      3) Any common audio files directly under dir (wav, mp3, flac, m4a, ogg)
    """
    if root.is_file():
        return [root]

    # 1) direct audio.wav
    direct = root / "audio.wav"
    if direct.exists():
        return [direct]

    # 2) nested audio.wav (segments root)
    nested = list(root.rglob("audio.wav"))
    if nested:
        return nested

    # 3) Any audio directly under root (not recursing)
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    any_audio = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in audio_exts]
    return any_audio


def _has_existing_outputs(out_dir: Path, export_format: str) -> bool:
    suffix = {
        "mp3": ".mp3",
        "wav": ".wav",
        "flac": ".flac",
    }[export_format]
    voice = out_dir / f"voice{suffix}"
    bg_dir = out_dir / "background_chunks"
    return voice.exists() and bg_dir.exists() and any(bg_dir.iterdir())


def separate_files_batch(
    files: List[Path],
    outputs: Dict[Path, Path],
    model: str,
    two_stems: str,
    export_format: str,
    jobs: Optional[int],
) -> BatchSeparationOutcome:
    if not files:
        return BatchSeparationOutcome(results={}, errors={})

    if not DEMUCS_BATCH_SCRIPT.exists():
        raise RuntimeError(f"未找到 Demucs 批处理脚本: {DEMUCS_BATCH_SCRIPT}")

    ext_flags = {
        "mp3": ["--mp3"],
        "flac": ["--flac"],
        "wav": [],
    }
    if export_format not in ext_flags:
        raise ValueError(f"Unsupported export format '{export_format}'")

    tmp_input = Path(tempfile.mkdtemp(prefix="demucs_inputs_"))
    tmp_output = Path(tempfile.mkdtemp(prefix="demucs_outputs_"))
    mapping: Dict[str, Path] = {}
    try:
        for idx, src in enumerate(files):
            seg_dir = src.parent
            video_root = seg_dir.parent if seg_dir.parent else seg_dir
            video_stem = video_root.name
            unique_name = f"{idx:05d}_{video_stem}_{seg_dir.name}_{src.stem}{src.suffix}"
            dst = tmp_input / unique_name
            src_abs = src.resolve()
            try:
                os.symlink(src_abs.as_posix(), dst.as_posix())
            except OSError:
                shutil.copy2(src_abs.as_posix(), dst.as_posix())
            if not dst.exists():
                raise RuntimeError(f"Demucs staging failed: {dst} 不存在 (source: {src})")
            mapping[unique_name] = src

        batch_size = max(1, jobs or 8)
        num_worker = max(1, jobs or 4)
        target_sr = 44100
        max_duration = 0.0
        for src in files:
            try:
                max_duration = max(max_duration, probe_duration_sec(src.as_posix()))
            except Exception:
                continue
        audiolength_override: Optional[int] = None
        if max_duration > 0:
            audiolength_override = int(math.ceil(max_duration * target_sr))

        demucs_cmd = [
            sys.executable,
            DEMUCS_BATCH_SCRIPT.as_posix(),
            tmp_input.as_posix(),
            "--two-stems",
            two_stems,
            "-n",
            model,
            "-b",
            str(batch_size),
            "--num_worker",
            str(num_worker),
            "-o",
            tmp_output.as_posix(),
            "-sr",
            str(target_sr),
            "--no-split",
            *ext_flags[export_format],
        ]
        if jobs is not None:
            demucs_cmd += ["-j", str(jobs)]
        if audiolength_override is not None:
            demucs_cmd += ["-l", str(audiolength_override)]

        subprocess.run(demucs_cmd, check=True)

        model_dir = tmp_output / model
        if not model_dir.exists():
            raise RuntimeError(f"Demucs 没有生成模型输出目录: {model_dir}")

        suffix = {
            "mp3": ".mp3",
            "wav": ".wav",
            "flac": ".flac",
        }[export_format]

        results: Dict[Path, SeparationResult] = {}
        errors: Dict[Path, str] = {}
        for link_name, original in mapping.items():
            track_stem = Path(link_name).stem
            track_dir = model_dir / track_stem
            try:
                if not track_dir.exists():
                    # 某些版本会直接输出成文件，不带子目录
                    direct_voice = model_dir / f"{track_stem}{suffix}"
                    if direct_voice.exists():
                        track_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(direct_voice.as_posix(), (track_dir / f"{two_stems}{suffix}").as_posix())
                    else:
                        available = ", ".join(sorted(p.name for p in model_dir.iterdir()))
                        raise RuntimeError(
                            f"找不到 Demucs 输出 {track_dir} (源 {original})，当前可用: {available}"
                        )

                voice_src = track_dir / f"{two_stems}{suffix}"
                background_src = track_dir / f"no_{two_stems}{suffix}"
                if not voice_src.exists() or not background_src.exists():
                    available = ", ".join(sorted(p.name for p in track_dir.iterdir()))
                    raise RuntimeError(
                        f"Demucs 输出缺少 {voice_src.name}/{background_src.name} (来源 {original})，现有: {available}"
                    )

                out_dir = outputs[original]
                out_dir.mkdir(parents=True, exist_ok=True)
                voice_dst = out_dir / f"voice{suffix}"
                background_dst = out_dir / f"background{suffix}"
                if voice_dst.exists():
                    voice_dst.unlink()
                if background_dst.exists():
                    background_dst.unlink()
                shutil.move(voice_src.as_posix(), voice_dst.as_posix())
                shutil.move(background_src.as_posix(), background_dst.as_posix())
                results[original] = SeparationResult(voice_path=voice_dst, background_path=background_dst)
            except Exception as exc:
                errors[original] = str(exc)
        return BatchSeparationOutcome(results=results, errors=errors)
    finally:
        shutil.rmtree(tmp_input, ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)


def process_input(
    input_path: Path,
    model: str = "mdx_extra",
    two_stems: str = "vocals",
    export_format: str = "mp3",
    jobs: Optional[int] = None,
    bg_seg_len: float = 15.0,
    bg_min_last: float = 5.0,
    force: bool = False,
) -> None:
    files = _find_segment_audio_files(input_path)
    if not files:
        print(f"No audio files found under {input_path}", file=sys.stderr)
        return
    outputs: Dict[Path, Path] = {}
    to_process: List[Path] = []
    for f in files:
        if f.name == "audio.wav":
            out_dir = f.parent / "audio_sep"
        else:
            out_dir = Path(str(f.with_suffix("")) + ".demucs")

        if force and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        if not force and _has_existing_outputs(out_dir, export_format):
            print(f"Skipping existing separation for {f}")
            continue

        outputs[f] = out_dir
        to_process.append(f)

    if not to_process:
        print("All files already separated. Nothing to do.")
        return

    print(f"Running Demucs on {len(to_process)} file(s)...")
    outcome = separate_files_batch(
        files=to_process,
        outputs=outputs,
        model=model,
        two_stems=two_stems,
        export_format=export_format,
        jobs=jobs,
    )

    for f, sep in outcome.results.items():
        out_dir = outputs[f]
        print(f"Cutting background into chunks for {f}...")
        chunk_dir = out_dir / "background_chunks"
        chunks = cut_background_chunks(
            background_file=sep.background_path,
            chunk_dir=chunk_dir,
            seg_len=bg_seg_len,
            min_last=bg_min_last,
        )
        print(f"Generated {len(chunks)} background chunks under {chunk_dir}")

    if outcome.errors:
        details = "; ".join(f"{path}: {msg}" for path, msg in outcome.errors.items())
        raise RuntimeError(f"Demucs 批处理部分失败 ({len(outcome.errors)} 个文件): {details}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Separate voice/background via Demucs and chunk background audio.")
    p.add_argument("--input", required=True, help="Path to a segment dir, a segments root, or an audio file")
    p.add_argument("--demucs-model", dest="model", default="mdx_extra", help="Demucs model name (e.g., mdx_extra)")
    p.add_argument("--two-stems", dest="two_stems", default="vocals", help="Two-stem target passed to Demucs (default: vocals)")
    p.add_argument(
        "--format",
        dest="export_format",
        choices=["mp3", "wav", "flac"],
        default="mp3",
        help="Output format for separated stems",
    )
    p.add_argument("--jobs", dest="jobs", type=int, default=None, help="Number of parallel jobs for Demucs (pass through to --jobs)")
    p.add_argument("--bg-seg-len", dest="bg_seg_len", type=float, default=15.0, help="Background chunk length seconds")
    p.add_argument("--bg-min-last", dest="bg_min_last", type=float, default=5.0, help="Merge last remainder if <= this many seconds")
    p.add_argument("--force", action="store_true", help="Re-run separation even if outputs already exist")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    try:
        process_input(
            input_path=input_path,
            model=args.model,
            two_stems=args.two_stems,
            export_format=args.export_format,
            jobs=args.jobs,
            bg_seg_len=args.bg_seg_len,
            bg_min_last=args.bg_min_last,
            force=args.force,
        )
    except Exception as e:
        print(f"Processing failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

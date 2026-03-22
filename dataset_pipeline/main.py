#!/usr/bin/env python3
"""
One-click long-video caption pipeline:
1) Segment the input video into per-segment silent mp4 + audio.wav
2) For each segment, run:
   - audio_preprocess (separate voice/background + cut bgm chunks)
   - bgm_caption (per-clip BGM captions)
   - vocal_caption (voice-only mux + ASR + caption)
   - visual_captioning (silent visual-only caption)
   - joint_caption (combine expert outputs into per-segment joint caption)
3) temporal_integration (integrate per-segment joint captions into full-video caption)
4) visual_filtering (review final caption against visual evidence via local Qwen)
5) audio_filtering (review final caption against the full audio via local Qwen)
6) joint_filtering (Qwen-based joint audio-visual review of the final caption)

Notes:
- Requires ffmpeg/ffprobe, local Qwen (via vLLM), and LALAL.AI + AssemblyAI keys for separation/ASR.
- Requires a local vLLM deployment serving Qwen3-Omni-30B-A3B-Thinking (or set QWEN_MODEL).
- Pass prompt files via --*-system/--*-user (optional). If omitted, each script uses its default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from dotenv import load_dotenv

from audio_preprocess import (
    BatchSeparationOutcome,
    SeparationResult,
    cut_background_chunks,
    probe_duration_sec,
    separate_files_batch,
)
from qwen_client import DEFAULT_MODEL as QWEN_DEFAULT_MODEL
from temp_utils import TMP_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
TMP_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_TEMP_ROOT = TMP_ROOT / "datasets"
SEGMENTS_TEMP_ROOT = TMP_ROOT / "segments"
CAPTION_OUTPUT_ROOT = Path("dataset") / "output"
LOG_DIR = SCRIPT_DIR.parent / "log"

def _load_env():
    load_dotenv()
    script_dir = Path(__file__).resolve().parent
    env_candidate = script_dir / ".env"
    if env_candidate.exists():
        load_dotenv(env_candidate.as_posix(), override=False)


def run(cmd: list[str]) -> int:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        sys.stdout.write(res.stdout.decode(errors="ignore"))
        sys.stderr.write(res.stderr.decode(errors="ignore"))
        return 0
    except subprocess.CalledProcessError as e:
        sys.stdout.write(e.stdout.decode(errors="ignore"))
        sys.stderr.write(e.stderr.decode(errors="ignore"))
        return e.returncode


def default_segments_root(video_path: Path) -> Path:
    return video_path.with_suffix("").parent / f"{video_path.stem}_segments"


def _opt_pair(flag: str, val: Optional[str]) -> list[str]:
    return [flag, val] if val else []


def _maybe_prompts(prefix: str, system: Optional[str], user: Optional[str]) -> list[str]:
    args: list[str] = []
    if system:
        args += ["--system", system]
    if user:
        args += ["--user", user]
    return args


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _trim_audio_to_duration(path: Path, duration: float) -> None:
    if duration <= 0 or not path.exists():
        return
    tmp_path = path.parent / f"{path.stem}.trimtmp{path.suffix}"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path.as_posix(),
        "-t",
        _format_duration(duration),
        "-c",
        "copy",
        tmp_path.as_posix(),
    ]
    rc, _, _ = _run_subprocess_capture(cmd)
    if rc == 0 and tmp_path.exists():
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)


def _list_segment_dirs(seg_root: Path) -> List[Path]:
    if not seg_root.exists():
        return []
    return [p for p in sorted(seg_root.iterdir()) if p.is_dir() and p.name.startswith("segment_")]


def _has_audio_preprocess_outputs(seg_dir: Path) -> bool:
    audio_sep = seg_dir / "audio_sep"
    if not audio_sep.exists() or not audio_sep.is_dir():
        return False
    voice_exists = any(p.is_file() and p.name.startswith("voice.") for p in audio_sep.iterdir())
    bg_dir = audio_sep / "background_chunks"
    bg_exists = bg_dir.exists() and bg_dir.is_dir() and any(f.is_file() for f in bg_dir.iterdir())
    return voice_exists and bg_exists


def _run_subprocess_capture(cmd: List[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return proc.returncode, stdout, stderr


_RETRYABLE_SNIPPETS = (
    "503",
    "status code: 503",
    "service unavailable",
    "model overload",
    "model overloaded",
    "temporarily unavailable",
    "try again later",
    "overloaded",
    "resource exhausted",
)


def _should_retry_on_output(stdout: str, stderr: str) -> bool:
    merged = f"{stdout}\n{stderr}".lower()
    return any(snippet in merged for snippet in _RETRYABLE_SNIPPETS)



@dataclass
class PreprocessResult:
    video: Path
    seg_root: Path
    segments: List[Path]
    cleanup_needed: bool


def _allocate_segments_root(video: Path, args: argparse.Namespace) -> tuple[Path, bool]:
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        return Path(out_dir), False
    temp_base = SEGMENTS_TEMP_ROOT
    temp_base.mkdir(parents=True, exist_ok=True)
    seg_root = temp_base / f"{video.stem}_{uuid4().hex[:8]}"
    cleanup_requested = bool(getattr(args, "cleanup_temp", False))
    return seg_root, cleanup_requested


def _maybe_segment_video(
    video: Path,
    seg_root: Path,
    args: argparse.Namespace,
    *,
    force: bool,
) -> Optional[List[Path]]:
    seg_root.mkdir(parents=True, exist_ok=True)
    existing_segments = _list_segment_dirs(seg_root)
    if existing_segments and not force:
        print(
            f"[preprocess] 检测到已有 {len(existing_segments)} 个 segment，跳过视频切分: {video}",
            flush=True,
        )
        return existing_segments

    seg_cmd = [
        sys.executable,
        (SCRIPT_DIR / "video_segmenter.py").as_posix(),
        video.as_posix(),
        "--segment-length",
        str(getattr(args, "segment_length", 60.0) or 60.0),
        "--min-last",
        str(getattr(args, "min_last", 10.0) or 10.0),
        "--video-output",
        "--output-dir",
        seg_root.as_posix(),
    ]
    print(f"[preprocess] 正在执行视频分段: {video} -> {seg_root}")
    rc, _, _ = _run_subprocess_capture(seg_cmd)
    if rc != 0:
        print(f"[preprocess] video_segmenter 失败，退出码 {rc}", file=sys.stderr)
        return None

    segments = _list_segment_dirs(seg_root)
    if not segments:
        print(f"[preprocess] 未找到 segment 目录，请检查输入视频或输出路径。", file=sys.stderr)
        return None
    return segments


def preprocess_video(
    video: Path,
    args: argparse.Namespace,
    *,
    seg_root: Optional[Path] = None,
    cleanup_override: Optional[bool] = None,
    skip_audio_sep: bool = False,
) -> tuple[int, Optional[PreprocessResult]]:
    force = bool(getattr(args, "force", False))
    if seg_root is None:
        seg_root, cleanup_needed = _allocate_segments_root(video, args)
    else:
        seg_root = Path(seg_root)
        cleanup_needed = bool(cleanup_override) if cleanup_override is not None else False

    segments = _maybe_segment_video(video, seg_root, args, force=force)
    if segments is None:
        return 1, None

    audio_files: List[Path] = []
    outputs: Dict[Path, Path] = {}
    for seg in segments:
        audio_file = seg / "audio.wav"
        if not audio_file.exists():
            print(f"[preprocess] 缺少音频文件: {audio_file}", file=sys.stderr)
            return 1, None
        if force:
            shutil.rmtree((seg / "audio_sep"), ignore_errors=True)
        if force or not _has_audio_preprocess_outputs(seg):
            outputs[audio_file] = seg / "audio_sep"
            audio_files.append(audio_file)

    if skip_audio_sep:
        if audio_files:
            print(f"[preprocess] 收集到 {len(audio_files)} 个 segment 等待后续音频分离。")
        else:
            print("[preprocess] 所有 segment 均已有音频分离结果，跳过后续音频处理。")
        return 0, PreprocessResult(video=video, seg_root=seg_root, segments=segments, cleanup_needed=cleanup_needed)

    if audio_files:
        try:
            demucs_outcome = _run_demucs_batch_for_segments(audio_files, outputs, args)
        except Exception as exc:
            print(f"[preprocess] Demucs 批处理失败: {exc}", file=sys.stderr)
            return 1, None

        demucs_results = demucs_outcome.results
        if demucs_outcome.errors:
            error_details = "; ".join(f"{path}: {msg}" for path, msg in demucs_outcome.errors.items())
            print(f"[preprocess] Demucs 缺失输出: {error_details}", file=sys.stderr)
            return 1, None

        missing = [audio for audio in audio_files if audio not in demucs_results]
        if missing:
            missing_labels = ", ".join(str(p.parent.name) for p in missing)
            print(f"[preprocess] 以下 segment 缺失音频分离结果: {missing_labels}", file=sys.stderr)
            return 1, None

        try:
            _finalize_demucs_results(demucs_results, args, force=force)
        except Exception as exc:
            print(f"[preprocess] 生成背景切片失败: {exc}", file=sys.stderr)
            return 1, None

        print(f"[preprocess] 全部 {len(audio_files)} 个 segment 的音频预处理已完成。")
    else:
        print("[preprocess] 所有 segment 均已有音频分离结果，跳过。")

    return 0, PreprocessResult(video=video, seg_root=seg_root, segments=segments, cleanup_needed=cleanup_needed)

def _run_demucs_batch_for_segments(
    audio_files: List[Path],
    outputs: Dict[Path, Path],
    args: argparse.Namespace,
) -> BatchSeparationOutcome:
    if not audio_files:
        return BatchSeparationOutcome(results={}, errors={})
    model = getattr(args, "audio_sep_model", None) or "mdx_extra"
    two_stems = getattr(args, "audio_sep_two_stems", None) or "vocals"
    export_format = getattr(args, "audio_sep_format", None) or "wav"
    jobs = getattr(args, "audio_sep_jobs", None)
    return separate_files_batch(
        files=audio_files,
        outputs=outputs,
        model=model,
        two_stems=two_stems,
        export_format=export_format,
        jobs=jobs,
    )


def _finalize_demucs_results(
    results: Dict[Path, SeparationResult],
    args: argparse.Namespace,
    *,
    force: bool = False,
) -> None:
    seg_len = getattr(args, "audio_bg_seg_len", None) or 15.0
    min_last = getattr(args, "audio_bg_min_last", None) or 5.0
    for audio_file, sep in results.items():
        try:
            duration = probe_duration_sec(audio_file.as_posix())
        except Exception:
            duration = None
        if duration:
            _trim_audio_to_duration(sep.voice_path, duration)
            _trim_audio_to_duration(sep.background_path, duration)
        chunk_dir = sep.background_path.parent / "background_chunks"
        if force:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        cut_background_chunks(
            background_file=sep.background_path,
            chunk_dir=chunk_dir,
            seg_len=seg_len,
            min_last=min_last,
        )



def _ensure_full_audio(video: Path, seg_root: Path, override: Optional[str]) -> Optional[Path]:
    if override:
        audio_path = Path(override)
        if not audio_path.exists():
            print(f"指定的音频文件不存在: {audio_path}", file=sys.stderr)
            return None
        return audio_path

    target = seg_root / "full_audio.wav"
    if target.exists():
        return target

    print("[Info] 提取完整音频轨...")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video.as_posix(),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        target.as_posix(),
    ]
    rc = run(cmd)
    if rc != 0:
        return None
    return target if target.exists() else None


def _prepare_joint_filter_video(video: Path, seg_root: Path) -> Path:
    """
    Create a lightweight re-encoded copy of the full video to improve
    compatibility with the Qwen video loader.
    """
    safe_dir = seg_root / "_joint_video"
    safe_dir.mkdir(parents=True, exist_ok=True)
    safe_path = safe_dir / f"{video.stem}_safe.mp4"
    if safe_path.exists():
        return safe_path

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video.as_posix(),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        safe_path.as_posix(),
    ]
    rc = run(cmd)
    if rc != 0 or not safe_path.exists():
        print(f"[Warning] Joint filtering 视频重编码失败，回退到原视频: {video}", file=sys.stderr)
        safe_path.unlink(missing_ok=True)
        return video
    return safe_path


def _gather_video_metadata(video: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video.as_posix(),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        return {"error": "ffprobe_not_found"}
    except subprocess.CalledProcessError as exc:
        return {
            "error": "ffprobe_failed",
            "stderr": exc.stderr.decode(errors="ignore"),
        }

    raw = res.stdout.decode(errors="ignore")
    try:
        info = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {
            "error": "ffprobe_invalid_json",
            "raw": raw,
        }

    if isinstance(info, dict):
        return info
    return {"raw": info}


def _load_json_file(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[Warning] 无法读取 {path}: {exc}", file=sys.stderr)
        return None
    try:
        return json.loads(text)
    except Exception as exc:
        print(f"[Warning] 无法解析 JSON {path}: {exc}", file=sys.stderr)
        return None


def _extract_score(payload: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    node = payload.get(key)
    if not isinstance(node, dict):
        return None
    val = node.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _compose_payload(
    video: Path,
    caption: str,
    metadata: Dict[str, Any],
    dataset: str,
    filters: Dict[str, Any],
    score_vector: List[Optional[float]],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "filename": video.name,
        "metadata": metadata,
        "caption": caption.strip(),
        "filters": filters,
        "score": score_vector,
    }


def _cleanup_intermediate(seg_root: Path) -> None:
    if not seg_root.exists():
        return
    try:
        shutil.rmtree(seg_root)
        print(f"[Cleanup] 已删除中间产物目录: {seg_root}")
    except Exception as exc:
        print(f"[Warning] 无法删除中间产物目录 {seg_root}: {exc}", file=sys.stderr)


def _parse_single_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click long video caption pipeline")
    p.add_argument("--video", required=True, help="Path to input long video")
    p.add_argument("--segment-length", type=float, default=60.0)
    p.add_argument("--min-last", type=float, default=10.0)
    p.add_argument("--model", default=QWEN_DEFAULT_MODEL)
    p.add_argument(
        "--force",
        action="store_true",
        help="忽略已有产物，强制重新执行预处理",
    )
    p.add_argument("--out-dir", default=None, help="Optional override for segments root")
    p.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="处理结束后清理中间分段目录（默认保留以便排查问题）",
    )
    p.add_argument("--dataset", default="captions", help="Dataset name stored in the final JSON output")
    p.add_argument("--output-json", default=None, help="Optional override path for the final caption JSON output")
    p.add_argument("--audio-sep-model", default=None, help="Demucs model override for audio preprocessing")
    p.add_argument("--audio-sep-two-stems", default=None, help="Demucs --two-stems value (default: vocals)")
    p.add_argument(
        "--audio-sep-format",
        choices=["mp3", "wav", "flac", "ogg"],
        default="wav",
        help="Separated audio export format (default: wav)",
    )
    p.add_argument("--audio-sep-jobs", type=int, default=None, help="Parallel jobs passed to Demucs")
    p.add_argument(
        "--audio-bg-seg-len",
        type=float,
        default=None,
        help="Background chunk duration seconds (default: script default 15)",
    )
    p.add_argument(
        "--audio-bg-min-last",
        type=float,
        default=None,
        help="Merge remainder when <= seconds (default: script default 5)",
    )
    # Optional prompts per stage
    p.add_argument("--visual-system", default=None)
    p.add_argument("--visual-user", default=None)
    p.add_argument("--bgm-system", default=None)
    p.add_argument("--bgm-user", default=None)
    p.add_argument("--vocal-system", default=None)
    p.add_argument("--vocal-user", default=None)
    p.add_argument("--joint-system", default=None)
    p.add_argument("--joint-user", default=None)
    p.add_argument("--temporal-system", default=None)
    p.add_argument("--temporal-user", default=None)
    p.add_argument("--filter-system", default=None)
    p.add_argument("--filter-user", default=None)
    p.add_argument("--visual-filter-system", default=None)
    p.add_argument("--visual-filter-user", default=None)
    p.add_argument("--joint-filter-system", default=None)
    p.add_argument("--joint-filter-user", default=None)

    # Model overrides
    p.add_argument("--filter-model", default=QWEN_DEFAULT_MODEL)
    p.add_argument("--filter-audio", default=None, help="Optional path to full-length audio track")
    p.add_argument("--visual-filter-model", default=QWEN_DEFAULT_MODEL)
    p.add_argument("--joint-filter-model", default=QWEN_DEFAULT_MODEL)
    p.add_argument(
        "--visual-filter-visual",
        default=None,
        help="Optional aggregated visual caption file for visual filtering",
    )
    p.add_argument(
        "--visual-filter-visual-filename",
        default=None,
        help="Override per-segment visual caption filename (default: caption.txt)",
    )
    p.add_argument(
        "--visual-filter-frame-interval",
        type=float,
        default=None,
        help="Seconds between sampled frames sent to visual filtering",
    )
    p.add_argument(
        "--visual-filter-max-total-frames",
        type=int,
        default=None,
        help="Maximum total frames uploaded for visual filtering",
    )
    p.add_argument(
        "--visual-filter-frame-width",
        type=int,
        default=None,
        help="Resize width for sampled frames before upload",
    )

    return p.parse_args(argv)


def _load_dataset_entries(dataset_path: Path) -> List[Dict[str, Any]]:
    try:
        raw = dataset_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"无法读取数据集文件 {dataset_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据集 JSON 解析失败: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("数据集 JSON 格式错误，期望为列表")
    entries: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("video_path"), str):
            entries.append(item)
    return entries


def _parse_batch_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch processing for long video caption pipeline")
    parser.add_argument("--dataset-json", required=True, help="Path to dataset JSON containing video_path entries")
    parser.add_argument(
        "--output-json",
        default=None,
        help=f"Aggregate caption JSON output (default: {CAPTION_OUTPUT_ROOT.as_posix()}/<dataset>_captions.json)",
    )
    parser.add_argument("--partial-save-interval", type=int, default=0, help="Write partial results after every N videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional limit on number of videos to process")
    parser.add_argument("--start-index", type=int, default=0, help="Start index within the dataset for batch processing")
    parser.add_argument(
        "--segments-dir",
        default=None,
        help=f"Directory to store intermediate segment data (default: {DATASET_TEMP_ROOT.as_posix()}/<dataset>/segments)",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help="Number of parallel workers for the preprocess stage (default: CPU count)",
    )
    parser.add_argument(
        "--caption-workers",
        type=int,
        default=1,
        help="Number of parallel workers for caption generation (default: 1)",
    )

    args, remaining = parser.parse_known_args(argv)
    args.single_cli = remaining
    return args



def run_single(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.exists():
        print(f"未找到输入视频: {video}", file=sys.stderr)
        return 2

    precomputed: Optional[PreprocessResult] = getattr(args, "_preprocessed", None)
    prep_result: Optional[PreprocessResult] = precomputed
    rc = 0
    if prep_result is None:
        rc, prep_result = preprocess_video(video, args)
        if rc != 0 or prep_result is None:
            return rc if rc != 0 else 1
    else:
        segments = prep_result.segments or _list_segment_dirs(prep_result.seg_root)
        prep_result = PreprocessResult(
            video=prep_result.video,
            seg_root=prep_result.seg_root,
            segments=segments,
            cleanup_needed=prep_result.cleanup_needed,
        )

    seg_root = prep_result.seg_root
    segments = prep_result.segments
    if not segments:
        print(f"未找到 segment 目录，请检查输入视频或输出路径: {seg_root}", file=sys.stderr)
        return 1

    video_label = video.name
    if precomputed is None:
        print(f"[1/6] {video_label} - 分段与音频预处理完成，共 {len(segments)} 段。")
    else:
        print(f"[1/6] {video_label} - 使用已有预处理结果，共 {len(segments)} 段。")

    try:
        for i, seg in enumerate(segments, 1):
            print(f"[2/6] {video_label} ({i}/{len(segments)}) - BGM caption...")
            rc = run([
                sys.executable,
                (SCRIPT_DIR / "bgm_caption.py").as_posix(),
                "--input",
                seg.as_posix(),
                "--model",
                args.model,
                *_maybe_prompts("bgm", args.bgm_system, args.bgm_user),
            ])
            if rc != 0:
                print("  -> BGM caption 失败，继续其他步骤", file=sys.stderr)

            print(f"[3/6] {video_label} ({i}/{len(segments)}) - Vocal caption...")
            rc = run([
                sys.executable,
                (SCRIPT_DIR / "vocal_caption.py").as_posix(),
                "--input",
                seg.as_posix(),
                "--model",
                args.model,
                *_maybe_prompts("vocal", args.vocal_system, args.vocal_user),
            ])
            if rc != 0:
                print("  -> Vocal caption 失败，继续其他步骤", file=sys.stderr)

            print(f"[4/6] {video_label} ({i}/{len(segments)}) - Visual caption...")
            rc = run([
                sys.executable,
                (SCRIPT_DIR / "visual_captioning.py").as_posix(),
                "--input",
                seg.as_posix(),
                "--model",
                args.model,
                "--require-silent",
                *_maybe_prompts("visual", args.visual_system, args.visual_user),
            ])
            if rc != 0:
                print("  -> Visual caption 失败，继续其他步骤", file=sys.stderr)

            print(f"[5/6] {video_label} ({i}/{len(segments)}) - Joint caption...")
            rc = run([
                sys.executable,
                (SCRIPT_DIR / "joint_caption.py").as_posix(),
                "--input",
                seg.as_posix(),
                "--model",
                args.model,
                *_maybe_prompts("joint", args.joint_system, args.joint_user),
            ])
            if rc != 0:
                print("  -> Joint caption 失败", file=sys.stderr)

        print(f"[Final 1/4] {video_label} - Temporal integration...")
        rc = run([
            sys.executable,
            (SCRIPT_DIR / "temporal_integration.py").as_posix(),
            "--video",
            video.as_posix(),
            "--segments-root",
            seg_root.as_posix(),
            "--model",
            args.model,
            *_maybe_prompts("temporal", args.temporal_system, args.temporal_user),
        ])
        if rc != 0:
            return rc

        print(f"[Final 2/4] {video_label} - Visual filtering review...")
        visual_filter_model = args.visual_filter_model or QWEN_DEFAULT_MODEL
        visual_cmd = [
            sys.executable,
            (SCRIPT_DIR / "visual_filtering.py").as_posix(),
            "--segments-root",
            seg_root.as_posix(),
            "--model",
            visual_filter_model,
            *_maybe_prompts("visual_filter", args.visual_filter_system, args.visual_filter_user),
        ]
        if args.visual_filter_visual:
            visual_cmd += ["--visual", args.visual_filter_visual]
        if args.visual_filter_visual_filename:
            visual_cmd += ["--visual-filename", args.visual_filter_visual_filename]
        if args.visual_filter_frame_interval is not None:
            visual_cmd += ["--frame-interval", str(args.visual_filter_frame_interval)]
        if args.visual_filter_max_total_frames is not None:
            visual_cmd += ["--max-total-frames", str(args.visual_filter_max_total_frames)]
        if args.visual_filter_frame_width is not None:
            visual_cmd += ["--frame-width", str(args.visual_filter_frame_width)]
        rc = run(visual_cmd)
        if rc != 0:
            return rc

        audio_path = _ensure_full_audio(video, seg_root, args.filter_audio)
        if audio_path is None:
            print("提取完整音频失败，无法执行音频过滤评分", file=sys.stderr)
            return 2

        print(f"[Final 3/4] {video_label} - Audio filtering review...")
        filter_model = args.filter_model or QWEN_DEFAULT_MODEL
        audio_rc = run([
            sys.executable,
            (SCRIPT_DIR / "audio_filtering.py").as_posix(),
            "--audio",
            audio_path.as_posix(),
            "--segments-root",
            seg_root.as_posix(),
            "--model",
            filter_model,
            *_maybe_prompts("filter", args.filter_system, args.filter_user),
        ])
        if audio_rc != 0:
            return audio_rc

        print(f"[Final 4/4] {video_label} - Joint filtering review...")
        joint_filter_model = args.joint_filter_model or QWEN_DEFAULT_MODEL
        joint_video = _prepare_joint_filter_video(video, seg_root)
        joint_cmd = [
            sys.executable,
            (SCRIPT_DIR / "joint_filtering.py").as_posix(),
            "--video",
            joint_video.as_posix(),
            "--segments-root",
            seg_root.as_posix(),
            "--model",
            joint_filter_model,
            *_maybe_prompts("joint_filter", args.joint_filter_system, args.joint_filter_user),
        ]
        try:
            joint_rc = run(joint_cmd)
        finally:
            if joint_video != video:
                safe_dir = joint_video.parent
                try:
                    joint_video.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    # remove directory if empty
                    safe_dir.rmdir()
                except Exception:
                    pass
        if joint_rc != 0:
            return joint_rc

        final_caption_path = seg_root / "temporal_caption.txt"
        if not final_caption_path.exists():
            print(f"未找到最终 caption 文件: {final_caption_path}", file=sys.stderr)
            return 2

        try:
            final_caption = final_caption_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"读取最终 caption 失败: {exc}", file=sys.stderr)
            return 1

        metadata = _gather_video_metadata(video)

        joint_raw = _load_json_file(seg_root / "joint_filtering.json")
        joint_score = joint_raw.get("score") if isinstance(joint_raw, dict) else None

        visual_raw = _load_json_file(seg_root / "visual_filtering.json")
        visual_score = visual_raw.get("score") if isinstance(visual_raw, dict) else None

        audio_raw = _load_json_file(seg_root / "audio_filtering.json")
        audio_score = audio_raw.get("score") if isinstance(audio_raw, dict) else None

        filters = {
            "joint": joint_score,
            "visual": visual_score,
            "audio": audio_score,
        }

        score_vector: List[Optional[float]] = [None] * 9
        if isinstance(joint_score, dict):
            val = joint_score.get("score")
            if isinstance(val, (int, float)):
                score_vector[0] = float(val)

        visual_eval = visual_score.get("evaluation") if isinstance(visual_score, dict) else None
        score_vector[1] = _extract_score(visual_eval, "visual_hallucinations")
        score_vector[2] = _extract_score(visual_eval, "visual_omissions")
        score_vector[3] = _extract_score(visual_eval, "visual_inaccuracies")
        score_vector[4] = _extract_score(visual_eval, "visual_granularity_and_detail")

        audio_eval = audio_score.get("evaluation") if isinstance(audio_score, dict) else None
        score_vector[5] = _extract_score(audio_eval, "hallucinations")
        score_vector[6] = _extract_score(audio_eval, "omissions")
        score_vector[7] = _extract_score(audio_eval, "inaccuracies")
        score_vector[8] = _extract_score(audio_eval, "granularity_and_detail")

        payload = _compose_payload(
            video,
            final_caption,
            metadata,
            dataset=args.dataset,
            filters=filters,
            score_vector=score_vector,
        )

        if args.output_json:
            output_path = Path(args.output_json)
        elif args.out_dir:
            out_dir_path = Path(args.out_dir)
            out_base = out_dir_path.parent if out_dir_path.parent != out_dir_path else video.parent
            output_path = out_base / f"{video.stem}_caption.json"
        else:
            output_path = video.parent / f"{video.stem}_caption.json"

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Output] Final caption JSON written to: {output_path}")
        except Exception as exc:
            print(f"写入最终 JSON 失败: {exc}", file=sys.stderr)
            return 1

        return 0
    finally:
        if prep_result and prep_result.cleanup_needed:
            _cleanup_intermediate(prep_result.seg_root)

def _truncate_log(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _log_batch_failure(
    video_path: str,
    *,
    exit_code: Optional[int] = None,
    message: Optional[str] = None,
    stdout: str = "",
    stderr: str = "",
) -> None:
    parts = [f"[Batch] 视频处理失败: {video_path}"]
    if exit_code is not None:
        parts[-1] += f" (exit {exit_code})"
    if message:
        parts[-1] += f" - {message}"
    print(parts[0], flush=True)

    def _emit(label: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        snippet = _truncate_log(content, limit=2000)
        print(f"[Batch] {label}日志:\n{snippet}", flush=True)

    _emit("stdout", stdout)
    _emit("stderr", stderr)


def _write_partial_results(base_path: Path, chunk_index: int, results: List[Dict[str, Any]]) -> None:
    """Persist a chunk of batch results to disk for incremental progress tracking."""
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "chunk_index": chunk_index,
        "chunk_size": len(results),
        "success_count": sum(1 for r in results if r.get("status") == "ok"),
        "failure_count": sum(1 for r in results if r.get("status") != "ok"),
        "items": results,
    }
    partial_name = f"{base_path.stem}_partial_{chunk_index:04d}.json"
    partial_path = base_path.parent / partial_name
    try:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Batch] Partial results saved: {partial_path}")
    except Exception as exc:
        print(f"[Warning] 无法写入部分结果 {partial_path}: {exc}", file=sys.stderr)


def _append_dataset_log(log_path: Path, payload: Dict[str, Any]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[Warning] 无法写入日志 {log_path}: {exc}", file=sys.stderr)


def _preprocess_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker wrapper for parallel preprocess execution."""
    index = payload["index"]
    video: Path = payload["video"]
    single_args: argparse.Namespace = payload["single_args"]
    seg_root: Path = payload["seg_root"]
    cleanup_flag: bool = payload["cleanup_flag"]
    skip_audio_sep: bool = payload.get("skip_audio_sep", False)
    try:
        _load_env()
        rc, prep_result = preprocess_video(
            video,
            single_args,
            seg_root=seg_root,
            cleanup_override=cleanup_flag,
            skip_audio_sep=skip_audio_sep,
        )
        return {
            "index": index,
            "rc": rc,
            "prep_result": prep_result,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - worker should rarely explode
        return {
            "index": index,
            "rc": 1,
            "prep_result": None,
            "error": str(exc),
        }


def _single_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker wrapper for running the full single-video pipeline."""
    index = payload["index"]
    video: Path = payload["video"]
    entry: Dict[str, Any] = payload["entry"]
    single_args: argparse.Namespace = payload["single_args"]
    prep_result: PreprocessResult = payload["prep_result"]
    per_video_output: Path = payload["per_video_output"]

    per_video_parent = per_video_output.parent
    try:
        _load_env()
        setattr(single_args, "_preprocessed", prep_result)
        result_code = run_single(single_args)
        item: Dict[str, Any] = {
            "video_path": str(video),
            "source_entry": entry,
            "status": "ok" if result_code == 0 else "error",
        }
        if result_code == 0:
            caption_payload = _load_json_file(per_video_output)
            if caption_payload is None:
                item["status"] = "error"
                item["error"] = f"缺少输出文件: {per_video_output}"
            else:
                item["caption"] = caption_payload
        else:
            item["exit_code"] = result_code
    except Exception as exc:  # pragma: no cover - defensive
        item = {
            "video_path": str(video),
            "source_entry": entry,
            "status": "error",
            "error": str(exc),
        }
    finally:
        try:
            per_video_output.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if per_video_parent.is_dir() and not any(per_video_parent.iterdir()):
                per_video_parent.rmdir()
        except Exception:
            pass

    return {
        "index": index,
        "item": item,
    }


def _collect_audio_sep_tasks(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gather audio separation tasks after segmentation has completed."""

    def _audio_config(args: argparse.Namespace) -> Tuple[Any, ...]:
        model = getattr(args, "audio_sep_model", None) or "mdx_extra"
        two_stems = getattr(args, "audio_sep_two_stems", None) or "vocals"
        export_format = getattr(args, "audio_sep_format", None) or "wav"
        jobs_override = getattr(args, "audio_sep_jobs", None)
        bg_seg_len = getattr(args, "audio_bg_seg_len", None) or 15.0
        bg_min_last = getattr(args, "audio_bg_min_last", None) or 5.0
        return (model, two_stems, export_format, jobs_override, bg_seg_len, bg_min_last)

    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for job in jobs:
        single_args = job.get("single_args")
        prep_result: Optional[PreprocessResult] = job.get("prep_result")
        if single_args is None or prep_result is None:
            continue
        if job.get("preprocess_rc") not in (0, None):
            continue
        segments = prep_result.segments or []
        if not segments:
            continue

        force = bool(getattr(single_args, "force", False))
        audio_files: List[Path] = []
        outputs: Dict[Path, Path] = {}
        missing_audio: List[Path] = []

        for seg in segments:
            audio_file = seg / "audio.wav"
            if not audio_file.exists():
                missing_audio.append(audio_file)
                continue
            audio_sep_dir = seg / "audio_sep"
            if force:
                shutil.rmtree(audio_sep_dir, ignore_errors=True)
            if force or not _has_audio_preprocess_outputs(seg):
                outputs[audio_file] = audio_sep_dir
                audio_files.append(audio_file)

        if missing_audio:
            job["preprocess_rc"] = 1
            job["preprocess_error"] = f"missing_audio_files: {', '.join(str(p) for p in missing_audio)}"
            continue
        if not audio_files:
            continue

        config = _audio_config(single_args)
        group = grouped.get(config)
        if group is None:
            group = {
                "config": config,
                "audio_files": [],
                "outputs": {},
                "jobs": [],
                "primary_args": single_args,
                "force": force,
                "file_to_job": {},
            }
            grouped[config] = group
        else:
            group["force"] = group["force"] or force

        for audio_file in audio_files:
            group["audio_files"].append(audio_file)
            group["outputs"][audio_file] = outputs[audio_file]
            group["file_to_job"][audio_file] = job
        if job not in group["jobs"]:
            group["jobs"].append(job)

    return list(grouped.values())


def _apply_preprocess_result(jobs: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    """Merge a worker result back into the job table."""
    index = result.get("index")
    if index is None or not (0 <= index < len(jobs)):
        return
    job = jobs[index]
    rc = result.get("rc", 1)
    job["preprocess_rc"] = rc
    job["prep_result"] = result.get("prep_result")
    error_detail = result.get("error")
    if error_detail:
        job["preprocess_error"] = error_detail
    if rc == 0 and job.get("prep_result") is not None and job.get("single_args") is not None:
        setattr(job["single_args"], "_preprocessed", job["prep_result"])
    elif rc != 0 and job.get("cleanup_flag") and job.get("seg_root") is not None:
        _cleanup_intermediate(job["seg_root"])



def run_batch(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset_json)
    try:
        entries = _load_dataset_entries(dataset_path)
    except Exception as exc:
        print(f"无法加载数据集 {dataset_path}: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print(f"数据集 {dataset_path} 中没有可用的 video_path 项。", file=sys.stderr)
        return 2

    DATASET_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_dir = DATASET_TEMP_ROOT / dataset_path.stem
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_progress_log = dataset_dir / "progress.log"

    segments_base = Path(args.segments_dir) if getattr(args, "segments_dir", None) else dataset_dir / "segments"
    segments_base.mkdir(parents=True, exist_ok=True)

    CAPTION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aggregated_output_path = (
        Path(args.output_json)
        if getattr(args, "output_json", None)
        else CAPTION_OUTPUT_ROOT / f"{dataset_path.stem}_captions.json"
    )
    args.output_json = aggregated_output_path.as_posix()

    start_index = max(0, int(args.start_index or 0))
    subset = entries[start_index:]
    if args.max_videos is not None:
        subset = subset[: max(0, args.max_videos)]

    available: List[Dict[str, Any]] = []
    missing: List[str] = []
    for entry in subset:
        video_path = entry.get("video_path")
        if not isinstance(video_path, str):
            continue
        if not Path(video_path).exists():
            missing.append(video_path)
            continue
        available.append(entry)

    if missing:
        print(f"[Warn] 跳过 {len(missing)} 个不存在的视频文件。", file=sys.stderr)
    if not available:
        print("选择的数据集中没有可处理的视频。", file=sys.stderr)
        return 2

    per_video_output_root = dataset_dir / "captions"
    per_video_output_root.mkdir(parents=True, exist_ok=True)

    partial_interval = max(0, int(args.partial_save_interval or 0))
    partial_bucket: List[Dict[str, Any]] = []
    partial_index = 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    batch_start = time.time()
    log_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_filename = f"{dataset_path.stem}_{log_timestamp}.log"
    batch_log_path = LOG_DIR / log_filename
    log_total_target = len(available)
    try:
        with batch_log_path.open("w", encoding="utf-8") as fh:
            fh.write(
                f"[{datetime.utcnow().isoformat()}Z] Batch started: dataset={dataset_path} total_candidates={len(subset)} available={log_total_target}\n"
            )
    except Exception:
        pass
    LOG_INTERVAL = 60.0
    last_log_emit = batch_start

    def _append_log_line(message: str) -> None:
        try:
            with batch_log_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except Exception:
            pass

    jobs = []
    for entry in available:
        video_path = entry.get("video_path")
        if not isinstance(video_path, str):
            continue
        video = Path(video_path)
        per_video_output = per_video_output_root / f"{video.stem}_caption.json"

        single_argv = ["--video", video_path, "--output-json", per_video_output.as_posix(), *getattr(args, "single_cli", [])]
        try:
            single_args = _parse_single_args(single_argv)
        except SystemExit as exc:
            _log_batch_failure(video_path, exit_code=exc.code if isinstance(exc.code, int) else 1, message="参数解析失败")
            jobs.append(
                {
                    "entry": entry,
                    "video": video,
                    "single_args": None,
                    "per_video_output": per_video_output,
                    "seg_root": None,
                    "cleanup_flag": False,
                    "preprocess_rc": 1,
                    "preprocess_error": "arg_parse_failed",
                    "prep_result": None,
                }
            )
            continue

        if single_args.out_dir:
            seg_root = Path(single_args.out_dir)
        else:
            seg_root = segments_base / f"{video.stem}_{uuid4().hex[:8]}"
        cleanup_flag = bool(getattr(single_args, "cleanup_temp", False)) and not single_args.out_dir

        jobs.append(
            {
                "entry": entry,
                "video": video,
                "single_args": single_args,
                "per_video_output": per_video_output,
                "seg_root": seg_root,
                "cleanup_flag": cleanup_flag,
                "preprocess_rc": None,
                "preprocess_error": None,
                "prep_result": None,
            }
        )

    preprocess_payloads: List[Dict[str, Any]] = []
    for index, job in enumerate(jobs):
        single_args = job.get("single_args")
        if single_args is None:
            continue
        payload = {
            "index": index,
            "video": job["video"],
            "single_args": single_args,
            "seg_root": job["seg_root"],
            "cleanup_flag": job["cleanup_flag"],
            "skip_audio_sep": True,
        }
        preprocess_payloads.append(payload)

    preprocess_total = len(preprocess_payloads)
    if preprocess_total:
        requested_workers = getattr(args, "preprocess_workers", None)
        if requested_workers is None or requested_workers <= 0:
            cpu_count = os.cpu_count() or 1
            preprocess_workers = min(preprocess_total, cpu_count)
        else:
            preprocess_workers = min(preprocess_total, requested_workers)
        preprocess_workers = max(1, preprocess_workers)
        print(f"[Batch] 启动预处理阶段，共 {preprocess_total} 个视频，workers={preprocess_workers}")

        if preprocess_workers == 1:
            for payload in preprocess_payloads:
                result = _preprocess_worker(payload)
                _apply_preprocess_result(jobs, result)
        else:
            with ProcessPoolExecutor(max_workers=preprocess_workers) as pool:
                future_map = {pool.submit(_preprocess_worker, payload): payload["index"] for payload in preprocess_payloads}
                completed = 0
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        result = {
                            "index": idx,
                            "rc": 1,
                            "prep_result": None,
                            "error": str(exc),
                        }
                    _apply_preprocess_result(jobs, result)
                    completed += 1
                    print(f"[Batch] Preprocess progress: {completed}/{preprocess_total}", flush=True)

        preprocess_success = sum(1 for job in jobs if job.get("preprocess_rc") == 0)
        preprocess_fail = sum(
            1
            for job in jobs
            if job.get("single_args") is not None and job.get("preprocess_rc") not in (0, None)
        )
        if preprocess_fail:
            print(f"[Batch] 预处理失败 {preprocess_fail} 个视频。", file=sys.stderr)
        print(f"[Batch] 预处理完成: {preprocess_success}/{preprocess_total}")
    else:
        print("[Batch] 无需预处理任务。")

    for job in jobs:
        if job.get("single_args") is not None and job.get("preprocess_rc") is None:
            job["preprocess_rc"] = 1
            job.setdefault("preprocess_error", "preprocess_not_run")

    audio_tasks = _collect_audio_sep_tasks(jobs)
    if audio_tasks:
        total_segments_for_sep = sum(len(task["audio_files"]) for task in audio_tasks)
        print(
            f"[Batch] 启动音频分离阶段，共 {len(audio_tasks)} 个配置，{total_segments_for_sep} 段待处理。"
        )
        for task in audio_tasks:
            audio_files = task["audio_files"]
            outputs = task["outputs"]
            args_for_group = task["primary_args"]
            jobs_in_group = task["jobs"]
            force_flag = task["force"]
            video_names = ", ".join(str(job["video"].name) for job in jobs_in_group)
            try:
                demucs_outcome = _run_demucs_batch_for_segments(
                    audio_files,
                    outputs,
                    args_for_group,
                )
            except Exception as exc:
                message = f"audio_sep_failed: {exc}"
                for job in jobs_in_group:
                    job["preprocess_rc"] = 1
                    job["preprocess_error"] = message
                    if job.get("cleanup_flag") and job.get("seg_root") is not None:
                        _cleanup_intermediate(job["seg_root"])
                print(f"[Batch] 音频分离失败 ({video_names}): {exc}", file=sys.stderr)
                continue

            demucs_results = demucs_outcome.results
            failure_map: Dict[Path, str] = dict(demucs_outcome.errors)
            # Guard against silent misses with no explicit error message
            for audio in audio_files:
                if audio not in demucs_results and audio not in failure_map:
                    failure_map[audio] = "missing_demucs_output"

            file_to_job: Dict[Path, Dict[str, Any]] = task.get("file_to_job", {})
            job_failures: Dict[int, Dict[str, Any]] = {}
            for audio_path, reason in failure_map.items():
                job = file_to_job.get(audio_path)
                if job is None:
                    continue
                key = id(job)
                entry = job_failures.setdefault(key, {"job": job, "details": []})
                entry["details"].append((audio_path, reason))

            failed_job_ids = set(job_failures.keys())
            for _, info in job_failures.items():
                job = info["job"]
                if job.get("preprocess_rc") not in (0, None):
                    continue
                details = "; ".join(f"{path}: {reason}" for path, reason in info["details"])
                job["preprocess_rc"] = 1
                job["preprocess_error"] = f"audio_sep_missing_outputs: {details}"
                if job.get("cleanup_flag") and job.get("seg_root") is not None:
                    _cleanup_intermediate(job["seg_root"])
                print(f"[Batch] 音频分离缺少输出 ({job['video'].name}): {details}", file=sys.stderr)

            success_results: Dict[Path, SeparationResult] = {}
            for audio_path, sep in demucs_results.items():
                job = file_to_job.get(audio_path)
                if job is None or id(job) in failed_job_ids:
                    continue
                success_results[audio_path] = sep

            if not failure_map and not success_results and audio_files:
                message = "audio_sep_missing_outputs: no_results"
                for job in jobs_in_group:
                    job["preprocess_rc"] = 1
                    job["preprocess_error"] = message
                    if job.get("cleanup_flag") and job.get("seg_root") is not None:
                        _cleanup_intermediate(job["seg_root"])
                print(f"[Batch] 音频分离缺少输出 ({video_names}): no_results", file=sys.stderr)
                continue

            if not success_results:
                # 所有相关视频均失败，跳过后续处理
                continue

            try:
                _finalize_demucs_results(
                    success_results,
                    args_for_group,
                    force=force_flag,
                )
            except Exception as exc:
                message = f"audio_finalize_failed: {exc}"
                for job in jobs_in_group:
                    if id(job) in failed_job_ids:
                        continue
                    job["preprocess_rc"] = 1
                    job["preprocess_error"] = message
                    if job.get("cleanup_flag") and job.get("seg_root") is not None:
                        _cleanup_intermediate(job["seg_root"])
                print(f"[Batch] 音频分离后处理失败 ({video_names}): {exc}", file=sys.stderr)
                continue

            jobs_with_success: Dict[int, Dict[str, Any]] = {}
            for audio_path in success_results:
                job = file_to_job.get(audio_path)
                if job is None:
                    continue
                jobs_with_success[id(job)] = job

            for job in jobs_in_group:
                if id(job) in failed_job_ids:
                    continue
                if jobs_with_success and id(job) not in jobs_with_success:
                    continue
                print(f"[Batch] 音频分离完成: {job['video']}")
    else:
        print("[Batch] 无需额外音频分离。")

    items: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    success_count = 0
    failure_count = 0
    progress_done = 0
    output_path = Path(args.output_json)
    total = len(jobs)

    def _maybe_emit_log(force: bool = False) -> None:
        nonlocal last_log_emit, success_count, failure_count, progress_done
        now = time.time()
        if not force and (now - last_log_emit) < LOG_INTERVAL:
            return
        processed = progress_done
        elapsed = now - batch_start
        avg = elapsed / processed if processed else 0.0
        summary = (
            f"[{datetime.utcnow().isoformat()}Z] processed={processed}/{log_total_target} "
            f"success={success_count} failure={failure_count} avg_sec_per_video={avg:.2f} "
            f"elapsed_sec={elapsed:.2f}"
        )
        _append_log_line(summary)
        last_log_emit = now

    _maybe_emit_log(force=True)

    def _record_item(index: int, item: Dict[str, Any]) -> None:
        nonlocal success_count, failure_count, progress_done, partial_index
        if index < 0 or index >= len(items):
            return
        if items[index] is not None:
            return
        status = item.get("status")
        if status == "ok":
            success_count += 1
        else:
            failure_count += 1
        items[index] = item
        if partial_interval:
            partial_bucket.append(item)
            if len(partial_bucket) >= partial_interval:
                partial_index += 1
                _write_partial_results(output_path, partial_index, list(partial_bucket))
                partial_bucket.clear()
        progress_done += 1
        status_label = status if isinstance(status, str) else "unknown"
        print(f"[Batch] Progress: {progress_done}/{total} - {status_label}", flush=True)
        _maybe_emit_log()

    caption_payloads: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        entry = job["entry"]
        video = job["video"]
        per_video_output = job["per_video_output"]
        single_args = job["single_args"]
        prep_result = job.get("prep_result")
        preprocess_rc = job.get("preprocess_rc")

        if preprocess_rc != 0 or prep_result is None or single_args is None:
            item = {
                "video_path": str(video),
                "source_entry": entry,
                "status": "error",
                "error": "preprocess_failed",
                "exit_code": preprocess_rc if isinstance(preprocess_rc, int) else 1,
            }
            if job.get("preprocess_error"):
                item["error_detail"] = job["preprocess_error"]
            _record_item(idx, item)
        else:
            caption_payloads.append(
                {
                    "index": idx,
                    "entry": entry,
                    "video": video,
                    "single_args": single_args,
                    "prep_result": prep_result,
                    "per_video_output": per_video_output,
                }
            )

    caption_total = len(caption_payloads)
    if caption_total:
        caption_workers = getattr(args, "caption_workers", 1) or 1
        caption_workers = max(1, caption_workers)
        caption_workers = min(caption_workers, caption_total)
        if caption_workers == 1:
            for payload in caption_payloads:
                result = _single_worker(payload)
                _record_item(result["index"], result["item"])
        else:
            with ProcessPoolExecutor(max_workers=caption_workers) as pool:
                future_map = {pool.submit(_single_worker, payload): payload for payload in caption_payloads}
                for future in as_completed(future_map):
                    payload = future_map[future]
                    idx = payload["index"]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        item = {
                            "video_path": str(payload["video"]),
                            "source_entry": payload["entry"],
                            "status": "error",
                            "error": str(exc),
                        }
                        result = {
                            "index": idx,
                            "item": item,
                        }
                    _record_item(result["index"], result["item"])
    else:
        print("[Batch] 无需 caption 执行任务。")

    for idx, item in enumerate(items):
        if item is None:
            job = jobs[idx]
            fallback = {
                "video_path": str(job["video"]),
                "source_entry": job["entry"],
                "status": "error",
                "error": "internal_no_result",
            }
            _record_item(idx, fallback)

    if partial_interval and partial_bucket:
        partial_index += 1
        _write_partial_results(output_path, partial_index, list(partial_bucket))

    _maybe_emit_log(force=True)

    finalized_items: List[Dict[str, Any]] = [item for item in items if item is not None]

    aggregated_payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_json": str(dataset_path),
        "start_index": start_index,
        "requested": len(subset),
        "processed": len(finalized_items),
        "success_count": success_count,
        "failure_count": failure_count,
        "items": finalized_items,
        "log_path": str(batch_log_path),
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Batch] 结果已写入: {output_path}")
    except Exception as exc:
        print(f"写入批处理输出失败: {exc}", file=sys.stderr)
        return 1

    success_rate = float(success_count) / total if total else 0.0
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_json": str(dataset_path),
        "success": success_count,
        "failure": failure_count,
        "total": total,
        "success_rate": success_rate,
        "segments_dir": str(segments_base),
        "output_json": str(output_path),
        "log_path": str(batch_log_path),
    }
    _append_dataset_log(dataset_progress_log, log_entry)

    print(f"[Batch] Log written to: {batch_log_path}")
    _maybe_emit_log(force=True)

    shutil.rmtree(per_video_output_root, ignore_errors=True)
    dataset_dir_resolved = dataset_dir.resolve()
    output_resolved = output_path.resolve()
    try:
        keep_dataset_dir = output_resolved.is_relative_to(dataset_dir_resolved)
    except AttributeError:
        keep_dataset_dir = dataset_dir_resolved in output_resolved.parents
    if not keep_dataset_dir:
        shutil.rmtree(dataset_dir, ignore_errors=True)
    shutil.rmtree(segments_base, ignore_errors=True)

    if failure_count:
        print(f"[Batch] 有 {failure_count} 个视频处理失败。", file=sys.stderr)
        return 1
    if not getattr(args, "segments_dir", None):
        shutil.rmtree(segments_base, ignore_errors=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_env()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    mode = "single"
    if raw_args and raw_args[0] in {"single", "batch"}:
        mode = raw_args.pop(0)
    elif any(arg.startswith("--video") for arg in raw_args):
        mode = "single"
    else:
        mode = "batch" if raw_args else "batch"

    if mode == "batch":
        args = _parse_batch_args(raw_args)
        return run_batch(args)

    args = _parse_single_args(raw_args)
    return run_single(args)


if __name__ == "__main__":
    sys.exit(main())

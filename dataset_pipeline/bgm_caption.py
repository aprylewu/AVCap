#!/usr/bin/env python3
"""
BGM captioning for per-segment videos using Qwen3-Omni Thinking (served via vLLM).

流程（严格限定在单个 segment 内，不跨段）：
1) 读取 segment 目录（形如 <video>_segments/segment_0000）或整个 segments 根目录。
2) 在每个 segment 下查找：
   - 基础视频：<seg_dir>/video.mp4（若存在音轨，仅取其视频流，不会带出原音频）
   - 经过 audio_preprocess 生成的 BGM 切片：<seg_dir>/audio_sep/background_chunks/bg_*.{wav,mp3,...}
3) 逐个 BGM 切片：
   - 以 segment 的 video.mp4 为第一个输入，按累计偏移 t0 和 bg 时长 t 进行精准剪裁（-ss/-t 放在 -i 之后）。
   - 以 bg_XXX.* 为第二个输入，仅映射其音频流，与剪裁后的视频 mux 在一起（-map 0:v -map 1:a）。
   - 输出到 <seg_dir>/bgm_chunks/clip_XXX.mp4（不会跨越 segment 时长）。
4) 对每个 clip_XXX.mp4 调用本地 Qwen 模型（你可自备 system / user prompt），得到 BGM caption。
5) 将每段的所有 caption 串联写入 <seg_dir>/bgm_captions.txt（或 --out-dir 镜像目录下）。

依赖：
- pip install python-dotenv
- 本机 ffmpeg/ffprobe
- 本地 vLLM 服务：vllm serve Qwen3-Omni-30B-A3B-Thinking --port 8901 ...

用法示例：
  # 针对单个分段目录
  python bgm_caption.py \
    --input /path/to/<video>_segments/segment_0000 \
    --model Qwen3-Omni-30B-A3B-Thinking \
    --system @path/to/system.txt \
    --user @path/to/user.txt

  # 针对整个 segments 根目录（逐段处理）
  python bgm_caption.py \
    --input /path/to/<video>_segments \
    --model Qwen3-Omni-30B-A3B-Thinking \
    --system @path/to/system.txt \
    --user @path/to/user.txt

注意：本脚本不会跨越 segment 的时间边界；BGM 切片与视频剪裁一一对应。
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
# Utils: env & deps
# ----------------------------

def _load_env() -> None:
    # 加载 .env（支持从 CWD 及脚本同目录）
    load_dotenv()
    env_candidate = SCRIPT_DIR / ".env"
    if env_candidate.exists():
        load_dotenv(env_candidate.as_posix(), override=False)


# ----------------------------
# ffmpeg helpers
# ----------------------------

class FFmpegError(RuntimeError):
    pass


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _require_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if _which(b) is None]
    if missing:
        raise EnvironmentError(
            f"缺少依赖: {', '.join(missing)}。请安装 ffmpeg (含 ffprobe)。"
        )


def _run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"命令失败: {' '.join(cmd)}\n"
            f"STDOUT: {e.stdout.decode(errors='ignore')}\n"
            f"STDERR: {e.stderr.decode(errors='ignore')}"
        )


def probe_duration_sec(media_path: Path) -> float:
    _require_binaries()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        media_path.as_posix(),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"ffprobe 失败: {e.stderr.decode(errors='ignore')}")
    import json as _json

    info = _json.loads(res.stdout.decode() or "{}")
    try:
        dur = float(info["format"]["duration"])  # seconds
    except Exception:
        raise FFmpegError("无法从 ffprobe 输出中解析 duration")
    return dur


def _format_ts(t: float) -> str:
    # ffmpeg 接受小数秒；保留毫秒
    return f"{t:.3f}"


def has_audio_stream(video_path: Path) -> Optional[bool]:
    if _which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    out = proc.stdout.decode(errors="ignore").strip()
    return bool(out)


def strip_audio_copy(video_path: Path, out_dir: Path) -> Optional[Path]:
    """生成无声拷贝（优先流复制，失败时再转码）。"""
    if _which("ffmpeg") is None:
        return None
    out_path = out_dir / f"{video_path.stem}.silent{video_path.suffix}"
    # 先尝试流复制
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path.as_posix(),
                "-c",
                "copy",
                "-an",
                out_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass
    # 回退：转码
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path.as_posix(),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                out_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        return None
    return None


# ----------------------------
# Scan helpers
# ----------------------------

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}


def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES


def find_segment_dirs(root: Path) -> List[Path]:
    """查找 segment_* 目录。
    规则：
      - 若 root 本身是 segment_* 目录，返回 [root]
      - 否则返回 root 下的所有 segment_* 子目录
    """
    if root.is_dir() and root.name.startswith("segment_"):
        return [root]
    if root.is_dir():
        segs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("segment_")]
        if segs:
            return segs
    return []


def resolve_segment_video(seg_dir: Path) -> Optional[Path]:
    """优先选择 <seg_dir>/video.mp4；若不存在，则返回 None。"""
    v = seg_dir / "video.mp4"
    if v.exists() and v.is_file():
        return v
    # 兜底：寻找任意已知视频文件，但强烈建议使用 video.mp4
    for p in seg_dir.iterdir():
        if _is_video_file(p):
            return p
    return None


def find_bgm_chunks(seg_dir: Path) -> List[Path]:
    """查找 BGM 切片：<seg_dir>/audio_sep/background_chunks/bg_*.* 按编号排序。"""
    bg_dir = seg_dir / "audio_sep" / "background_chunks"
    if not bg_dir.exists() or not bg_dir.is_dir():
        return []
    items = [p for p in bg_dir.iterdir() if p.is_file() and p.name.startswith("bg_")]
    def _key(p: Path) -> Tuple[int, str]:
        stem = p.stem  # e.g., bg_001
        try:
            idx = int(stem.split("_")[1])
        except Exception:
            idx = 10**9
        return (idx, p.suffix.lower())
    items.sort(key=_key)
    return items


# ----------------------------
# BGM clip muxing
# ----------------------------

def mux_video_with_bgm(
    base_video: Path,
    out_clip: Path,
    t0: float,
    dur: float,
    bgm_audio: Path,
    reencode_video: bool = True,
) -> None:
    """从 base_video 精准截取 [t0, t0+dur)，并与 bgm_audio 合并为一个 mp4（仅映射视频+bgm）。"""
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"] if reencode_video else ["-c:v", "copy"]
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        # precise trim as INPUT option for #0 (applies only to base video)
        "-ss",
        _format_ts(max(0.0, t0)),
        "-t",
        _format_ts(max(0.0, dur)),
        # input #0: base video
        "-i",
        base_video.as_posix(),
        # input #1: bgm audio (already a chunk)
        "-i",
        bgm_audio.as_posix(),
        # map video from #0, audio from #1
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *vcodec,
        "-c:a",
        "aac",
        "-shortest",
        out_clip.as_posix(),
    ]
    _run(cmd)


# ----------------------------
# Gemini upload & caption
# ----------------------------

PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "bgm_caption_system.md"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "bgm_caption_user.md"


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
    system_prompt: Optional[str],
    user_prompt: str,
    *,
    model: Optional[str],
    audio_path: Optional[Path] = None,
) -> str:
    text_chunks: List[str] = []
    if user_prompt:
        text_chunks.append(user_prompt)
    else:
        text_chunks.append("请描述该视频片段的背景音乐。")

    return client.generate(
        system_prompt=system_prompt,
        text_chunks=text_chunks,
        video_path=media_path,
        audio_path=audio_path,
        model=model,
        temperature=0.3,
        top_p=0.9,
    )


# ----------------------------
# Orchestration per segment
# ----------------------------

@dataclass
class BgmJob:
    seg_dir: Path
    video: Path
    bgms: List[Path]
    out_dir: Path  # where to write clips and captions


def plan_jobs(input_path: Path, out_root: Optional[Path]) -> List[BgmJob]:
    segs = find_segment_dirs(input_path)
    jobs: List[BgmJob] = []
    for seg in segs:
        v = resolve_segment_video(seg)
        if v is None:
            print(f"[跳过] 未找到视频: {seg}")
            continue
        bg = find_bgm_chunks(seg)
        if not bg:
            print(f"[跳过] 未找到 BGM 切片: {seg}/audio_sep/background_chunks")
            continue
        if out_root is None:
            out_dir = seg / "bgm_chunks"
        else:
            out_dir = Path(out_root) / seg.name / "bgm_chunks"
        jobs.append(BgmJob(seg_dir=seg, video=v, bgms=bg, out_dir=out_dir))
    # 单文件路径（某个 clip）不支持；明确要求输入为段目录或根目录
    return jobs


def process_segment(
    job: BgmJob,
    client,
    model: Optional[str],
    system_prompt: Optional[str],
    user_prompt: str,
) -> Tuple[List[Path], List[str]]:
    """生成该 segment 的每个 bgm+video clip，并逐个调用 Gemini 生成 caption。
    返回 (clips, captions)。
    """
    # 计算 segment 总时长与每个 bgm 切片时长
    try:
        seg_total = probe_duration_sec(job.video)
    except Exception as e:
        print(f"  -> 获取视频时长失败: {job.video} -> {e}", file=sys.stderr)
        return ([], [])

    bg_durs: List[float] = []
    for bg in job.bgms:
        try:
            bg_durs.append(probe_duration_sec(bg))
        except Exception as e:
            print(f"  -> 获取 BGM 时长失败: {bg} -> {e}", file=sys.stderr)
            return ([], [])

    # 逐个输出 clip
    tmp_dir = prepare_temp_dir(job.seg_dir)
    try:
        staged_bgms: Dict[Path, Path] = {}
        clips: List[Path] = []
        captions: List[str] = []
        t0 = 0.0
        for idx, (bg_path, dur) in enumerate(zip(job.bgms, bg_durs)):
            if t0 >= seg_total - 1e-3:
                break  # 已到达或超过 segment 尾部
            # clamp dur 使得不超 segment_end
            max_dur = max(0.0, seg_total - t0)
            use_dur = min(dur, max_dur)
            if use_dur <= 1e-3:
                break

            out_clip = job.out_dir / f"clip_{idx:03d}.mp4"
            temp_clip = tmp_dir / f"bgm_clip_{idx:03d}.mp4"
            staged_bg = bg_path
            try:
                staged_bg = staged_bgms.get(bg_path) or stage_file(
                    bg_path,
                    tmp_dir,
                    name=f"{bg_path.stem}_{idx:03d}{bg_path.suffix}",
                )
                staged_bgms[bg_path] = staged_bg
            except Exception as exc:
                print(f"  -> 临时目录写入 BGM 音频失败，改为直接使用原路径: {exc}", file=sys.stderr)
                staged_bg = bg_path
            try:
                mux_video_with_bgm(
                    base_video=job.video,
                    out_clip=temp_clip,
                    t0=t0,
                    dur=use_dur,
                    bgm_audio=staged_bg,
                )
                out_clip.parent.mkdir(parents=True, exist_ok=True)
                if out_clip.resolve() != temp_clip.resolve():
                    shutil.copy2(temp_clip, out_clip)
                clips.append(out_clip)
            except Exception as e:
                print(f"  -> 生成 clip 失败: {out_clip} -> {e}", file=sys.stderr)
                t0 += dur
                continue

            # 调用 Gemini 生成该 clip 的 BGM caption
            try:
                text = upload_and_caption(
                    client=client,
                    media_path=temp_clip,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    audio_path=staged_bg,
                )
                captions.append(text)
                # 写入每个 clip 的 caption 单文件（便于调试）
                per_txt = out_clip.with_suffix(".txt")
                try:
                    per_txt.write_text(text, encoding="utf-8")
                except Exception:
                    pass
            except QwenClientError as e:
                print(f"  -> Caption 失败: {out_clip} -> {e}", file=sys.stderr)
                captions.append("")

            t0 += dur

        return (clips, captions)
    finally:
        cleanup_temp_dir(tmp_dir)


def write_concat_caption(seg_dir: Path, out_root: Optional[Path], seg_name: str, captions: List[str]) -> Optional[Path]:
    if not captions:
        return None
    if out_root is None:
        out_path = seg_dir / "bgm_captions.txt"
    else:
        out_path = Path(out_root) / seg_name / "bgm_captions.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled: List[str] = []
    for idx, text in enumerate(captions, 1):
        cleaned = text.strip()
        if not cleaned:
            continue
        heading = f"Segment {idx}"
        labeled.append(f"{heading}\n{cleaned}")
    if not labeled:
        return None
    body = "\n\n".join(labeled)
    out_path.write_text(body, encoding="utf-8")
    return out_path


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-segment BGM captioning using Gemini 2.5 Pro")
    p.add_argument("--input", required=True, help="Path to a segment dir or segments root")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt text or @file (default: prompts/bgm_caption_system.md)",
    )
    p.add_argument(
        "--user",
        default=None,
        help="User prompt text or @file (default: prompts/bgm_caption_user.md)",
    )
    p.add_argument(
        "--model",
        default=QWEN_DEFAULT_MODEL,
        help=f"Qwen model identifier (default: {QWEN_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="If set and input is a directory, write outputs under <out-dir>/<segment>/...",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only, do not generate or upload")
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
        print("未找到可处理的 segment（需要 video.mp4 与 audio_sep/background_chunks）。", file=sys.stderr)
        return 1

    if args.dry_run:
        print("将要处理以下 segments：")
        for j in jobs:
            print(f"- {j.seg_dir} -> {len(j.bgms)} bgm chunks, out: {j.out_dir}")
        return 0

    try:
        client = default_client()
    except Exception as exc:
        print(f"初始化 Qwen 客户端失败: {exc}", file=sys.stderr)
        return 2

    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] 处理 segment: {job.seg_dir}")
        try:
            clips, captions = process_segment(
                job,
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
            )
        except Exception as e:
            print(f"  -> 失败: {e}", file=sys.stderr)
            continue

        out_txt = write_concat_caption(job.seg_dir, out_root, job.seg_dir.name, captions)
        if out_txt is not None:
            print(f"  -> 已写入合并 caption: {out_txt}")
        else:
            print("  -> 无可写入的 caption（全部失败或为空）")

    return 0


if __name__ == "__main__":
    sys.exit(main())

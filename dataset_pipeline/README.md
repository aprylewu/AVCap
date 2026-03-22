# AVCaption (code)

This folder contains the end-to-end audio-visual captioning pipeline scripts.
The repo has been cleaned to avoid hard-coded absolute paths; use relative paths
or environment variables instead.

## Pipeline overview (typical order)
1. `video_segmenter.py` — split long video into per-segment video/audio.
2. `audio_preprocess.py` — separate vocals/background and chunk BGM.
3. `bgm_caption.py` — caption background music clips.
4. `vocal_caption.py` — ASR + vocal captioning.
5. `visual_captioning.py` — caption silent visual segments.
6. `joint_caption.py` — fuse expert captions per segment.
7. `temporal_integration.py` — integrate per-segment captions.
8. `visual_filtering.py` / `audio_filtering.py` / `joint_filtering.py` — review scores.
9. `main.py` — one-click orchestration.

## Qwen/vLLM settings (recommended)
Set these in `code/.env` or your shell:
- `QWEN_BASE_URL` (default: `http://127.0.0.1:8901/v1`)
- `QWEN_MODEL` (default: `Qwen3-Omni-30B-A3B-Thinking`)

`TORCH_HOME` is now relative (`.cache/torch`) to avoid machine-specific paths.
Adjust as needed for your environment.

## Dataset path notes
- `extract_bbc_news_video_paths.py` defaults to `dataset/selected_100k/bbc_news` and
  writes `dataset/avcaption/bbc_news_video_paths.json` with **relative** video paths.
- You can override with `BBC_NEWS_SOURCE_DIR` / `BBC_NEWS_OUTPUT_PATH`.

## Environment export
A sanitized conda export for the `qwen` environment is stored at:
- `code/qwen_env.yml` (prefix removed to avoid absolute paths)

## Quick example
```bash
python code/main.py --video path/to/video.mp4
```

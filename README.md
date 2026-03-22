# AVCap

AVCap is an audio-visual captioning project focused on long-form video understanding, fine-grained multi-stage caption generation, and evaluation tooling for audio-visual reasoning. This repository is the public code release for the AVCap pipeline and evaluation scripts. Model weights and dataset assets are released separately on Hugging Face.

## Release Links

- Model: [Apryle/AVCap-30B-RL](https://huggingface.co/Apryle/AVCap-30B-RL)
- Dataset: [Apryle/AVCap-Dataset](https://huggingface.co/datasets/Apryle/AVCap-Dataset)
- Base inference environment reference: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- Qwen3-Omni Transformers documentation: [qwen3_omni_moe](https://huggingface.co/docs/transformers/en/model_doc/qwen3_omni_moe)

The AVCap model and dataset releases are currently distributed as gated Hugging Face repositories.

## What This Repository Contains

This code release is organized around two main components:

1. `dataset_pipeline/`
   End-to-end long-video captioning utilities, including segmentation, audio preprocessing, per-modality caption generation, multimodal caption fusion, temporal integration, and filtering-based review.

2. `evaluation/`
   Evaluation scripts and prompt templates for AVCap Score and additional benchmark-style evaluation packages.

The repository is intentionally code-centric. Large raw video assets are not included in this GitHub release.

## Project Overview

AVCap follows a staged audio-visual captioning workflow rather than a single monolithic inference pass. In broad terms, the pipeline:

1. Splits a long video into manageable segments.
2. Separates audio streams and prepares vocal and background-music views.
3. Generates modality-specific captions for visual content, vocals, and background music.
4. Fuses these expert views into a higher-granularity joint caption for each segment.
5. Integrates segment-level captions into a full-video caption.
6. Applies review and filtering stages to check visual, audio, and joint consistency.

This design makes the pipeline easier to inspect, adapt, and evaluate for research or production experiments involving long-form multimodal content.

## Repository Structure

```text
AVCap/
├── README.md
├── dataset_pipeline/
│   ├── main.py
│   ├── video_segmenter.py
│   ├── audio_preprocess.py
│   ├── visual_captioning.py
│   ├── bgm_caption.py
│   ├── vocal_caption.py
│   ├── joint_caption.py
│   ├── temporal_integration.py
│   ├── visual_filtering.py
│   ├── audio_filtering.py
│   ├── joint_filtering.py
│   ├── qwen_client.py
│   └── prompts/
└── evaluation/
    ├── avcap_score/
    └── exsisting/
```

Notes:

- `evaluation/exsisting/` keeps the original internal directory naming for compatibility with the provided scripts.
- `dataset_pipeline/qwen_env.yml` is a sanitized environment export included as a reference snapshot.

## Environment and Inference Setup

For model serving, dependencies, and runtime details, please follow the official setup guidance from [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct). This repository assumes an OpenAI-compatible multimodal endpoint and was written to work with a locally served Qwen3-Omni-family model.

At minimum, the full pipeline expects:

- Python environment with the dependencies required by the scripts
- A running OpenAI-compatible multimodal server for Qwen3-Omni-family inference
- `ffmpeg` and `ffprobe`
- GPU resources appropriate for your served model

Some data-processing stages additionally rely on external services or credentials mentioned in the scripts, including:

- LALAL.AI for source separation-related workflow components
- AssemblyAI for parts of the ASR-oriented workflow

The code uses the following environment variables for the local inference endpoint:

```bash
export QWEN_BASE_URL=http://127.0.0.1:8901/v1
export QWEN_MODEL=Apryle/AVCap-30B-RL
```

If you prefer to evaluate against another compatible checkpoint, simply point `QWEN_MODEL` to that served model instead.

## Quick Start

The main long-video pipeline entry point is:

```bash
python dataset_pipeline/main.py --video /path/to/video.mp4
```

Key command-line options exposed by `dataset_pipeline/main.py` include:

- `--video`: path to the input long video
- `--segment-length`: segment duration in seconds
- `--min-last`: minimum duration for the final segment
- `--model`: model name used by the multimodal inference backend
- `--out-dir`: optional override for the segment output directory
- `--output-json`: optional override for the final caption JSON

The scripts in `dataset_pipeline/` can also be run independently when you want to inspect or replace a specific stage.

## Dataset Pipeline

The `dataset_pipeline/` directory implements the staged captioning workflow:

- `video_segmenter.py`
  Splits a long video into segment-level media files.

- `audio_preprocess.py`
  Prepares audio branches for downstream captioning, including separation-oriented processing.

- `visual_captioning.py`
  Generates captions from visual-only evidence.

- `bgm_caption.py`
  Captions background music and non-speech audio cues.

- `vocal_caption.py`
  Handles vocal/audio-language-focused captioning and ASR-related processing.

- `joint_caption.py`
  Merges modality-specific evidence into a joint segment caption.

- `temporal_integration.py`
  Integrates multiple segment captions into a coherent full-video description.

- `visual_filtering.py`, `audio_filtering.py`, `joint_filtering.py`
  Review the generated caption against visual, audio, or joint evidence.

- `main.py`
  Orchestrates the end-to-end workflow in a single command.

Prompt templates used by the pipeline are stored in `dataset_pipeline/prompts/`.

## Evaluation

### AVCap Score

`evaluation/avcap_score/` provides the AVCap Score evaluation pipeline, including:

- caption generation scripts
- QA-answer generation
- grading prompts
- merged testset construction
- pipeline orchestration
- summary utilities

The main files include:

- `run_evaluation.py`
- `evaluate_qa.sh`
- `synthesize_dataset.py`
- `build_testset_oneclick.py`
- `step1_generate_captions*.py`
- `step2_generate_qa_answers.py`
- `step3_grade_answers.py`

For details, see:

- [`evaluation/avcap_score/README.md`](evaluation/avcap_score/README.md)
- [`evaluation/avcap_score/README_private_dataset.md`](evaluation/avcap_score/README_private_dataset.md)

### Additional Evaluation Packages

`evaluation/exsisting/` includes benchmark-style evaluation packages for:

- `UGC-VideoCap`
- `video-SALMONN2-testset`

These folders contain scripts, annotations, and evaluation wrappers, but not the large raw benchmark video folders that were present in the original Hugging Face repository layout.

## Open-Source Notes

This GitHub repository intentionally excludes large binary evaluation assets, especially raw benchmark videos. In particular, the following directories are not included here:

- `evaluation/avcap_score/data/`
- `evaluation/exsisting/UGC-VideoCap/video/`
- `evaluation/exsisting/video-SALMONN2-testset/video/`

This keeps the code release lightweight and practical for browsing, cloning, and development. For model artifacts and dataset access, please use the dedicated Hugging Face repositories:

- [Apryle/AVCap-30B-RL](https://huggingface.co/Apryle/AVCap-30B-RL)
- [Apryle/AVCap-Dataset](https://huggingface.co/datasets/Apryle/AVCap-Dataset)

## Practical Notes

- This repository does not package model weights.
- This repository does not include the full benchmark video assets.
- Some evaluation scripts support online evaluation flows that require an `apikey.txt` file in the corresponding evaluation directory.
- Paths and environment references were cleaned for public release so the code is easier to adapt to other machines.

## Acknowledgement

AVCap builds on the Qwen3-Omni ecosystem for multimodal inference and serving. Please refer to the official Qwen release for model-specific deployment instructions, hardware requirements, and backend compatibility details:

- [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)

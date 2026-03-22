# Private Dataset Full-Pipeline Guide (Video + GT Caption)

This guide explains how to run the **full AVCap QA evaluation pipeline** on a private dataset
that only has **videos + GT captions** (no QA yet). It covers:
- how to prepare the data
- how to generate QA (questions + answers)
- how to run the full evaluation pipeline

All commands below are relative to:
`/data/yuexiangyu-folder/xiangyu1/workspace/evaluation/avcap_score`

---

## 0) Prerequisites

- A machine with GPUs and a working vLLM environment.
- A text model for question/answer generation (e.g. Qwen2.5-Instruct).
- An Omni (audio-visual) model for Step 1 captioning.
- This repo checked out locally.

Optional but recommended:
- Use a dedicated conda env (same as main pipeline).
- Set cache dirs (HF / ModelScope) if you need local storage.

---

## 1) Prepare Your Private Dataset (Video + GT Caption)

Create a **base file** in JSONL or JSON with at least:
- `video_path`: absolute path to the video file
- `ground_truth_caption`: GT caption text

### Minimal JSONL format
```json
{"video_id":"video_0001","video_path":"/abs/path/to/video_0001.mp4","ground_truth_caption":"..."}
{"video_id":"video_0002","video_path":"/abs/path/to/video_0002.mp4","ground_truth_caption":"..."}
```

Notes:
- Use **absolute paths** so the pipeline can find videos.
- `video_id` is optional but recommended.

### Base dataset schema (what the QA generator accepts)
Your "base" file can be **JSONL** or **JSON (list)**. Each item must provide:
- a **video path**, and
- a **ground-truth caption** (GT caption).

The scripts accept multiple field names, so you can keep your existing structure:

#### Required (at least one of each group)
- **Video path**: `video_path` OR `videos[0]` OR `video`
- **GT caption**: `ground_truth_caption` OR `caption` OR `qa_caption` OR `messages` (assistant content)

#### Optional
- `video_id` (or `id`), otherwise we fall back to the filename stem.
- `system_prompt`, `user_prompt` (if absent, defaults are filled in).

#### Example JSON (list)
```json
[
  {
    "video_id": "video_0001",
    "video_path": "/abs/path/to/video_0001.mp4",
    "ground_truth_caption": "..."
  }
]
```

#### Example JSONL (line-delimited)
```json
{"video_id":"video_0001","video_path":"/abs/path/to/video_0001.mp4","ground_truth_caption":"..."}
{"video_id":"video_0002","video_path":"/abs/path/to/video_0002.mp4","ground_truth_caption":"..."}
```

#### Example (messages-style)
```json
{
  "video": "/abs/path/to/video_0003.mp4",
  "messages": [
    {"role": "system", "content": "You are ..."},
    {"role": "user", "content": "<video> ..."},
    {"role": "assistant", "content": "GT caption here"}
  ]
}
```

Notes:
- **Option A (one-click)** supports JSON or JSONL base files.
- **Option B (two-step)**: `generate_questions_from_gt.py` expects **JSON list** input (not JSONL).
---

## 2) Generate QA from GT (Questions + Answers)

We provide two options:

### Option A) One-click (recommended)
This generates **20 questions** (1 request per video),
then **20 answers** (20 requests per video),
and merges everything into `data/testset.json`.
We recommend using Qwen3-30B-A3B-Instruct as the text model

```bash
python build_testset_oneclick.py \
  --base /path/to/your_base.jsonl \
  --output data/testset.json \
  --model_path /path/to/text_model
```


### Option B) Two-step (questions then answers)
If you want to inspect/edit questions before answering:

1) Generate questions:
```bash
python generate_questions_from_gt.py \
  --input /path/to/your_base.jsonl \
  --output data/answers/questions_generated.json \
  --model_path /path/to/text_model
```

2) Generate answers (uses `answer_generation_prompt.txt`):
```bash
python generate_answers_from_gt.py \
  --input data/answers/questions_generated.json \
  --output data/answers/qa_generated.json \
  --model_path /path/to/text_model
```

3) Merge into unified testset (same format as pipeline expects):
```bash
python merge_qa_to_testset.py \
  --base /path/to/your_base.jsonl \
  --qa data/answers/qa_generated.json \
  --output data/testset.json
```

---

## 3) Wire the Pipeline to Your testset.json

The pipeline reads:
`data/testset.json` by default.

---

## synthesize_dataset.py vs build_testset_oneclick.py

**synthesize_dataset.py**
- Use when you already have **test set JSON/JSONL** + **QA annotations**.
- It merges them into a unified `data/testset.json`.
- No model inference happens in this step.

**build_testset_oneclick.py**
- Use when you only have **videos + GT captions** (no QA yet).
- It runs LLM inference to **generate 20 questions** and **20 answers**, then merges them.
- Output is a ready-to-use `data/testset.json`.

## 4) Run the Full Evaluation Pipeline

```bash
bash evaluate_qa.sh \
  --baseline my_private_set \
  --omni-model /path/to/omni_model \
  --text-model /path/to/text_model
```

Outputs go to:
- `outputs/step1_predictions_<baseline>.json`
- `outputs/step2_qa_answers_<baseline>.json`
- `outputs/step3_grades_<baseline>.json`

---

## 5) Output Schema (What testset.json Looks Like)

Each item in `data/testset.json` will be:
```json
{
  "video_id": "video_0001",
  "video_path": "/abs/path/to/video_0001.mp4",
  "system_prompt": "...",
  "user_prompt": "<video> ...",
  "ground_truth_caption": "...",
  "qa_caption": "...",
  "questions": {"question_01": "..."},
  "ground_truth_answers": {"answer_01": "..."}
}
```

`qa_caption` is the same as `ground_truth_caption` by default.

---

## 6) Troubleshooting

- **Video not found**: use absolute paths in `video_path`.
- **OOM**: reduce `--tensor_parallel_size` or `--gpu_memory_utilization`.

---

## 7) Question Quality Rules (Reminder)

Your QA quality matters. We follow:
1) Deterministic answers
2) Inferable from caption
3) Audiovisual granularity

Question distribution (20 total):
- Visual Details: 5
- Audio Details: 5
- Audio-Visual Joint: 10

---

## synthesize_dataset.py vs build_testset_oneclick.py vs merge_qa_to_testset.py

**synthesize_dataset.py**
- Use when you already have **test set JSON/JSONL** + **QA annotations** in a separate folder.
- It merges them into a unified `data/testset.json`.
- No model inference happens in this step.

**build_testset_oneclick.py**
- Use when you only have **videos + GT captions** (no QA yet).
- It runs LLM inference to **generate 20 questions** and **20 answers**, then merges them.
- Output is a ready-to-use `data/testset.json`.

**merge_qa_to_testset.py**
- Use when you already generated **questions + answers** (two-step flow).
- It merges base data + QA JSON into `data/testset.json` without running any model.

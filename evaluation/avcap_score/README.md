# AVCap Score QA Evaluation (Quick Guide)

## What’s Included
- `answer_generation_prompt.txt` / `grading_prompt.txt`: prompt templates
- `synthesize_dataset.py`: build the unified QA evaluation set
- `step1_generate_captions*.py`: caption generation (vLLM / Gemini / Qwen2-VL)
- `step2_generate_qa_answers.py`: QA answering
- `step3_grade_answers.py`: grading
- `run_evaluation.py`: pipeline orchestrator
- `evaluate_qa.sh`: convenience wrapper
- `summarize_all_grades.py`: quick summary table

## Environment
Follow the **Qwen3Omni** repository setup instructions (vLLM + dependencies).
Make sure the runtime environment can load your Omni model and text model (GPU + CUDA + vLLM).

## Quick Start (Bundled Testset)
The repo already ships a unified test set in `data/testset.json`, and the pipeline
reads this file by default.

Run the full pipeline:
```bash
bash evaluate_qa.sh \
  --baseline my_model \
  --omni-model /path/to/omni_model \
  --text-model /path/to/text_model
```

## Data Layout
- Unified test set JSON: `data/testset.json`
- Videos (hashed filenames): `data/videos/*.mp4`

### Dataset Schema (data/testset.json)
Each item is a dict with:
- `video_id` (str): sample ID
- `video_path` (str): path to video file (relative to this folder by default)
- `system_prompt` (str): system message for captioning
- `user_prompt` (str): user instruction (usually contains `<video>`)
- `ground_truth_caption` (str): reference caption from the test set
- `qa_caption` (str): caption from the QA annotation file
- `questions` (dict): QA questions (`question_01` ~ `question_20`)
- `ground_truth_answers` (dict): QA answers (`answer_01` ~ `answer_20`)

Notes:
- For the bundled `data/testset.json`, **`qa_caption` == `ground_truth_caption` for all 1000 samples**.
- `video_path` is relative; run from this directory or convert to absolute paths.

## Step 0: (Optional) Synthesize the QA Dataset
If you have your own test set and ground-truth Q&A files, synthesize a unified dataset:
```bash
python synthesize_dataset.py \
  --test-set data/testset.json \
  --answers-dir /path/to/answers \
  --output-path data/testset.json
```

## Use Your Own Video Series / Dataset (Recommended Workflow)
If you want to evaluate a different dataset, follow these steps:

### 1) Prepare your **test set** (JSON or JSONL)
Each test item must contain:
- `id`: unique sample ID
- `videos`: list of video paths (first one will be used)
- `messages`: list of chat messages that include:
  - `role: "system"` with the system prompt (optional)
  - `role: "user"` with the user prompt (usually contains `<video>`)
  - `role: "assistant"` with the reference caption (optional)

Example JSONL line:
```json
{"id":"video_0001","videos":["/abs/path/to/video_0001.mp4"],"messages":[{"role":"system","content":"You are an audio-visual captioner."},{"role":"user","content":"<video> Please describe the clip."},{"role":"assistant","content":"(optional) reference caption"}]}
```

### 2) Prepare your **QA annotations** directory
Create a folder with multiple `*.json` files, each being a list of items:
```json
[
  {
    "video_path": "/abs/path/to/video_0001.mp4",
    "caption": "reference caption used for QA",
    "questions": {"question_01": "..."},
    "answers": {"answer_01": "..."}
  }
]
```

### 3) Synthesize unified evaluation data
```bash
python synthesize_dataset.py \
  --test-set /path/to/your_test.jsonl \
  --answers-dir /path/to/your_answers_dir \
  --output-path data/testset.json
```

### 4) Run evaluation
```bash
bash evaluate_qa.sh \
  --baseline my_dataset \
  --omni-model /path/to/omni_model \
  --text-model /path/to/text_model
```

### Notes / Gotchas
- `video_path` is validated with `os.path.exists`; use absolute paths if running from another directory.
- If you only want **captioning** (no QA), run Step 1 only:
  ```bash
  bash evaluate_qa.sh --baseline my_dataset --steps 1 --omni-model /path/to/omni_model
  ```

## If You Only Have Video + GT Caption (No QA Yet)
You must **create 20 QA probes** per video and their answers before running Steps 2/3.

### Question Generation Principles
The validity of this pipeline relies on the quality of the probes. We adhere to three principles:
1) **Deterministic**: the answer is unique, not ambiguous.  
2) **Inferable from context**: the answer is directly supported by the caption.  
3) **Audiovisual granularity**: questions focus on fine-grained visual/audio details.  

The 20 questions are distributed across three categories:
- **Visual Details (Qv, 5 questions)**: object colors, camera movements, OCR text, spatial relationships.  
- **Audio Details (Qa, 5 questions)**: timbre, pitch, specific instruments, background noise, speaker identity.  
- **Audio-Visual Joint (Qav, 10 questions)**: temporal sync, causality, source grounding.  

### Recommended Workflow
1) Start from your **GT caption** and generate 20 questions following the principles above.  
2) Answer each question **only using the GT caption**.  
3) Save QA annotations in `data/answers/*.json` format (see below).  
4) Run `synthesize_dataset.py` to build `data/testset.json`.  

### QA Annotation JSON Format
Create one or more JSON files under `data/answers/`:
```json
[
  {
    "video_path": "/abs/path/to/video_0001.mp4",
    "caption": "GT caption used for QA",
    "questions": {
      "question_01": "...",
      "question_02": "...",
      "question_03": "...",
      "question_04": "...",
      "question_05": "...",
      "question_06": "...",
      "question_07": "...",
      "question_08": "...",
      "question_09": "...",
      "question_10": "...",
      "question_11": "...",
      "question_12": "...",
      "question_13": "...",
      "question_14": "...",
      "question_15": "...",
      "question_16": "...",
      "question_17": "...",
      "question_18": "...",
      "question_19": "...",
      "question_20": "..."
    },
    "answers": {
      "answer_01": "...",
      "answer_02": "...",
      "answer_03": "...",
      "answer_04": "...",
      "answer_05": "...",
      "answer_06": "...",
      "answer_07": "...",
      "answer_08": "...",
      "answer_09": "...",
      "answer_10": "...",
      "answer_11": "...",
      "answer_12": "...",
      "answer_13": "...",
      "answer_14": "...",
      "answer_15": "...",
      "answer_16": "...",
      "answer_17": "...",
      "answer_18": "...",
      "answer_19": "...",
      "answer_20": "..."
    }
  }
]
```

### Built-in Scripts (Questions + Answers from GT)
We include two ready-to-run scripts with embedded prompts:

1) **Generate questions** from GT captions:
```bash
python generate_questions_from_gt.py \
  --input data/testset.json \
  --output data/answers/questions_generated.json \
  --model_path /path/to/text_model
```

2) **Generate answers** from GT captions + questions:
```bash
python generate_answers_from_gt.py \
  --input data/answers/questions_generated.json \
  --output data/answers/qa_generated.json \
  --model_path /path/to/text_model
```

Answer generation uses the unified prompt in:
`answer_generation_prompt.txt`

These scripts output JSON compatible with `synthesize_dataset.py`.

### One-Click: Build data/testset.json (Questions + Answers + Merge)
If you only have video + GT captions, you can generate QA and merge in one run:
```bash
python build_testset_oneclick.py \
  --base /path/to/your_base.jsonl \
  --output data/testset.json \
  --model_path /path/to/text_model
```

Notes:
- Question generation is **1 request per video** (20 questions returned together).
- Answer generation is **20 requests per video** (one per question), per evaluation protocol.
- Answers are generated with `answer_generation_prompt.txt`.

### synthesize_dataset.py vs build_testset_oneclick.py vs merge_qa_to_testset.py
- **synthesize_dataset.py**: merge an existing test set + QA annotations into `data/testset.json`. No model inference.
- **build_testset_oneclick.py**: generate questions/answers from GT captions, then output `data/testset.json`. Uses LLM inference.
- **merge_qa_to_testset.py**: merge base data + pre-generated QA JSON into `data/testset.json`. No model inference.

### Final Unified Testset (data/testset.json)
After `synthesize_dataset.py`, each sample in the unified JSON looks like:
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
If you prefer, you can directly build `data/testset.json` in this exact format and skip
`synthesize_dataset.py`.

## Step 1–3: Run the Pipeline
```bash
bash evaluate_qa.sh \
  --baseline my_model \
  --omni-model models/your_omni_model \
  --text-model models/your_text_model
```

Or run directly:
```bash
python run_evaluation.py \
  --steps all \
  --baseline my_model \
  --omni_model_path models/your_omni_model \
  --text_model_path models/your_text_model
```

## Optional: Gemini Captioning (Step 1)
```bash
python step1_generate_captions_gemini.py \
  --input_data data/testset.json \
  --output_path outputs/step1_predictions_my_model.json \
  --api-key YOUR_GEMINI_KEY
```

## Outputs
All outputs are written to `outputs/`, for example:
- `outputs/step1_predictions_<baseline>.json`
- `outputs/step2_qa_answers_<baseline>.json`
- `outputs/step3_grades_<baseline>.json`

## Notes
- API keys are **not** included. Put `apikey.txt` in this folder or pass `--api-key` for Gemini.
- Paths are cleaned to be **relative** and easy to relocate.
- `evaluate_qa.sh` runs from this directory, so relative `video_path` works out of the box.

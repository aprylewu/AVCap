#!/usr/bin/env python3
"""
Generate 20 QA questions from GT captions using a text LLM (vLLM).
Output is compatible with data/answers/*.json format (without answers).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a dataset curation assistant. Generate QA probes from captions.\n"
    "Principles:\n"
    "1) Deterministic: the answer must be unique and unambiguous.\n"
    "2) Inferable from context: the answer must be directly supported by the caption.\n"
    "3) Audiovisual granularity: focus on fine-grained visual/audio details.\n"
    "\n"
    "Question distribution (exactly 20):\n"
    "- Visual Details (Qv, 5): object colors, camera movements, OCR text, spatial relations.\n"
    "- Audio Details (Qa, 5): timbre, pitch, instruments, background noise, speaker identity.\n"
    "- Audio-Visual Joint (Qav, 10): temporal sync, causality, source grounding.\n"
    "\n"
    "Output format:\n"
    "Return ONLY a valid JSON object with keys question_01 ... question_20.\n"
    "Keep each question concise and answerable from the caption.\n"
    "Use the same language as the caption."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA questions from GT captions")
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--input",
        type=str,
        default=str(base_dir / "data" / "testset.json"),
        help="Input JSON list containing GT captions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "data" / "answers" / "questions_generated.json"),
        help="Output JSON list with generated questions",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-7B-Instruct",
        help="Text model path (vLLM)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max model length",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--caption_field",
        type=str,
        default="ground_truth_caption",
        help="Caption field name to use",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip items with questions)",
    )
    return parser.parse_args()


def load_input(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def build_prompt(caption: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Caption:\n{caption}\n\n"
        "Task: Generate the 20 questions now. Return JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_questions(text: str) -> Dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    obj = json.loads(text[start : end + 1])
    # Basic validation
    expected = [f"question_{i:02d}" for i in range(1, 21)]
    for key in expected:
        if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"Missing or empty key: {key}")
    return obj


def get_caption(item: Dict, preferred_field: str) -> Tuple[str, str]:
    if preferred_field in item and isinstance(item[preferred_field], str):
        return preferred_field, item[preferred_field]
    for k in ("ground_truth_caption", "caption", "qa_caption"):
        if k in item and isinstance(item[k], str):
            return k, item[k]
    raise ValueError("No caption field found in item")


def main() -> None:
    args = parse_args()

    items = load_input(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
        for it in existing_list:
            key = it.get("video_path") or it.get("video_id")
            if key:
                existing[key] = it

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    results: List[Dict] = []
    for idx, item in enumerate(items, start=1):
        key = item.get("video_path") or item.get("video_id") or str(idx)
        if key in existing:
            results.append(existing[key])
            continue

        _, caption = get_caption(item, args.caption_field)
        prompt = build_prompt(caption)
        out = llm.generate([prompt], sampling)[0].outputs[0].text

        try:
            questions = parse_questions(out)
        except Exception as e:
            raise RuntimeError(f"Failed to parse questions for item {key}: {e}\nRaw output:\n{out}")

        result = {
            "video_path": item.get("video_path"),
            "caption": caption,
            "questions": questions,
        }
        if "video_id" in item:
            result["video_id"] = item["video_id"]
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} items to {output_path}")


if __name__ == "__main__":
    main()

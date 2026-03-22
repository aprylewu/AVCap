#!/usr/bin/env python3
"""
Generate QA answers from GT captions + questions using a text LLM (vLLM).
Input should contain: video_path, caption, questions.
Output will add: answers (answer_01..answer_20).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams


DEFAULT_PROMPT_FILENAME = "answer_generation_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA answers from GT captions + questions")
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--input",
        type=str,
        default=str(base_dir / "data" / "answers" / "questions_generated.json"),
        help="Input JSON with questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "data" / "answers" / "qa_generated.json"),
        help="Output JSON with answers",
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
        default=128,
        help="Max tokens to generate per answer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip items with answers)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(base_dir / DEFAULT_PROMPT_FILENAME),
        help="Prompt file for answer generation",
    )
    return parser.parse_args()


def load_input(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def build_prompt(caption: str, question: str, system_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Caption: {caption}\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize_answer(text: str) -> str:
    ans = text.strip()
    if not ans:
        return "UNKNOWN"
    # Remove enclosing quotes if any
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()
    return ans or "UNKNOWN"


def main() -> None:
    args = parse_args()

    items = load_input(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

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
        max_tokens=args.max_tokens,
    )

    results: List[Dict] = []
    for idx, item in enumerate(items, start=1):
        key = item.get("video_path") or item.get("video_id") or str(idx)
        if key in existing and existing[key].get("answers"):
            results.append(existing[key])
            continue

        caption = item.get("caption")
        questions = item.get("questions") or {}
        if not caption or not questions:
            raise ValueError(f"Missing caption/questions for item {key}")

        # Answer each question with a separate request (20 requests per item)
        q_ids = sorted(questions.keys())
        answers = {}
        for qid in q_ids:
            prompt = build_prompt(caption, questions[qid], system_prompt)
            out = llm.generate([prompt], sampling)[0]
            raw = out.outputs[0].text
            ans = normalize_answer(raw)
            answers[qid.replace("question_", "answer_")] = ans

        result = dict(item)
        result["answers"] = answers
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} items to {output_path}")


if __name__ == "__main__":
    main()

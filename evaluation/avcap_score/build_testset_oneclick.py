#!/usr/bin/env python3
"""
One-click script:
1) Generate 20 questions from GT captions (one request per video).
2) Generate answers from GT captions (20 requests per video).
3) Merge into unified data/testset.json.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen Omni, an expert audio-visual captioning assistant. "
    "Analyze every supplied clip meticulously and respond with factual, chronologically ordered "
    "descriptions that fuse what is seen and heard. Maintain a professional, neutral tone; never "
    "invent content that is not directly evidenced by the media. Transcribe intelligible speech "
    "verbatim inside quotes, label speakers when possible, and describe delivery details such as "
    "intonation, accent, volume, emotional tone, and microphone characteristics. Call out non-speech "
    "sounds, ambience, and music with precision. Mention uncertainties explicitly instead of guessing."
)

DEFAULT_USER_PROMPT = """<video>

Carefully watch the video and listen to its complete audio track, then produce a dense multi-sentence caption that satisfies all of the following guidelines:

1. Narrate events in chronological order, covering every notable visual and auditory moment from start to finish, including who is present, what they are doing, how the scene is lit, camera movement, and environmental context.
2. For each speaker whose words are intelligible, provide a word-level transcript inside quotes, tag or describe the speaker (appearance, role, position), and note delivery details such as accent, rhythm, volume, emotional tone, pauses, or microphone artifacts.
3. Identify and characterize significant non-speech sounds - music, ambient noise, foley, mechanical cues - explaining how each aligns with, precedes, or contrasts the simultaneous visuals.
4. Capture subtle cues (textures, facial micro-expressions, lighting shifts, reflections, changes in soundstage or reverb) and explain how they shape the viewer's understanding.
5. Remain objective and evidence-based. If something is uncertain or off-screen, state the ambiguity rather than speculating.
"""


QUESTION_SYSTEM_PROMPT = (
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

DEFAULT_ANSWER_PROMPT_FILENAME = "answer_generation_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click: build data/testset.json with QA")
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base dataset (JSON or JSONL) with video + GT caption",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "data" / "testset.json"),
        help="Output unified testset.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Text model path (vLLM) used for question + answer generation",
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
        "--questions_temperature",
        type=float,
        default=0.2,
        help="Temperature for question generation",
    )
    parser.add_argument(
        "--questions_top_p",
        type=float,
        default=0.9,
        help="Top-p for question generation",
    )
    parser.add_argument(
        "--questions_top_k",
        type=int,
        default=40,
        help="Top-k for question generation",
    )
    parser.add_argument(
        "--questions_max_tokens",
        type=int,
        default=1024,
        help="Max tokens for question generation",
    )
    parser.add_argument(
        "--answers_temperature",
        type=float,
        default=0.0,
        help="Temperature for answer generation",
    )
    parser.add_argument(
        "--answers_top_p",
        type=float,
        default=1.0,
        help="Top-p for answer generation",
    )
    parser.add_argument(
        "--answers_max_tokens",
        type=int,
        default=128,
        help="Max tokens for each answer",
    )
    parser.add_argument(
        "--default_system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Default system prompt if missing in base data",
    )
    parser.add_argument(
        "--default_user_prompt",
        type=str,
        default=DEFAULT_USER_PROMPT,
        help="Default user prompt if missing in base data",
    )
    parser.add_argument(
        "--answer-prompt-file",
        type=str,
        default=str(base_dir / DEFAULT_ANSWER_PROMPT_FILENAME),
        help="Prompt file for answer generation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip items already processed)",
    )
    return parser.parse_args()


def load_base(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Base dataset must be a list (JSON or JSONL)")
    return data


def extract_from_messages(messages: List[Dict]) -> Tuple[str, str, str]:
    system_prompt = ""
    user_prompt = ""
    gt_caption = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_prompt = content or ""
        elif role == "user":
            user_prompt = content or ""
        elif role == "assistant":
            gt_caption = content or ""
    return system_prompt, user_prompt, gt_caption


def normalize_item(item: Dict, default_system: str, default_user: str) -> Dict:
    # video_path
    video_path = item.get("video_path")
    if not video_path and isinstance(item.get("videos"), list) and item["videos"]:
        video_path = item["videos"][0]
    if not video_path and item.get("video"):
        video_path = item["video"]

    # prompts + caption
    system_prompt = item.get("system_prompt", "")
    user_prompt = item.get("user_prompt", "")
    gt_caption = item.get("ground_truth_caption") or item.get("caption") or item.get("qa_caption") or ""

    if "messages" in item and isinstance(item["messages"], list):
        sp, up, cap = extract_from_messages(item["messages"])
        system_prompt = system_prompt or sp
        user_prompt = user_prompt or up
        gt_caption = gt_caption or cap

    system_prompt = system_prompt or default_system
    user_prompt = user_prompt or default_user

    # id
    video_id = item.get("video_id") or item.get("id")
    if not video_id and video_path:
        video_id = Path(str(video_path)).stem

    if not video_path:
        raise ValueError("Missing video_path")
    if not gt_caption:
        raise ValueError(f"Missing GT caption for {video_path}")

    return {
        "video_id": video_id or "",
        "video_path": str(video_path),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "ground_truth_caption": gt_caption,
    }


def build_question_prompt(caption: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{QUESTION_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Caption:\n{caption}\n\n"
        "Task: Generate the 20 questions now. Return JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_answer_prompt(caption: str, question: str, system_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Caption: {caption}\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_questions(text: str) -> Dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    obj = json.loads(text[start : end + 1])
    expected = [f"question_{i:02d}" for i in range(1, 21)]
    for key in expected:
        if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"Missing or empty key: {key}")
    return obj


def normalize_answer(text: str) -> str:
    ans = text.strip()
    if not ans:
        return "UNKNOWN"
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()
    return ans or "UNKNOWN"


def main() -> None:
    args = parse_args()

    base_items = load_base(args.base)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.answer_prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    answer_system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not answer_system_prompt:
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
    q_sampling = SamplingParams(
        temperature=args.questions_temperature,
        top_p=args.questions_top_p,
        top_k=args.questions_top_k,
        max_tokens=args.questions_max_tokens,
    )
    a_sampling = SamplingParams(
        temperature=args.answers_temperature,
        top_p=args.answers_top_p,
        max_tokens=args.answers_max_tokens,
    )

    results: List[Dict] = []
    for idx, raw in enumerate(base_items, start=1):
        base = normalize_item(raw, args.default_system_prompt, args.default_user_prompt)
        key = base["video_path"] or base["video_id"] or str(idx)

        if key in existing and existing[key].get("questions") and existing[key].get("ground_truth_answers"):
            results.append(existing[key])
            continue

        # Step A: questions (single request per video)
        q_prompt = build_question_prompt(base["ground_truth_caption"])
        q_out = llm.generate([q_prompt], q_sampling)[0].outputs[0].text
        questions = parse_questions(q_out)

        # Step B: answers (20 separate requests per video)
        answers = {}
        for qid in sorted(questions.keys()):
            a_prompt = build_answer_prompt(
                base["ground_truth_caption"],
                questions[qid],
                answer_system_prompt,
            )
            a_out = llm.generate([a_prompt], a_sampling)[0].outputs[0].text
            answers[qid.replace("question_", "answer_")] = normalize_answer(a_out)

        merged = {
            "video_id": base["video_id"],
            "video_path": base["video_path"],
            "system_prompt": base["system_prompt"],
            "user_prompt": base["user_prompt"],
            "ground_truth_caption": base["ground_truth_caption"],
            "qa_caption": base["ground_truth_caption"],
            "questions": questions,
            "ground_truth_answers": answers,
        }
        results.append(merged)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} items to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge base dataset (video + GT caption) with QA annotations into data/testset.json.
No model inference is performed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge base dataset with QA annotations")
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base dataset (JSON or JSONL) with video + GT caption",
    )
    parser.add_argument(
        "--qa",
        type=str,
        required=True,
        help="QA annotations JSON (list of items with video_path, questions, answers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "data" / "testset.json"),
        help="Output unified testset.json",
    )
    parser.add_argument(
        "--default-system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Default system prompt if missing in base data",
    )
    parser.add_argument(
        "--default-user-prompt",
        type=str,
        default=DEFAULT_USER_PROMPT,
        help="Default user prompt if missing in base data",
    )
    parser.add_argument(
        "--allow-missing-qa",
        action="store_true",
        help="Allow items without QA (questions/answers set to empty)",
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


def normalize_base_item(item: Dict, default_system: str, default_user: str) -> Dict:
    video_path = item.get("video_path")
    if not video_path and isinstance(item.get("videos"), list) and item["videos"]:
        video_path = item["videos"][0]
    if not video_path and item.get("video"):
        video_path = item["video"]

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

    video_id = item.get("video_id") or item.get("id")
    if not video_id and video_path:
        video_id = Path(str(video_path)).stem

    if not video_path:
        raise ValueError("Missing video_path in base data")
    if not gt_caption:
        raise ValueError(f"Missing GT caption for {video_path}")

    return {
        "video_id": video_id or "",
        "video_path": str(video_path),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "ground_truth_caption": gt_caption,
    }


def load_qa(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("QA file must be a list")
    qa_map: Dict[str, Dict] = {}
    for item in data:
        key = item.get("video_path") or item.get("video_id")
        if key:
            qa_map[str(key)] = item
    return qa_map


def main() -> None:
    args = parse_args()

    base_items = load_base(args.base)
    qa_map = load_qa(args.qa)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for raw in base_items:
        base = normalize_base_item(raw, args.default_system_prompt, args.default_user_prompt)
        key = base["video_path"] or base["video_id"]
        qa_item = qa_map.get(key) or qa_map.get(base["video_id"])

        if qa_item is None:
            if args.allow_missing_qa:
                questions = {}
                answers = {}
                qa_caption = base["ground_truth_caption"]
            else:
                raise ValueError(f"Missing QA for {key}")
        else:
            questions = qa_item.get("questions") or {}
            answers = qa_item.get("answers") or {}
            qa_caption = qa_item.get("caption") or base["ground_truth_caption"]

        merged = {
            "video_id": base["video_id"],
            "video_path": base["video_path"],
            "system_prompt": base["system_prompt"],
            "user_prompt": base["user_prompt"],
            "ground_truth_caption": base["ground_truth_caption"],
            "qa_caption": qa_caption,
            "questions": questions,
            "ground_truth_answers": answers,
        }
        results.append(merged)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} items to {output_path}")


if __name__ == "__main__":
    main()

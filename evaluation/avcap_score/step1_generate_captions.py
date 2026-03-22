#!/usr/bin/env python3
"""
Step 1: Generate video captions using vLLM batch inference.
Uses Qwen3-Omni model to generate predicted captions for evaluation.
"""

import os
import json
import argparse
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force vLLM v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
FPS_MAX_FRAMES = 64
IMAGE_MAX_TOKEN_NUM = 1024
VIDEO_MAX_TOKEN_NUM = 512
SAMPLING_RATE = 16000
USE_AUDIO_IN_VIDEO_DEFAULT = True

DEFAULT_QWEN_OMNI_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)
DEFAULT_USER_PROMPT_TEMPLATES = [
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
]
FALLBACK_USER_PROMPT = (
    "Provide a comprehensive description of all the content in the video, "
    "leaving out no details. Be sure to include as much of the audio "
    "information as possible, and ensure that your descriptions of the "
    "audio and video are closely aligned."
)

# Keep Qwen3-Omni runtime knobs explicit in the script for release stability.
os.environ["VIDEO_MAX_PIXELS"] = str(VIDEO_TOTAL_PIXELS)
os.environ["VIDEO_TOTAL_PIXELS"] = str(VIDEO_TOTAL_PIXELS)
os.environ["FPS_MAX_FRAMES"] = str(FPS_MAX_FRAMES)
os.environ["IMAGE_MAX_TOKEN_NUM"] = str(IMAGE_MAX_TOKEN_NUM)
os.environ["VIDEO_MAX_TOKEN_NUM"] = str(VIDEO_MAX_TOKEN_NUM)
os.environ["SAMPLING_RATE"] = str(SAMPLING_RATE)
os.environ["USE_AUDIO_IN_VIDEO"] = "True"


def parse_reasoning_and_content(text: str, reasoning_parser: Optional[str]) -> Tuple[Optional[str], str]:
    """Split model output into (reasoning, content) and return content for downstream tasks.

    vLLM OpenAI server can expose `reasoning_content`; this pipeline uses the Python API, so
    we parse common formats (e.g. Qwen3 `<think>...</think>`).
    """
    if not text:
        return None, ""
    rp = (reasoning_parser or "").strip().lower()
    if not rp:
        return None, text.strip()

    if rp in {"qwen3", "qwen"}:
        # Common Qwen3 thinking format: <think> ... </think>\n<final content>
        m = re.search(r"<think>(.*?)</think>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip() or None
            content = (m.group(2) or "").strip()
            return reasoning, content
        # Fallback: sometimes uses markdown headings
        m2 = re.search(r"(?s)^(?:#\s*thoughts\s*|##\s*thoughts\s*|thoughts:\s*)(.*?)(?:\n\s*(?:final:|answer:|caption:)|$)(.*)$", text, flags=re.IGNORECASE)
        if m2:
            reasoning = (m2.group(1) or "").strip() or None
            content = (m2.group(2) or "").strip() or ""
            return reasoning, content
        return None, text.strip()

    raise ValueError(f"Unsupported reasoning_parser: {reasoning_parser}. Supported: qwen3")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video captions using vLLM")
    base_dir = Path(__file__).resolve().parent

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the model"
    )

    # Data paths
    parser.add_argument(
        "--input_data",
        type=str,
        default=str(base_dir / "data" / "testset.json"),
        help="Input synthesized test set JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(base_dir / "outputs" / "step1_predictions.json"),
        help="Output predictions JSON file"
    )

    # vLLM configuration
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.93,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=65536,
        help="Maximum model length"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=64,
        help="Maximum number of sequences"
    )

    # Sampling configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (>1 discourages repetitive loops)"
    )

    # Processing configuration
    parser.add_argument(
        "--use_audio_in_video",
        action="store_true",
        default=USE_AUDIO_IN_VIDEO_DEFAULT,
        help="Use audio in video"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for input preparation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions"
    )

    parser.add_argument(
        "--reasoning_parser",
        "--reasoning-parser",
        type=str,
        default="",
        help="Optional: parse/strip thinking content from model output (e.g., 'qwen3').",
    )

    parser.add_argument(
        "--save_reasoning",
        action="store_true",
        help="If set, save extracted reasoning into `reasoning_content` field in output JSON.",
    )

    return parser.parse_args()


def load_test_data(path: str) -> List[Dict[str, Any]]:
    """Load synthesized test data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


def load_existing_predictions(path: str) -> Dict[str, Dict]:
    """Load existing predictions to support resume."""
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    pred_map = {}
    for item in predictions:
        video_id = item['video_id']
        pred_map[video_id] = item

    print(f"Loaded {len(pred_map)} existing predictions from {path}")
    return pred_map


def save_predictions(predictions: List[Dict[str, Any]], path: str):
    """Save predictions to JSON file."""
    # Create output directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} predictions to {path}")


def build_input(processor, messages, use_audio_in_video):
    """Build vLLM input format"""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    return inputs


def normalize_user_prompt(user_prompt: Optional[str]) -> str:
    return (user_prompt or "").replace("<video>", "").strip()

def select_user_prompt(item: Dict[str, Any], prompt_templates: List[str]) -> str:
    if not prompt_templates:
        user_prompt = normalize_user_prompt(item.get("user_prompt", ""))
        return user_prompt or FALLBACK_USER_PROMPT
    stable_key = str(item.get("video_id") or item.get("video_path") or "")
    digest = hashlib.md5(stable_key.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(prompt_templates)
    return prompt_templates[index]


def process_single_sample(
    item: Dict[str, Any],
    processor,
    use_audio_in_video: bool,
    user_prompt_templates: List[str],
) -> Optional[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
    """
    Process a single sample and return (input_data, video_id) or None.
    This function will be called in parallel.
    """
    video_path = item['video_path']
    if isinstance(video_path, str) and not os.path.isabs(video_path):
        video_path = str((Path(__file__).resolve().parent / video_path).resolve())
    video_id = item['video_id']

    # Check video file
    if not os.path.exists(video_path):
        return None

    # Use canonical Qwen3-Omni system prompt for more stable behavior.
    messages = [{
        "role": "system",
        "content": DEFAULT_QWEN_OMNI_SYSTEM
    }]

    # Add user message with video.
    user_text = select_user_prompt(item, user_prompt_templates)

    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": VIDEO_MAX_PIXELS},
            {"type": "text", "text": user_text}
        ]
    })

    try:
        input_data = build_input(processor, messages, use_audio_in_video)
        return (input_data, video_id, item)
    except Exception as e:
        print(f"\nError processing {video_id}: {str(e)}")
        return None


def main():
    args = parse_args()

    print("=" * 80)
    print("Step 1: Generate Video Captions - vLLM Batch Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Tensor Parallel: {args.tensor_parallel_size}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Max Num Seqs: {args.max_num_seqs}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Use Audio in Video: {args.use_audio_in_video}")
    print(f"Video Max Pixels (per frame): {VIDEO_MAX_PIXELS}")
    print(f"Video Total Pixels: {VIDEO_TOTAL_PIXELS}")
    print(f"FPS Max Frames: {FPS_MAX_FRAMES}")
    print(f"Image Max Token Num: {IMAGE_MAX_TOKEN_NUM}")
    print(f"Video Max Token Num: {VIDEO_MAX_TOKEN_NUM}")
    print(f"Sampling Rate: {SAMPLING_RATE}")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Resume: {args.resume}")
    print(f"Reasoning Parser: {args.reasoning_parser or 'none'}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print(f"User Prompt Templates: built-in ({len(DEFAULT_USER_PROMPT_TEMPLATES)})")
    print("=" * 80)

    # Load test data
    test_data = load_test_data(args.input_data)
    user_prompt_templates = list(DEFAULT_USER_PROMPT_TEMPLATES)
    print("Prompt mode: fixed system + built-in multi user prompt templates")

    # Load existing predictions for resume support
    existing_predictions = {} if not args.resume else load_existing_predictions(args.output_path)

    # Separate completed and pending samples
    completed_predictions = []
    pending_samples = []

    for item in test_data:
        video_id = item['video_id']

        if video_id in existing_predictions:
            completed_predictions.append(existing_predictions[video_id])
        else:
            pending_samples.append(item)

    print(f"\nTotal samples: {len(test_data)}")
    print(f"Already completed: {len(completed_predictions)}")
    print(f"Pending: {len(pending_samples)}")

    if len(pending_samples) == 0:
        print("\nAll samples already processed!")
        return

    # Initialize vLLM
    print("\nInitializing vLLM engine...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={'image': 0, 'video': 1, 'audio': 1},
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    print("vLLM engine initialized successfully!")

    # Batch processing
    print(f"\nProcessing {len(pending_samples)} samples in batches...")
    print(f"Using {args.num_workers} parallel workers for input preparation")

    # Prepare all inputs - use parallel processing
    all_inputs = []
    all_video_ids = []
    all_items = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(
                process_single_sample,
                item,
                processor,
                args.use_audio_in_video,
                user_prompt_templates,
            ): item
            for item in pending_samples
        }

        # Use tqdm to show progress
        with tqdm(total=len(pending_samples), desc="Preparing inputs") as pbar:
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        input_data, video_id, original_item = result
                        all_inputs.append(input_data)
                        all_video_ids.append(video_id)
                        all_items.append(original_item)
                except Exception as e:
                    video_id = item['video_id']
                    print(f"\nError preparing input for {video_id}: {str(e)}")
                finally:
                    pbar.update(1)

    print(f"\nPrepared {len(all_inputs)} valid inputs")

    if len(all_inputs) == 0:
        print("No valid inputs to process!")
        return

    # Batch inference
    print("\nRunning batch inference...")
    try:
        outputs = llm.generate(all_inputs, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during batch inference: {str(e)}")
        return

    # Collect results
    new_predictions = []
    for i, output in enumerate(outputs):
        video_id = all_video_ids[i]
        item = all_items[i]
        raw_text = output.outputs[0].text
        reasoning, predicted_caption = parse_reasoning_and_content(raw_text, args.reasoning_parser)

        prediction = {
            "video_id": video_id,
            "video_path": item['video_path'],
            "predicted_caption": predicted_caption,
            "ground_truth_caption": item['ground_truth_caption'],
            "questions": item['questions'],
            "ground_truth_answers": item['ground_truth_answers']
        }
        if args.save_reasoning:
            prediction["reasoning_content"] = reasoning
            prediction["raw_output_text"] = raw_text
        new_predictions.append(prediction)

    # Merge all predictions
    all_predictions = completed_predictions + new_predictions

    # Sort by original order
    video_id_to_pred = {item['video_id']: item for item in all_predictions}
    final_predictions = []
    for item in test_data:
        video_id = item['video_id']
        if video_id in video_id_to_pred:
            final_predictions.append(video_id_to_pred[video_id])
        else:
            # Add error placeholder
            final_predictions.append({
                "video_id": video_id,
                "video_path": item['video_path'],
                "predicted_caption": "[ERROR: Not processed]",
                "ground_truth_caption": item['ground_truth_caption'],
                "questions": item['questions'],
                "ground_truth_answers": item['ground_truth_answers']
            })

    # Save results
    save_predictions(final_predictions, args.output_path)

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total samples: {len(test_data)}")
    print(f"  Successfully generated: {len([p for p in final_predictions if not p['predicted_caption'].startswith('[ERROR')])}")
    print(f"  Failed: {len([p for p in final_predictions if p['predicted_caption'].startswith('[ERROR')])}")
    print(f"  Output saved to: {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 1 (Qwen2-VL): Generate video captions using vLLM batch inference.

This is a drop-in replacement for `step1_generate_captions.py` when the caption model
is Qwen2-VL style (model_type: qwen2_vl). It uses `Qwen2VLProcessor`.

Output schema matches Step2 expectations.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import Qwen2VLProcessor
from qwen_omni_utils import process_mm_info


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video captions (Qwen2-VL) using vLLM")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2-VL model")
    parser.add_argument("--input_data", type=str, required=True, help="Input synthesized test set JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output predictions JSON")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_seqs", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--video_max_pixels", type=int, default=None, help="Optional max pixels for video inputs")
    parser.add_argument("--legacy_root", type=str, default="", help="Optional legacy root to remap absolute paths")
    parser.add_argument("--workspace_root", type=str, default="", help="Workspace root for remapping legacy paths")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    preds = load_json(path)
    return {x["video_id"]: x for x in preds if "video_id" in x}


def save_predictions(predictions: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def resolve_video_path(path: str, legacy_root: str, workspace_root: str) -> str:
    if not path:
        return path
    if not os.path.isabs(path):
        candidate = os.path.join(workspace_root, path)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(path):
        return path
    if legacy_root and workspace_root and path.startswith(legacy_root):
        candidate = workspace_root + path[len(legacy_root) :]
        if os.path.exists(candidate):
            return candidate
    return path


def build_input(processor, messages, use_audio_in_video: bool):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios

    return inputs


def process_single_sample(
    item: Dict[str, Any],
    processor,
    use_audio_in_video: bool,
    video_max_pixels: Optional[int],
    legacy_root: str,
    workspace_root: str,
) -> Optional[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
    video_id = item["video_id"]
    video_path = resolve_video_path(item["video_path"], legacy_root, workspace_root)

    if not os.path.exists(video_path):
        return None

    system_text = item.get("system_prompt")
    user_text = (item.get("user_prompt") or "").replace("<video>", "").strip()

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})

    video_obj: Dict[str, Any] = {"type": "video", "video": video_path}
    if video_max_pixels is not None:
        video_obj["max_pixels"] = video_max_pixels

    messages.append(
        {
            "role": "user",
            "content": [video_obj, {"type": "text", "text": user_text}],
        }
    )

    input_data = build_input(processor, messages, use_audio_in_video)
    return input_data, video_id, item


def main():
    args = parse_args()

    print("=" * 80)
    print("Step 1: Generate Video Captions (Qwen2-VL) - vLLM Batch Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Tensor Parallel: {args.tensor_parallel_size}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Max Num Seqs: {args.max_num_seqs}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Use Audio in Video: {args.use_audio_in_video}")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Resume: {args.resume}")
    print("=" * 80)

    test_data = load_json(args.input_data)
    existing_predictions = {} if not args.resume else load_existing_predictions(args.output_path)

    completed_predictions: List[Dict[str, Any]] = []
    pending_samples: List[Dict[str, Any]] = []
    for item in test_data:
        vid = item["video_id"]
        if vid in existing_predictions:
            completed_predictions.append(existing_predictions[vid])
        else:
            pending_samples.append(item)

    print(f"\nTotal samples: {len(test_data)}")
    print(f"Already completed: {len(completed_predictions)}")
    print(f"Pending: {len(pending_samples)}")
    if not pending_samples:
        print("\nAll samples already processed!")
        return

    print("\nInitializing vLLM engine...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1},
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # NOTE: use_fast=False avoids Qwen2TokenizerFast missing `.image_token` in some checkpoints.
    processor = Qwen2VLProcessor.from_pretrained(args.model_path, use_fast=False)
    print("vLLM engine initialized successfully!")

    video_max_pixels = args.video_max_pixels
    workspace_root = args.workspace_root or str(Path(__file__).resolve().parent)
    legacy_root = args.legacy_root

    all_inputs: List[Dict[str, Any]] = []
    all_video_ids: List[str] = []
    all_items: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_item = {
            executor.submit(
                process_single_sample,
                item,
                processor,
                args.use_audio_in_video,
                video_max_pixels,
                legacy_root,
                workspace_root,
            ): item
            for item in pending_samples
        }
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
                    print(f"\nError preparing input for {item.get('video_id', 'unknown')}: {e}")
                finally:
                    pbar.update(1)

    print(f"\nPrepared {len(all_inputs)} valid inputs")
    if not all_inputs:
        print("No valid inputs to process!")
        return

    print("\nRunning batch inference...")
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)

    new_predictions: List[Dict[str, Any]] = []
    for i, output in enumerate(outputs):
        video_id = all_video_ids[i]
        item = all_items[i]
        predicted_caption = output.outputs[0].text
        new_predictions.append(
            {
                "video_id": video_id,
                "video_path": item["video_path"],
                "predicted_caption": predicted_caption,
                "ground_truth_caption": item.get("ground_truth_caption"),
                "questions": item.get("questions"),
                "ground_truth_answers": item.get("ground_truth_answers"),
            }
        )

    all_predictions = completed_predictions + new_predictions
    video_id_to_pred = {x["video_id"]: x for x in all_predictions if "video_id" in x}

    final_predictions: List[Dict[str, Any]] = []
    for item in test_data:
        vid = item["video_id"]
        if vid in video_id_to_pred:
            final_predictions.append(video_id_to_pred[vid])
        else:
            final_predictions.append(
                {
                    "video_id": vid,
                    "video_path": item["video_path"],
                    "predicted_caption": "[ERROR: Not processed]",
                    "ground_truth_caption": item.get("ground_truth_caption"),
                    "questions": item.get("questions"),
                    "ground_truth_answers": item.get("ground_truth_answers"),
                }
            )

    save_predictions(final_predictions, args.output_path)

    ok = sum(1 for p in final_predictions if not str(p.get("predicted_caption", "")).startswith("[ERROR"))
    bad = len(final_predictions) - ok

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total samples: {len(final_predictions)}")
    print(f"  Successfully generated: {ok}")
    print(f"  Failed: {bad}")
    print(f"  Output saved to: {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

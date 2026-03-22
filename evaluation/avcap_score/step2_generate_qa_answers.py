#!/usr/bin/env python3
"""
Step 2: Generate answers to questions based on predicted captions.
Uses text LLM (vLLM) to answer questions based on the predicted captions.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Force vLLM v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA answers using text LLM")

    # Model configuration
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="models/Qwen2.5-72B-Instruct",
        help="Path to the text LLM model"
    )

    # Data paths
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Input predictions from Step 1 (JSON file)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output QA answers JSON file"
    )

    # vLLM configuration
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
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
        default=16384,
        help="Maximum model length"
    )

    # Sampling configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,  # Lower temperature to reduce randomness
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,  # Limit max tokens to avoid overly long outputs
        help="Maximum tokens to generate per answer"
    )

    # Processing configuration
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions"
    )

    return parser.parse_args()


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load Step 1 predictions."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} predictions from {path}")
    return data


def load_existing_qa_answers(path: str) -> Dict[str, Dict]:
    """Load existing QA answers to support resume."""
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    qa_map = {}
    for item in qa_data:
        video_id = item['video_id']
        qa_map[video_id] = item

    print(f"Loaded {len(qa_map)} existing QA answers from {path}")
    return qa_map


def save_qa_answers(qa_answers: List[Dict[str, Any]], path: str):
    """Save QA answers to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(qa_answers, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qa_answers)} QA answers to {path}")


def build_qa_prompt(caption: str, question: str, system_instruction: str) -> str:
    """Build prompt for QA using the provided system instruction."""
    # Use chat format to constrain the model to answer only
    prompt = f"""<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
Caption: {caption}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def load_system_instruction(prompt_file: str = "answer_generation_prompt.txt") -> str:
    """Load system instruction from prompt file."""
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_file)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        print(f"Warning: Prompt file not found at {prompt_path}, using default")
        return """You are a "Grounded Caption Analyst." Your task is to provide a single concise, factually correct answer to the provided question based on the video caption.

**Answering Rules:**
1.  **Source of Truth:** Base the answer *solely* on the caption. Use the **exact terminology** found in the text.
2.  **Conciseness:** Provide a descriptive phrase or sentence fragment (under 20 words).
3.  **Directness:** Do NOT use filler words like "The answer is..." or "According to the caption...". Start directly with the answer content.

**Output Requirement:**
Return **only** the raw answer text string."""


def prepare_qa_prompts(predictions: List[Dict], system_instruction: str) -> tuple:
    """
    Prepare all QA prompts for batch processing.
    Returns: (prompts, metadata)
    """
    prompts = []
    metadata = []

    for item in predictions:
        video_id = item['video_id']
        predicted_caption = item['predicted_caption']
        questions = item.get('questions', {})

        if not questions:
            continue

        # Skip if prediction failed
        if predicted_caption.startswith('[ERROR'):
            continue

        # Generate prompts for all 20 questions
        for q_id in sorted(questions.keys()):
            question = questions[q_id]
            prompt = build_qa_prompt(predicted_caption, question, system_instruction)

            prompts.append(prompt)
            metadata.append({
                'video_id': video_id,
                'question_id': q_id
            })

    return prompts, metadata


def main():
    args = parse_args()

    print("=" * 80)
    print("Step 2: Generate QA Answers - Text LLM Batch Inference")
    print("=" * 80)
    print(f"Text Model: {args.text_model_path}")
    print(f"Tensor Parallel: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Resume: {args.resume}")
    print("=" * 80)

    # Load Step 1 predictions
    predictions = load_predictions(args.input_data)

    # Load existing QA answers for resume support
    existing_qa = {} if not args.resume else load_existing_qa_answers(args.output_path)

    # Separate completed and pending samples
    completed_qa = []
    pending_predictions = []

    for item in predictions:
        video_id = item['video_id']

        if video_id in existing_qa:
            completed_qa.append(existing_qa[video_id])
        else:
            pending_predictions.append(item)

    print(f"\nTotal samples: {len(predictions)}")
    print(f"Already completed: {len(completed_qa)}")
    print(f"Pending: {len(pending_predictions)}")

    if len(pending_predictions) == 0:
        print("\nAll samples already processed!")
        return

    # Load system instruction from prompt file
    print("\nLoading system instruction from answer_generation_prompt.txt...")
    system_instruction = load_system_instruction()
    print(f"Loaded system instruction ({len(system_instruction)} characters)")

    # Prepare QA prompts
    print("\nPreparing QA prompts...")
    prompts, metadata = prepare_qa_prompts(pending_predictions, system_instruction)

    print(f"Prepared {len(prompts)} QA prompts (20 questions per video)")

    if len(prompts) == 0:
        print("No valid prompts to process!")
        return

    # Initialize text LLM
    print("\nInitializing text LLM engine...")
    llm = LLM(
        model=args.text_model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "\n\n"],  # Stop on end tokens or double newline
    )

    print("Text LLM engine initialized successfully!")

    # Batch inference
    print("\nRunning batch inference for QA...")
    try:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during batch inference: {str(e)}")
        return

    # Organize results by video_id
    print("\nOrganizing results...")
    video_qa_map = {}

    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        meta = metadata[i]
        video_id = meta['video_id']
        question_id = meta['question_id']
        answer = output.outputs[0].text.strip()

        # Clean answer: remove common leading phrases
        prefixes_to_remove = [
            "我需要你根据提供的视频字幕，回答问题。",
            "根据字幕内容，",
            "根据字幕，",
            "答案是：",
            "答案：",
            "The answer is:",
            "Answer:",
            "According to the caption,",
        ]

        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # If the answer includes a leading question line, strip it
        if "问题是：" in answer or "Question:" in answer:
            # Remove up to the first newline
            parts = answer.split('\n', 1)
            if len(parts) > 1:
                answer = parts[1].strip()

        # Remove trailing explanatory phrases
        endings_to_remove = [
            "。字幕中提到：",
            "。字幕提到：",
            "，字幕中",
        ]
        for ending in endings_to_remove:
            if ending in answer:
                answer = answer.split(ending)[0].strip()

        if video_id not in video_qa_map:
            video_qa_map[video_id] = {}

        video_qa_map[video_id][question_id] = answer

    # Build final QA answers
    new_qa_answers = []
    for item in pending_predictions:
        video_id = item['video_id']

        qa_item = {
            "video_id": video_id,
            "video_path": item['video_path'],
            "predicted_caption": item['predicted_caption'],
            "questions": item['questions'],
            "predicted_answers": video_qa_map.get(video_id, {}),
            "ground_truth_answers": item['ground_truth_answers']
        }
        new_qa_answers.append(qa_item)

    # Merge all QA answers
    all_qa_answers = completed_qa + new_qa_answers

    # Sort by original order
    video_id_to_qa = {item['video_id']: item for item in all_qa_answers}
    final_qa_answers = []
    for item in predictions:
        video_id = item['video_id']
        if video_id in video_id_to_qa:
            final_qa_answers.append(video_id_to_qa[video_id])
        else:
            # Add error placeholder
            final_qa_answers.append({
                "video_id": video_id,
                "video_path": item['video_path'],
                "predicted_caption": item['predicted_caption'],
                "questions": item['questions'],
                "predicted_answers": {},
                "ground_truth_answers": item['ground_truth_answers']
            })

    # Save results
    save_qa_answers(final_qa_answers, args.output_path)

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Successfully generated answers: {len([qa for qa in final_qa_answers if qa['predicted_answers']])}")
    print(f"  Output saved to: {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

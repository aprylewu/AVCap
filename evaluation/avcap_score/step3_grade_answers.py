#!/usr/bin/env python3
"""
Step 3: Grade predicted answers against ground truth.
Uses text LLM (vLLM) to grade answers on a 0-5 scale.
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Force vLLM v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Grade QA answers using text LLM")

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
        help="Input QA answers from Step 2 (JSON file)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output grades JSON file"
    )

    # vLLM configuration
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
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
        default=0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
        help="Maximum tokens to generate per grade"
    )

    # Processing configuration
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing grades"
    )

    return parser.parse_args()


def load_qa_answers(path: str) -> List[Dict[str, Any]]:
    """Load Step 2 QA answers."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QA answers from {path}")
    return data


def load_existing_grades(path: str) -> Dict[str, Dict]:
    """Load existing grades to support resume."""
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        grades_data = json.load(f)

    grades_map = {}
    for item in grades_data:
        video_id = item['video_id']
        grades_map[video_id] = item

    print(f"Loaded {len(grades_map)} existing grades from {path}")
    return grades_map


def save_grades(grades: List[Dict[str, Any]], path: str):
    """Save grades to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(grades, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(grades)} grades to {path}")


def load_grading_instruction(prompt_file: str = "grading_prompt.txt") -> str:
    """Load grading instruction from prompt file."""
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_file)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        print(f"Warning: Prompt file not found at {prompt_path}, using default")
        return """You are an expert grader evaluating answers about video content.

**Grading Scale:**
- 5: Perfect match or semantically equivalent
- 4: Mostly correct with minor differences
- 3: Partially correct, captures main idea
- 2: Somewhat related but missing key information
- 1: Incorrect but shows some understanding
- 0: Completely incorrect or irrelevant

**Output Requirement:**
Provide ONLY the numeric score (0-5)."""


def build_grading_prompt(question: str, predicted_answer: str, ground_truth_answer: str, grading_instruction: str) -> str:
    """Build grading prompt using chat format."""
    prompt = f"""<|im_start|>system
{grading_instruction}<|im_end|>
<|im_start|>user
Question: {question}

Ground Truth Answer: {ground_truth_answer}

Predicted Answer: {predicted_answer}<|im_end|>
<|im_start|>assistant
Score: """
    return prompt


def parse_score_from_response(response: str) -> int:
    """Parse score from LLM response."""
    # Try to extract a number 0-5
    match = re.search(r'\b([0-5])\b', response.strip())
    if match:
        return int(match.group(1))

    # Default to 0 if parsing fails
    print(f"Warning: Could not parse score from response: {response}")
    return 0


def prepare_grading_prompts(qa_answers: List[Dict], grading_instruction: str) -> tuple:
    """
    Prepare all grading prompts for batch processing.
    Returns: (prompts, metadata)
    """
    prompts = []
    metadata = []

    for item in qa_answers:
        video_id = item['video_id']
        questions = item.get('questions', {})
        predicted_answers = item.get('predicted_answers', {})
        ground_truth_answers = item.get('ground_truth_answers', {})

        if not questions or not predicted_answers:
            continue

        # Generate grading prompts for all questions
        for q_id in sorted(questions.keys()):
            question = questions[q_id]
            pred_answer = predicted_answers.get(q_id, "")

            # Convert question_XX to answer_XX for ground truth lookup
            answer_id = q_id.replace('question_', 'answer_')
            gt_answer = ground_truth_answers.get(answer_id, "")

            if not pred_answer or not gt_answer:
                continue

            prompt = build_grading_prompt(question, pred_answer, gt_answer, grading_instruction)

            prompts.append(prompt)
            metadata.append({
                'video_id': video_id,
                'question_id': q_id
            })

    return prompts, metadata


def calculate_statistics(grades_data: List[Dict]) -> Dict:
    """Calculate grading statistics."""
    total_videos = len(grades_data)
    all_scores = []
    per_question_scores = {f"question_{i:02d}": [] for i in range(1, 21)}

    for item in grades_data:
        grades = item.get('grades', {})
        if grades:
            video_scores = list(grades.values())
            all_scores.extend(video_scores)

            for q_id, score in grades.items():
                if q_id in per_question_scores:
                    per_question_scores[q_id].append(score)

    def to_100(score_0_5: float) -> float:
        return score_0_5 * 20.0

    avg_score_0_5 = sum(all_scores) / len(all_scores) if all_scores else 0.0
    avg_score_100 = to_100(avg_score_0_5)

    # Score distribution
    score_dist = {}
    for item in grades_data:
        avg = item.get('average_score', 0)
        bucket = f"{int(avg)}.0-{int(avg)}.9" if avg < 5 else "5.0"
        score_dist[bucket] = score_dist.get(bucket, 0) + 1

    # Per-question averages
    per_q_avg_0_5 = {}
    per_q_avg_100 = {}
    for q_id, scores in per_question_scores.items():
        if scores:
            v = sum(scores) / len(scores)
            per_q_avg_0_5[q_id] = v
            per_q_avg_100[q_id] = to_100(v)

    # Section averages (weighted by number of available scores)
    section_defs = {
        "visual": [f"question_{i:02d}" for i in range(1, 6)],
        "audio": [f"question_{i:02d}" for i in range(6, 11)],
        "joint": [f"question_{i:02d}" for i in range(11, 21)],
    }
    section_avg_0_5 = {}
    section_avg_100 = {}
    for name, q_ids in section_defs.items():
        sec_scores = []
        for q_id in q_ids:
            sec_scores.extend(per_question_scores.get(q_id, []))
        if sec_scores:
            v = sum(sec_scores) / len(sec_scores)
        else:
            v = 0.0
        section_avg_0_5[name] = v
        section_avg_100[name] = to_100(v)

    return {
        'total_videos': total_videos,
        'average_score_0_5': avg_score_0_5,
        'average_score_100': avg_score_100,
        'score_distribution': score_dist,
        'per_question_averages_0_5': per_q_avg_0_5,
        'per_question_averages_100': per_q_avg_100,
        'section_averages_0_5': section_avg_0_5,
        'section_averages_100': section_avg_100,
    }


def print_statistics(stats: Dict):
    """Print evaluation statistics."""
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total Videos: {stats['total_videos']}")
    print(f"Average Score (100): {stats['average_score_100']:.2f} / 100.00")

    sec_100 = stats.get('section_averages_100', {})
    print("\nSection Average Scores (100):")
    print(f"  Visual (Q01-Q05): {sec_100.get('visual', 0.0):.2f}")
    print(f"  Audio  (Q06-Q10): {sec_100.get('audio', 0.0):.2f}")
    print(f"  Joint  (Q11-Q20): {sec_100.get('joint', 0.0):.2f}")

    print("\nScore Distribution:")
    for bucket, count in sorted(stats['score_distribution'].items()):
        percentage = count / stats['total_videos'] * 100
        print(f"  {bucket}: {count} videos ({percentage:.1f}%)")

    print("\nPer-Question Average Scores (100):")
    for q_id, avg in sorted(stats.get('per_question_averages_100', {}).items()):
        q_num = q_id.split('_')[1]
        print(f"  Q{q_num}: {avg:.2f}")

    print("=" * 80)


def main():
    args = parse_args()

    print("=" * 80)
    print("Step 3: Grade Answers - Text LLM Batch Inference")
    print("=" * 80)
    print(f"Text Model: {args.text_model_path}")
    print(f"Tensor Parallel: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Resume: {args.resume}")
    print("=" * 80)

    # Load Step 2 QA answers
    qa_answers = load_qa_answers(args.input_data)

    # Load existing grades for resume support
    existing_grades = {} if not args.resume else load_existing_grades(args.output_path)

    # Separate completed and pending samples
    completed_grades = []
    pending_qa = []

    for item in qa_answers:
        video_id = item['video_id']

        if video_id in existing_grades:
            completed_grades.append(existing_grades[video_id])
        else:
            pending_qa.append(item)

    print(f"\nTotal samples: {len(qa_answers)}")
    print(f"Already completed: {len(completed_grades)}")
    print(f"Pending: {len(pending_qa)}")

    if len(pending_qa) == 0:
        print("\nAll samples already processed!")
        # Print statistics for existing grades
        stats = calculate_statistics(completed_grades)
        print_statistics(stats)
        return

    # Load grading instruction from prompt file
    print("\nLoading grading instruction from grading_prompt.txt...")
    grading_instruction = load_grading_instruction()
    print(f"Loaded grading instruction ({len(grading_instruction)} characters)")

    # Prepare grading prompts
    print("\nPreparing grading prompts...")
    prompts, metadata = prepare_grading_prompts(pending_qa, grading_instruction)

    print(f"Prepared {len(prompts)} grading prompts")

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
        stop=["<|im_end|>", "\n"],
    )

    print("Text LLM engine initialized successfully!")

    # Batch inference
    print("\nRunning batch inference for grading...")
    try:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during batch inference: {str(e)}")
        return

    # Organize results by video_id
    print("\nOrganizing results...")
    video_grades_map = {}

    for i, output in enumerate(tqdm(outputs, desc="Processing grades")):
        meta = metadata[i]
        video_id = meta['video_id']
        question_id = meta['question_id']
        response = output.outputs[0].text.strip()
        score = parse_score_from_response(response)

        if video_id not in video_grades_map:
            video_grades_map[video_id] = {}

        video_grades_map[video_id][question_id] = score

    # Build final grades
    new_grades = []
    for item in pending_qa:
        video_id = item['video_id']
        grades = video_grades_map.get(video_id, {})

        # Calculate average score
        if grades:
            avg_score = sum(grades.values()) / len(grades)
        else:
            avg_score = 0.0

        grade_item = {
            "video_id": video_id,
            "video_path": item['video_path'],
            "grades": grades,
            "average_score": avg_score,
            "predicted_answers": item['predicted_answers'],
            "ground_truth_answers": item['ground_truth_answers']
        }
        new_grades.append(grade_item)

    # Merge all grades
    all_grades = completed_grades + new_grades

    # Sort by original order
    video_id_to_grade = {item['video_id']: item for item in all_grades}
    final_grades = []
    for item in qa_answers:
        video_id = item['video_id']
        if video_id in video_id_to_grade:
            final_grades.append(video_id_to_grade[video_id])
        else:
            # Add error placeholder
            final_grades.append({
                "video_id": video_id,
                "video_path": item['video_path'],
                "grades": {},
                "average_score": 0.0,
                "predicted_answers": item['predicted_answers'],
                "ground_truth_answers": item['ground_truth_answers']
            })

    # Save results
    save_grades(final_grades, args.output_path)

    # Calculate and print statistics
    stats = calculate_statistics(final_grades)
    print_statistics(stats)

    print(f"\nOutput saved to: {args.output_path}")


if __name__ == "__main__":
    main()

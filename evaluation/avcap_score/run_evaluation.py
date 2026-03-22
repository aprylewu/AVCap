#!/usr/bin/env python3
"""
Main orchestrator script to run QA evaluation pipeline.
Supports running individual steps or the complete pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps with default model
  python run_evaluation.py --steps all --baseline qwen3omni

  # Run only Step 1 (caption generation)
  python run_evaluation.py --steps 1 --baseline qwen3omni --omni_model_path /path/to/model

  # Run Steps 2 and 3 (QA and grading)
  python run_evaluation.py --steps 2 3 --baseline qwen3omni --text_model_path /path/to/text/model

  # Resume from checkpoint
  python run_evaluation.py --steps all --baseline qwen3omni --resume
        """
    )

    # Step selection
    parser.add_argument(
        "--steps",
        nargs='+',
        choices=['1', '2', '3', 'all'],
        default=['all'],
        help="Which steps to run (1=caption, 2=qa, 3=grade, all=all steps)"
    )

    # Baseline name for output files
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline name for output files (e.g., qwen3omni, videollama2)"
    )

    # Model paths
    parser.add_argument(
        "--omni_model_path",
        type=str,
        default="modelscope/Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Path to Omni model for Step 1"
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Path to text LLM for Steps 2 and 3"
    )

    # vLLM configuration for Step 1 (Omni model)
    parser.add_argument(
        "--omni_gpu_memory",
        type=float,
        default=0.7,
        help="GPU memory utilization for Omni model"
    )
    parser.add_argument(
        "--omni_tensor_parallel",
        type=int,
        default=8,
        help="Tensor parallel size for Omni model"
    )
    parser.add_argument(
        "--omni_temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for Step 1 caption generation"
    )
    parser.add_argument(
        "--omni_repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty for Step 1 caption generation"
    )

    # vLLM configuration for Steps 2 & 3 (Text model)
    parser.add_argument(
        "--text_gpu_memory",
        type=float,
        default=0.8,
        help="GPU memory utilization for text model"
    )
    parser.add_argument(
        "--text_tensor_parallel",
        type=int,
        default=8,
        help="Tensor parallel size for text model"
    )

    # Processing options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for input preparation"
    )

    return parser.parse_args()


def run_command(cmd: list, step_name: str):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"Running {step_name}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: {step_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{step_name} completed successfully!")


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent

    # Define output paths based on baseline name
    output_dir = base_dir / "outputs"
    step1_output = output_dir / f"step1_predictions_{args.baseline}.json"
    step2_output = output_dir / f"step2_qa_answers_{args.baseline}.json"
    step3_output = output_dir / f"step3_grades_{args.baseline}.json"

    # Input data (unified test set) - fixed relative path
    input_data = base_dir / "data" / "testset.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input data exists
    if not input_data.exists():
        print(f"Error: Input data not found at {input_data}")
        print("Please create data/testset.json (or run synthesize_dataset.py with --output-path data/testset.json)!")
        sys.exit(1)

    # Determine which steps to run
    steps_to_run = set(args.steps)
    if 'all' in steps_to_run:
        steps_to_run = {'1', '2', '3'}

    print("=" * 80)
    print("QA Evaluation Pipeline")
    print("=" * 80)
    print(f"Baseline: {args.baseline}")
    print(f"Steps to run: {', '.join(sorted(steps_to_run))}")
    print(f"Resume: {args.resume}")
    print(f"Omni Model: {args.omni_model_path}")
    print(f"Text Model: {args.text_model_path}")
    print(f"Omni Temperature: {args.omni_temperature}")
    print(f"Omni Repetition Penalty: {args.omni_repetition_penalty}")
    print("=" * 80)

    # Step 1: Generate captions
    if '1' in steps_to_run:
        cmd = [
            "python", "step1_generate_captions.py",
            "--model_path", args.omni_model_path,
            "--input_data", str(input_data),
            "--output_path", str(step1_output),
            "--gpu_memory_utilization", str(args.omni_gpu_memory),
            "--tensor_parallel_size", str(args.omni_tensor_parallel),
            "--num_workers", str(args.num_workers),
            "--temperature", str(args.omni_temperature),
            "--repetition_penalty", str(args.omni_repetition_penalty),
        ]
        if args.resume:
            cmd.append("--resume")

        run_command(cmd, "Step 1: Caption Generation")

    # Step 2: Generate QA answers
    if '2' in steps_to_run:
        # Check if Step 1 output exists
        if not step1_output.exists():
            print(f"\nError: Step 1 output not found at {step1_output}")
            print("Please run Step 1 first!")
            sys.exit(1)

        cmd = [
            "python", "step2_generate_qa_answers.py",
            "--text_model_path", args.text_model_path,
            "--input_data", str(step1_output),
            "--output_path", str(step2_output),
            "--gpu_memory_utilization", str(args.text_gpu_memory),
            "--tensor_parallel_size", str(args.text_tensor_parallel),
        ]
        if args.resume:
            cmd.append("--resume")

        run_command(cmd, "Step 2: QA Answer Generation")

    # Step 3: Grade answers
    if '3' in steps_to_run:
        # Check if Step 2 output exists
        if not step2_output.exists():
            print(f"\nError: Step 2 output not found at {step2_output}")
            print("Please run Step 2 first!")
            sys.exit(1)

        cmd = [
            "python", "step3_grade_answers.py",
            "--text_model_path", args.text_model_path,
            "--input_data", str(step2_output),
            "--output_path", str(step3_output),
            "--gpu_memory_utilization", str(args.text_gpu_memory),
            "--tensor_parallel_size", str(args.text_tensor_parallel),
        ]
        if args.resume:
            cmd.append("--resume")

        run_command(cmd, "Step 3: Answer Grading")

    # Print final summary
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"Baseline: {args.baseline}")
    print("\nOutput files:")
    if '1' in steps_to_run or 'all' in args.steps:
        print(f"  Step 1 (Captions): {step1_output}")
    if '2' in steps_to_run or 'all' in args.steps:
        print(f"  Step 2 (QA Answers): {step2_output}")
    if '3' in steps_to_run or 'all' in args.steps:
        print(f"  Step 3 (Grades): {step3_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/bin/bash
# QA Evaluation Pipeline (AVCap Score)
# Usage: bash evaluate_qa.sh [--baseline name] [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE="AVCap"
STEPS="all"
OMNI_MODEL_PATH="models/your_omni_model"
TEXT_MODEL_PATH="models/your_text_model"
OMNI_GPU_MEMORY="0.8"
OMNI_TEMPERATURE="0.6"
OMNI_REPETITION_PENALTY="1.05"
TEXT_GPU_MEMORY="0.8"
NUM_WORKERS="32"
RESUME=0

usage() {
  cat << 'USAGE'
Usage:
  bash evaluate_qa.sh [options]

Options:
  --baseline <name>       Baseline name for output files
  --steps <all|1,2,3>     Steps to run (default: all)
  --omni-model <path>     Omni caption model path (Step 1)
  --text-model <path>     Text model path (Steps 2 & 3)
  --omni-gpu <float>      Omni GPU memory utilization (default: 0.7)
  --omni-tp <int>         Omni tensor parallel size (default: 8)
  --omni-temperature <f>  Step1 temperature (default: 0.6)
  --omni-repetition-penalty <f>
                          Step1 repetition penalty (default: 1.05)
  --text-gpu <float>      Text GPU memory utilization (default: 0.8)
  --text-tp <int>         Text tensor parallel size (default: 4)
  --num-workers <int>     Parallel workers (default: 16)
  --resume               Resume from existing outputs
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      BASELINE="$2"; shift 2;;
    --steps)
      STEPS="$2"; shift 2;;
    --omni-model)
      OMNI_MODEL_PATH="$2"; shift 2;;
    --text-model)
      TEXT_MODEL_PATH="$2"; shift 2;;
    --omni-gpu)
      OMNI_GPU_MEMORY="$2"; shift 2;;
    --omni-tp)
      OMNI_TENSOR_PARALLEL="$2"; shift 2;;
    --omni-temperature)
      OMNI_TEMPERATURE="$2"; shift 2;;
    --omni-repetition-penalty)
      OMNI_REPETITION_PENALTY="$2"; shift 2;;
    --text-gpu)
      TEXT_GPU_MEMORY="$2"; shift 2;;
    --text-tp)
      TEXT_TENSOR_PARALLEL="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    --resume)
      RESUME=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

IFS=',' read -r -a STEPS_ARRAY <<< "$STEPS"

CMD=(
  python "$SCRIPT_DIR/run_evaluation.py"
  --steps "${STEPS_ARRAY[@]}"
  --baseline "$BASELINE"
  --omni_model_path "$OMNI_MODEL_PATH"
  --text_model_path "$TEXT_MODEL_PATH"
  --omni_gpu_memory "$OMNI_GPU_MEMORY"
  --omni_tensor_parallel "$OMNI_TENSOR_PARALLEL"
  --omni_temperature "$OMNI_TEMPERATURE"
  --omni_repetition_penalty "$OMNI_REPETITION_PENALTY"
  --text_gpu_memory "$TEXT_GPU_MEMORY"
  --text_tensor_parallel "$TEXT_TENSOR_PARALLEL"
  --num_workers "$NUM_WORKERS"
)

if [[ $RESUME -eq 1 ]]; then
  CMD+=(--resume)
fi

cd "$SCRIPT_DIR"
"${CMD[@]}"

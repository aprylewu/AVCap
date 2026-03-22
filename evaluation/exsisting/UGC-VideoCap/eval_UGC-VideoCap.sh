#!/bin/bash

# ============================================================================
# AVoCaDO UGC-VideoCap Evaluation Pipeline
# ============================================================================
#
# Usage:
#   1) Edit the config variables below
#   2) Run: bash eval_UGC-VideoCap.sh
#
# ============================================================================

# ========================= Configuration =========================

# Option 1: specify model name directly (for existing results)
# If MODEL_NAME is set, MODEL_PATH is ignored.
# Example: MODEL_NAME="avocado"
MODEL_NAME=""

# Option 2: specify model path (model name will be inferred from the path)
# If MODEL_NAME is empty, this path is used.
MODEL_PATH="models/your_model"

# Video directory (relative to this folder)
VIDEO_DIR="video"

# Inference mode: "transformers" or "vllm"
# - transformers: single-video inference (slower, compatible)
# - vllm: batched inference (faster, needs more GPU memory)
INFERENCE_MODE="vllm"

# vLLM inference settings (only for INFERENCE_MODE="vllm")
TENSOR_PARALLEL_SIZE=""   # Leave empty to auto-detect GPUs
GPU_MEMORY_UTILIZATION="0.8"
MAX_MODEL_LEN="32768"
MAX_NUM_SEQS="64"
NUM_WORKERS="8"

# Evaluation mode: "local" or "online"
# - local: vLLM + local eval model (free, requires multi-GPU)
# - online: GPT-4o via OpenRouter (paid, needs apikey.txt)
EVAL_MODE="online"

# Eval model path (only for local mode)
EVAL_MODEL_PATH="models/Qwen3-8B"

# Base directory (auto-detected)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

# OpenMP threads for preprocessing
export OMP_NUM_THREADS=28

# ====================== End Configuration ========================

set -e  # Exit on error

# Convert relative paths to absolute paths
convert_to_absolute_path() {
    local path="$1"
    if [ -z "$path" ]; then
        echo ""
        return
    fi

    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$(cd "$BASE_DIR" && cd "$path" && pwd)"
    fi
}

if [ -n "$MODEL_PATH" ]; then
    MODEL_PATH=$(convert_to_absolute_path "$MODEL_PATH")
fi

if [ -n "$VIDEO_DIR" ]; then
    VIDEO_DIR=$(convert_to_absolute_path "$VIDEO_DIR")
fi

if [ -n "$EVAL_MODEL_PATH" ]; then
    EVAL_MODEL_PATH=$(convert_to_absolute_path "$EVAL_MODEL_PATH")
fi

# Resolve model name
if [ -z "$MODEL_NAME" ]; then
    if [ -z "$MODEL_PATH" ]; then
        echo "[ERROR] MODEL_NAME and MODEL_PATH are both empty. Set at least one."
        exit 1
    fi

    if [ ! -d "$MODEL_PATH" ]; then
        echo "[ERROR] Model path does not exist: $MODEL_PATH"
        exit 1
    fi

    MODEL_NAME=$(basename "$MODEL_PATH")
    USE_MODEL_PATH=true
else
    USE_MODEL_PATH=false
fi

# Validate modes
if [ "$INFERENCE_MODE" != "transformers" ] && [ "$INFERENCE_MODE" != "vllm" ]; then
    echo "[ERROR] Invalid INFERENCE_MODE: $INFERENCE_MODE"
    echo "        Must be 'transformers' or 'vllm'"
    exit 1
fi

if [ "$EVAL_MODE" != "local" ] && [ "$EVAL_MODE" != "online" ]; then
    echo "[ERROR] Invalid EVAL_MODE: $EVAL_MODE"
    echo "        Must be 'local' or 'online'"
    exit 1
fi

# Check video directory when generating captions
if [ "$USE_MODEL_PATH" = true ]; then
    if [ ! -d "$VIDEO_DIR" ]; then
        echo "[ERROR] Video directory does not exist: $VIDEO_DIR"
        echo "        Please set VIDEO_DIR to the correct path"
        exit 1
    fi
fi

# Output paths
OUTPUT_DIR="$BASE_DIR/results/ugc_videocap_eval/$MODEL_NAME"
mkdir -p "$OUTPUT_DIR"
CAPTION_FILE="$OUTPUT_DIR/model_caption.jsonl"

# Show config
echo "============================================"
echo "  AVoCaDO UGC-VideoCap Evaluation"
echo "============================================"
echo "Model name        : $MODEL_NAME"
if [ "$USE_MODEL_PATH" = true ]; then
    echo "Model path        : $MODEL_PATH"
fi
echo "Video directory   : $VIDEO_DIR"
echo "Inference mode    : $INFERENCE_MODE"
echo "Evaluation mode   : $EVAL_MODE"
if [ "$EVAL_MODE" = "local" ]; then
    echo "Eval model        : $(basename "$EVAL_MODEL_PATH")"
fi
echo "Output directory  : $OUTPUT_DIR"
echo "============================================"
echo ""

# ==================== Step 1: Generate captions ====================

if [ -f "$CAPTION_FILE" ]; then
    echo "========================================="
    echo "Step 1: Generate captions [skipped]"
    echo "========================================="
    echo "✓ Found existing file: $CAPTION_FILE"
    echo ""
else
    if [ "$USE_MODEL_PATH" = false ]; then
        echo "========================================="
        echo "Step 1: Generate captions [skipped]"
        echo "========================================="
        echo "[WARNING] Caption file not found and MODEL_PATH is not set."
        echo "          Please ensure the file exists: $CAPTION_FILE"
        echo "          Or set MODEL_PATH to generate captions."
        exit 1
    fi

    echo "========================================="
    echo "Step 1: Generate captions"
    echo "========================================="
    echo "Generating captions (mode: $INFERENCE_MODE)..."
    echo ""

    if [ "$INFERENCE_MODE" = "transformers" ]; then
        python "$SCRIPT_DIR/generate_caption.py" \
            --model_path "$MODEL_PATH" \
            --video_dir "$VIDEO_DIR" \
            --save_path "$CAPTION_FILE"

    elif [ "$INFERENCE_MODE" = "vllm" ]; then
        VLLM_CMD="python $SCRIPT_DIR/generate_caption_vllm.py \
            --model_path $MODEL_PATH \
            --video_dir $VIDEO_DIR \
            --save_path $CAPTION_FILE \
            --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
            --max_model_len $MAX_MODEL_LEN \
            --max_num_seqs $MAX_NUM_SEQS \
            --num_workers $NUM_WORKERS"

        if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
            VLLM_CMD="$VLLM_CMD --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
        fi

        eval $VLLM_CMD
    fi

    echo ""
    echo "✓ Caption generation finished"
    echo ""
fi

# ==================== Step 2: Evaluate captions ====================

echo "========================================="
echo "Step 2: Evaluate captions"
echo "========================================="

if [ "$EVAL_MODE" = "local" ]; then
    echo "Using local evaluation (vLLM)..."
    echo ""

    if [ ! -d "$EVAL_MODEL_PATH" ]; then
        echo "[ERROR] Eval model path not found: $EVAL_MODEL_PATH"
        exit 1
    fi

    EVAL_MODEL_NAME=$(basename "$EVAL_MODEL_PATH")
    EVAL_RESULT_FILE="$OUTPUT_DIR/eval_results_${EVAL_MODEL_NAME}.json"

    python "$SCRIPT_DIR/evaluation_vllm.py" "$CAPTION_FILE" "$EVAL_RESULT_FILE" \
        --model_path "$EVAL_MODEL_PATH"

elif [ "$EVAL_MODE" = "online" ]; then
    echo "Using online evaluation (GPT-4o)..."
    echo ""

    if [ ! -f "apikey.txt" ]; then
        echo "[WARNING] apikey.txt not found. Please add your OpenRouter key."
    fi

    EVAL_RESULT_FILE="$OUTPUT_DIR/eval_results_gpt4o.json"
    python "$SCRIPT_DIR/evaluation.py" "$CAPTION_FILE" "$EVAL_RESULT_FILE"
fi

echo ""
echo "✓ Evaluation complete"
echo ""

# ==================== Summary ====================

echo "============================================"
echo "  Done"
echo "============================================"
echo "Model            : $MODEL_NAME"
echo "Inference mode   : $INFERENCE_MODE"
echo "Evaluation mode  : $EVAL_MODE"
echo ""
echo "Outputs:"
echo "  Caption file   : $CAPTION_FILE"
echo "  Eval results   : $EVAL_RESULT_FILE"
echo "============================================"
echo ""

if [ -f "$EVAL_RESULT_FILE" ]; then
    echo "Metrics:"
    echo "--------------------------------------------"
    python3 << PYEOF
import json
with open("$EVAL_RESULT_FILE", "r") as f:
    results = json.load(f)
    print(f"  Visual Avg   : {results.get('overall_visual_average', 0):.2f}%")
    print(f"  Audio Avg    : {results.get('overall_audio_average', 0):.2f}%")
    print(f"  Details Avg  : {results.get('overall_details_average', 0):.2f}%")
    print(f"  Overall Avg  : {results.get('overall_average_percent', 0):.2f}%")
PYEOF
    echo "--------------------------------------------"
fi

echo ""
echo "✓ All done!"
echo ""

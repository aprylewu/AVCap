#!/bin/bash

# ============================================================================
# AVoCaDO Video-SALMONN2 Evaluation Pipeline
# ============================================================================
#
# Usage:
#   1) Edit the config variables below
#   2) Run: bash eval_video-SALMONN2-testset.sh
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

# Evaluation mode: "local" or "online"
# - local: vLLM + local eval model (free, requires multi-GPU)
# - online: OpenRouter GPT-4.1 (paid, needs apikey.txt)
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

MODEL_PATH=$(convert_to_absolute_path "$MODEL_PATH")
EVAL_MODEL_PATH=$(convert_to_absolute_path "$EVAL_MODEL_PATH")

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

# Validate eval mode
if [ "$EVAL_MODE" != "local" ] && [ "$EVAL_MODE" != "online" ]; then
    echo "[ERROR] Invalid EVAL_MODE: $EVAL_MODE"
    echo "        Must be 'local' or 'online'"
    exit 1
fi

# Output paths
OUTPUT_DIR="$BASE_DIR/results/salmonn2_eval/$MODEL_NAME"
mkdir -p "$OUTPUT_DIR"
CAPTION_FILE="$OUTPUT_DIR/model_caption.json"

# Show config
echo "============================================"
echo "  AVoCaDO Evaluation"
echo "============================================"
echo "Model name       : $MODEL_NAME"
if [ "$USE_MODEL_PATH" = true ]; then
    echo "Model path       : $MODEL_PATH"
fi
echo "Evaluation mode  : $EVAL_MODE"
if [ "$EVAL_MODE" = "local" ]; then
    echo "Eval model       : $(basename "$EVAL_MODEL_PATH")"
fi
echo "Output directory : $OUTPUT_DIR"
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
    echo "Generating captions..."
    echo ""

    python "$SCRIPT_DIR/qwen3omni_generate_caption.py" \
        --model_path "$MODEL_PATH" \
        --save_path "$CAPTION_FILE"

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
    LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_results_${EVAL_MODEL_NAME}.log"

    python "$SCRIPT_DIR/evaluation_vllm.py" "$CAPTION_FILE" --model_path "$EVAL_MODEL_PATH"

elif [ "$EVAL_MODE" = "online" ]; then
    echo "Using online evaluation (GPT-4.1)..."
    echo ""

    if [ ! -f "apikey.txt" ]; then
        echo "[WARNING] apikey.txt not found. Please add your OpenRouter key."
    fi

    LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_results_gpt41.log"
    python "$SCRIPT_DIR/evaluation.py" "$CAPTION_FILE"
fi

echo ""
echo "✓ Evaluation complete"
echo ""

# ==================== Summary ====================

echo "============================================"
echo "  Done"
echo "============================================"
echo "Model            : $MODEL_NAME"
echo "Evaluation mode  : $EVAL_MODE"
echo ""
echo "Outputs:"
echo "  Caption file   : $CAPTION_FILE"
echo "  Eval log       : $LOG_FILE"
echo "  Eval details   : ${CAPTION_FILE%.json}_eval_results_gpt41.json"
echo "============================================"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "Metrics:"
    echo "--------------------------------------------"
    cat "$LOG_FILE"
    echo "--------------------------------------------"
fi

echo ""
echo "✓ All done!"
echo ""

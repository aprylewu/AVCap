"""
Configuration file for the QA evaluation pipeline.
Centralized configuration for all steps.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TEST_SET_PATH = str(DATA_DIR / "testset.json")
ANSWERS_DIR = str(DATA_DIR / "answers")
OUTPUT_DIR = str(BASE_DIR / "outputs")

# ============================================================================
# Model Paths
# ============================================================================
# Omni model for video captioning (Step 1)
OMNI_MODEL_PATH = "models/Qwen3-Omni-30B-A3B-Instruct"

# Text LLM for QA and grading (Steps 2 & 3)
TEXT_MODEL_PATH = "models/Qwen2.5-72B-Instruct"

# ============================================================================
# vLLM Configuration
# ============================================================================
# Omni model (Step 1)
OMNI_GPU_MEMORY = 0.7
OMNI_TENSOR_PARALLEL = 8
OMNI_MAX_MODEL_LEN = 65536
OMNI_MAX_NUM_SEQS = 64

# Text model (Steps 2 & 3)
TEXT_GPU_MEMORY = 0.8
TEXT_TENSOR_PARALLEL = 4
TEXT_MAX_MODEL_LEN = 32768

# ============================================================================
# Sampling Parameters
# ============================================================================
# Step 1: Caption generation
CAPTION_TEMPERATURE = 0.6
CAPTION_TOP_P = 0.95
CAPTION_TOP_K = 20
CAPTION_MAX_TOKENS = 4096

# Step 2: QA answer generation
QA_TEMPERATURE = 0.3
QA_MAX_TOKENS = 512

# Step 3: Answer grading
GRADE_TEMPERATURE = 0.0  # Deterministic
GRADE_MAX_TOKENS = 10

# ============================================================================
# Processing Configuration
# ============================================================================
# Parallel workers for input preparation
NUM_WORKERS = 16

# Use audio in video
USE_AUDIO_IN_VIDEO = True

# ============================================================================
# Output File Templates
# ============================================================================
def get_output_paths(baseline: str):
    """Get output paths for a given baseline."""
    return {
        "step1": str(Path(OUTPUT_DIR) / f"step1_predictions_{baseline}.json"),
        "step2": str(Path(OUTPUT_DIR) / f"step2_qa_answers_{baseline}.json"),
        "step3": str(Path(OUTPUT_DIR) / f"step3_grades_{baseline}.json"),
    }

# ============================================================================
# Prompt Templates
# ============================================================================
QA_PROMPT_TEMPLATE = """You are an expert at answering questions about video content based on detailed captions.

Caption: {caption}

Question: {question}

Provide a concise, accurate answer based solely on the information in the caption. If the caption doesn't contain enough information to answer, say "Cannot determine from caption."

Answer:"""

GRADING_PROMPT_TEMPLATE = """You are an expert grader evaluating answers about video content.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Predicted Answer: {predicted_answer}

Grade the predicted answer on a scale of 0-5:
- 5: Perfect match or semantically equivalent
- 4: Mostly correct with minor differences
- 3: Partially correct, captures main idea
- 2: Somewhat related but missing key information
- 1: Incorrect but shows some understanding
- 0: Completely incorrect or irrelevant

Provide only the numeric score (0-5) without explanation.

Score:"""

# Video-SALMONN2 Evaluation (Quick Guide)

## Directory Layout
- `video/`: SALMONN2 test videos
- `video_salmonn2_test.json`: ground-truth / event annotations
- `eval_video-SALMONN2-testset.sh`: one-click evaluation script
- Other `*.py`: inference / evaluation scripts

## Environment
Follow the setup instructions from the **Qwen3Omni** repository.

## Usage
1. Open `eval_video-SALMONN2-testset.sh` and set these variables (relative or absolute paths):
   - `MODEL_PATH`: model to evaluate
   - `EVAL_MODEL_PATH`: local evaluation model (only when `EVAL_MODE=local`)
2. For online evaluation, put `apikey.txt` (OpenRouter key) in this folder.
3. Run:
   ```bash
   bash eval_video-SALMONN2-testset.sh
   ```

## Outputs
Results are written to: `results/salmonn2_eval/<model_name>/`

Note: This package contains no model weights; paths are cleaned to be relative.

# UGC-VideoCap Evaluation (Quick Guide)

## Directory Layout
- `video/`: UGC-VideoCap videos
- `final_caption_qa.json`: Ground-truth annotations
- `eval_UGC-VideoCap.sh`: One-click evaluation script
- Other `*.py`: inference / evaluation scripts

## Environment
Follow the setup instructions from the **Qwen3Omni** repository.

## Usage
1. Open `eval_UGC-VideoCap.sh` and set these variables (relative or absolute paths):
   - `MODEL_PATH`: model to evaluate
   - `EVAL_MODEL_PATH`: local evaluation model (only when `EVAL_MODE=local`)
   - If you want to skip inference, set `MODEL_NAME` to an existing results folder name
2. For online evaluation, put `apikey.txt` (OpenRouter key) in this folder.
3. Run:
   ```bash
   bash eval_UGC-VideoCap.sh
   ```

## Outputs
Results are written to: `results/ugc_videocap_eval/<model_name>/`

Note: This package contains no model weights; paths are cleaned to be relative.

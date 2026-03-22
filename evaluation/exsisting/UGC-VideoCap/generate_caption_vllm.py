import os
import torch
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)

# vLLM engine v1 not supported yet
os.environ['VLLM_USE_V1'] = '0'

script_dir = Path(__file__).resolve().parent
video_dir = str(script_dir / "video")


parser = argparse.ArgumentParser(description="Generate captions using vLLM for UGC-VideoCap dataset.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--video_dir", type=str, default=video_dir, help="Path to the video directory.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results.")
parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size for vLLM.")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization.")
parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length.")
parser.add_argument("--max_num_seqs", type=int, default=64, help="Maximum number of sequences.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel data preprocessing.")
args = parser.parse_args()

model_path = args.model_path
fout_path = args.save_path
if args.video_dir:
    video_dir = args.video_dir

video_dir_path = Path(video_dir)
if not video_dir_path.is_absolute():
    video_dir_path = (script_dir / video_dir_path).resolve()
video_dir = str(video_dir_path)

# Initialize vLLM
# If tensor_parallel_size is None, default to 8 (or use GPU count if less than 8)
if args.tensor_parallel_size is None:
    tensor_parallel_size = min(8, torch.cuda.device_count())
else:
    tensor_parallel_size = args.tensor_parallel_size

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    gpu_memory_utilization=args.gpu_memory_utilization,
    tensor_parallel_size=tensor_parallel_size,
    limit_mm_per_prompt={'image': 0, 'video': 1, 'audio': 1},
    max_num_seqs=args.max_num_seqs,
    max_model_len=args.max_model_len,
)

# Sampling params matching the original script: do_sample=False, thinker_max_new_tokens=2048
sampling_params = SamplingParams(
    temperature=0.0,  # do_sample=False means greedy decoding
    max_tokens=8192,  # thinker_max_new_tokens=2048
)

processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)


def build_input(processor, messages, use_audio_in_video):
    """Build input for vLLM from conversation messages."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    return inputs


def prepare_single_input(video_id, processor, use_audio_in_video):
    """Prepare input for a single video (used for parallel preprocessing)."""
    video_path = os.path.join(video_dir, video_id)

    # Use the same prompt as in generate_caption.py
    prompt = ("You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. "
              "Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; "
              "(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
              "(4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
              "Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style.")

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        },
    ]

    inputs = build_input(processor, conversation, use_audio_in_video)
    return inputs, video_id, prompt


# Get all video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
num_workers = args.num_workers

print(f"Processing {len(video_files)} videos with {num_workers} workers for parallel preprocessing")

# Step 1: Parallel data preprocessing for ALL videos
print("Step 1: Preprocessing all videos in parallel...")
prepare_func = partial(prepare_single_input, processor=processor, use_audio_in_video=USE_AUDIO_IN_VIDEO)

all_inputs = []
all_video_ids = []
all_prompts = []

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(prepare_func, video_id) for video_id in video_files]

    for future in tqdm(as_completed(futures), total=len(video_files), desc="Preprocessing"):
        inputs, video_id, prompt = future.result()
        all_inputs.append(inputs)
        all_video_ids.append(video_id)
        all_prompts.append(prompt)

print(f"Step 2: Sending all {len(all_inputs)} requests to model for batch inference...")

# Step 2: Send all requests to model at once
outputs = llm.generate(all_inputs, sampling_params=sampling_params)

print("Step 3: Collecting results and saving to JSONL...")

# Step 3: Collect results and write to JSONL
with open(fout_path, 'w', encoding='utf-8') as fout:
    for video_id, prompt, output in zip(all_video_ids, all_prompts, outputs):
        model_generation = output.outputs[0].text
        out_data = {
            "video_id": video_id.replace(".mp4", ""),
            "caption": model_generation,
        }
        fout.write(json.dumps(out_data, ensure_ascii=False) + '\n')

print(f"Results saved to {fout_path}")

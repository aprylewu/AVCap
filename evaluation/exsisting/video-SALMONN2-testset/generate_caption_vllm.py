import os
import torch
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
VIDEO_MAX_PIXELS = 401408
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)
os.environ['VLLM_USE_V1'] = '0'

script_dir = Path(__file__).resolve().parent
fin_path = script_dir / "video_salmonn2_test.json"
video_dir = str(script_dir)
default_save_dir = str(script_dir / "results" / "salmonn2_eval" / "avocado" / "model_caption_vllm.json")
default_model_path = "models/your_model"

parser = argparse.ArgumentParser(description="Evaluate a model using vLLM and save results.")
parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the model checkpoint.")
parser.add_argument("--save_path", type=str, default=default_save_dir, help="Path to save the evaluation results.")
parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size for vLLM.")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization.")
parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length.")
parser.add_argument("--max_num_seqs", type=int, default=64, help="Maximum number of sequences.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel data preprocessing.")
args = parser.parse_args()

model_path = args.model_path
fout_path = args.save_path
Path(fout_path).parent.mkdir(parents=True, exist_ok=True)

with open(fin_path, 'r', encoding='utf-8') as fin:
    annotations = json.load(fin)

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

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=2048,
)

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."


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


def prepare_single_input(anno, processor, use_audio_in_video):
    """Prepare input for a single video."""
    video_path = os.path.join(video_dir, anno["video"])
    prompt = anno["conversations"][0]["value"].replace("<image>\n", "")

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
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
    return inputs, video_path, prompt


num_workers = args.num_workers
print(f"Processing {len(annotations)} videos with {num_workers} workers for parallel preprocessing")

print("Step 1: Preprocessing all videos in parallel...")
prepare_func = partial(prepare_single_input, processor=processor, use_audio_in_video=USE_AUDIO_IN_VIDEO)

all_inputs = []
all_video_paths = []
all_prompts = []

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(prepare_func, anno) for anno in annotations]

    for future in tqdm(as_completed(futures), total=len(annotations), desc="Preprocessing"):
        inputs, video_path, prompt = future.result()
        all_inputs.append(inputs)
        all_video_paths.append(video_path)
        all_prompts.append(prompt)

print(f"Step 2: Sending all {len(all_inputs)} requests to model for batch inference...")

outputs = llm.generate(all_inputs, sampling_params=sampling_params)

print("Step 3: Collecting results...")

updated_annos = []
for video_path, prompt, output in zip(all_video_paths, all_prompts, outputs):
    model_generation = output.outputs[0].text
    out_data = {
        "id": [video_path],
        "prompt": prompt,
        "pred": model_generation,
    }
    updated_annos.append(out_data)

with open(fout_path, 'w', encoding='utf-8') as fout:
    json.dump(updated_annos, fout, indent=4, ensure_ascii=False)

print(f"Results saved to {fout_path}")

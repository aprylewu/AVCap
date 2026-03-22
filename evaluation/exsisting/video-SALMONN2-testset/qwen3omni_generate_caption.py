import os
import torch
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
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
fin_path = script_dir / "video_salmonn2_test.json"
default_video_dir = script_dir


parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
parser.add_argument("--model_path", type=str, default="models/your_model", help="Path to the model checkpoint.")
parser.add_argument("--video_dir", type=str, default=str(default_video_dir), help="Path to the video directory.")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the evaluation results. If not specified, auto-generated based on model name.")
parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size for vLLM.")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization.")
parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length.")
parser.add_argument("--max_num_seqs", type=int, default=64, help="Maximum number of sequences.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel data preprocessing.")
parser.add_argument(
    "--reasoning_parser",
    type=str,
    default="auto",
    help=(
        "Reasoning parser backend for extracting thinking content. "
        "Use 'qwen3' for Qwen3/Qwen3-Omni *-Thinking models. "
        "Use 'none' to disable. Default: auto."
    ),
)
args = parser.parse_args()

model_path = args.model_path
video_dir = args.video_dir
video_dir_path = Path(video_dir)
if not video_dir_path.is_absolute():
    video_dir_path = (script_dir / video_dir_path).resolve()
video_dir = str(video_dir_path)

reasoning_parser_name: str
if args.reasoning_parser.lower() in {"none", "", "false", "0"}:
    reasoning_parser_name = ""
elif args.reasoning_parser.lower() == "auto":
    model_basename = Path(model_path).name.lower()
    if "qwen3" in model_basename and "thinking" in model_basename:
        reasoning_parser_name = "qwen3"
    else:
        reasoning_parser_name = ""
else:
    reasoning_parser_name = args.reasoning_parser

# Auto-generate save path based on model name
if args.save_path is None:
    model_name = Path(model_path).name
    results_dir = script_dir / "results" / "salmonn2_eval" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    fout_path = str(results_dir / "model_caption.json")
    print(f"Auto-generated save path: {fout_path}")
else:
    fout_path = args.save_path
    Path(fout_path).parent.mkdir(parents=True, exist_ok=True)

fin = open(fin_path, 'r', encoding='utf-8')
fout = open(fout_path, 'w', encoding='utf-8')

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
    reasoning_parser=reasoning_parser_name,
)

reasoner = None
if reasoning_parser_name:
    tokenizer = llm.get_tokenizer(None)
    reasoner = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(
        tokenizer=tokenizer)
    reasoning_request = ChatCompletionRequest(messages=[], model=str(model_path))

# Sampling params matching the original script: do_sample=False, thinker_max_new_tokens=2048
sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
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


def prepare_single_input(anno, processor, use_audio_in_video):
    """Prepare input for a single video (used for parallel preprocessing)."""
    video_path = os.path.join(video_dir, anno["video"])

    # Use the official prompt from the dataset (same as generate_caption.py)
    prompt = anno["conversations"][0]["value"].replace("<image>\n", "")

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
    return inputs, video_path, prompt


annotations = json.load(fin)
num_workers = args.num_workers

print(f"Processing {len(annotations)} videos with {num_workers} workers for parallel preprocessing")

# Step 1: Parallel data preprocessing for ALL videos
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

# Step 2: Send all requests to model at once
outputs = llm.generate(all_inputs, sampling_params=sampling_params)

print("Step 3: Collecting results...")

# Step 3: Collect results
updated_annos = []
for video_path, prompt, output in zip(all_video_paths, all_prompts, outputs):
    raw_generation = output.outputs[0].text
    reasoning_content = None
    model_generation = raw_generation
    if reasoner is not None:
        reasoning_content, content = reasoner.extract_reasoning_content(
            model_output=raw_generation, request=reasoning_request)
        if content is not None:
            model_generation = content
    out_data = {
        "id": [video_path],
        "prompt": prompt,
        "pred": model_generation,
    }
    if reasoning_content is not None:
        out_data["reasoning_content"] = reasoning_content
    updated_annos.append(out_data)

json.dump(updated_annos, fout, indent=4, ensure_ascii=False)
print(f"Results saved to {fout_path}")

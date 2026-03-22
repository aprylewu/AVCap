"""
Dataset Synthesis Script
Merge the test set with ground-truth Q&A to create a unified evaluation dataset.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def load_test_set(path: str) -> List[Dict]:
    """Load test set from JSON or JSONL."""
    print(f"Loading test set from: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON test set must be a list of items")
        print(f"Loaded {len(data)} test samples")
        return data

    test_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            test_data.append(item)
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def load_ground_truth_answers(answers_dir: str) -> Dict[str, Dict]:
    """Load all ground truth answer files and index by video_path"""
    print(f"Loading ground truth data from: {answers_dir}")
    gt_data = {}

    answer_files = sorted(Path(answers_dir).glob("*.json"))
    print(f"Found {len(answer_files)} answer files")

    for answer_file in tqdm(answer_files, desc="Loading answer files"):
        with open(answer_file, 'r', encoding='utf-8') as f:
            answers_list = json.load(f)

        for item in answers_list:
            video_path = item['video_path']
            gt_data[video_path] = {
                'caption': item['caption'],
                'questions': item['questions'],
                'answers': item['answers']
            }

    print(f"Loaded ground truth for {len(gt_data)} videos")
    return gt_data


def merge_test_and_ground_truth(test_data: List[Dict], gt_data: Dict[str, Dict]) -> List[Dict]:
    """Merge test set with ground truth data"""
    print("\nMerging test set with ground truth data...")
    merged_data = []
    matched_count = 0
    missing_count = 0

    for item in tqdm(test_data, desc="Merging data"):
        # Extract video path from test item
        video_path = item['videos'][0] if 'videos' in item and item['videos'] else None

        if not video_path:
            print(f"Warning: No video path found for item {item.get('id', 'unknown')}")
            continue

        # Extract system and user prompts from messages
        system_prompt = None
        user_prompt = None
        ground_truth_caption = None

        for msg in item.get('messages', []):
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']
            elif msg['role'] == 'assistant':
                ground_truth_caption = msg['content']

        # Create merged item
        merged_item = {
            'video_id': item.get('id', ''),
            'video_path': video_path,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'ground_truth_caption': ground_truth_caption,
            'questions': None,
            'ground_truth_answers': None
        }

        # Try to match with ground truth Q&A data
        if video_path in gt_data:
            matched_count += 1
            merged_item['questions'] = gt_data[video_path]['questions']
            merged_item['ground_truth_answers'] = gt_data[video_path]['answers']
            # Also store the caption from Q&A file (should match ground_truth_caption)
            merged_item['qa_caption'] = gt_data[video_path]['caption']
        else:
            missing_count += 1

        merged_data.append(merged_item)

    print(f"\nMatching statistics:")
    print(f"  Total test samples: {len(test_data)}")
    print(f"  Matched with Q&A: {matched_count}")
    print(f"  Missing Q&A data: {missing_count}")
    print(f"  Match rate: {matched_count/len(test_data)*100:.2f}%")

    return merged_data


def save_json(data: List[Dict], output_path: str):
    """Save data to JSON file"""
    print(f"\nSaving merged dataset to: {output_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} samples to {output_path}")


def parse_args():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    parser = argparse.ArgumentParser(description="Synthesize QA evaluation dataset")
    parser.add_argument(
        "--test-set",
        dest="test_set_path",
        default=str(data_dir / "qwen_omni_caption_test.jsonl"),
        help="Path to the test set JSONL",
    )
    parser.add_argument(
        "--answers-dir",
        default=str(data_dir / "answers"),
        help="Directory containing ground-truth answer JSONs",
    )
    parser.add_argument(
        "--output-path",
        default=str(base_dir / "data" / "testset.json"),
        help="Output JSON path for the synthesized dataset",
    )
    parser.add_argument(
        "--legacy-root",
        default="",
        help="Optional legacy root to remap absolute video paths",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(base_dir),
        help="Workspace root for remapping legacy paths",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load test set
    test_data = load_test_set(args.test_set_path)

    # Load ground truth answers
    gt_data = load_ground_truth_answers(args.answers_dir)

    # Merge datasets
    merged_data = merge_test_and_ground_truth(test_data, gt_data)

    # Map legacy absolute video paths to current workspace if needed
    if args.legacy_root:
        for item in merged_data:
            vp = item.get("video_path")
            if isinstance(vp, str) and vp.startswith(args.legacy_root):
                item["video_path"] = str(args.workspace_root) + vp[len(args.legacy_root) :]

    # Save merged dataset
    save_json(merged_data, args.output_path)

    # Print some statistics
    print("\n" + "="*60)
    print("Dataset Synthesis Complete!")
    print("="*60)
    print(f"Output file: {args.output_path}")
    print(f"Total samples: {len(merged_data)}")

    # Count samples with Q&A data
    with_qa = sum(1 for item in merged_data if item['questions'] is not None)
    print(f"Samples with Q&A: {with_qa}")
    print(f"Samples without Q&A: {len(merged_data) - with_qa}")

    # Show a sample
    print("\nSample item structure:")
    if merged_data:
        sample = merged_data[0]
        print(json.dumps({
            'video_id': sample['video_id'],
            'video_path': sample['video_path'][:80] + '...',
            'has_prompts': bool(sample['system_prompt'] and sample['user_prompt']),
            'has_ground_truth_caption': bool(sample['ground_truth_caption']),
            'has_questions': bool(sample['questions']),
            'has_answers': bool(sample['ground_truth_answers']),
            'num_questions': len(sample['questions']) if sample['questions'] else 0
        }, indent=2))


if __name__ == "__main__":
    main()

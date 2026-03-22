#!/usr/bin/env python3
"""
Step 1: Generate video captions using Google Gemini API.
Uses Gemini multimodal capability to caption local video files.
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_BASE = "https://generativelanguage.googleapis.com"
UPLOAD_ENDPOINT = f"{API_BASE}/upload/v1beta/files"
DEFAULT_KEY_PATH = str(Path(__file__).with_name("apikey.txt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video captions using Gemini API")

    # Compatibility args (ignored but accepted to keep evaluate_qa.sh stable)
    parser.add_argument("--model_path", type=str, default="", help="Unused (compat)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.0, help="Unused (compat)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Unused (compat)")
    parser.add_argument("--num_workers", type=int, default=1, help="Unused (compat)")

    # Data paths
    parser.add_argument("--input_data", type=str, required=True, help="Input synthesized test set JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Output predictions JSON file")

    # Gemini config
    parser.add_argument("--api-key", type=str, default="", help="Gemini API key")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model")
    parser.add_argument("--cache-dir", type=str, default="", help="Cache dir for uploaded file URIs")
    parser.add_argument("--inline-threshold-mb", type=int, default=20, help="Use inline data when file <= MB")
    parser.add_argument("--use-file-api", action="store_true", help="Force Files API upload (kept for compat; Files API is always used)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling API")
    parser.add_argument("--log-file", type=str, default="", help="Optional log file path")
    parser.add_argument("--retry-errors", action="store_true", help="When resuming, re-run items with [ERROR] outputs")

    # Retries
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier in seconds")

    return parser.parse_args()


def load_api_key(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_existing_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    preds = load_json(path)
    pred_map = {}
    for item in preds:
        pred_map[item["video_id"]] = item
    return pred_map


def get_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "video/mp4"


def file_cache_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def load_file_cache(cache_dir: str) -> Dict[str, Dict[str, str]]:
    if not cache_dir:
        return {}
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "gemini_file_cache.json")
    if not os.path.exists(cache_path):
        return {}
    return load_json(cache_path)


def save_file_cache(cache_dir: str, cache: Dict[str, Dict[str, str]]) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "gemini_file_cache.json")
    save_json(cache_path, cache)


def request_json(url: str, headers: Dict[str, str], data: Dict[str, Any], timeout: int = 600) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    payload = json.dumps(data).encode("utf-8")
    req = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else {}, dict(resp.headers)


def request_binary(url: str, headers: Dict[str, str], data: bytes, timeout: int = 600) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else {}, dict(resp.headers)


def upload_file(api_key: str, video_path: str, mime_type: str, display_name: str, max_retries: int, backoff: float) -> Tuple[str, str]:
    num_bytes = os.path.getsize(video_path)
    headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(num_bytes),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    body = {"file": {"display_name": display_name}}

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            status, _, resp_headers = request_json(UPLOAD_ENDPOINT, headers, body)
            upload_url = resp_headers.get("X-Goog-Upload-URL") or resp_headers.get("x-goog-upload-url")
            if not upload_url:
                raise RuntimeError("Upload URL missing in response headers")

            with open(video_path, "rb") as f:
                data = f.read()

            up_headers = {
                "Content-Length": str(num_bytes),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }
            status, resp_json, _ = request_binary(upload_url, up_headers, data)
            file_info = resp_json.get("file", {})
            file_uri = file_info.get("uri")
            file_mime = file_info.get("mimeType") or mime_type
            if not file_uri:
                raise RuntimeError("File URI missing in upload response")
            return file_uri, file_mime
        except (HTTPError, URLError, RuntimeError, ValueError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            raise

    raise last_err or RuntimeError("Upload failed")


def generate_with_file_uri(api_key: str, model: str, file_uri: str, mime_type: str, prompt_text: str) -> str:
    url = f"{API_BASE}/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                    {"text": prompt_text},
                ],
            }
        ]
    }
    status, resp_json, _ = request_json(url, headers, body)
    _ = status
    return extract_text_from_response(resp_json)


def generate_with_inline(api_key: str, model: str, video_path: str, mime_type: str, prompt_text: str) -> str:
    url = f"{API_BASE}/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    with open(video_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                    {"text": prompt_text},
                ],
            }
        ]
    }
    status, resp_json, _ = request_json(url, headers, body)
    _ = status
    return extract_text_from_response(resp_json)


def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    candidates = resp_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates in response: {resp_json}")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = []
    for part in parts:
        if "text" in part:
            texts.append(part["text"])
    if not texts:
        raise RuntimeError(f"No text parts in response: {resp_json}")
    return "".join(texts).strip()


def build_prompt(system_prompt: str, user_prompt: str) -> str:
    user_text = (user_prompt or "").replace("<video>", "").strip()
    system_text = (system_prompt or "").strip()
    if system_text:
        if user_text:
            return system_text + "\n\n" + user_text
        return system_text
    return user_text


def process_item(
    item: Dict[str, Any],
    api_key: str,
    model: str,
    cache_dir: str,
    file_cache: Dict[str, Dict[str, str]],
    cache_lock: Optional[threading.Lock],
    inline_threshold_mb: int,
    use_file_api: bool,
    max_retries: int,
    backoff: float,
    dry_run: bool,
) -> str:
    video_path = item.get("video_path")
    if isinstance(video_path, str) and not os.path.isabs(video_path):
        video_path = str((Path(__file__).resolve().parent / video_path).resolve())
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")

    prompt_text = build_prompt(item.get("system_prompt", ""), item.get("user_prompt", ""))
    mime_type = get_mime_type(video_path)
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    if dry_run:
        return f"[DRY-RUN] Would caption video {os.path.basename(video_path)} with {model}"

    def call_with_retry(fn, *fn_args):
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn(*fn_args)
            except (HTTPError, URLError, RuntimeError, ValueError) as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Request failed")

    if True:
        key = file_cache_key(video_path)
        cached = None
        if cache_lock is not None:
            with cache_lock:
                cached = file_cache.get(key)
        else:
            cached = file_cache.get(key)
        if cached and cached.get("uri") and cached.get("mime"):
            file_uri = cached["uri"]
            file_mime = cached["mime"]
        else:
            display_name = os.path.basename(video_path)
            file_uri, file_mime = upload_file(api_key, video_path, mime_type, display_name, max_retries, backoff)
            if cache_lock is not None:
                with cache_lock:
                    file_cache[key] = {"uri": file_uri, "mime": file_mime}
                    save_file_cache(cache_dir, file_cache)
            else:
                file_cache[key] = {"uri": file_uri, "mime": file_mime}
                save_file_cache(cache_dir, file_cache)
        return call_with_retry(generate_with_file_uri, api_key, model, file_uri, file_mime, prompt_text)

    return call_with_retry(generate_with_inline, api_key, model, video_path, mime_type, prompt_text)


def main() -> int:
    args = parse_args()
    if not args.api_key:
        args.api_key = load_api_key(DEFAULT_KEY_PATH)
    max_workers = 10

    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=log_handlers,
    )

    print("=" * 80)
    print("Step 1: Generate Video Captions - Gemini API")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Input: {args.input_data}")
    print(f"Output: {args.output_path}")
    print(f"Cache Dir: {args.cache_dir or 'disabled'}")
    print(f"Inline Threshold MB: {args.inline_threshold_mb}")
    print(f"Use File API: {args.use_file_api}")
    print(f"Resume: {args.resume}")
    print(f"Dry Run: {args.dry_run}")
    print(f"Log File: {args.log_file or 'stdout only'}")
    print(f"Workers: {max_workers}")
    print("=" * 80)

    if not args.dry_run and not args.api_key:
        print("[ERROR] Gemini API key is required (use --api-key or apikey.txt)")
        return 1

    test_data = load_json(args.input_data)
    existing_predictions = load_existing_predictions(args.output_path) if args.resume else {}

    completed = []
    pending = []
    for item in test_data:
        if item["video_id"] in existing_predictions:
            pred = existing_predictions[item["video_id"]]
            if args.retry_errors and str(pred.get("predicted_caption", "")).startswith("[ERROR"):
                pending.append(item)
            else:
                completed.append(pred)
        else:
            pending.append(item)

    print(f"Total samples: {len(test_data)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")

    file_cache = load_file_cache(args.cache_dir) if args.cache_dir else {}
    cache_lock = threading.Lock()
    results_lock = threading.Lock()

    results: List[Dict[str, Any]] = []
    results.extend(completed)

    def worker(item: Dict[str, Any]) -> Dict[str, Any]:
        video_id = item.get("video_id", "unknown")
        try:
            logging.info("Processing %s", video_id)
            caption = process_item(
                item,
                args.api_key,
                args.model,
                args.cache_dir,
                file_cache,
                cache_lock,
                args.inline_threshold_mb,
                args.use_file_api,
                args.max_retries,
                args.retry_backoff,
                args.dry_run,
            )
            logging.info("Gemini caption for %s: %s", video_id, caption)
            return {
                "video_id": item["video_id"],
                "video_path": item["video_path"],
                "predicted_caption": caption,
                "ground_truth_caption": item.get("ground_truth_caption", ""),
                "questions": item.get("questions", {}),
                "ground_truth_answers": item.get("ground_truth_answers", {}),
            }
        except Exception as e:
            logging.error("Error on %s: %s", video_id, str(e))
            return {
                "video_id": item.get("video_id", "unknown"),
                "video_path": item.get("video_path", ""),
                "predicted_caption": f"[ERROR: {str(e)}]",
                "ground_truth_caption": item.get("ground_truth_caption", ""),
                "questions": item.get("questions", {}),
                "ground_truth_answers": item.get("ground_truth_answers", {}),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        with tqdm(total=len(futures), desc="Gemini Step1", unit="video") as pbar:
            for future in as_completed(futures):
                pred = future.result()
                with results_lock:
                    results.append(pred)
                    save_json(args.output_path, results)
                    logging.info("Progress: %d/%d (saved)", len(results) - len(completed), len(pending))
                pbar.update(1)

    save_json(args.output_path, results)
    logging.info("Saved %d predictions to %s", len(results), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

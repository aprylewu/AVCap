#!/usr/bin/env python3
"""
Upload this folder to a Hugging Face Hub repo.

Usage:
  export HF_TOKEN=hf_xxx
  python upload_hf.py --repo Apryle/AVCap-30B
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi
try:
    from huggingface_hub import CommitOperationDelete  # type: ignore
except Exception:  # pragma: no cover - older hub versions
    from huggingface_hub._commit_api import CommitOperationDelete  # type: ignore


DEFAULT_IGNORE = [
    ".git/**",
    ".hg/**",
    ".svn/**",
    "__pycache__/**",
    "*.pyc",
    ".DS_Store",
    ".env",
    ".env.*",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub.")
    p.add_argument("--repo", required=True, help="Repo ID, e.g. Apryle/AVCap-30B")
    p.add_argument(
        "--folder",
        default=None,
        help="Folder to upload (default: directory containing this script)",
    )
    p.add_argument("--branch", default="main", help="Target branch (default: main)")
    p.add_argument(
        "--path-in-repo",
        default="caption_pipeline",
        help="Upload folder destination within the repo (default: caption_pipeline)",
    )
    p.add_argument(
        "--commit-message",
        default="Upload avcaption code",
        help="Commit message for the upload",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token (default: HF_TOKEN env var)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist",
    )
    p.add_argument(
        "--include-env",
        action="store_true",
        help="Include .env files (excluded by default)",
    )
    p.add_argument(
        "--delete-remote",
        action="store_true",
        help="Delete existing files under --delete-prefix before upload",
    )
    p.add_argument(
        "--delete-prefix",
        default="",
        help="Remote path prefix to delete (default: empty = delete all files)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Missing HF token. Set HF_TOKEN or pass --token.")

    folder = Path(args.folder) if args.folder else Path(__file__).resolve().parent
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    ignore: List[str] = list(DEFAULT_IGNORE)
    if args.include_env:
        ignore = [p for p in ignore if not p.startswith(".env")]

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    )

    if args.delete_remote:
        prefix = args.delete_prefix.strip()
        if prefix in (".", "./"):
            prefix = ""
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        files = api.list_repo_files(
            repo_id=args.repo,
            repo_type="model",
            revision=args.branch,
        )
        to_delete = [
            f for f in files if (not prefix or f.startswith(prefix))
        ]
        if to_delete:
            ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
            api.create_commit(
                repo_id=args.repo,
                repo_type="model",
                revision=args.branch,
                operations=ops,
                commit_message=f"Delete {prefix or 'all'} before upload",
            )

    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(folder),
        path_in_repo=args.path_in_repo,
        revision=args.branch,
        commit_message=args.commit_message,
        ignore_patterns=ignore,
    )

    print(
        f"Uploaded {folder} to https://huggingface.co/{args.repo} "
        f"(branch: {args.branch}, path: {args.path_in_repo})"
    )


if __name__ == "__main__":
    main()

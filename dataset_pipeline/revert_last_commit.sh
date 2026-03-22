#!/usr/bin/env bash
set -euo pipefail

REPO_ID=""
BRANCH="main"
YES="no"
TMP_DIR=""

usage() {
  cat <<EOF
Revert the last commit on a Hugging Face repo branch (force push).

Usage:
  HF_TOKEN=hf_xxx ./revert_last_commit.sh --repo Apryle/AVCap-30B [--branch main] [--tmp /tmp/hf_revert] --yes

Notes:
  - Uses GIT_LFS_SKIP_SMUDGE=1 to avoid downloading large LFS files.
  - This rewrites history on the target branch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ID="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --tmp) TMP_DIR="$2"; shift 2 ;;
    --yes) YES="yes"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${REPO_ID}" ]]; then
  echo "Missing --repo" >&2
  usage
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Missing HF_TOKEN env var" >&2
  exit 2
fi

if [[ "${YES}" != "yes" ]]; then
  echo "Refusing to proceed without --yes (this will force-push)" >&2
  exit 2
fi

if [[ -z "${TMP_DIR}" ]]; then
  TMP_DIR="$(mktemp -d -t hf_revert_XXXXXX)"
fi

export GIT_LFS_SKIP_SMUDGE=1

git clone --branch "${BRANCH}" "https://${HF_TOKEN}@huggingface.co/${REPO_ID}" "${TMP_DIR}"
cd "${TMP_DIR}"

echo "Last two commits on ${BRANCH}:"
git --no-pager log -n 2 --oneline

git reset --hard HEAD~1
git push --force origin "${BRANCH}"

echo "Reverted last commit on ${REPO_ID} (${BRANCH})."

#!/bin/bash
# Retry HF model download until it completes. Resume on each attempt.
#
# Usage:
#   bash env_tests/retry_hf_download.sh <model-id>
#
# Example:
#   bash env_tests/retry_hf_download.sh Qwen/Qwen2.5-Math-1.5B
#                                       mlx-community/Qwen2.5-Math-1.5B-bf16

set -u

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-id>" >&2
  exit 2
fi
MODEL="$1"
HF=~/.venv-vllm-metal/bin/hf
export HF_XET_HIGH_PERFORMANCE=1

attempt=1
while true; do
  echo "=== Attempt $attempt at $(date) for $MODEL ==="
  if $HF download "$MODEL" 2>&1; then
    echo "=== SUCCESS on attempt $attempt for $MODEL ==="
    exit 0
  fi
  echo "=== Attempt $attempt failed for $MODEL, retrying in 5s ==="
  attempt=$((attempt+1))
  sleep 5
done

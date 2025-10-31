#!/bin/bash
# Inference script for GPT-5-chat model
# Usage: ./run_gpt5_chat.sh [additional_args]

set -e

MODEL_NAME="gpt-5-chat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

# Default arguments
DEFAULT_ARGS=(
    --model_name "${MODEL_NAME}"
    --benchmark_file benchmark/text/benchmark.csv
    --eval_mode mcq
    --result_path "results/text/${MODEL_NAME}"
    --size 32k
)

# Run inference with default args plus any additional args passed to script
python inference.py "${DEFAULT_ARGS[@]}" "$@"

echo "Inference completed for ${MODEL_NAME} with text mode"

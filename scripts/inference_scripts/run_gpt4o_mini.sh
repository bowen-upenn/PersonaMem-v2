#!/bin/bash
# Inference script for GPT-4o-mini model
# Usage: ./run_gpt4o_mini.sh [additional_args]

set -e

MODEL_NAME="gpt-4o-mini"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

# Default arguments
DEFAULT_ARGS=(
    --model_name "${MODEL_NAME}"
    --eval_mode mcq
    --result_path "results/${MODEL_NAME}/"
)

# Run inference with default args plus any additional args passed to script
python inference.py "${DEFAULT_ARGS[@]}" "$@"

echo "Inference completed for ${MODEL_NAME}"

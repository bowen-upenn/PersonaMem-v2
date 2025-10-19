#!/bin/bash
# Inference script for Claude 3.5 Haiku model
# Usage: ./run_claude_haiku.sh [additional_args]

set -e

MODEL_NAME="claude-3-5-haiku-20241022"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

# Default arguments
DEFAULT_ARGS=(
    --model_name "${MODEL_NAME}"
    --benchmark_file benchmark/multimodal/benchmark.csv
    --eval_mode mcq
    --use_multimodal
    --result_path "results/multimodal/${MODEL_NAME}"
    --size 128k
)

# Run inference with default args plus any additional args passed to script
python inference.py "${DEFAULT_ARGS[@]}" "$@"

echo "Inference completed for ${MODEL_NAME} with multimodal mode"


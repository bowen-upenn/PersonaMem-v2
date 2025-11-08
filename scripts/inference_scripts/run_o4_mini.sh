#!/bin/bash
set -e

MODEL_NAME="o4-mini"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

# Default arguments
DEFAULT_ARGS=(
    --model_name "${MODEL_NAME}"
    --benchmark_file data/benchmark/multimodal/benchmark.csv
    --eval_mode both
    --use_multimodal
    --result_path "results/multimodal/${MODEL_NAME}"
    --size 128k
)

# Run inference with default args plus any additional args passed to script
python inference.py "${DEFAULT_ARGS[@]}" "$@"

echo "Inference completed for ${MODEL_NAME} with multimodal mode"

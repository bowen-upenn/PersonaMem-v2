#!/bin/bash
set -e

MODEL_NAME="gpt-5-chat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEM0_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MEM0_DIR}/.." && pwd)"

cd "${MEM0_DIR}"

echo "Running Mem0 inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

python inference_mem0.py \
    --model_name "${MODEL_NAME}" \
    --benchmark_file "${PROJECT_ROOT}/data/benchmark/multimodal/benchmark.csv" \
    --config_path "${PROJECT_ROOT}/config.yaml" \
    --eval_mode both \
    --result_path "${PROJECT_ROOT}/results/mem0/${MODEL_NAME}" \
    --size both \
    --max_items 1000 \
    --parallel 1 \
    --mem0_top_k 10 \
    "$@"

echo "Mem0 inference completed for ${MODEL_NAME}"

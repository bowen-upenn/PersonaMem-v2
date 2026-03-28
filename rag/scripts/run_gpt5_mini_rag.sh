#!/bin/bash
set -e

MODEL_NAME="gpt-5-mini"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running RAG inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

DEFAULT_ARGS=(
    --parquet_file "rag/data/benchmark_text_32k_mcq.parquet"
    --model_name "${MODEL_NAME}"
    --result_path "rag/results/${MODEL_NAME}"
    --eval_mode mcq
    --parallel 4
)

python rag/inference_rag.py "${DEFAULT_ARGS[@]}" "$@"

echo "RAG inference completed for ${MODEL_NAME}"

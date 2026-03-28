#!/bin/bash
# Full RAG pipeline: preprocess (32k + 128k, RAG-only + RAG+full-context) → MCQ + open-ended inference
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p rag/data rag/results rag/logs

echo "============================================================"
echo "PersonaMem-v2 RAG Pipeline"
echo "Project root: ${PROJECT_ROOT}"
echo "Started: $(date)"
echo "============================================================"

# ---- Step 1: Download data from HuggingFace ----
if [ ! -d "data/benchmark" ]; then
    echo ""
    echo "[Step 1/4] Downloading benchmark data from HuggingFace..."
    python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="bowen-upenn/PersonaMem-v2",
    repo_type="dataset",
    local_dir="data",
    ignore_patterns=["*.git*"],
)
print("Download complete.")
EOF
else
    echo "[Step 1/4] Data already present, skipping download."
fi

# ---- Step 2a: RAG preprocessing — 32k ----
PARQUET_32K_MCQ="rag/data/benchmark_text_32k_mcq.parquet"
if [ ! -f "${PARQUET_32K_MCQ}" ]; then
    echo ""
    echo "[Step 2a/4] Running RAG preprocessing (32k)..."
    python verl_custom/rag.py \
        --benchmark_csv data/benchmark/text/benchmark.csv \
        --output_dir rag/data \
        --chunk_size 6 \
        --chunk_overlap 2 \
        --top_k 10 \
        --path_prefix data/ \
        --context_size 32k \
        2>&1 | tee rag/logs/rag_preprocess_32k.log
    echo "RAG preprocessing (32k) complete."
else
    echo "[Step 2a/4] 32k RAG parquet already exists, skipping."
fi

PARQUET_32K_OE="rag/data/benchmark_text_32k.parquet"

# ---- Step 2b: RAG preprocessing — 128k ----
PARQUET_128K_MCQ="rag/data/benchmark_text_128k_mcq.parquet"
PARQUET_128K_OE="rag/data/benchmark_text_128k.parquet"
if [ ! -f "${PARQUET_128K_MCQ}" ]; then
    echo ""
    echo "[Step 2b/4] Running RAG preprocessing (128k)..."
    python verl_custom/rag.py \
        --benchmark_csv data/benchmark/text/benchmark.csv \
        --output_dir rag/data \
        --chunk_size 6 \
        --chunk_overlap 2 \
        --top_k 10 \
        --path_prefix data/ \
        --context_size 128k \
        2>&1 | tee rag/logs/rag_preprocess_128k.log
    echo "RAG preprocessing (128k) complete."
else
    echo "[Step 2b/4] 128k RAG parquet already exists, skipping."
fi

for CONTEXT in 32k 128k; do
    if [ "${CONTEXT}" = "32k" ]; then
        PARQUET_MCQ="${PARQUET_32K_MCQ}"
        PARQUET_OE="${PARQUET_32K_OE}"
    else
        PARQUET_MCQ="${PARQUET_128K_MCQ}"
        PARQUET_OE="${PARQUET_128K_OE}"
    fi

    for MODEL in gpt-5-chat gpt-5-mini; do
        # ---- MCQ: RAG-only ----
        echo ""
        echo "MCQ inference: ${MODEL} | ${CONTEXT} | rag-only"
        python rag/inference_rag.py \
            --parquet_file "${PARQUET_MCQ}" \
            --model_name "${MODEL}" \
            --result_path "rag/results/${MODEL}/${CONTEXT}" \
            --eval_mode mcq \
            --context_size "${CONTEXT}" \
            --parallel 4 \
            2>&1 | tee "rag/logs/inference_${MODEL//-/_}_${CONTEXT}_mcq.log"

        # ---- MCQ: RAG + full context ----
        echo ""
        echo "MCQ inference: ${MODEL} | ${CONTEXT} | rag+full"
        python rag/inference_rag.py \
            --parquet_file "${PARQUET_MCQ}" \
            --model_name "${MODEL}" \
            --result_path "rag/results/${MODEL}/${CONTEXT}_full" \
            --eval_mode mcq \
            --context_size "${CONTEXT}" \
            --full_context \
            --parallel 4 \
            2>&1 | tee "rag/logs/inference_${MODEL//-/_}_${CONTEXT}_full_mcq.log"

        # ---- Open-ended: RAG-only ----
        echo ""
        echo "Open-ended inference: ${MODEL} | ${CONTEXT} | rag-only"
        python rag/inference_rag.py \
            --parquet_file "${PARQUET_OE}" \
            --model_name "${MODEL}" \
            --result_path "rag/results/${MODEL}/${CONTEXT}" \
            --eval_mode generative \
            --context_size "${CONTEXT}" \
            --parallel 4 \
            2>&1 | tee "rag/logs/inference_${MODEL//-/_}_${CONTEXT}_oe.log"

        # ---- Open-ended: RAG + full context ----
        echo ""
        echo "Open-ended inference: ${MODEL} | ${CONTEXT} | rag+full"
        python rag/inference_rag.py \
            --parquet_file "${PARQUET_OE}" \
            --model_name "${MODEL}" \
            --result_path "rag/results/${MODEL}/${CONTEXT}_full" \
            --eval_mode generative \
            --context_size "${CONTEXT}" \
            --full_context \
            --parallel 4 \
            2>&1 | tee "rag/logs/inference_${MODEL//-/_}_${CONTEXT}_full_oe.log"
    done
done

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "Results in rag/results/"
echo "============================================================"

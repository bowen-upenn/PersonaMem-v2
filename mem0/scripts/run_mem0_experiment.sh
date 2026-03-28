#!/bin/bash
# Mem0 experiment runner for PersonaMem-v2
# Runs gpt-5-chat with Mem0 memory on the benchmark (MCQ + generative, 32k + 128k)
# Usage: bash run_mem0_experiment.sh [--max_items N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEM0_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MEM0_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
DATA_DIR="${PROJECT_ROOT}/data"
BENCHMARK_FILE="${DATA_DIR}/benchmark/multimodal/benchmark.csv"
RESULTS_DIR="${PROJECT_ROOT}/results/mem0/gpt-5-chat"
LOG_FILE="${RESULTS_DIR}/run.log"

MAX_ITEMS="${1:-1000}"
# Parse --max_items argument if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_items) MAX_ITEMS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo " PersonaMem-v2 Mem0 Experiment"
echo " Started: $(date)"
echo " Project root: ${PROJECT_ROOT}"
echo " Max items: ${MAX_ITEMS}"
echo "=============================================="

# ── Step 1: Download data if not present ─────────────────────────────────────
if [ ! -f "${BENCHMARK_FILE}" ]; then
    echo ""
    echo "[Step 1] Downloading benchmark data from HuggingFace..."
    conda run -n personamem-v2 pip install huggingface_hub -q
    conda run -n personamem-v2 python "${MEM0_DIR}/download_data.py"

else
    echo "[Step 1] Benchmark data already present at ${BENCHMARK_FILE}, skipping download."
fi

# ── Step 2: Verify benchmark file exists ─────────────────────────────────────
if [ ! -f "${BENCHMARK_FILE}" ]; then
    echo "ERROR: Benchmark file not found at ${BENCHMARK_FILE} after download attempt."
    echo "Please check your HuggingFace access or download manually."
    exit 1
fi

echo ""
echo "[Step 2] Benchmark file confirmed: ${BENCHMARK_FILE}"
echo "         $(wc -l < "${BENCHMARK_FILE}") rows (including header)"

# ── Step 3: Run Mem0 inference ────────────────────────────────────────────────
echo ""
echo "[Step 3] Running Mem0 inference with gpt-5-chat..."
echo "         eval_mode=both, size=both, parallel=1"

cd "${PROJECT_ROOT}"

# Clear stale Qdrant local DB from previous runs
rm -rf /tmp/qdrant_personamem

conda run -n personamem-v2 \
    --no-capture-output \
    python "${MEM0_DIR}/inference_mem0.py" \
        --model_name "gpt-5-chat" \
        --benchmark_file "${BENCHMARK_FILE}" \
        --config_path "${PROJECT_ROOT}/config.yaml" \
        --eval_mode both \
        --result_path "${RESULTS_DIR}" \
        --size both \
        --max_items "${MAX_ITEMS}" \
        --parallel 1 \
        --mem0_top_k 10 \
        --mem0_llm "gpt-5-chat" \
        --mem0_embedding "text-embedding-3-large"

echo ""
echo "=============================================="
echo " Experiment complete: $(date)"
echo " Results saved to: ${RESULTS_DIR}"
echo "=============================================="

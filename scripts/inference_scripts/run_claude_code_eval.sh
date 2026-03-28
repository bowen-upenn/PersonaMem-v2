#!/bin/bash
# Initialize a PersonaMem-v2 Claude Code eval run.
#
# Usage:
#   bash scripts/inference_scripts/run_claude_code_eval.sh <model_name> [max_items] [subagent_model]
#
# Examples:
#   bash scripts/inference_scripts/run_claude_code_eval.sh claude-opus-4-6 1000
#   bash scripts/inference_scripts/run_claude_code_eval.sh claude-sonnet-4-6 5 sonnet
set -e

MODEL_NAME="${1:-claude-opus-4-6}"
MAX_ITEMS="${2:-}"
SUBAGENT_MODEL="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RESULT_PATH="results/multimodal/${MODEL_NAME}-codeeval"

INIT_ARGS=(
    init
    --benchmark_file data/benchmark/multimodal/benchmark.csv
    --result_path "${RESULT_PATH}"
    --eval_mode both
    --size both
    --model_name "${MODEL_NAME}"
)

if [ -n "${MAX_ITEMS}" ]; then
    INIT_ARGS+=(--max_items "${MAX_ITEMS}")
fi

if [ -n "${SUBAGENT_MODEL}" ]; then
    INIT_ARGS+=(--subagent_model "${SUBAGENT_MODEL}")
fi

echo "Initializing eval run for model: ${MODEL_NAME}"
echo "Result path: ${RESULT_PATH}"
python3 eval_orchestrator.py "${INIT_ARGS[@]}"

echo ""
echo "Initialized. Now run the eval loop inside Claude Code:"
echo ""
echo "  /persona-eval ${RESULT_PATH}"
echo ""
echo "Or manually:"
echo "  python3 eval_orchestrator.py next --result_path ${RESULT_PATH}"

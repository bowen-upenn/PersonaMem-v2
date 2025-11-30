#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Default CSV file path
DEFAULT_CSV_PATH="results/multimodal/gpt-5-mini/evaluation_results_both_multimodal_11182025_021532.csv"

# Get CSV file path from command line argument or use default
CSV_PATH="${1:-$DEFAULT_CSV_PATH}"

# Check if file exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    echo ""
    echo "Usage: $0 [path_to_results_csv]"
    exit 1
fi

echo "Running judge evaluation on results..."
echo "Project root: ${PROJECT_ROOT}"
echo "Results CSV: ${CSV_PATH}"
echo ""

# Run judge evaluation
python inference.py \
    --run_judges \
    --results_csv_path "${CSV_PATH}" \
    --verbose \
    --max_items 500

echo ""
echo "Judge evaluation completed!"
echo "Results have been updated in: ${CSV_PATH}"

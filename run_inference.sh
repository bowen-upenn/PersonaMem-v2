#!/bin/bash

# Inference script runner for Azure OpenAI models
# Usage: ./run_inference.sh [benchmark_file] [model_name] [batch_size] [max_items]

# Default values
BENCHMARK_FILE=${1:-"data/benchmark/benchmark_32k.csv"}
MODEL_NAME=${2:-"gpt-4"}
BATCH_SIZE=${3:-10}
MAX_ITEMS=${4:-100}  # Start with 100 items for testing

echo "Running inference with:"
echo "  Benchmark file: $BENCHMARK_FILE"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Max items: $MAX_ITEMS"
echo ""

# Note: Environment variables are loaded from .env file by the Python script
echo "Environment variables will be loaded from .env file"

# Run the inference script
python inference.py \
    --benchmark_file "$BENCHMARK_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --max_items "$MAX_ITEMS"

echo ""
echo "Inference completed!"

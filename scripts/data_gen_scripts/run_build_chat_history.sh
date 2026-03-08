#!/bin/bash

# Loop through both 32k and 128k versions
# for version in 128k; do
for version in 32k 128k; do
  echo "============================================"
  echo "Building chat history for version: $version"
  echo "============================================"

  PYTHONPATH=. python data_generation/main.py \
    --model gpt-5-chat \
    --step build_chat_history \
    --conv_output_dir data/raw_data/ \
    --persona_start_idx 0 \
    --persona_end_idx -1 \
    --version $version \
    # --verbose
  
  echo "Completed version: $version"
  echo ""
done

echo "All versions completed!"

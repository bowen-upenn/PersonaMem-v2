#!/bin/bash

python main.py \
  --model gpt-4.1 \
  --step categorize_topics \
  --conv_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx -1 \
  --rate_limit_per_min 10 \
  --parallel \
  --refresh_mem 100
  # --verbose

# Example usage:
# ./scripts/run_categorize_topics.sh

# To categorize topics for specific persona range:
# python main.py --step categorize_topics --persona_start_idx 0 --persona_end_idx 50 --parallel --verbose

# To categorize topics sequentially:
# python main.py --step categorize_topics --conv_output_dir data/raw_data/ --verbose

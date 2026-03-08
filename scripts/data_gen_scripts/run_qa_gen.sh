#!/bin/bash

PYTHONPATH=. python data_generation/main.py \
  --model gpt-5-chat \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx -1 \
  --validate_qa \
  --rate_limit_per_min 10 \
  --parallel \

# Examples of specifying persona ranges:
# --persona_start_idx 0 --persona_end_idx 49  # Process personas 0-49
# --persona_start_idx 50 --persona_end_idx 99  # Process personas 50-99
# --persona_start_idx -1 --persona_end_idx -1  # Process all personas (default)
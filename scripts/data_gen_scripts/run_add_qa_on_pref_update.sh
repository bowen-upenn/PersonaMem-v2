#!/bin/bash

PYTHONPATH=. python data_generation/main.py \
  --model gpt-5-chat \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx 20 \
  --persona_end_idx -1 \
  --validate_qa \
  --add_more_minority \
  --rate_limit_per_min 10 \
  --parallel \
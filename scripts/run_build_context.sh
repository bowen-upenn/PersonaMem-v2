#!/bin/bash

python main.py \
  --model gpt-4.1 \
  --step build_context \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --verbose

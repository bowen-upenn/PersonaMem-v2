#!/bin/bash

python main.py \
  --model gpt-5-chat \
  --step build_chat_history \
  --conv_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx -1 \
  --verbose

#!/bin/bash

python main.py \
  --model gpt-4.1 \
  --step build_context \
  --conv_output_path data/interactions.json \
  --qa_output_path data/interactions.json \
  --verbose

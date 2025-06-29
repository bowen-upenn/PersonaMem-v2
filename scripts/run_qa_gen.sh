#!/bin/bash

python main.py \
  --model gpt-4.1 \
  --step generate_qa \
  --conv_output_path data/interactions.json \
  --qa_output_path data/interactions.json \
  --verbose

#!/bin/bash

python main.py \
  --model gpt-4.1 \
  --conv_output_path data/interactions.jsonl \
  --result_path results/ \
  --num_persona 1 \
  --data_types email creative_writing professional_writing chat_message \
  --context_length 32000 \
  --clean \
  --verbose

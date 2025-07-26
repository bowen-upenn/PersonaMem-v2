#!/bin/bash

python main.py \
  --model o4_mini \
  --step generate_data \
  --conv_output_dir data/raw_data/ \
  --num_persona 1000 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult knowledge_query \
  --rate_limit_per_min 10 \
  --parallel

python main.py \
  --model o4-mini \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx -1 \
  --persona_end_idx -1 \
  --rate_limit_per_min 10 \
  --parallel
#!/bin/bash

python main.py \
  --model gpt-5-chat \
  --step generate_conv \
  --conv_output_dir data/raw_data/ \
  --num_persona 1000 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult social_media_post knowledge_query \
  --self_verify \
  --rate_limit_per_min 5 \
  --parallel

python main.py \
  --model gpt-5-chat \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx -1 \
  --persona_end_idx -1 \
  --validate_qa \
  --rate_limit_per_min 5 \
  --parallel

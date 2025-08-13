#!/bin/bash

# python main.py \
#   --model o3_mini \
#   --step generate_convo \
#   --conv_output_dir data/raw_data/ \
#   --num_persona 1000 \
#   --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult social_media_post knowledge_query \
#   --rate_limit_per_min 20 \
#   --parallel

# python main.py \
#   --model o4-mini \
#   --step generate_qa \
#   --conv_output_dir data/raw_data/ \
#   --qa_output_dir data/raw_data/ \
#   --persona_start_idx -1 \
#   --persona_end_idx -1 \
#   --validate_qa \
#   --rate_limit_per_min 20 \
#   --parallel


python main.py \
  --model o3_mini \
  --step generate_convo \
  --conv_output_dir data/raw_data/ \
  --num_persona 1 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult social_media_post knowledge_query \
  --verbose
  # --rate_limit_per_min 20 \
  # --parallel

python main.py \
  --model o3-mini \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx 0 \
  --validate_qa \
  --verbose
  # --rate_limit_per_min 20 \
  # --parallel

python main.py \
  --model gpt-4.1 \
  --step categorize_topics \
  --conv_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx 0 \
  --refresh_mem 100 \
  --verbose
  # --rate_limit_per_min 10 \
  # --parallel \
#!/bin/bash

python main.py \
  --model o4_mini \
  --step generate_convo \
  --conv_output_dir data/raw_data/ \
  --num_persona 114 \
  --data_types personal_email professional_email social_media_post \
  --rate_limit_per_min 10 \
  --parallel

  # --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult knowledge_query \
#!/bin/bash

python main.py \
  --model gpt-5-chat \
  --step generate_conv \
  --conv_output_dir data/raw_data/ \
  --num_persona 1000 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult social_media_post knowledge_query \
  --rate_limit_per_min 5 \
  --parallel

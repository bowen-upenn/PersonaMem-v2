#!/bin/bash

python main.py \
  --model o4-mini \
  --step generate_data \
  --conv_output_path data/interactions.json \
  --num_persona 100 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult knowledge_query \
  --clean

  # --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult knowledge_query \
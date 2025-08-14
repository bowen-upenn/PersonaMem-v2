#!/bin/bash

# If you want to add new keys to the persona, you can specify them after persona_keys_to_add.
# This script will not modify or regenerate any existing conversations,
# but only create new keys and conversations for the new key under all data types specified after data_types.
# Existing personas will be loaded, not overwitten.
# python main.py \
#   --model gpt-5-chat \
#   --step update_conv \
#   --conv_output_dir data/raw_data/ \
#   --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult knowledge_query social_media_post \
#   --persona_keys_to_add health_and_medical_conditions \
#   --persona_start_idx 0 \
#   --persona_end_idx -1 \
#   --rate_limit_per_min 5 \
#   --parallel

# If you want to regenerate specific data types, you can specify them after data_types, without adding persona_keys_to_add.
# This script will remove existing conversations under these data types.
# You can also add new data types. 
# Existing personas will be loaded, not overwitten.
python main.py \
  --model gpt-5-chat \
  --step update_conv \
  --conv_output_dir data/raw_data/ \
  --data_types translation trouble_consult chat_message social_media_post \
  --persona_start_idx 0 \
  --persona_end_idx -1 \
  --rate_limit_per_min 5 \
  --parallel

# Examples of specifying persona ranges:
# --persona_start_idx 0 --persona_end_idx 49  # Update personas 0-49
# --persona_start_idx 50 --persona_end_idx 99  # Update personas 50-99
# --persona_start_idx 800 --persona_end_idx -1  # Update personas 800 to end
# --persona_start_idx -1 --persona_end_idx -1  # Update all personas (default)

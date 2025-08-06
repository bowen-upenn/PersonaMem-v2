#!/bin/bash

python main.py \
  --model o3_mini \
  --step update_conv \
  --conv_output_dir data/raw_data/ \
  --data_types personal_email professional_email social_media_post \
  --persona_start_idx 0 \
  --persona_end_idx -1 \

# Examples of specifying persona ranges:
# --persona_start_idx 0 --persona_end_idx 49  # Update personas 0-49
# --persona_start_idx 50 --persona_end_idx 99  # Update personas 50-99
# --persona_start_idx 800 --persona_end_idx -1  # Update personas 800 to end
# --persona_start_idx -1 --persona_end_idx -1  # Update all personas (default)

#!/bin/bash

# Script to run add_pref_others functionality
# This script processes existing JSON files to regenerate conversations for "others" preferences
# 
# What it does:
# - Iterates through existing persona files in data/raw_data/
# - For each conversation item where who="self" and updated=False
# - Checks if the preference appears in any "prev_pref" in the file
# - If not, with 30% probability, regenerates conversations and QA using who="others"
# - Removes existing who, conversations, user_query, topic_query, correct_answer, incorrect_answers
# - Generates new conversations from "others" perspective and corresponding QA pairs

# Basic usage - process all personas
python main.py \
  --model gpt-5-chat \
  --step add_pref_others \
  --conv_output_dir data/raw_data/ \
  --persona_start_idx 166 \
  --persona_end_idx -1 \
  --rate_limit_per_min 2 \
  --parallel \

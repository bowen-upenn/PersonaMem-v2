#!/bin/bash

# Script to run add_sensitive_info functionality
# This script processes existing JSON files to add sensitive information conversations
# 
# What it does:
# - Iterates through existing persona files in data/raw_data/
# - For each conversation item that doesn't have "sensitive_info" key
# - With 10% probability, randomly selects the item
# - Extracts random sensitive information from the persona's sensitive_information data
# - Generates new conversations involving the sensitive information
# - Generates corresponding QA pairs for the sensitive information conversations
# - Adds the new conversation element to the existing conversations

# Basic usage - process all personas
python main.py \
  --model o3-mini \
  --step add_sensitive_info \
  --conv_output_dir data/raw_data/ \
  --persona_start_idx 590 \
  --persona_end_idx -1 \
  --rate_limit_per_min 10 \
  --parallel \
#   --verbose

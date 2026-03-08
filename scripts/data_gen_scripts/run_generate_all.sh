#!/bin/bash

# Step 1: Prepare image embeddings database (required for multimodal persona matching)
PYTHONPATH=. python data_generation/image_matcher.py \
    --model gpt-5-chat \
    --recreate \
    --parallel \
    --rate_limit_per_min 5 \
    --verbose

# Step 2: Generate user personas, preferences, and conversations
PYTHONPATH=. python data_generation/main.py \
  --model gpt-5-chat \
  --step generate_conv \
  --conv_output_dir data/raw_data/ \
  --num_persona 1000 \
  --data_types personal_email professional_email creative_writing professional_writing chat_message translation trouble_consult social_media_post knowledge_query \
  --rate_limit_per_min 5 \
  --parallel

# Step 3: Generate Q&A pairs with quality validation
PYTHONPATH=. python data_generation/main.py \
  --model gpt-5-chat \
  --step generate_qa \
  --conv_output_dir data/raw_data/ \
  --qa_output_dir data/raw_data/ \
  --persona_start_idx 0 \
  --persona_end_idx -1 \
  --validate_qa \
  --rate_limit_per_min 5 \
  --parallel

# Step 4: Build chat history (32k and 128k token versions)
for version in 32k 128k; do
  echo "Building chat history for version: $version"
  PYTHONPATH=. python data_generation/main.py \
    --model gpt-5-chat \
    --step build_chat_history \
    --conv_output_dir data/raw_data/ \
    --persona_start_idx 0 \
    --persona_end_idx -1 \
    --version $version
  echo "Completed version: $version"
done

# Step 5: Build benchmark CSV for evaluation
PYTHONPATH=. python data_generation/prepare_benchmark.py 
  --split \
  --benchmark-size 5000 \
  --train-val-split 0.9 \
  --random-seed 42

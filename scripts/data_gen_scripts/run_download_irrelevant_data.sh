#!/bin/bash

PYTHONPATH=. python data_generation/download_irrelevant_data.py \
  --model o3-mini \
  --rate-limit-per-min 30 \
  --parallel \
  --process-queries \
  --add-code \
  --sample-size 1000

# PYTHONPATH=. python data_generation/download_irrelevant_data.py --add-code --model o3-mini --parallel --rate-limit-per-min 30

#!/bin/bash

python download_irrelevant_data.py \
  --model o3-mini \
  --rate-limit-per-min 20 \
  --parallel \
  --process-queries \
  --sample-size 1000

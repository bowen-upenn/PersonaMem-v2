#!/bin/bash

PYTHONPATH=. python data_generation/image_matcher.py \
    --model gpt-5-chat \
    --recreate \
    --parallel \
    --rate_limit_per_min 5 \
    --verbose
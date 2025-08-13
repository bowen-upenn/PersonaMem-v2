#!/bin/bash

python image_matcher.py \
    --model gpt-4.1 \
    --recreate \
    --parallel \
    --rate_limit_per_min 20 \
    --verbose
#!/bin/bash

# Script to build comprehensive benchmark CSV from raw persona data
# This script creates a single CSV file with one row per user_query for evaluation

echo "Starting benchmark preparation..."
echo "Input directory: data/raw_data"
echo "Output file: data/benchmark.csv"

PYTHONPATH=. python data_generation/prepare_benchmark.py --split --benchmark-size 5000 --train-val-split 0.9 --random-seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "Benchmark preparation completed successfully!"
    echo "Output file: data/benchmark.csv"
    
    # Show some basic stats about the generated file
    if [ -f "data/benchmark.csv" ]; then
        echo ""
        echo "=== File Statistics ==="
        echo "Total lines (including header): $(wc -l < data/benchmark.csv)"
        echo "Total data rows: $(($(wc -l < data/benchmark.csv) - 1))"
        echo "File size: $(du -h data/benchmark.csv | cut -f1)"
        
        echo ""
        echo "=== Column Headers ==="
        head -1 data/benchmark.csv | tr ',' '\n' | nl
        
        echo ""
        echo "Benchmark CSV is ready for evaluation!"
    fi
else
    echo ""
    echo "Error: Benchmark preparation failed!"
    exit 1
fi


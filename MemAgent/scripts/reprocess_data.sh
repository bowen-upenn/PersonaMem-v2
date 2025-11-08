#!/bin/bash
set -x

# Reprocess the ImplicitPersona dataset with system message filtering
# This script only regenerates the parquet files without retraining

echo "========================================="
echo "Reprocessing ImplicitPersona Dataset"
echo "Filtering out system messages from context"
echo "========================================="

cd "$(dirname "$0")/.."

python3 data/data_preprocess.py \
    --text_train_csv="../data/benchmark/text/train.csv" \
    --text_val_csv="../data/benchmark/text/val.csv" \
    --local_dir="data/implicit_persona" \
    --config_file="verl/trainer/config/ppo_trainer.yaml" \
    --script_file="run_qwen3_4b_grpo.sh" \
    --model_path="../verl_custom/ckpt_sft/global_step_400"

echo "========================================="
echo "Data reprocessing complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify the data with: python3 data/data_preprocess.py --check-only"
echo "2. Start training with: bash run_qwen3_4b_grpo.sh"

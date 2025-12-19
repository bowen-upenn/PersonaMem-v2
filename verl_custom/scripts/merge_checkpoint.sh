#!/bin/bash
set -e

# Configuration
# CHECKPOINT_DIR="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/global_step_200/actor"
# CHECKPOINT_DIR="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251112_203834/global_step_375/actor"
# CHECKPOINT_DIR="checkpoints/implicit_persona_verl_ablation_openonly/verl_qwen3_4b_grpo_20251129_171107/global_step_100/actor"
CHECKPOINT_DIR="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251129_165940/global_step_175/actor"
HF_MODEL_PATH="verl_custom/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
TARGET_DIR="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251129_165940/merged"

echo "==========================================="
echo "Merging FSDP checkpoint to HuggingFace format"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Reference model: $HF_MODEL_PATH"
echo "Target directory: $TARGET_DIR"
echo "==========================================="

# Create target directory
mkdir -p "$TARGET_DIR"

# # Clean up any existing merged model files to avoid conflicts
# echo "Cleaning up existing merged files..."
# rm -f "$TARGET_DIR"/model-*.safetensors
# rm -f "$TARGET_DIR"/model.safetensors.index.json
# rm -f "$TARGET_DIR"/*.json
# rm -f "$TARGET_DIR"/tokenizer*
# rm -f "$TARGET_DIR"/vocab*
# rm -f "$TARGET_DIR"/merges.txt

# Run the merger
python verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "$HF_MODEL_PATH" \
    --local_dir "$CHECKPOINT_DIR" \
    --target_dir "$TARGET_DIR"

echo "==========================================="
echo "Merge completed!"
echo "Merged model saved to: $TARGET_DIR"
echo "==========================================="

# Copy tokenizer files from original model
echo "Copying tokenizer files..."
cp "$HF_MODEL_PATH"/*.json "$TARGET_DIR/" 2>/dev/null || echo "No JSON files to copy"
cp "$HF_MODEL_PATH"/tokenizer* "$TARGET_DIR/" 2>/dev/null || echo "No tokenizer files to copy"
cp "$HF_MODEL_PATH"/vocab* "$TARGET_DIR/" 2>/dev/null || echo "No vocab files to copy"
cp "$HF_MODEL_PATH"/merges.txt "$TARGET_DIR/" 2>/dev/null || echo "No merges.txt to copy"

echo "==========================================="
echo "Tokenizer files copied!"
echo "You can now use $TARGET_DIR as a complete HuggingFace model"
echo "==========================================="
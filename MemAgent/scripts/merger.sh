CKPT=/path/to/your/ckpt/global_step_400
BASE=Qwen/Qwen2.5-7B-Instruct
# BASE=Qwen/Qwen2.5-14B-Instruct

TARGET=$CKPT/huggingface
python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $BASE \
    --local_dir $CKPT/actor \
    --target_dir $TARGET
cp $BASE/token*json $TARGET
cp $BASE/vocab.json $TARGET
# Original path in verl: verl/examples/ppo_trainer/run_deepseek7b_llm.sh
# Usage: ./run_qwen3_4b_inference.sh --model [base|sft|grpo] --data [ours|implicit_persona|personamem_v2|prefeval|longmemeval]

# Parse arguments
MODEL_TYPE="grpo"
DATA_TYPE="ours"
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_TYPE="$2"; shift 2 ;;
        --data) DATA_TYPE="$2"; shift 2 ;;
        *) break ;;
    esac
done

# Set model path
case $MODEL_TYPE in
    base) MODEL_PATH="verl_custom/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554" ;;
    sft) MODEL_PATH="verl_custom/ckpt_sft/global_step_400" ;;
    # grpo) MODEL_PATH="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/merged" ;;
    grpo) MODEL_PATH="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251112_203834/merged" ;;
    grpo_ablation_openonly) MODEL_PATH="checkpoints/implicit_persona_verl_ablation_openonly/verl_qwen3_4b_grpo_20251129_171107/merged" ;;
    *) echo "Error: Invalid --model '$MODEL_TYPE'. Use: base, sft, or grpo"; exit 1 ;;
esac

# Set data path
case $DATA_TYPE in
    ours|implicit_persona|personamem_v2) DATA_PATH="verl_custom/data/implicit_persona_benchmark/benchmark_text_32k.parquet" ;;
    prefeval) DATA_PATH="verl_custom/data/prefeval/prefeval_train.parquet" ;;
    longmemeval) DATA_PATH="verl_custom/data/longmemeval/longmemeval_s.parquet" ;;
    *) echo "Error: Invalid --data '$DATA_TYPE'. Use: ours, implicit_persona, personamem_v2, prefeval, or longmemeval"; exit 1 ;;
esac

echo "Model: $MODEL_TYPE | Data: $DATA_TYPE"

set -x  # Enable debug mode

python3 -m verl_custom.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=verl_custom/data/implicit_persona_rft/train_text_32k.parquet \
    data.val_files=$DATA_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=37000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.enable_thinking=True \
    reward_model.reward_manager=custom_naive \
    reward_model.eval_method=judge \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=39048 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=39048 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='implicit_persona_verl' \
    trainer.experiment_name='verl_qwen3_4b_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=0 $@ \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++critic.model.fsdp_config.model_dtype=bfloat16 \

# Original path in verl: verl/examples/ppo_trainer/run_deepseek7b_llm.sh

# Base: verl_custom/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
# SFT: verl_custom/ckpt_sft/global_step_400
# GRPO: checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/merged

set -x  # Enable debug mode

python3 -m verl_custom.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=verl_custom/data/implicit_persona_rft/train_text_32k.parquet \
    data.val_files=verl_custom/data/implicit_persona_benchmark/benchmark_text_32k.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=37000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.enable_thinking=True \
    reward_model.reward_manager=custom_naive \
    reward_model.eval_method=judge \
    actor_rollout_ref.model.path=checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/merged \
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

# set -x

# data_path==verl_custom/data/implicit_persona_benchmark/benchmark_text_32k.parquet
# save_path=verl_custom/data/implicit_persona_benchmark/
# model_path=checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/merged

# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=1 \
#     trainer.n_gpus_per_node=8 \
#     data.path=$data_path \
#     data.prompt_key=prompt \
#     data.n_samples=1 \
#     data.output_path=$save_path \
#     model.path=$model_path \
#     +model.trust_remote_code=True \
#     rollout.temperature=1.0 \
#     rollout.top_k=50 \
#     rollout.top_p=0.7 \
#     rollout.prompt_length=36000 \
#     rollout.response_length=2048 \
#     rollout.tensor_model_parallel_size=1 \
#     rollout.gpu_memory_utilization=0.8

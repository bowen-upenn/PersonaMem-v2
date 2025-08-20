# Original path in verl: verl/examples/ppo_trainer/run_deepseek7b_llm.sh

set -x  # Enable debug mode

python3 -m verl_custom.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=verl_custom/data/werewolf_game/train.parquet \
    data.val_files=verl_custom/data/werewolf_game/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=32000 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.enable_thinking=False \
    reward_model.reward_manager=custom_naive \
    actor_rollout_ref.model.path=verl_custom/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=36096 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=verl_custom/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.ppo_max_token_len_per_gpu=36096 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='implicit_persona_verl' \
    trainer.experiment_name='verl_qwen3_8b_ppo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.rollout_data_dir=rollout \
    trainer.total_epochs=5 $@
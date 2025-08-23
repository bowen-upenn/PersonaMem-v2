#!/bin/bash
set -x

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=1
NGPUS_PER_NODE=8
PROJ_ROOT=
DATASET_ROOT=../../verl_custom/data/implicit_persona/
MODEL_PATH=../../verl_custom/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
VAL_PATH="${DATASET_ROOT}/test.parquet"
TRAIN_PATH="${DATASET_ROOT}/train.parquet"
EXP=verl_memagent_implicit_persona_qwen3_8b_ppo
PROJ_DIR=${PROJ_ROOT}/${EXP}

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=32000 
MAX_NEW_TOKEN=4096
TOTAL_MAX=36096


python3 -m verl_custom.main_ppo \
    recurrent.enable=memory \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=True \
    trainer.save_freq=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=16 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager=custom_naive \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOTAL_MAX \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOTAL_MAX \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOTAL_MAX \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$TOTAL_MAX \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${EXP} \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=1 \
    trainer.test_freq_batch=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 \
    trainer.parallelize_sessions=False \
    # trainer.resume_mode=resume_path \
    # trainer.resume_from_path=checkpoints/verl_memagent_implicit_persona_qwen3_8b_ppo/verl_memagent_implicit_persona_qwen3_8b_ppo_20250822_120826/global_step_53

set -x

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl_custom.fsdp_sft_trainer \
    data.train_files=verl_custom/data/implicit_persona_sft/train.parquet \
    data.val_files=verl_custom/data/implicit_persona_sft/val.parquet \
    optim.lr=1e-5 \
    optim.clip_grad=0.5 \
    optim.warmup_steps_ratio=0.05 \
    data.train_batch_size=64 \
    data.micro_batch_size=8 \
    data.max_length=40960 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    model.partial_pretrain=verl_custom/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    trainer.default_local_dir=verl_custom/ckpt_sft \
    trainer.project_name='implicit_persona_verl' \
    trainer.experiment_name='verl_qwen3_4b_sft' \
    trainer.logger='["console","wandb"]' \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=4
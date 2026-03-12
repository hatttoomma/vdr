#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
TRAIN_FILE=${TRAIN_FILE:-"./mmsearch_data/train.parquet"}
VAL_FILE=${VAL_FILE:-"./mmsearch_data/train.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
ROLLOUT_N_PER_ITER=${ROLLOUT_N_PER_ITER:-1}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
VALIDATION_LOG_PATH=${VALIDATION_LOG_PATH:-"./logs/val_records.jsonl"}
VLLM_ATTENTION_BACKEND=FLASHINFER
NUM_GPUS=2

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=8 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  +actor_rollout_ref.rollout.n_per_iter="${ROLLOUT_N_PER_ITER}" \
  actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  custom_reward_function.path="$(pwd)/verl_grpo/reward_vdr.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function_val.path="$(pwd)/verl_grpo/reward_vdr.py" \
  +custom_reward_function_val.name=compute_score_ground_truth \
  reward_model.reward_manager=batch \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='vdr_qwen3vl_grpo' \
  trainer.experiment_name='qwen2.5-vl-3b-instruct_grpo' \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=2 \
  trainer.total_epochs=20 \
  +trainer.validation_log_path="${VALIDATION_LOG_PATH}" \
  "$@"

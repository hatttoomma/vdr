#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
TRAIN_FILE=${TRAIN_FILE:-"./data/vdr_bench_verl/train.parquet"}
VAL_FILE=${VAL_FILE:-"./data/vdr_bench_verl/val.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=64 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  custom_reward_function.path="$(pwd)/verl_grpo/reward_vdr.py" \
  custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='vdr_qwen3vl_grpo' \
  trainer.experiment_name='qwen3vl_2b_minimal_grpo' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  "$@"

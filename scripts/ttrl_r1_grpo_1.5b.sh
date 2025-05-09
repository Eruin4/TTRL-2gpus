set -x
export TRANSFORMERS_NO_FLASH_ATTN=1
export RAY_MAX_RUNTIME_ENV_SIZE_BYTES=100000000 
export CUDA_LAUNCH_BLOCKING=1
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0
# 메모리 관리를 위한 변수 추가
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=1000000000
# VLLM KV 캐시 크기 제한
export VLLM_GPU_MEMORY_UTILIZATION=0.3
# GPU 할당
export CUDA_VISIBLE_DEVICES=0,1
# Tear down any old cluster, then start fresh and local-only
ray stop || true
ray start --head --dashboard-port=8265

# …set up your ROOT_DIR, MODEL_DIR, etc…

ROOT_DIR=$1
WANDB_KTY=$2
# ------------------------------------------------------------

DATE=$(date +%m%d)
EXPERIMENT="TTRL"
ADVANTAGE="group_norm"
TASK="open_r1"
BACKBONE="Qwen/Qwen2.5-Math-1.5B-Instruct"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE//\//-}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}"

# ------------------------------------------------------------

EXP="${DATE}-${TASK}-${EXPERIMENT}-${BACKBONE//\//-}-${ADVANTAGE}"
LOG_FILE="${ROOT_DIR}/logs/${EXP}.log"

# 대신 로컬에서 직접 실행
cd ${ROOT_DIR}
python -m ttrl.cli.train_ppo_naive \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --colocate_critic_reward \
   --pretrain "${BACKBONE}" \
   --save_path "${OUTPUT_DIR}/model" \
   --verify_task "ttt" \
   --verify_task_eval "math" \
   --micro_train_batch_size 2 \
   --train_batch_size 2 \
   --num_episodes 10 \
   --save_steps 1 \
   --eval_steps 1 \
   --logging_steps 1 \
   --max_samples 100000 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 2 \
   --n_samples_per_prompt 2 \
   --n_votes_per_prompt 8 \
   --extra_eval_task "" \
   --training_mode "rl" \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --advantage_estimator ${ADVANTAGE} \
   --use_kl_loss \
   --temperature 1.0 \
   --eval_temperature 0.0 \
   --lambd 1.0 \
   --gamma 1.0 \
   --zero_stage 1 \
   --zpg 1 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.00 \
   --prompt_data "json@${ROOT_DIR}/data/${TASK}" \
   --input_key "prompt" \
   --label_key "answer" \
   --max_ckpt_num 10 \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --use_wandb ${WANDB_KTY} \
   --wandb_project TTRL \
   --wandb_run_name ${EXP} \
   --use_tensorboard "${ROOT_DIR}/logs/${EXP}" \
   --ckpt_path "${OUTPUT_DIR}/ckpt" \
#  > ${LOG_FILE} 2>&1 &

echo "Model Training started in background. Check logs at ${LOG_FILE}"

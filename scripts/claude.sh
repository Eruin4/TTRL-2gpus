set -x

# Increase both GCS storage limit and Dashboard HTTP server request limit
export RAY_MAX_RUNTIME_ENV_SIZE_BYTES=209715200
export RAY_AIOHTTP_CLIENT_MAX_SIZE=209715200 # Crucial for allowing larger uploads to the dashboard

# Tear down any old cluster, then start fresh and local-only
ray stop || true
# Ensure the Ray head starts with the new environment variable settings
ray start --head --dashboard-port=8265

# …set up your ROOT_DIR, MODEL_DIR, etc…

ROOT_DIR=$1
MODEL_DIR=$2
WANDB_KTY=$3
# ------------------------------------------------------------

DATE=$(date +%m%d)
EXPERIMENT="TTRL"
ADVANTAGE="group_norm"
TASK="open_r1"
BACKBONE="Qwen2.5-Math-1.5B-Instruct"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE}"
BACKBONE_PATH="${MODEL_DIR}/${BACKBONE}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}"

# ------------------------------------------------------------

EXP="${DATE}-${TASK}-${EXPERIMENT}-${BACKBONE}-${ADVANTAGE}"
LOG_FILE="${ROOT_DIR}/logs/${EXP}.log"

# Best Practice: Exclude the virtual environment and other unnecessary large files/dirs
# from the runtime environment package to reduce its size and speed up submission.
# The path "scripts/.venv" is relative to the working_dir "/home/dshs-wallga/TTRL".
# You might also want to exclude ".git", "data/" (if not needed in the package), etc.
# For this specific case, excluding "scripts/.venv" would likely bring the size below 100MB.
# If you apply this exclude, you might not even need to raise RAY_AIOHTTP_CLIENT_MAX_SIZE above default.
RUNTIME_ENV_JSON_CONTENT='{
    "pip": ["ray[default]"],
    "working_dir": "/home/dshs-wallga/TTRL",
    "excludes": ["scripts/.venv", ".git"]
}'

ray job submit --address="http://localhost:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON_CONTENT}" \
   -- python -m ttrl.cli.train_ppo_naive \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain "${BACKBONE_PATH}" \
   --save_path "${OUTPUT_DIR}/model" \
   --verify_task "ttt" \
   --verify_task_eval "math" \
   --micro_train_batch_size 2 \
   --train_batch_size 2 \
   --num_episodes 40 \
   --save_steps 2 \
   --eval_steps 1 \
   --logging_steps 1 \
   --max_samples 400000 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 16 \
   --n_votes_per_prompt 32 \
   --extra_eval_task "" \
   --training_mode "rl" \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 3072 \
   --advantage_estimator ${ADVANTAGE} \
   --use_kl_loss \
   --temperature 1.0 \
   --eval_temperature 0.0 \
   --lambd 1.0 \
   --gamma 1.0 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.00 \
   --prompt_data "json@${ROOT_DIR}/data/${TASK}" \
   --input_key "prompt" \
   --label_key "answer" \
   --max_ckpt_num 10 \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --flash_attn \
   --use_wandb ${WANDB_KTY} \
   --wandb_project TTRL \
   --wandb_run_name ${EXP} \
   --use_tensorboard "${ROOT_DIR}/logs/${EXP}" \
   --ckpt_path "${OUTPUT_DIR}/ckpt"
#  > ${LOG_FILE} 2>&1 &

echo "Model Training started in background. Check logs at ${LOG_FILE}"
#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 <model-name-or-path> <dataset-mixture> <epochs> <learning-rate> [output-dir]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VILA_DIR="${VILA_DIR:-${REPO_ROOT}/VILA}"

NETWORK="$1"
DATASET="$2"
EPOCHS="$3"
LEARNING_RATE="$4"

: "${DATA_ROOT:?Set DATA_ROOT to the dataset root described in README.md}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${REPO_ROOT}/checkpoints/pretrained}"
FINETUNED_ROOT="${FINETUNED_ROOT:-${REPO_ROOT}/checkpoints/finetuned}"
OUTPUT_DIR="${5:-${FINETUNED_ROOT}/${NETWORK##*/}-${DATASET}-e${EPOCHS}-LR${LEARNING_RATE}}"

if [[ -e "${NETWORK}" ]]; then
    MODEL_PATH="$(cd -- "$(dirname -- "${NETWORK}")" && pwd)/$(basename -- "${NETWORK}")"
elif [[ "${NETWORK}" == *robopoint* ]]; then
    MODEL_PATH="${FINETUNED_ROOT}/${NETWORK}"
else
    MODEL_PATH="${PRETRAIN_ROOT}/${NETWORK}"
fi

ROBOPOINT_JSON="${ROBOPOINT_JSON:-${DATA_ROOT}/robopoint/robopoint_1432k.json}"
ROBOPOINT_MEDIA_DIR="${ROBOPOINT_MEDIA_DIR:-${DATA_ROOT}/robopoint/images}"
MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${DATA_ROOT}/3DWay-Data}"
DROID_JSON="${DROID_JSON:-${MULTIVIEW_ROOT}/droid.json}"
RLBENCH_JSON="${RLBENCH_JSON:-${MULTIVIEW_ROOT}/rlbench.json}"
RH20T_JSON="${RH20T_JSON:-${MULTIVIEW_ROOT}/rh20t.json}"
MULTIVIEW_MEDIA_DIR="${MULTIVIEW_MEDIA_DIR:-${MULTIVIEW_ROOT}}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-25010}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
TIME_LIMIT_MINUTES="${TIME_LIMIT_MINUTES:-1680}"
PRE_TERMINATE_MINUTES="${PRE_TERMINATE_MINUTES:-10}"
MAX_RETRIES="${MAX_RETRIES:-100}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"
REPORT_TO="${REPORT_TO:-wandb}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export NCCL_IB_SL="${NCCL_IB_SL:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-NVILA}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -d "${VILA_DIR}" ]]; then
    echo "VILA submodule is missing: ${VILA_DIR}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model checkpoint does not exist: ${MODEL_PATH}" >&2
    echo "Set PRETRAIN_ROOT/FINETUNED_ROOT or pass an existing model path." >&2
    exit 1
fi

IFS='+' read -r -a requested_datasets <<<"${DATASET}"
for requested_dataset in "${requested_datasets[@]}"; do
    case "${requested_dataset}" in
        robopoint_1432k) required_path="${ROBOPOINT_JSON}" ;;
        droid) required_path="${DROID_JSON}" ;;
        rlbench) required_path="${RLBENCH_JSON}" ;;
        rh20t) required_path="${RH20T_JSON}" ;;
        *) continue ;;
    esac
    if [[ ! -f "${required_path}" ]]; then
        echo "Dataset metadata does not exist: ${required_path}" >&2
        exit 1
    fi
done

bash "${SCRIPT_DIR}/apply_vila_patch.sh" --require-applied
mkdir -p "${OUTPUT_DIR}"

dataset_registry="$(mktemp "${TMPDIR:-/tmp}/3dway-vila-datasets.XXXXXX.yaml")"
cleanup() {
    rm -f -- "${dataset_registry}"
}
trap cleanup EXIT

yaml_quote() {
    local value="${1//\'/\'\'}"
    printf "'%s'" "${value}"
}

{
    printf 'robopoint_1432k:\n'
    printf '    _target_: llava.data.LLaVADataset\n'
    printf '    data_path: %s\n' "$(yaml_quote "${ROBOPOINT_JSON}")"
    printf '    media_dir: %s\n' "$(yaml_quote "${ROBOPOINT_MEDIA_DIR}")"
    printf 'droid:\n'
    printf '    _target_: llava.data.LLaVADataset\n'
    printf '    data_path: %s\n' "$(yaml_quote "${DROID_JSON}")"
    printf '    media_dir: %s\n' "$(yaml_quote "${MULTIVIEW_MEDIA_DIR}")"
    printf 'rlbench:\n'
    printf '    _target_: llava.data.LLaVADataset\n'
    printf '    data_path: %s\n' "$(yaml_quote "${RLBENCH_JSON}")"
    printf '    media_dir: %s\n' "$(yaml_quote "${MULTIVIEW_MEDIA_DIR}")"
    printf 'rh20t:\n'
    printf '    _target_: llava.data.LLaVADataset\n'
    printf '    data_path: %s\n' "$(yaml_quote "${RH20T_JSON}")"
    printf '    media_dir: %s\n' "$(yaml_quote "${MULTIVIEW_MEDIA_DIR}")"
} >"${dataset_registry}"
export VILA_DATASETS="${dataset_registry}"

train_command=(
    torchrun
    "--nnodes=${NNODES}"
    "--nproc_per_node=${GPUS_PER_NODE}"
    "--node_rank=${NODE_RANK}"
    "--master_addr=${MASTER_ADDR}"
    "--master_port=${MASTER_PORT}"
    llava/train/train_mem.py
    --deepspeed scripts/zero3.json
    --model_name_or_path "${MODEL_PATH}"
    --data_mixture "${DATASET}"
    --ignore_data_skip
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448
    --mm_vision_select_feature cls_patch
    --mm_projector mlp_downsample_3x3_fix
    --tune_vision_tower True
    --tune_mm_projector True
    --tune_language_model True
    --lora_enable False
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio resize
    --bf16 True
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 100
    --save_total_limit 1
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 4096
    --gradient_checkpointing True
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --lazy_preprocess True
    --vflan_no_system_prompt True
    --total_time_limit "${TIME_LIMIT_MINUTES}"
    --pre_terminate_time "${PRE_TERMINATE_MINUTES}"
    --report_to "${REPORT_TO}"
)

echo "Model:   ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Output:  ${OUTPUT_DIR}"

for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
    if [[ -e "${OUTPUT_DIR}/done.md" ]]; then
        echo "Training already completed: ${OUTPUT_DIR}"
        exit 0
    fi

    echo "Training attempt ${attempt}/${MAX_RETRIES}"
    set +e
    (
        cd -- "${VILA_DIR}"
        "${train_command[@]}"
    ) 2>&1 | tee -a "${OUTPUT_DIR}/terminal.log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        printf 'Done!\n' >"${OUTPUT_DIR}/done.md"
        echo "Training completed: ${OUTPUT_DIR}"
        exit 0
    fi
    if [[ ${status} -ne 124 ]]; then
        echo "Training failed with exit code ${status}; not retrying." >&2
        exit "${status}"
    fi

    echo "Time limit reached; checkpoint saved. Retrying in ${RETRY_DELAY_SECONDS}s."
    sleep "${RETRY_DELAY_SECONDS}"
done

echo "Training did not finish after ${MAX_RETRIES} attempts." >&2
exit 1

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "baseline" && "$1" != "frangi" ) ]]; then
    printf 'Usage: %s {baseline|frangi}\n' "$0" >&2
    exit 2
fi

VARIANT="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${CRACKSAM_ROOT}/../.." && pwd)"
DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${CRACKSAM_ROOT}/data}"
ARTIFACT_ROOT="${CRACKSAM2_ARTIFACT_ROOT:-${CRACKSAM_ROOT}/artifacts}"
PROMPT_ROOT="${CRACKSAM2_PROMPT_ROOT:-${CRACKSAM_ROOT}/prompt_cache}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CRACKSAM_ROOT}/checkpoints/sam2_hiera_large.pt}"
RANK="${CRACKSAM2_RANK:-4}"
EPOCHS="${CRACKSAM2_EPOCHS:-70}"
NUM_WORKERS="${CRACKSAM2_NUM_WORKERS:-8}"
DEVICE="${CRACKSAM2_DEVICE:-cuda}"
AMP_DTYPE="${CRACKSAM2_AMP_DTYPE:-bfloat16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIST_ROOT="${CRACKSAM_ROOT}/protocol/cracksam_paper/lists"
OUTPUT="${ARTIFACT_ROOT}/${VARIANT}_r${RANK}"

mkdir -p "${OUTPUT}"
export PYTHONPATH="${CRACKSAM_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

command=(
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/train_sam2.py"
    --train-root "${DATA_ROOT}/khanhha/train"
    --val-root "${DATA_ROOT}/khanhha/train"
    --train-list "${LIST_ROOT}/lists_khanhha/train.txt"
    --val-list "${LIST_ROOT}/lists_khanhha/val_vol.txt"
    --sam2-checkpoint "${SAM2_CHECKPOINT}"
    --output "${OUTPUT}"
    --variant "${VARIANT}"
    --rank "${RANK}"
    --epochs "${EPOCHS}"
    --batch-size 8
    --num-workers "${NUM_WORKERS}"
    --base-lr 0.0004
    --warmup-steps 300
    --poly-power 6
    --weight-decay 0.01
    --ce-weight 0.2
    --threshold 0.5
    --checkpoint-every-steps 50
    --device "${DEVICE}"
    --amp-dtype "${AMP_DTYPE}"
)

if [[ "${VARIANT}" == "frangi" ]]; then
    command+=(
        --train-prompt-cache "${PROMPT_ROOT}/khanhha/train"
        --val-prompt-cache "${PROMPT_ROOT}/khanhha/val"
    )
fi
if [[ -f "${OUTPUT}/latest.pt" ]]; then
    command+=(--resume)
fi

exec "${command[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${CRACKSAM_ROOT}/../.." && pwd)"

DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${CRACKSAM_ROOT}/data}"
ARTIFACT_ROOT="${CRACKSAM2_ARTIFACT_ROOT:-${CRACKSAM_ROOT}/artifacts}"
PROMPT_ROOT="${CRACKSAM2_PROMPT_ROOT:-${CRACKSAM_ROOT}/prompt_cache}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CRACKSAM_ROOT}/checkpoints/sam2_hiera_large.pt}"
SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_CHECKPOINT_SHA256="7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
RANK="${CRACKSAM2_RANK:-4}"
EPOCHS="${CRACKSAM2_EPOCHS:-70}"
NUM_WORKERS="${CRACKSAM2_NUM_WORKERS:-8}"
DEVICE="${CRACKSAM2_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LIST_ROOT="${CRACKSAM_ROOT}/protocol/cracksam_paper/lists"
KHANHHA_TRAIN_LIST="${LIST_ROOT}/lists_khanhha/train.txt"
KHANHHA_VAL_LIST="${LIST_ROOT}/lists_khanhha/val_vol.txt"
KHANHHA_TEST_LIST="${LIST_ROOT}/lists_khanhha/test_vol.txt"

mkdir -p "${DATA_ROOT}" "${ARTIFACT_ROOT}" "${PROMPT_ROOT}" "$(dirname "${SAM2_CHECKPOINT}")"
export PYTHONPATH="${CRACKSAM_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${SAM2_CHECKPOINT}" ]]; then
    temporary="${SAM2_CHECKPOINT}.part"
    printf '[checkpoint] Downloading SAM 2 Hiera Large...\n'
    curl --location --fail --retry 5 --continue-at - \
        --output "${temporary}" "${SAM2_CHECKPOINT_URL}"
    printf '%s  %s\n' "${SAM2_CHECKPOINT_SHA256}" "${temporary}" | sha256sum --check -
    mv "${temporary}" "${SAM2_CHECKPOINT}"
fi

printf '%s  %s\n' "${SAM2_CHECKPOINT_SHA256}" "${SAM2_CHECKPOINT}" | \
    sha256sum --check -

"${PYTHON_BIN}" - <<'PY' "${SAM2_CHECKPOINT}"
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(state, dict) or "model" not in state:
    raise SystemExit(f"Invalid SAM 2 checkpoint: {path}")
print(f"[checkpoint] Validated {path}")
PY

"${PYTHON_BIN}" "${CRACKSAM_ROOT}/prepare_cracksam2_data.py" \
    --output "${DATA_ROOT}" \
    --datasets khanhha road420 facade390 concrete3k

CRACKSAM2_DATA_ROOT="${DATA_ROOT}" \
CRACKSAM2_PROMPT_ROOT="${PROMPT_ROOT}" \
CRACKSAM2_DEVICE="${DEVICE}" \
PYTHON_BIN="${PYTHON_BIN}" \
    "${SCRIPT_DIR}/precompute_all_cracksam2_prompts.sh"

train_variant() {
    local variant="$1"
    local output="${ARTIFACT_ROOT}/${variant}_r${RANK}"
    local -a command=(
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/train_sam2.py"
        --train-root "${DATA_ROOT}/khanhha/train"
        --val-root "${DATA_ROOT}/khanhha/train"
        --train-list "${KHANHHA_TRAIN_LIST}"
        --val-list "${KHANHHA_VAL_LIST}"
        --sam2-checkpoint "${SAM2_CHECKPOINT}"
        --output "${output}"
        --variant "${variant}"
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
        --amp-dtype bfloat16
    )
    if [[ "${variant}" == "frangi" ]]; then
        command+=(
            --train-prompt-cache "${PROMPT_ROOT}/khanhha/train"
            --val-prompt-cache "${PROMPT_ROOT}/khanhha/val"
        )
    fi
    if [[ -f "${output}/latest.pt" ]]; then
        command+=(--resume)
    fi
    "${command[@]}"
}

train_variant baseline
train_variant frangi

evaluate_variant() {
    local variant="$1"
    local checkpoint="${ARTIFACT_ROOT}/${variant}_r${RANK}/best.pt"
    local -a command=(
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/evaluate_sam2.py"
        --data-root "${DATA_ROOT}"
        --sam2-checkpoint "${SAM2_CHECKPOINT}"
        --adapter-checkpoint "${checkpoint}"
        --output "${ARTIFACT_ROOT}/${variant}_r${RANK}/evaluation"
        --batch-size 1
        --num-workers "${NUM_WORKERS}"
        --threshold 0.5
        --device "${DEVICE}"
        --amp-dtype bfloat16
    )
    if [[ "${variant}" == "frangi" ]]; then
        command+=(--prompt-cache-root "${PROMPT_ROOT}")
    fi
    if [[ "${CRACKSAM2_WASSERSTEIN_EXACT:-0}" == "1" ]]; then
        command+=(--wasserstein-exact)
    elif [[ -n "${CRACKSAM2_WASSERSTEIN_MAX_POINTS:-}" ]]; then
        command+=(--wasserstein-max-points "${CRACKSAM2_WASSERSTEIN_MAX_POINTS}")
    fi
    "${command[@]}"
}

evaluate_variant baseline
evaluate_variant frangi

printf '[complete] Results: %s\n' "${ARTIFACT_ROOT}"

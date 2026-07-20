#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${CRACKSAM_ROOT}/../.." && pwd)"
DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${CRACKSAM_ROOT}/data}"
PROMPT_ROOT="${CRACKSAM2_PROMPT_ROOT:-${CRACKSAM_ROOT}/prompt_cache}"
DEVICE="${CRACKSAM2_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIST_ROOT="${CRACKSAM_ROOT}/protocol/cracksam_paper/lists"

export PYTHONPATH="${CRACKSAM_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

precompute() {
    local data_root="$1"
    local list_file="$2"
    local cache_dir="$3"
    local noise="$4"
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/precompute_frangi_prompts.py" \
        --data-root "${data_root}" \
        --list-file "${list_file}" \
        --cache-dir "${cache_dir}" \
        --noise "${noise}" \
        --device "${DEVICE}"
}

precompute "${DATA_ROOT}/khanhha/train" \
    "${LIST_ROOT}/lists_khanhha/train.txt" \
    "${PROMPT_ROOT}/khanhha/train" none
precompute "${DATA_ROOT}/khanhha/train" \
    "${LIST_ROOT}/lists_khanhha/val_vol.txt" \
    "${PROMPT_ROOT}/khanhha/val" none
precompute "${DATA_ROOT}/khanhha/test" \
    "${LIST_ROOT}/lists_khanhha/test_vol.txt" \
    "${PROMPT_ROOT}/khanhha/test_original" none
precompute "${DATA_ROOT}/khanhha/test" \
    "${LIST_ROOT}/lists_khanhha/test_vol.txt" \
    "${PROMPT_ROOT}/khanhha/test_noisy1" noisy1
precompute "${DATA_ROOT}/khanhha/test" \
    "${LIST_ROOT}/lists_khanhha/test_vol.txt" \
    "${PROMPT_ROOT}/khanhha/test_noisy2" noisy2
precompute "${DATA_ROOT}/road420" \
    "${LIST_ROOT}/lists_road420/test_vol.txt" \
    "${PROMPT_ROOT}/road420/test_original" none
precompute "${DATA_ROOT}/facade390" \
    "${LIST_ROOT}/lists_facade390/test_vol.txt" \
    "${PROMPT_ROOT}/facade390/test_original" none
precompute "${DATA_ROOT}/concrete3k" \
    "${LIST_ROOT}/lists_concrete3k/test_vol.txt" \
    "${PROMPT_ROOT}/concrete3k/test_original" none

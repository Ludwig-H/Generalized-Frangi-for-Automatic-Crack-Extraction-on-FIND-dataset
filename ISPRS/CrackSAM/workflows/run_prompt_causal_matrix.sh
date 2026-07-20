#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${CRACKSAM_ROOT}/data}"
ARTIFACT_ROOT="${CRACKSAM2_ARTIFACT_ROOT:-${CRACKSAM_ROOT}/artifacts}"
PROMPT_ROOT="${CRACKSAM2_PROMPT_ROOT:-${CRACKSAM_ROOT}/prompt_cache}"
CHECKPOINT_ROOT="${CRACKSAM2_CHECKPOINT_ROOT:-${ARTIFACT_ROOT}}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CRACKSAM_ROOT}/checkpoints/sam2_hiera_large.pt}"
BASELINE_EPOCH20_CHECKPOINT="${BASELINE_EPOCH20_CHECKPOINT:-${CHECKPOINT_ROOT}/baseline_r4/best.pt}"
FRANGI_EPOCH20_CHECKPOINT="${FRANGI_EPOCH20_CHECKPOINT:-${CHECKPOINT_ROOT}/frangi_r4/milestone_epoch20.pt}"
FRANGI_BEST_CHECKPOINT="${FRANGI_BEST_CHECKPOINT:-${CHECKPOINT_ROOT}/frangi_r4/best.pt}"
OUTPUT_ROOT="${CRACKSAM2_CAUSAL_OUTPUT:-${ARTIFACT_ROOT}/causal_prompt_matrix}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_WORKERS="${CRACKSAM2_NUM_WORKERS:-8}"
BATCH_SIZE="${CRACKSAM2_CAUSAL_BATCH_SIZE:-1}"
PERMUTATION_SEED="${CRACKSAM2_PROMPT_PERMUTATION_SEED:-3407}"
SHIFT_DY="${CRACKSAM2_PROMPT_SHIFT_DY:-32}"
SHIFT_DX="${CRACKSAM2_PROMPT_SHIFT_DX:-32}"

for required in \
    "${SAM2_CHECKPOINT}" \
    "${BASELINE_EPOCH20_CHECKPOINT}" \
    "${FRANGI_EPOCH20_CHECKPOINT}" \
    "${FRANGI_BEST_CHECKPOINT}"; do
    test -f "${required}" || {
        printf 'Fichier requis absent: %s\n' "${required}" >&2
        exit 1
    }
done
test -d "${DATA_ROOT}" || { printf 'Données absentes: %s\n' "${DATA_ROOT}" >&2; exit 1; }
test -d "${PROMPT_ROOT}" || { printf 'Cache Frangi absent: %s\n' "${PROMPT_ROOT}" >&2; exit 1; }

export PYTHONPATH="${CRACKSAM_ROOT}:${PYTHONPATH:-}"

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --sam2-checkpoint "${SAM2_CHECKPOINT}"
    --prompt-cache-root "${PROMPT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --threshold 0.5
    --skip-wasserstein
    --prompt-permutation-seed "${PERMUTATION_SEED}"
    --prompt-shift "${SHIFT_DY}" "${SHIFT_DX}"
    --device cuda
    --amp-dtype bfloat16
)

if [[ "${CRACKSAM2_CAUSAL_SAVE_LOGITS:-1}" == "1" ]]; then
    COMMON_ARGS+=(--save-logits --logit-dtype float16)
fi
if [[ "${CRACKSAM2_CAUSAL_SAVE_PREDICTIONS:-0}" == "1" ]]; then
    COMMON_ARGS+=(--save-predictions)
fi
if [[ -n "${CRACKSAM2_CAUSAL_MAX_SAMPLES:-}" ]]; then
    COMMON_ARGS+=(--max-samples "${CRACKSAM2_CAUSAL_MAX_SAMPLES}")
fi
if [[ -n "${CRACKSAM2_CAUSAL_DATASETS:-}" ]]; then
    # Intentional shell splitting: this variable is a whitespace-separated list
    # of evaluator dataset identifiers.
    read -r -a selected_datasets <<< "${CRACKSAM2_CAUSAL_DATASETS}"
    COMMON_ARGS+=(--datasets "${selected_datasets[@]}")
fi

run_matrix() {
    local label="$1"
    local checkpoint="$2"
    shift 2
    printf '[causal] %s\n' "${label}"
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/evaluate_sam2.py" \
        "${COMMON_ARGS[@]}" \
        --adapter-checkpoint "${checkpoint}" \
        --output "${OUTPUT_ROOT}/${label}" \
        --prompt-conditions "$@"
}

# C0, C1, C4, C5, C6 with exactly the same baseline checkpoint.
run_matrix baseline_epoch20 "${BASELINE_EPOCH20_CHECKPOINT}" \
    none frangi zero_logit permuted shifted

# C2 and C3 at the common training horizon.
run_matrix frangi_epoch20 "${FRANGI_EPOCH20_CHECKPOINT}" none frangi

# Secondary selected-checkpoint analysis (historical Frangi best, epoch 25).
run_matrix frangi_best "${FRANGI_BEST_CHECKPOINT}" none frangi

printf '[complete] Matrice causale: %s\n' "${OUTPUT_ROOT}"

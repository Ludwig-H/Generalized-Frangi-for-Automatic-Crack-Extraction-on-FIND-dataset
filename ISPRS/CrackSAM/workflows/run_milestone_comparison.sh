#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${HOME}/cracksam2-data}"
ARTIFACT_ROOT="${CRACKSAM2_ARTIFACT_ROOT:-${HOME}/cracksam2-artifacts}"
PROMPT_ROOT="${CRACKSAM2_PROMPT_ROOT:-${HOME}/cracksam2-prompts}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CRACKSAM_ROOT}/checkpoints/sam2_hiera_large.pt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_WORKERS="${CRACKSAM2_NUM_WORKERS:-8}"
EXACT_WORKERS="${CRACKSAM2_EXACT_WORKERS:-8}"
EXACT_MEMORY_GB="${CRACKSAM2_EXACT_MEMORY_GB:-140}"

FRANGI_ROOT="${ARTIFACT_ROOT}/frangi_r4"
COMPARISON_ROOT="${FRANGI_ROOT}/milestone_comparison"
BASELINE_EVALUATION="${ARTIFACT_ROOT}/baseline_r4/final_evaluation"

MILESTONES=(
    "epoch25_best:milestone_epoch25_best.pt"
    "epoch20:milestone_epoch20.pt"
    "epoch30:milestone_epoch30.pt"
    "epoch55:milestone_epoch55.pt"
    "epoch70:milestone_epoch70.pt"
)

export PYTHONPATH="${CRACKSAM_ROOT}:${PYTHONPATH:-}"

evaluate_milestones() {
    mkdir -p "${COMPARISON_ROOT}"
    local entry label checkpoint output
    for entry in "${MILESTONES[@]}"; do
        label="${entry%%:*}"
        checkpoint="${FRANGI_ROOT}/${entry#*:}"
        output="${COMPARISON_ROOT}/${label}"
        test -f "${checkpoint}"
        printf '[evaluate] %s -> %s\n' "${checkpoint}" "${output}"
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/evaluate_sam2.py" \
            --data-root "${DATA_ROOT}" \
            --sam2-checkpoint "${SAM2_CHECKPOINT}" \
            --adapter-checkpoint "${checkpoint}" \
            --output "${output}" \
            --prompt-cache-root "${PROMPT_ROOT}" \
            --batch-size 1 \
            --num-workers "${NUM_WORKERS}" \
            --threshold 0.5 \
            --skip-wasserstein \
            --save-predictions \
            --device cuda \
            --amp-dtype bfloat16
    done
}

evaluation_roots() {
    printf '%s\n' "${BASELINE_EVALUATION}"
    local entry label
    for entry in "${MILESTONES[@]}"; do
        label="${entry%%:*}"
        printf '%s\n' "${COMPARISON_ROOT}/${label}"
    done
}

scan_wasserstein() {
    local evaluation_root
    while IFS= read -r evaluation_root; do
        test -f "${evaluation_root}/summary.csv"
        printf '[wasserstein-scan] %s\n' "${evaluation_root}"
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/compute_exact_wasserstein.py" \
            --data-root "${DATA_ROOT}" \
            --evaluation-root "${evaluation_root}" \
            --workers "${EXACT_WORKERS}" \
            --memory-budget-gb "${EXACT_MEMORY_GB}" \
            --skip-oversized \
            --scan-only
    done < <(evaluation_roots)
}

compute_wasserstein() {
    local evaluation_root
    while IFS= read -r evaluation_root; do
        test -f "${evaluation_root}/summary.csv"
        printf '[wasserstein-exact] %s\n' "${evaluation_root}"
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/compute_exact_wasserstein.py" \
            --data-root "${DATA_ROOT}" \
            --evaluation-root "${evaluation_root}" \
            --workers "${EXACT_WORKERS}" \
            --memory-budget-gb "${EXACT_MEMORY_GB}" \
            --skip-oversized
    done < <(evaluation_roots)
}

case "${1:-all}" in
    evaluate)
        evaluate_milestones
        ;;
    scan)
        scan_wasserstein
        ;;
    exact)
        compute_wasserstein
        ;;
    all)
        evaluate_milestones
        scan_wasserstein
        compute_wasserstein
        ;;
    *)
        printf 'Usage: %s [evaluate|scan|exact|all]\n' "$0" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
set -Eeuo pipefail

# Leak-free, Spot-resumable FrangiGraph residual + logistic-gate pilot.
#
# This workflow is deliberately compute-provider agnostic: it never creates,
# starts, stops, or inspects a cloud resource.  Run it *inside* the selected
# worker after mounting durable DATA_ROOT and FRANGIGRAPH_RUN_ROOT paths.

umask 027

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRACKSAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${CRACKSAM_ROOT}/../.." && pwd)"

usage() {
    printf '%s\n' \
        'Usage:' \
        '  run_frangigraph_logistic_gate_pilot.sh --mode SMOKE|FULL [--resume]' \
        '' \
        'Required environment variables:' \
        '  CRACKSAM2_DATA_ROOT       Prepared dataset root (contains khanhha/train).' \
        '  SAM2_CHECKPOINT           Frozen SAM 2 foundation checkpoint.' \
        '  BASELINE_CHECKPOINT       Frozen CrackSAM 2 baseline adapter checkpoint.' \
        '  FRANGIGRAPH_RUN_ROOT      New durable output directory for this exact run.' \
        '' \
        'Important optional variables (all have explicit defaults below):' \
        '  FRANGIGRAPH_PYTHON_BIN, FRANGIGRAPH_DEVICE, FRANGIGRAPH_NUM_WORKERS' \
        '  FRANGIGRAPH_GRAPH_CACHE (absolute reusable/resumable cache directory)' \
        '  FRANGIGRAPH_RASTER_CONDITION (correct [default] or no_evidence control)' \
        '  FRANGIGRAPH_EPOCHS, FRANGIGRAPH_BATCH_SIZE, FRANGIGRAPH_EVAL_BATCH_SIZE' \
        '  FRANGIGRAPH_LABEL_MINIMUM_GAIN, FRANGIGRAPH_SEGMENTATION_THRESHOLD' \
        '  FRANGIGRAPH_SMOKE_SAMPLES_PER_FOLD, FRANGIGRAPH_GIT_COMMIT' \
        '' \
        'Resume contract:' \
        '  --resume is mandatory after interruption. Completed phase markers are' \
        '  revalidated and skipped; partial cache/evaluations and training latest.pt' \
        '  are resumed by their native entry points.'
}

MODE=""
RESUME=0
while (($#)); do
    case "$1" in
        --mode)
            (($# >= 2)) || { printf 'Missing value after --mode\n' >&2; exit 2; }
            MODE="${2^^}"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${MODE}" in
    SMOKE|FULL) ;;
    *)
        printf '%s\n' '--mode must be exactly SMOKE or FULL' >&2
        exit 2
        ;;
esac

: "${CRACKSAM2_DATA_ROOT:?Set CRACKSAM2_DATA_ROOT to the prepared dataset root}"
: "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to the frozen SAM 2 checkpoint}"
: "${BASELINE_CHECKPOINT:?Set BASELINE_CHECKPOINT to the frozen baseline checkpoint}"
: "${FRANGIGRAPH_RUN_ROOT:?Set FRANGIGRAPH_RUN_ROOT to a durable run directory}"

PYTHON_BIN="${FRANGIGRAPH_PYTHON_BIN:-python3}"
DEVICE="${FRANGIGRAPH_DEVICE:-cuda}"
NUM_WORKERS="${FRANGIGRAPH_NUM_WORKERS:-8}"
AMP_DTYPE="${FRANGIGRAPH_AMP_DTYPE:-bfloat16}"
SEED="${FRANGIGRAPH_SEED:-3407}"
KHANHHA_DATA_ROOT="${FRANGIGRAPH_KHANHHA_DATA_ROOT:-${CRACKSAM2_DATA_ROOT}/khanhha/train}"
RUN_ROOT="${FRANGIGRAPH_RUN_ROOT}"

CANONICAL_TRAIN_LIST="${FRANGIGRAPH_TRAIN_LIST:-${CRACKSAM_ROOT}/protocol/cracksam_paper/lists/lists_khanhha/train.txt}"
CANONICAL_FOLD_CSV="${FRANGIGRAPH_FOLD_CSV:-${CRACKSAM_ROOT}/protocol/frangigraph_v1/train_group_folds.csv}"
CANONICAL_PROTOCOL_MANIFEST="${FRANGIGRAPH_PROTOCOL_MANIFEST:-${CRACKSAM_ROOT}/protocol/frangigraph_v1/manifest.json}"

SMOKE_SAMPLES_PER_FOLD="${FRANGIGRAPH_SMOKE_SAMPLES_PER_FOLD:-12}"
if [[ "${MODE}" == "SMOKE" ]]; then
    EPOCHS="${FRANGIGRAPH_EPOCHS:-1}"
    BATCH_SIZE="${FRANGIGRAPH_BATCH_SIZE:-2}"
    EVAL_BATCH_SIZE="${FRANGIGRAPH_EVAL_BATCH_SIZE:-2}"
    WARMUP_STEPS="${FRANGIGRAPH_WARMUP_STEPS:-1}"
    CHECKPOINT_EVERY_STEPS="${FRANGIGRAPH_CHECKPOINT_EVERY_STEPS:-1}"
else
    EPOCHS="${FRANGIGRAPH_EPOCHS:-30}"
    BATCH_SIZE="${FRANGIGRAPH_BATCH_SIZE:-8}"
    EVAL_BATCH_SIZE="${FRANGIGRAPH_EVAL_BATCH_SIZE:-4}"
    WARMUP_STEPS="${FRANGIGRAPH_WARMUP_STEPS:-300}"
    CHECKPOINT_EVERY_STEPS="${FRANGIGRAPH_CHECKPOINT_EVERY_STEPS:-250}"
fi

# Cache parameters are preregistered here and bound into the cache manifest.
FRANGI_SCALES_TEXT="${FRANGIGRAPH_SCALES:-1.0 3.0 5.0 9.0 15.0}"
read -r -a FRANGI_SCALES <<< "${FRANGI_SCALES_TEXT}"
FRANGI_RADIUS="${FRANGIGRAPH_RADIUS:-3}"
FRANGI_SS="${FRANGIGRAPH_SS:-1.0}"
FRANGI_SI="${FRANGIGRAPH_SI:-0.25}"
FRANGI_SA="${FRANGIGRAPH_SA:-0.3}"
FRANGI_TAU="${FRANGIGRAPH_TAU:-0.18}"
FRANGI_MIN_REL_SIZE="${FRANGIGRAPH_MIN_REL_SIZE:-120.0}"
FRANGI_GRAPH_ORDER="${FRANGIGRAPH_GRAPH_ORDER:-1}"

# ``correct`` is the scientific candidate. ``no_evidence`` keeps the same
# trainable residual capacity and replaces all Frangi rasters with their
# canonical absent-evidence encoding. An unset, empty, or explicitly correct
# variable therefore has exactly the same downstream value and contract bytes.
RASTER_CONDITION="${FRANGIGRAPH_RASTER_CONDITION:-correct}"
case "${RASTER_CONDITION}" in
    correct|no_evidence) ;;
    *)
        printf '%s\n' \
            'FRANGIGRAPH_RASTER_CONDITION must be exactly correct or no_evidence' >&2
        exit 2
        ;;
esac

# Residual parameters. Validation is descriptive only; latest.pt at the fixed
# final epoch is always used for OOF prediction.
HIDDEN_CHANNELS="${FRANGIGRAPH_HIDDEN_CHANNELS:-32}"
BASE_LR="${FRANGIGRAPH_BASE_LR:-0.0004}"
POLY_POWER="${FRANGIGRAPH_POLY_POWER:-2.0}"
WEIGHT_DECAY="${FRANGIGRAPH_WEIGHT_DECAY:-0.01}"
CE_WEIGHT="${FRANGIGRAPH_CE_WEIGHT:-0.2}"
TOPOLOGY_WEIGHT="${FRANGIGRAPH_TOPOLOGY_WEIGHT:-0.1}"
SAFETY_WEIGHT="${FRANGIGRAPH_SAFETY_WEIGHT:-1.0}"
SAFETY_MARGIN="${FRANGIGRAPH_SAFETY_MARGIN:-0.0}"
SKELETON_ITERATIONS="${FRANGIGRAPH_SKELETON_ITERATIONS:-10}"
GRADIENT_CLIP="${FRANGIGRAPH_GRADIENT_CLIP:-1.0}"
SEGMENTATION_THRESHOLD="${FRANGIGRAPH_SEGMENTATION_THRESHOLD:-0.5}"
if [[ "${MODE}" == "SMOKE" ]]; then
    DEFAULT_LABEL_MINIMUM_GAIN="0.0"
else
    # The scientific pilot targets a practically visible improvement rather
    # than treating numerical ties as positive examples. Sensitivities at 0
    # and 0.01 remain analysis-only follow-ups after this primary fit is frozen.
    DEFAULT_LABEL_MINIMUM_GAIN="0.005"
fi
LABEL_MINIMUM_GAIN="${FRANGIGRAPH_LABEL_MINIMUM_GAIN:-${DEFAULT_LABEL_MINIMUM_GAIN}}"

# Logistic fit and fold-4-only calibration parameters. No historical test list
# is accepted or referenced anywhere in this workflow.
GATE_L2="${FRANGIGRAPH_GATE_L2:-1.0}"
GATE_MINIMUM_PRECISION="${FRANGIGRAPH_GATE_MINIMUM_PRECISION:-0.80}"
GATE_MINIMUM_SELECTED="${FRANGIGRAPH_GATE_MINIMUM_SELECTED:-10}"
GATE_MINIMUM_SELECTED_GROUPS="${FRANGIGRAPH_GATE_MINIMUM_SELECTED_GROUPS:-10}"
GATE_MINIMUM_MEAN_DELTA_IOU="${FRANGIGRAPH_GATE_MINIMUM_MEAN_DELTA_IOU:-0.0}"
GATE_SEVERE_LOSS_THRESHOLD="${FRANGIGRAPH_GATE_SEVERE_LOSS_THRESHOLD:--0.05}"
GATE_MAXIMUM_SEVERE_LOSS_RATE="${FRANGIGRAPH_GATE_MAXIMUM_SEVERE_LOSS_RATE:-0.05}"

if [[ "${RUN_ROOT}" != /* || "${RUN_ROOT}" == "/" ]]; then
    printf 'FRANGIGRAPH_RUN_ROOT must be an absolute, non-root path: %s\n' "${RUN_ROOT}" >&2
    exit 2
fi

for required_file in \
    "${SAM2_CHECKPOINT}" \
    "${BASELINE_CHECKPOINT}" \
    "${CANONICAL_TRAIN_LIST}" \
    "${CANONICAL_FOLD_CSV}" \
    "${CANONICAL_PROTOCOL_MANIFEST}"; do
    [[ -f "${required_file}" ]] || {
        printf 'Required file is missing: %s\n' "${required_file}" >&2
        exit 2
    }
done
[[ -d "${KHANHHA_DATA_ROOT}" ]] || {
    printf 'Prepared Khanhha training directory is missing: %s\n' "${KHANHHA_DATA_ROOT}" >&2
    exit 2
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 2
}

if [[ -e "${RUN_ROOT}" && ! -d "${RUN_ROOT}" ]]; then
    printf 'Run root exists but is not a directory: %s\n' "${RUN_ROOT}" >&2
    exit 2
fi
if ((RESUME)); then
    [[ -d "${RUN_ROOT}" ]] || {
        printf -- '--resume requires an existing run root: %s\n' "${RUN_ROOT}" >&2
        exit 2
    }
    [[ -f "${RUN_ROOT}/workflow_contract.json" ]] || {
        printf -- '--resume requires workflow_contract.json in %s\n' "${RUN_ROOT}" >&2
        exit 2
    }
else
    if [[ -d "${RUN_ROOT}" ]] && [[ -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Refusing non-empty run root without --resume: %s\n' "${RUN_ROOT}" >&2
        exit 2
    fi
fi

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd -- "${RUN_ROOT}" && pwd -P)"
STATE_ROOT="${RUN_ROOT}/state"
LOG_ROOT="${RUN_ROOT}/logs"
PROTOCOL_ROOT="${RUN_ROOT}/protocol"
CACHE_ROOT="${FRANGIGRAPH_GRAPH_CACHE:-${RUN_ROOT}/graph_cache}"
if [[ "${CACHE_ROOT}" != /* || "${CACHE_ROOT}" == "/" ]]; then
    printf 'FRANGIGRAPH_GRAPH_CACHE must be an absolute, non-root path: %s\n' \
        "${CACHE_ROOT}" >&2
    exit 2
fi
TRAIN_ROOT="${RUN_ROOT}/residual_folds"
OOF_ROOT="${RUN_ROOT}/oof_evaluations"
ASSEMBLY_ROOT="${RUN_ROOT}/gate_data"
GATE_ROOT="${RUN_ROOT}/gate"
GATE_JSON="${GATE_ROOT}/logistic_gate.json"
GATE_ANALYSIS_ROOT="${GATE_ROOT}/oof_analysis"
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

# One local process owns a run root. On filesystems without flock, fail closed.
command -v flock >/dev/null 2>&1 || {
    printf 'flock is required to protect the run root\n' >&2
    exit 2
}
exec 9>"${RUN_ROOT}/.workflow.lock"
flock -n 9 || {
    printf 'Another workflow process owns this run root: %s\n' "${RUN_ROOT}" >&2
    exit 3
}

GIT_COMMIT="${FRANGIGRAPH_GIT_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
if [[ ! "${GIT_COMMIT}" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
    printf 'FRANGIGRAPH_GIT_COMMIT must be a full lowercase commit hash\n' >&2
    exit 2
fi
if [[ "${MODE}" == "FULL" && "${FRANGIGRAPH_ALLOW_DIRTY_TREE:-0}" != "1" ]]; then
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
        printf '%s\n' \
            'FULL mode refuses uncommitted or untracked working-tree changes because gate provenance' \
            'must identify the implementation commit. Commit first, or explicitly' \
            'set FRANGIGRAPH_ALLOW_DIRTY_TREE=1 for a non-confirmatory diagnostic run.' >&2
        exit 2
    fi
fi

if [[ "${MODE}" == "SMOKE" ]]; then
    ACTIVE_TRAIN_LIST="${PROTOCOL_ROOT}/train.txt"
    ACTIVE_FOLD_CSV="${PROTOCOL_ROOT}/train_group_folds.csv"
    ACTIVE_PROTOCOL_MANIFEST="${PROTOCOL_ROOT}/manifest.json"
else
    ACTIVE_TRAIN_LIST="${CANONICAL_TRAIN_LIST}"
    ACTIVE_FOLD_CSV="${CANONICAL_FOLD_CSV}"
    ACTIVE_PROTOCOL_MANIFEST="${CANONICAL_PROTOCOL_MANIFEST}"
fi

export PYTHONPATH="${CRACKSAM_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# Immutable workflow-level contract: --resume must use byte-identical inputs
# and hyperparameters. The phase tools add content hashes to their own outputs.
CONTRACT_TMP="$(mktemp "${RUN_ROOT}/.workflow_contract.XXXXXX")"
"${PYTHON_BIN}" - "${CONTRACT_TMP}" \
    "${MODE}" "${KHANHHA_DATA_ROOT}" "${SAM2_CHECKPOINT}" \
    "${BASELINE_CHECKPOINT}" "${CACHE_ROOT}" "${ACTIVE_TRAIN_LIST}" "${ACTIVE_FOLD_CSV}" \
    "${ACTIVE_PROTOCOL_MANIFEST}" "${GIT_COMMIT}" "${DEVICE}" \
    "${NUM_WORKERS}" "${AMP_DTYPE}" "${SEED}" "${EPOCHS}" \
    "${BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WARMUP_STEPS}" \
    "${CHECKPOINT_EVERY_STEPS}" "${SMOKE_SAMPLES_PER_FOLD}" \
    "${FRANGI_SCALES_TEXT}" "${FRANGI_RADIUS}" "${FRANGI_SS}" \
    "${FRANGI_SI}" "${FRANGI_SA}" "${FRANGI_TAU}" \
    "${FRANGI_MIN_REL_SIZE}" "${FRANGI_GRAPH_ORDER}" \
    "${RASTER_CONDITION}" \
    "${HIDDEN_CHANNELS}" "${BASE_LR}" "${POLY_POWER}" \
    "${WEIGHT_DECAY}" "${CE_WEIGHT}" "${TOPOLOGY_WEIGHT}" \
    "${SAFETY_WEIGHT}" "${SAFETY_MARGIN}" "${SKELETON_ITERATIONS}" \
    "${GRADIENT_CLIP}" "${SEGMENTATION_THRESHOLD}" \
    "${LABEL_MINIMUM_GAIN}" "${GATE_L2}" "${GATE_MINIMUM_PRECISION}" \
    "${GATE_MINIMUM_SELECTED}" "${GATE_MINIMUM_SELECTED_GROUPS}" \
    "${GATE_MINIMUM_MEAN_DELTA_IOU}" "${GATE_SEVERE_LOSS_THRESHOLD}" \
    "${GATE_MAXIMUM_SEVERE_LOSS_RATE}" <<'PY'
import json
import os
import sys

keys = (
    "mode", "khanhha_data_root", "sam2_checkpoint", "baseline_checkpoint",
    "graph_cache_root", "train_list", "fold_csv", "protocol_manifest", "git_commit", "device",
    "num_workers", "amp_dtype", "seed", "epochs", "batch_size",
    "eval_batch_size", "warmup_steps", "checkpoint_every_steps",
    "smoke_samples_per_fold", "frangi_scales", "frangi_radius", "frangi_ss",
    "frangi_si", "frangi_sa", "frangi_tau", "frangi_min_rel_size",
    "frangi_graph_order", "raster_condition", "hidden_channels", "base_lr", "poly_power",
    "weight_decay", "ce_weight", "topology_weight", "safety_weight",
    "safety_margin", "skeleton_iterations", "gradient_clip",
    "segmentation_threshold", "label_minimum_gain", "gate_l2",
    "gate_minimum_precision", "gate_minimum_selected",
    "gate_minimum_selected_groups", "gate_minimum_mean_delta_iou",
    "gate_severe_loss_threshold", "gate_maximum_severe_loss_rate",
)
destination, *values = sys.argv[1:]
if len(values) != len(keys):
    raise SystemExit("internal workflow contract arity mismatch")
payload = {
    "schema": "cracksam2.frangigraph-logistic-gate-workflow",
    "schema_version": 2,
    "parameters": dict(zip(keys, values, strict=True)),
    "historical_test_inputs_used": False,
    "gate_fit_folds": [0, 1, 2, 3],
    "gate_calibration_fold": 4,
}
temporary = destination
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

if [[ -f "${RUN_ROOT}/workflow_contract.json" ]]; then
    if ! cmp -s "${CONTRACT_TMP}" "${RUN_ROOT}/workflow_contract.json"; then
        rm -f -- "${CONTRACT_TMP}"
        printf 'Resume parameters differ from immutable workflow_contract.json\n' >&2
        exit 2
    fi
    rm -f -- "${CONTRACT_TMP}"
elif ((RESUME)); then
    rm -f -- "${CONTRACT_TMP}"
    printf 'Cannot resume without an immutable workflow contract\n' >&2
    exit 2
else
    mv -- "${CONTRACT_TMP}" "${RUN_ROOT}/workflow_contract.json"
fi

CURRENT_PHASE="initialization"
workflow_exit() {
    local status=$?
    if ((status != 0)); then
        printf '%s\tphase=%s\texit=%s\n' \
            "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${CURRENT_PHASE}" "${status}" \
            >> "${STATE_ROOT}/failures.log"
        printf '[failed] phase=%s exit=%s; resume with --resume\n' \
            "${CURRENT_PHASE}" "${status}" >&2
    fi
}
trap workflow_exit EXIT

validate_smoke_protocol() {
    [[ -s "${ACTIVE_TRAIN_LIST}" && -s "${ACTIVE_FOLD_CSV}" && -s "${ACTIVE_PROTOCOL_MANIFEST}" ]]
}

validate_cache() {
    [[ -s "${CACHE_ROOT}/.cracksam2-frangi-graph-v2.json" ]]
}

validate_training_fold() {
    local fold="$1"
    [[ -s "${TRAIN_ROOT}/fold_${fold}/config.json" && -s "${TRAIN_ROOT}/fold_${fold}/latest.pt" ]]
}

validate_oof_fold() {
    local fold="$1"
    [[ -s "${OOF_ROOT}/fold_${fold}/evaluation_contract.json" && \
       -s "${OOF_ROOT}/fold_${fold}/per_image.csv" && \
       -s "${OOF_ROOT}/fold_${fold}/summary.json" ]]
}

validate_assembly() {
    [[ -s "${ASSEMBLY_ROOT}/gate_fit.csv" && \
       -s "${ASSEMBLY_ROOT}/gate_calibration.csv" && \
       -s "${ASSEMBLY_ROOT}/oof_manifest.json" ]]
}

validate_gate() {
    [[ -s "${GATE_JSON}" ]]
}

validate_gate_analysis() {
    [[ -s "${GATE_ANALYSIS_ROOT}/summary.json" && \
       -s "${GATE_ANALYSIS_ROOT}/per_image_gated.csv" && \
       -s "${GATE_ANALYSIS_ROOT}/risk_coverage.csv" ]]
}

run_phase() {
    local phase="$1"
    local validator="$2"
    shift 2
    local marker="${STATE_ROOT}/${phase}.done"
    local log="${LOG_ROOT}/${phase}.log"
    local status
    local -a pipeline_status

    CURRENT_PHASE="${phase}"
    if [[ -f "${marker}" ]]; then
        ((RESUME)) || {
            printf 'Unexpected completed marker without --resume: %s\n' "${marker}" >&2
            return 2
        }
        "${validator}"
        printf '[resume] validated and skipped %s\n' "${phase}" | tee -a "${log}"
        return 0
    fi

    {
        printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${phase}"
        printf 'command:'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "${log}"

    if "$@" 2>&1 | tee -a "${log}"; then
        pipeline_status=("${PIPESTATUS[@]}")
    else
        pipeline_status=("${PIPESTATUS[@]}")
    fi
    status="${pipeline_status[0]}"
    if ((status == 0 && pipeline_status[1] != 0)); then
        status="${pipeline_status[1]}"
    fi
    ((status == 0)) || return "${status}"
    "${validator}"

    local temporary_marker="${marker}.tmp.$$"
    printf 'completed_at=%s\nphase=%s\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${phase}" > "${temporary_marker}"
    mv -- "${temporary_marker}" "${marker}"
    printf '[complete] %s\n' "${phase}" | tee -a "${log}"
}

if [[ "${MODE}" == "SMOKE" ]]; then
    run_phase protocol_smoke validate_smoke_protocol \
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/protocol/build_smoke_protocol.py" \
        --train-list "${CANONICAL_TRAIN_LIST}" \
        --fold-csv "${CANONICAL_FOLD_CSV}" \
        --output "${PROTOCOL_ROOT}" \
        --samples-per-fold "${SMOKE_SAMPLES_PER_FOLD}"
fi

CACHE_COMMAND=(
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/precompute_frangi_graph_cache.py"
    --data-root "${KHANHHA_DATA_ROOT}"
    --list-file "${ACTIVE_TRAIN_LIST}"
    --cache-dir "${CACHE_ROOT}"
    --noise original
    --image-size 448
    --device "${DEVICE}"
    --scales "${FRANGI_SCALES[@]}"
    --radius "${FRANGI_RADIUS}"
    --ss "${FRANGI_SS}"
    --si "${FRANGI_SI}"
    --sa "${FRANGI_SA}"
    --tau "${FRANGI_TAU}"
    --min-rel-size "${FRANGI_MIN_REL_SIZE}"
    --graph-order "${FRANGI_GRAPH_ORDER}"
    --failure-log "${LOG_ROOT}/cache_failures.json"
)
run_phase cache_frangigraph validate_cache "${CACHE_COMMAND[@]}"

for fold in 0 1 2 3 4; do
    fold_output="${TRAIN_ROOT}/fold_${fold}"
    TRAIN_COMMAND=(
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/train_frangi_graph_residual.py"
        --data-root "${KHANHHA_DATA_ROOT}"
        --train-list "${ACTIVE_TRAIN_LIST}"
        --fold-csv "${ACTIVE_FOLD_CSV}"
        --protocol-manifest "${ACTIVE_PROTOCOL_MANIFEST}"
        --fold "${fold}"
        --graph-cache "${CACHE_ROOT}"
        --sam2-checkpoint "${SAM2_CHECKPOINT}"
        --baseline-checkpoint "${BASELINE_CHECKPOINT}"
        --output "${fold_output}"
        --hidden-channels "${HIDDEN_CHANNELS}"
        --raster-condition "${RASTER_CONDITION}"
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --base-lr "${BASE_LR}"
        --warmup-steps "${WARMUP_STEPS}"
        --poly-power "${POLY_POWER}"
        --weight-decay "${WEIGHT_DECAY}"
        --ce-weight "${CE_WEIGHT}"
        --topology-weight "${TOPOLOGY_WEIGHT}"
        --safety-weight "${SAFETY_WEIGHT}"
        --safety-margin "${SAFETY_MARGIN}"
        --skeleton-iterations "${SKELETON_ITERATIONS}"
        --gradient-clip "${GRADIENT_CLIP}"
        --threshold "${SEGMENTATION_THRESHOLD}"
        --val-every 1
        --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --amp-dtype "${AMP_DTYPE}"
    )
    if ((fold < 4)); then
        # Strict OOF guard: fold 4 is invisible to every gate-fit residual.
        TRAIN_COMMAND+=(--exclude-training-fold 4)
    fi
    if ((RESUME)) && [[ ! -f "${STATE_ROOT}/train_fold_${fold}.done" ]]; then
        if [[ -f "${fold_output}/config.json" && -f "${fold_output}/latest.pt" ]]; then
            TRAIN_COMMAND+=(--resume)
        elif [[ -e "${fold_output}" ]] && \
             [[ -n "$(find "${fold_output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            printf 'Fold %s has partial output but no resumable config/latest.pt pair: %s\n' \
                "${fold}" "${fold_output}" >&2
            exit 4
        fi
    fi
    validate_this_training_fold() { validate_training_fold "${fold}"; }
    run_phase "train_fold_${fold}" validate_this_training_fold "${TRAIN_COMMAND[@]}"
done

for fold in 0 1 2 3 4; do
    role="gate_fit"
    [[ "${fold}" == "4" ]] && role="gate_calibration"
    EVALUATE_COMMAND=(
        "${PYTHON_BIN}" "${CRACKSAM_ROOT}/evaluate_frangi_graph_residual.py"
        --data-root "${KHANHHA_DATA_ROOT}"
        --list-file "${ACTIVE_TRAIN_LIST}"
        --split train
        --dataset-name khanhha_train_oof
        --noise original
        --graph-cache "${CACHE_ROOT}"
        --sam2-checkpoint "${SAM2_CHECKPOINT}"
        --baseline-checkpoint "${BASELINE_CHECKPOINT}"
        --residual-checkpoint "${TRAIN_ROOT}/fold_${fold}/latest.pt"
        --raster-condition "${RASTER_CONDITION}"
        --output "${OOF_ROOT}/fold_${fold}"
        --role "${role}"
        --fold "${fold}"
        --group-folds-csv "${ACTIVE_FOLD_CSV}"
        --batch-size "${EVAL_BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --segmentation-threshold "${SEGMENTATION_THRESHOLD}"
        --label-minimum-gain "${LABEL_MINIMUM_GAIN}"
        --device "${DEVICE}"
        --amp-dtype "${AMP_DTYPE}"
    )
    validate_this_oof_fold() { validate_oof_fold "${fold}"; }
    run_phase "evaluate_oof_fold_${fold}" validate_this_oof_fold "${EVALUATE_COMMAND[@]}"
done

ASSEMBLE_COMMAND=(
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/assemble_logistic_gate_data.py"
    --fold-dir "0=${OOF_ROOT}/fold_0"
    --fold-dir "1=${OOF_ROOT}/fold_1"
    --fold-dir "2=${OOF_ROOT}/fold_2"
    --fold-dir "3=${OOF_ROOT}/fold_3"
    --fold-dir "4=${OOF_ROOT}/fold_4"
    --output "${ASSEMBLY_ROOT}"
)
run_phase assemble_oof_gate_data validate_assembly "${ASSEMBLE_COMMAND[@]}"

TRAIN_GATE_COMMAND=(
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/train_logistic_gate.py"
    --train-csv "${ASSEMBLY_ROOT}/gate_fit.csv"
    --calibration-csv "${ASSEMBLY_ROOT}/gate_calibration.csv"
    --oof-manifest "${ASSEMBLY_ROOT}/oof_manifest.json"
    --output "${GATE_JSON}"
    --git-commit "${GIT_COMMIT}"
    --label-minimum-gain "${LABEL_MINIMUM_GAIN}"
    --l2 "${GATE_L2}"
    --minimum-precision "${GATE_MINIMUM_PRECISION}"
    --minimum-selected "${GATE_MINIMUM_SELECTED}"
    --minimum-selected-groups "${GATE_MINIMUM_SELECTED_GROUPS}"
    --minimum-mean-delta-iou "${GATE_MINIMUM_MEAN_DELTA_IOU}"
    --severe-loss-threshold "${GATE_SEVERE_LOSS_THRESHOLD}"
    --maximum-severe-loss-rate "${GATE_MAXIMUM_SEVERE_LOSS_RATE}"
)
run_phase train_calibrate_logistic_gate validate_gate "${TRAIN_GATE_COMMAND[@]}"

EVALUATE_GATE_COMMAND=(
    "${PYTHON_BIN}" "${CRACKSAM_ROOT}/evaluate_logistic_gate.py"
    --gate-json "${GATE_JSON}"
    --oof-manifest "${ASSEMBLY_ROOT}/oof_manifest.json"
    --evaluation-dir "${OOF_ROOT}/fold_0"
    --evaluation-dir "${OOF_ROOT}/fold_1"
    --evaluation-dir "${OOF_ROOT}/fold_2"
    --evaluation-dir "${OOF_ROOT}/fold_3"
    --evaluation-dir "${OOF_ROOT}/fold_4"
    --output "${GATE_ANALYSIS_ROOT}"
)
run_phase evaluate_frozen_gate_oof validate_gate_analysis "${EVALUATE_GATE_COMMAND[@]}"

CURRENT_PHASE="complete"
printf '[complete] %s pilot: %s\n' "${MODE}" "${RUN_ROOT}"

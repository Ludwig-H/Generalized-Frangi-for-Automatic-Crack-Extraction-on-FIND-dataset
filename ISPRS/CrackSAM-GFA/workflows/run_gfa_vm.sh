#!/usr/bin/env bash
# Orchestration complète de CrackSAM-GFA sur une VM G4.
#
# Reprenable après préemption Spot : chaque étape écrit un jalon dans
# ${GFA_RUN_ROOT}/state et est sautée si son jalon existe. Aucune étape ne
# supprime les résultats d'une étape antérieure.
#
# La porte 0 (reproduction de la baseline) et la porte 1 (oracle de source)
# sont bloquantes : un échec interrompt le pipeline, conformément au
# pré-enregistrement de DESIGN.md.

set -euo pipefail

readonly REPO_ROOT="${GFA_REPO_ROOT:-${HOME}/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset}"
readonly GFA="${REPO_ROOT}/ISPRS/CrackSAM-GFA"
readonly LISTS="${REPO_ROOT}/ISPRS/CrackSAM/protocol/cracksam_paper/lists/lists_khanhha"
readonly CKPT_DIR="${REPO_ROOT}/ISPRS/CrackSAM/artifacts/vm_backup_20260714T1435Z_final_checkpoints"

readonly DATA_ROOT="${CRACKSAM2_DATA_ROOT:-${HOME}/cracksam2-data}"
readonly RUN_ROOT="${GFA_RUN_ROOT:-${HOME}/gfa-run}"
readonly STATE="${RUN_ROOT}/state"
readonly ANALYSIS_SAMPLES="${GFA_ANALYSIS_SAMPLES:-1500}"

mkdir -p "${STATE}" "${RUN_ROOT}/baseline" "${RUN_ROOT}/evidence" "${RUN_ROOT}/oracles"

log() { printf '\n[GFA %(%H:%M:%S)T] %s\n' -1 "$*"; }

done_marker() { [[ -f "${STATE}/$1.done" ]]; }
mark_done() { touch "${STATE}/$1.done"; }

step() {
    local name="$1"; shift
    if done_marker "${name}"; then
        log "SAUT ${name} (jalon présent)"
        return 0
    fi
    log "DÉBUT ${name}"
    "$@"
    mark_done "${name}"
    log "FIN ${name}"
}

# --- 0. Environnement ---------------------------------------------------------
check_gpu() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible'; \
print('torch', torch.__version__, torch.cuda.get_device_name(0))"
}

install_deps() {
    SAM2_BUILD_CUDA=0 pip install --quiet --no-build-isolation \
        -r "${REPO_ROOT}/ISPRS/CrackSAM/requirements-sam2.txt"
    pip install --quiet scikit-image joblib tqdm pandas matplotlib
    python -c "import sam2; print('sam2 prêt')"
}

prepare_data() {
    if [[ -d "${DATA_ROOT}/khanhha/test/images" ]]; then
        log "données déjà présentes dans ${DATA_ROOT}"
        return 0
    fi
    python "${REPO_ROOT}/ISPRS/CrackSAM/prepare_cracksam2_data.py" \
        --output "${DATA_ROOT}" --datasets khanhha road420 facade390
}

# --- 1. Porte 0 : reproduction de la baseline ---------------------------------
gate0() {
    python "${GFA}/scripts/01_cache_baseline.py" \
        --data-root "${DATA_ROOT}/khanhha" \
        --list-file "${LISTS}/test_vol.txt" \
        --checkpoint "${CKPT_DIR}/baseline_r4/best.pt" \
        --sam2-checkpoint "${CKPT_DIR}/foundation/sam2_hiera_large.pt" \
        --output "${RUN_ROOT}/baseline" \
        --split test_vol --noise original \
        --batch-size 8 --num-workers 12 --save-logits
}

baseline_analysis() {
    # Partition d'analyse : sous-ensemble déterministe du split train officiel.
    python "${GFA}/scripts/01_cache_baseline.py" \
        --data-root "${DATA_ROOT}/khanhha" \
        --list-file "${LISTS}/train.txt" \
        --checkpoint "${CKPT_DIR}/baseline_r4/best.pt" \
        --sam2-checkpoint "${CKPT_DIR}/foundation/sam2_hiera_large.pt" \
        --output "${RUN_ROOT}/baseline_analysis" \
        --split train --noise original --condition-name analysis \
        --batch-size 8 --num-workers 12 --max-samples "${ANALYSIS_SAMPLES}" \
        --save-logits
}

# --- 2. Évidence et fragments (CPU, 48 vCPU) ----------------------------------
calibrate_thresholds() {
    python "${GFA}/scripts/02_cache_evidence.py" \
        --data-root "${DATA_ROOT}/khanhha" --split train \
        --list-file "${LISTS}/train.txt" \
        --output "${RUN_ROOT}/evidence" \
        --thresholds "${RUN_ROOT}/evidence/thresholds.json" \
        --calibrate --calibration-samples 120
}

evidence_analysis() {
    python "${GFA}/scripts/02_cache_evidence.py" \
        --data-root "${DATA_ROOT}/khanhha" --split train \
        --list-file "${LISTS}/train.txt" \
        --output "${RUN_ROOT}/evidence" \
        --thresholds "${RUN_ROOT}/evidence/thresholds.json" \
        --max-samples "${ANALYSIS_SAMPLES}" --jobs -1
}

evidence_test() {
    python "${GFA}/scripts/02_cache_evidence.py" \
        --data-root "${DATA_ROOT}/khanhha" --split test \
        --list-file "${LISTS}/test_vol.txt" \
        --output "${RUN_ROOT}/evidence" \
        --thresholds "${RUN_ROOT}/evidence/thresholds.json" \
        --jobs -1
}

# --- 3. Porte 1 : oracle de source sur la partition d'analyse -----------------
gate1() {
    python "${GFA}/scripts/03_run_oracles.py" \
        --data-root "${DATA_ROOT}/khanhha" --split train \
        --fragments "${RUN_ROOT}/evidence/fragments_train.npz" \
        --index "${RUN_ROOT}/evidence/evidence_summary_train.json" \
        --z0-dir "${RUN_ROOT}/baseline_analysis/z0" \
        --output "${RUN_ROOT}/oracles" --save-utilities
    python - <<'PY'
import json, os, sys
root = os.environ["GFA_RUN_ROOT"] if "GFA_RUN_ROOT" in os.environ else os.path.expanduser("~/gfa-run")
summary = json.load(open(f"{root}/oracles/source_oracle_train.json", encoding="utf-8"))
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["gate1_verdict"] != "GO":
    print("\nPORTE 1 : NO-GO pré-enregistré.")
    if summary["negative_is_conclusive"]:
        print("La borne SUPÉRIEURE est elle aussi sous le seuil : réfutation concluante.")
    else:
        print("La borne supérieure dépasse le seuil : le no-go porte sur l'assignation,")
        print("pas sur la famille de candidats elle-même.")
    sys.exit(3)
PY
}

# --- 4. Arbitre et évaluation (seulement si porte 1 = GO) ---------------------
train_arbiter() {
    python "${GFA}/scripts/04_train_arbiter.py" \
        --run-root "${RUN_ROOT}" --folds 5 --epochs 40
}

evaluate() {
    python "${GFA}/scripts/05_evaluate.py" \
        --run-root "${RUN_ROOT}" \
        --data-root "${DATA_ROOT}/khanhha" \
        --sam2-checkpoint "${CKPT_DIR}/foundation/sam2_hiera_large.pt" \
        --checkpoint "${CKPT_DIR}/baseline_r4/best.pt" \
        --list-file "${LISTS}/test_vol.txt" \
        --controls none null permuted shifted random_support
}

main() {
    log "racine de run : ${RUN_ROOT}"
    step environnement check_gpu
    step dependances install_deps
    step donnees prepare_data
    step porte0_baseline gate0
    step baseline_analyse baseline_analysis
    step seuils calibrate_thresholds
    step evidence_analyse evidence_analysis
    step evidence_test evidence_test
    step porte1_oracle gate1
    step arbitre train_arbiter
    step evaluation evaluate
    log "TERMINÉ. Résultats dans ${RUN_ROOT}"
    log "Ne pas oublier : ./gcp-migration/stop_and_verify.sh depuis le poste local."
}

main "$@"

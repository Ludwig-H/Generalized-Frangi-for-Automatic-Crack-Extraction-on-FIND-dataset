#!/usr/bin/env bash
# Chaîne complète sur une VM G4, reprenable après préemption Spot.
#
# Chaque étape écrit un jalon dans ${IRT_RUN_ROOT}/state et est sautée si le
# jalon existe. Une préemption ne coûte donc au pire qu'une étape.
#
#   export IRT_DATA_ROOT="$HOME/irt-crack"
#   export IRT_RUN_ROOT="$HOME/irt-run"
#   export SAM2_CHECKPOINT="$HOME/checkpoints/sam2_hiera_large.pt"
#   export CRACKSAM_LORA_CHECKPOINT="$HOME/checkpoints/tol3_best.pt"
#   bash ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/workflows/run_irt_vm.sh
#
# Ce script ne démarre ni n'arrête aucune VM : ces actions sont explicites et
# passent par gcp-migration/start_and_verify.sh et stop_and_verify.sh.

set -euo pipefail

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${STUDY_ROOT}/../../.." && pwd)"

: "${IRT_DATA_ROOT:?définir IRT_DATA_ROOT (racine du jeu IRT-Crack décompressé)}"
: "${IRT_RUN_ROOT:?définir IRT_RUN_ROOT (dossier de travail des caches et résultats)}"
: "${SAM2_CHECKPOINT:?définir SAM2_CHECKPOINT (poids SAM 2 Hiera-L)}"
: "${CRACKSAM_LORA_CHECKPOINT:?définir CRACKSAM_LORA_CHECKPOINT (barreau tol3 de GeoLoRA)}"

DEVICE="${IRT_DEVICE:-cuda}"
SEEDS="${IRT_SEEDS:-13 37 73}"
STATE="${IRT_RUN_ROOT}/state"
DATA="${IRT_RUN_ROOT}/data"
CACHE="${IRT_RUN_ROOT}/cache"
RESULTS="${IRT_RUN_ROOT}/results"
mkdir -p "${STATE}" "${DATA}" "${CACHE}" "${RESULTS}"

cd "${STUDY_ROOT}"

step() {
  local name="$1"; shift
  if [[ -f "${STATE}/${name}.done" ]]; then
    echo "[jalon] ${name} — déjà fait"
    return 0
  fi
  echo
  echo "=== ${name} ==="
  "$@"
  touch "${STATE}/${name}.done"
}

manifest() {
  local args=(--dataset-root "${IRT_DATA_ROOT}" --output "${DATA}/manifest.csv")
  if [[ -n "${IRT_OFFICIAL_SPLIT:-}" ]]; then
    args+=(--official-split "${IRT_OFFICIAL_SPLIT}")
  else
    echo "AVERTISSEMENT : IRT_OFFICIAL_SPLIT n'est pas défini."
    echo "  Le split officiel 358/90 n'est pas distribué sur Zenodo ; il vit dans le"
    echo "  dossier 00_List du Google Drive du benchmark IRFusionFormer. Sans lui, un"
    echo "  split dérivé déterministe de même effectif est construit, et le rapport"
    echo "  doit le signaler."
  fi
  python scripts/00_build_manifest.py "${args[@]}"
}

audit() {
  python scripts/01_audit_dataset.py \
    --manifest "${DATA}/manifest.csv" \
    --thermal-encoding auto \
    --output "${RESULTS}/dataset_audit"
}

cache_thermal() {
  python scripts/03_cache_thermal_frangi.py \
    --manifest "${DATA}/manifest.csv" \
    --config configs/irt_signed_abstention_v1.yaml \
    --output "${CACHE}/thermal_frangi" \
    --device "${DEVICE}"
}

cache_baseline() {
  python scripts/02_cache_cracksam_logits.py \
    --manifest "${DATA}/manifest.csv" \
    --sam2-checkpoint "${SAM2_CHECKPOINT}" \
    --lora-checkpoint "${CRACKSAM_LORA_CHECKPOINT}" \
    --output "${CACHE}/baseline" \
    --device "${DEVICE}"
}

ceiling() {
  python scripts/08_correction_ceiling.py \
    --manifest "${DATA}/manifest.csv" \
    --split-file "${DATA}/split.json" \
    --baseline-cache "${CACHE}/baseline/manifest.json" \
    --output "${RESULTS}/correction_ceiling" \
    --split validation
  echo
  echo "PORTE DE PLAFOND : lire le tableau ci-dessus avant d'entraîner quoi que ce soit."
  echo "  Si la marge oracle-baseline est du même ordre que le plancher de détection,"
  echo "  relever delta_max dans les SEPT configurations, puis relancer — jamais après"
  echo "  avoir vu les résultats des bras."
  if [[ "${IRT_CONFIRM_CEILING:-}" != "1" ]]; then
    echo
    echo "Relancer avec IRT_CONFIRM_CEILING=1 pour poursuivre vers la campagne."
    return 1
  fi
}

ablations() {
  # shellcheck disable=SC2086
  python scripts/06_run_ablations.py \
    --protocol configs/ablation_matrix.yaml \
    --manifest "${DATA}/manifest.csv" \
    --split-file "${DATA}/split.json" \
    --baseline-cache "${CACHE}/baseline/manifest.json" \
    --thermal-cache "${CACHE}/thermal_frangi/manifest.json" \
    --output "${RESULTS}/ablation_matrix" \
    --seeds ${SEEDS} \
    --device "${DEVICE}" \
    --eval-device cpu
}

report() {
  python scripts/07_report.py \
    --results "${RESULTS}/ablation_matrix" \
    --protocol configs/ablation_matrix.yaml \
    --bootstrap 10000
}

step "00_tests"        python -m pytest tests -q
step "01_manifest"     manifest
step "02_audit"        audit
step "03_thermal"      cache_thermal
step "04_baseline"     cache_baseline
step "05_ceiling"      ceiling
step "06_ablations"    ablations
step "07_report"       report

echo
echo "Terminé. Résultats : ${RESULTS}/ablation_matrix"
echo
echo "AVANT D'ARRÊTER : les checkpoints ne survivent PAS à la destruction du disque"
echo "Spot. Ceux de la campagne GeoLoRA d'août ont été perdus ainsi — il ne reste"
echo "d'eux qu'un chemin absolu figé dans un JSON. Copier d'abord :"
echo "  cp -r ${RESULTS} ${IRT_RUN_ROOT}/vm_backup_\$(date -u +%Y%m%dT%H%MZ)/"
echo "  sha256sum ${RESULTS}/ablation_matrix/*/*/best.pt"
echo
echo "PUIS arrêter — en NOMMANT la cible. Les défauts de stop_and_verify.sh sont"
echo "europe-west4-a / frangi-blackwell-spot, c'est-à-dire la VM historique et non"
echo "celle-ci : sans ces deux variables, le script vise la mauvaise machine."
echo "  GCP_ZONE=europe-west8-c \\"
echo "  GCP_INSTANCE_NAME=cracksam-frangigraph-g4-spot-ew8c \\"
echo "  ${REPO_ROOT}/gcp-migration/stop_and_verify.sh"
echo "puis vérifier TERMINATED."

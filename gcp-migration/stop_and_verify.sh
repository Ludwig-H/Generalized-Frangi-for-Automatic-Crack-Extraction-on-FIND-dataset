#!/usr/bin/env bash
set -euo pipefail

readonly INSTANCE_NAME="frangi-blackwell-spot"
readonly ZONE="europe-west4-a"
readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly STOP_TIMEOUT_SECONDS=180

PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

instance_status() {
    gcloud compute instances describe "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format='value(status)'
}

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."

configured_project="$(gcloud config get-value project 2>/dev/null || true)"
if [[ "${configured_project}" != "${PROJECT_ID}" ]]; then
    die "Projet actif « ${configured_project:-non configuré} » différent de « ${PROJECT_ID} »."
fi

status="$(instance_status)" || die "Instance ${INSTANCE_NAME} introuvable dans ${PROJECT_ID}/${ZONE}."
if [[ "${status}" == "TERMINATED" ]]; then
    printf '[OK] %s est déjà arrêtée (état GCE TERMINATED).\n' "${INSTANCE_NAME}"
    exit 0
fi

printf '[STOP] État actuel de %s : %s.\n' "${INSTANCE_NAME}" "${status}"
[[ -t 0 ]] || die "Confirmation interactive requise pour arrêter la VM."
expected_confirmation="STOPPER ${INSTANCE_NAME}"
read -r -p "Tapez exactement « ${expected_confirmation} » : " confirmation
[[ "${confirmation}" == "${expected_confirmation}" ]] || die "Arrêt annulé."

gcloud compute instances stop "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet

deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
    if current_status="$(instance_status 2>/dev/null)"; then
        status="${current_status}"
        if [[ "${status}" == "TERMINATED" ]]; then
            printf '[OK] %s est arrêtée et vérifiée (état GCE TERMINATED).\n' "${INSTANCE_NAME}"
            exit 0
        fi
        printf '[ATTENTE] État GCE : %s\n' "${status}"
    else
        printf '[ATTENTION] Lecture GCE transitoirement impossible ; nouvel essai dans 5 s.\n' >&2
    fi
    sleep 5
done

die "Délai dépassé : état final ${status}. Vérifiez immédiatement dans la console GCP."

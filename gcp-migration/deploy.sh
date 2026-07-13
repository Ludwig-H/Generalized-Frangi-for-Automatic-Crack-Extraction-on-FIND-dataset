#!/usr/bin/env bash
set -euo pipefail

readonly INSTANCE_NAME="frangi-blackwell-spot"
readonly ZONE="europe-west4-a"
readonly MACHINE_TYPE="g4-standard-48"
readonly IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
readonly IMAGE_PROJECT="deeplearning-platform-release"
readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"

PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
MAX_RUN_DURATION="${GCP_MAX_RUN_DURATION:-8h}"

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."

configured_project="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "${configured_project}" || "${configured_project}" == "(unset)" ]]; then
    die "Aucun projet gcloud configuré. Exécutez : gcloud config set project ${PROJECT_ID}"
fi
if [[ "${configured_project}" != "${PROJECT_ID}" ]]; then
    die "Le projet actif (${configured_project}) diffère de ${PROJECT_ID}. Configurez-le explicitement ou exportez GCP_PROJECT_ID."
fi

account="$(gcloud config get-value account 2>/dev/null || true)"
[[ -n "${account}" && "${account}" != "(unset)" ]] || die "Aucun compte gcloud actif."

if ! existing_instance="$(gcloud compute instances list \
    --project="${PROJECT_ID}" \
    --zones="${ZONE}" \
    --filter="name=${INSTANCE_NAME}" \
    --limit=1 \
    --format='value(name)')"; then
    die "Impossible de vérifier si ${INSTANCE_NAME} existe déjà ; création refusée."
fi
if [[ -n "${existing_instance}" ]]; then
    die "L'instance ${INSTANCE_NAME} existe déjà. Ce script ne la supprime ni ne la recrée."
fi

printf '%s\n' \
    "[DEPLOY] Création demandée :" \
    "  compte      : ${account}" \
    "  projet      : ${PROJECT_ID}" \
    "  instance    : ${INSTANCE_NAME}" \
    "  zone        : ${ZONE}" \
    "  machine     : ${MACHINE_TYPE} (1 x RTX PRO 6000 Blackwell, 96 Go)" \
    "  image       : ${IMAGE_FAMILY} (CUDA 12.9 / pilote 580)" \
    "  Spot        : oui, action de terminaison STOP" \
    "  coupe-circuit GCE : ${MAX_RUN_DURATION}"

[[ -t 0 ]] || die "Confirmation interactive requise pour créer une ressource facturable."
expected_confirmation="CREER ${INSTANCE_NAME} DANS ${PROJECT_ID}"
read -r -p "Tapez exactement « ${expected_confirmation} » : " confirmation
[[ "${confirmation}" == "${expected_confirmation}" ]] || die "Création annulée."

gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model="SPOT" \
    --instance-termination-action="STOP" \
    --max-run-duration="${MAX_RUN_DURATION}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="100GB" \
    --boot-disk-type="hyperdisk-balanced" \
    --network-interface="network=default,access-config-type=ONE_TO_ONE_NAT" \
    --metadata="install-nvidia-driver=true"

printf '[SUCCÈS] %s créée. Lancez immédiatement le preflight Blackwell.\n' "${INSTANCE_NAME}"

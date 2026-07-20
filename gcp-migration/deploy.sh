#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
readonly REGION="${GCP_REGION:-europe-west4}"
readonly ZONE="${GCP_ZONE:-europe-west4-a}"
readonly INSTANCE_NAME="${GCP_INSTANCE_NAME:-frangi-blackwell-spot}"
readonly MACHINE_TYPE="g4-standard-48"
readonly IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
readonly IMAGE_PROJECT="deeplearning-platform-release"
readonly BOOT_DISK_IOPS=3600
readonly BOOT_DISK_THROUGHPUT=290
readonly BOOT_DISK_SIZE_GB="${GCP_BOOT_DISK_SIZE_GB:-100}"
readonly NETWORK_INTERFACE="${GCP_NETWORK_INTERFACE:-network=default,nic-type=GVNIC}"
readonly RUNTIME_SERVICE_ACCOUNT="${GCP_RUNTIME_SERVICE_ACCOUNT:-}"
readonly MAX_RUN_DURATION="${GCP_MAX_RUN_DURATION:-8h}"

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

duration_seconds() {
    python3 - "$1" <<'PY'
import re
import sys

match = re.fullmatch(r"([1-9][0-9]*)([smh])", sys.argv[1])
if not match:
    raise SystemExit(2)
value = int(match.group(1))
factor = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
print(value * factor)
PY
}

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."
command -v python3 >/dev/null 2>&1 || die "python3 est requis."
[[ "${ZONE}" == "${REGION}-"* ]] || \
    die "La zone ${ZONE} n'appartient pas à la région déclarée ${REGION}."

if ! max_run_seconds="$(duration_seconds "${MAX_RUN_DURATION}")"; then
    die "GCP_MAX_RUN_DURATION doit être une durée GCE simple, par exemple 3600s, 60m ou 8h."
fi
((max_run_seconds >= 30 && max_run_seconds <= 28800)) || \
    die "GCP_MAX_RUN_DURATION doit rester entre 30 secondes et 8 heures."
[[ "${BOOT_DISK_SIZE_GB}" =~ ^[1-9][0-9]*$ ]] || \
    die "GCP_BOOT_DISK_SIZE_GB doit être un entier strictement positif."

configured_project="$(gcloud config get-value project 2>/dev/null || true)"
[[ "${configured_project}" == "${PROJECT_ID}" ]] || \
    die "Projet actif « ${configured_project:-non configuré} » différent de « ${PROJECT_ID} »."
account="$(gcloud config get-value account 2>/dev/null || true)"
[[ -n "${account}" && "${account}" != "(unset)" ]] || die "Aucun compte gcloud actif."

if [[ "${ZONE}" == *-ai* ]]; then
    ai_zone_status="$(gcloud compute preview-features describe ai-zones-visibility \
        --project="${PROJECT_ID}" \
        --format='value(activationStatus)')" || \
        die "Impossible de lire l'activation de ai-zones-visibility."
    if [[ "${ai_zone_status}" != "ENABLED" && \
        "${ai_zone_status}" != "ACTIVATION_STATE_ENABLED" ]]; then
        die "La zone IA ${ZONE} exige ai-zones-visibility=ENABLED; activation automatique interdite."
    fi
fi

gcloud compute machine-types describe "${MACHINE_TYPE}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --format='value(name)' >/dev/null || \
    die "${MACHINE_TYPE} n'est pas visible dans ${ZONE}."

require_external_address=0
if [[ "${NETWORK_INTERFACE}" != *"no-address"* ]]; then
    require_external_address=1
fi
GCP_PROJECT_ID="${PROJECT_ID}" \
GCP_REGION="${REGION}" \
GCP_ZONE="${ZONE}" \
GCP_BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB}" \
GCP_REQUIRE_EXTERNAL_ADDRESS="${require_external_address}" \
    "$(dirname "${BASH_SOURCE[0]}")/check_quotas.sh" || \
    die "Les quotas strictement nécessaires à une G4 Spot ne sont pas disponibles."

resolved_image="$(gcloud compute images describe-from-family "${IMAGE_FAMILY}" \
    --project="${IMAGE_PROJECT}" \
    --format='value(name)')" || \
    die "Impossible de résoudre la famille ${IMAGE_PROJECT}/${IMAGE_FAMILY}."
[[ -n "${resolved_image}" ]] || die "La famille d'image n'a renvoyé aucun nom."

existing_instance="$(gcloud compute instances list \
    --project="${PROJECT_ID}" \
    --zones="${ZONE}" \
    --filter="name=${INSTANCE_NAME}" \
    --limit=1 \
    --format='value(name)')" || \
    die "Impossible de vérifier l'existence de ${INSTANCE_NAME}."
[[ -z "${existing_instance}" ]] || \
    die "L'instance ${INSTANCE_NAME} existe déjà; ce script ne l'écrase pas."

active_g4="$(gcloud compute instances list \
    --project="${PROJECT_ID}" \
    --filter='machineType:g4-standard-48 AND status!=TERMINATED' \
    --format='value(name,zone.basename(),status)')" || \
    die "Impossible de vérifier les autres G4 actives."
[[ -z "${active_g4}" ]] || \
    die "Une autre G4 est active ou transitoire; création refusée : ${active_g4}"

printf '%s\n' \
    '[DEPLOY] Création facturable demandée :' \
    "  compte      : ${account}" \
    "  projet      : ${PROJECT_ID}" \
    "  instance    : ${INSTANCE_NAME}" \
    "  région/zone : ${REGION}/${ZONE}" \
    "  machine     : ${MACHINE_TYPE} (48 vCPU, 180 Go RAM, RTX PRO 6000 96 Go)" \
    "  image       : ${resolved_image}" \
    "  disque      : Hyperdisk Balanced ${BOOT_DISK_SIZE_GB} Go" \
    "  réseau      : ${NETWORK_INTERFACE}" \
    "  identité VM : ${RUNTIME_SERVICE_ACCOUNT:-aucune}" \
    "  Spot STOP   : oui, durée maximale ${MAX_RUN_DURATION}"

[[ -t 0 ]] || die "Confirmation interactive requise pour créer une ressource facturable."
expected_confirmation="CREER ${INSTANCE_NAME} DANS ${PROJECT_ID}/${ZONE}"
read -r -p "Tapez exactement « ${expected_confirmation} » : " confirmation
[[ "${confirmation}" == "${expected_confirmation}" ]] || die "Création annulée."

service_account_args=(--no-service-account --no-scopes)
if [[ -n "${RUNTIME_SERVICE_ACCOUNT}" ]]; then
    service_account_args=(
        "--service-account=${RUNTIME_SERVICE_ACCOUNT}"
        "--scopes=cloud-platform"
    )
fi

gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model="SPOT" \
    --instance-termination-action="STOP" \
    --max-run-duration="${MAX_RUN_DURATION}" \
    --maintenance-policy="TERMINATE" \
    --no-restart-on-failure \
    --image="${resolved_image}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type="hyperdisk-balanced" \
    --boot-disk-provisioned-iops="${BOOT_DISK_IOPS}" \
    --boot-disk-provisioned-throughput="${BOOT_DISK_THROUGHPUT}" \
    --network-interface="${NETWORK_INTERFACE}" \
    --metadata="enable-oslogin=TRUE" \
    --labels="project=generalized-frangi,workload=cracksam-frangi,managed-by=manual-script" \
    --deletion-protection \
    "${service_account_args[@]}"

printf '[SÉCURITÉ] La création démarre la VM; arrêt immédiat et certification.\n'
GCP_PROJECT_ID="${PROJECT_ID}" \
GCP_REGION="${REGION}" \
GCP_ZONE="${ZONE}" \
GCP_INSTANCE_NAME="${INSTANCE_NAME}" \
    "$(dirname "${BASH_SOURCE[0]}")/stop_and_verify.sh" --yes

printf '[SUCCÈS] %s créée puis certifiée TERMINATED. Utilisez start_and_verify.sh pour une session.\n' \
    "${INSTANCE_NAME}"

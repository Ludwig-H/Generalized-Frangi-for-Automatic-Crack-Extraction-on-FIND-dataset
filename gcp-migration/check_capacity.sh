#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly MACHINE_TYPE="g4-standard-48"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
readonly REGION="${GCP_REGION:-europe-west4}"
readonly ZONE="${GCP_ZONE:-europe-west4-a}"

die() {
    printf '[ÉCHEC] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage : ./gcp-migration/check_capacity.sh

Contrôle strictement non mutateur :
  1. vérifie que g4-standard-48 est visible dans GCP_ZONE ;
  2. affiche les zones où ce type est visible pour le projet ;
  3. interroge Capacity Advisor Spot pour la zone choisie.

Un score favorable ne réserve rien et ne garantit pas qu'une création réussira.
Le script ne choisit ni ne crée automatiquement une autre zone.
EOF
}

if (($# > 0)); then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Option inconnue : $1"
            ;;
    esac
fi

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."

configured_project="$(gcloud config get-value project 2>/dev/null || true)"
[[ "${configured_project}" == "${PROJECT_ID}" ]] || \
    die "Projet actif « ${configured_project:-non configuré} » différent de « ${PROJECT_ID} »."

printf '[CHECK] Visibilité de %s dans %s/%s...\n' \
    "${MACHINE_TYPE}" "${PROJECT_ID}" "${ZONE}"
gcloud compute machine-types describe "${MACHINE_TYPE}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --format='table(name,zone.basename(),guestCpus,memoryMb)' >/dev/null || \
    die "${MACHINE_TYPE} n'est pas visible dans ${ZONE} pour ce projet."

printf '[INFO] Zones visibles pour %s (ceci ne mesure pas le stock instantané) :\n' \
    "${MACHINE_TYPE}"
gcloud compute machine-types list \
    --project="${PROJECT_ID}" \
    --filter="name=${MACHINE_TYPE}" \
    --format='table(zone.basename():label=ZONE,name:label=MACHINE)'

printf '[CHECK] Capacity Advisor Spot pour %s/%s...\n' "${REGION}" "${ZONE}"
if ! gcloud beta compute advice capacity \
    --project="${PROJECT_ID}" \
    --provisioning-model=SPOT \
    --instance-selection-machine-types="${MACHINE_TYPE}" \
    --target-distribution-shape=ANY_SINGLE_ZONE \
    --size=1 \
    --region="${REGION}" \
    --zones="${ZONE}"; then
    die "Capacity Advisor est indisponible ou n'a renvoyé aucun conseil exploitable."
fi

printf '%s\n' \
    '[INFO] Le conseil de capacité est ponctuel. Quota, visibilité et capacité' \
    '       sont trois contrôles distincts; seule une création explicitement' \
    '       autorisée prouve que la ressource était disponible.'

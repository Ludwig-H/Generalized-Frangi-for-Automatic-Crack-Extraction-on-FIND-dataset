#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
readonly ZONE="${GCP_ZONE:-europe-west4-a}"
readonly INSTANCE_NAME="${GCP_INSTANCE_NAME:-frangi-blackwell-spot}"
readonly STOP_TIMEOUT_SECONDS=300
readonly GCLOUD_READ_TIMEOUT_SECONDS=30
readonly GCLOUD_MUTATION_TIMEOUT_SECONDS=180

ASSUME_YES=0
EXPECTED_LAST_START_TIMESTAMP=""

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage : ./gcp-migration/stop_and_verify.sh [options]

Options :
  --yes
  --expected-last-start-timestamp HORODATAGE
  -h, --help

Arrête uniquement GCP_PROJECT_ID/GCP_ZONE/GCP_INSTANCE_NAME et exige l'état
TERMINATED. La génération optionnelle empêche d'arrêter une session redémarrée
par un autre contrôleur.
EOF
}

while (($# > 0)); do
    case "$1" in
        --yes)
            ASSUME_YES=1
            shift
            ;;
        --expected-last-start-timestamp)
            (($# >= 2)) || die "Valeur manquante après --expected-last-start-timestamp."
            [[ -z "${EXPECTED_LAST_START_TIMESTAMP}" ]] || die "Option de génération répétée."
            EXPECTED_LAST_START_TIMESTAMP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Option inconnue : $1"
            ;;
    esac
done

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."
command -v timeout >/dev/null 2>&1 || die "GNU timeout est requis."

gcloud_read() {
    timeout --kill-after=10s "${GCLOUD_READ_TIMEOUT_SECONDS}s" gcloud "$@"
}

gcloud_mutation() {
    timeout --kill-after=10s "${GCLOUD_MUTATION_TIMEOUT_SECONDS}s" gcloud "$@"
}

instance_field() {
    gcloud_read compute instances describe "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format="value($1)"
}

verify_generation() {
    local observed=""
    [[ -n "${EXPECTED_LAST_START_TIMESTAMP}" ]] || return 0
    observed="$(instance_field lastStartTimestamp)" || \
        die "Impossible de relire lastStartTimestamp."
    [[ "${observed}" == "${EXPECTED_LAST_START_TIMESTAMP}" ]] || \
        die "Génération différente (${observed:-vide}); arrêt refusé pour protéger une session concurrente."
}

verify_target_identity() {
    local label=""
    local machine_type=""

    machine_type="$(instance_field 'machineType.basename()')" || \
        die "Type de machine illisible."
    [[ "${machine_type}" == "g4-standard-48" ]] || \
        die "La cible est ${machine_type:-inconnue}, pas g4-standard-48."

    label="$(instance_field 'labels.project')" || die "Label project illisible."
    if [[ "${label}" == "generalized-frangi" ]]; then
        return 0
    fi

    if [[ -z "${label}" && \
        "${PROJECT_ID}" == "${DEFAULT_PROJECT_ID}" && \
        "${ZONE}" == "europe-west4-a" && \
        "${INSTANCE_NAME}" == "frangi-blackwell-spot" ]]; then
        printf '[ATTENTION] Cible historique sans label acceptée par triplet exact; adoption GCP encore requise.\n' >&2
        return 0
    fi
    die "La cible ne porte pas labels.project=generalized-frangi et n'est pas le triplet legacy autorisé."
}

report_other_g4_instances() {
    local inventory=""
    if inventory="$(gcloud_read compute instances list \
        --project="${PROJECT_ID}" \
        --filter='machineType:g4-standard' \
        --format='csv[no-heading](name,zone.basename(),status)')"; then
        printf '[INFO] Inventaire G4 du projet (aucune autre cible n’est modifiée) :\n'
        printf '%s\n' "${inventory:-  aucune}"
    else
        printf '[ATTENTION] Inventaire des autres G4 illisible.\n' >&2
    fi
}

configured_project="$(gcloud_read config get-value project 2>/dev/null || true)"
[[ "${configured_project}" == "${PROJECT_ID}" ]] || \
    die "Projet actif « ${configured_project:-non configuré} » différent de « ${PROJECT_ID} »."

existing_instance="$(gcloud_read compute instances list \
    --project="${PROJECT_ID}" \
    --zones="${ZONE}" \
    --filter="name=${INSTANCE_NAME}" \
    --limit=1 \
    --format='value(name)')" || die "Impossible de vérifier la cible."
[[ "${existing_instance}" == "${INSTANCE_NAME}" ]] || \
    die "Instance ${INSTANCE_NAME} absente de ${PROJECT_ID}/${ZONE}."

verify_target_identity
verify_generation
status="$(instance_field status)" || die "État de la cible illisible."

if [[ "${status}" == "TERMINATED" ]]; then
    verify_generation
    printf '[OK] %s est déjà TERMINATED.\n' "${INSTANCE_NAME}"
    report_other_g4_instances
    exit 0
fi

printf '[STOP] Cible exacte %s/%s/%s, état %s.\n' \
    "${PROJECT_ID}" "${ZONE}" "${INSTANCE_NAME}" "${status}"
if [[ "${status}" != "STOPPING" && ${ASSUME_YES} -eq 0 ]]; then
    [[ -t 0 ]] || die "Confirmation interactive requise; --yes exige une autorisation préalable."
    expected_confirmation="STOPPER ${INSTANCE_NAME} DANS ${ZONE}"
    read -r -p "Tapez exactement « ${expected_confirmation} » : " confirmation
    [[ "${confirmation}" == "${expected_confirmation}" ]] || die "Arrêt annulé."
fi

if [[ "${status}" != "STOPPING" ]]; then
    verify_generation
    gcloud_mutation compute instances stop "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --quiet || die "La commande d'arrêt a échoué ou expiré."
fi

deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
    if status="$(instance_field status 2>/dev/null)"; then
        if [[ "${status}" == "TERMINATED" ]]; then
            verify_generation
            printf '[OK] %s est arrêtée et certifiée TERMINATED.\n' "${INSTANCE_NAME}"
            report_other_g4_instances
            exit 0
        fi
        printf '[ATTENTE] État GCE : %s\n' "${status}"
    fi
    sleep 5
done

die "Délai dépassé; dernier état ${status:-inconnu}. Vérifiez immédiatement la console GCP."

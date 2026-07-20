#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
readonly REGION="${GCP_REGION:-europe-west4}"
readonly ZONE="${GCP_ZONE:-europe-west4-a}"
readonly INSTANCE_NAME="${GCP_INSTANCE_NAME:-frangi-blackwell-spot}"
readonly GUEST_SHUTDOWN_MINUTES="${GCP_GUEST_SHUTDOWN_MINUTES:-240}"
readonly START_TIMEOUT_SECONDS=300

ASSUME_YES=0
SESSION_STARTED=0
START_GENERATION=""
PRE_START_GENERATION=""

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage : ./gcp-migration/start_and_verify.sh [--yes]

Vérifie la cible G4 Spot, la démarre, identifie sa génération et arme un
shutdown invité. En cas d'échec avant certification de la garde, le script
tente d'arrêter uniquement cette génération.
EOF
}

while (($# > 0)); do
    case "$1" in
        --yes)
            ASSUME_YES=1
            shift
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
[[ "${GUEST_SHUTDOWN_MINUTES}" =~ ^[1-9][0-9]*$ ]] || \
    die "GCP_GUEST_SHUTDOWN_MINUTES doit être un entier."
((GUEST_SHUTDOWN_MINUTES >= 5 && GUEST_SHUTDOWN_MINUTES <= 480)) || \
    die "Le shutdown invité doit rester entre 5 et 480 minutes."
[[ "${ZONE}" == "${REGION}-"* ]] || \
    die "La zone ${ZONE} n'appartient pas à ${REGION}."

gcloud_read() {
    timeout --kill-after=10s 30s gcloud "$@"
}

gcloud_mutation() {
    timeout --kill-after=10s 180s gcloud "$@"
}

instance_field() {
    gcloud_read compute instances describe "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format="value($1)"
}

verify_identity() {
    local label=""
    local machine_type=""

    machine_type="$(instance_field 'machineType.basename()')" || return 1
    [[ "${machine_type}" == "g4-standard-48" ]] || \
        die "La cible est ${machine_type:-inconnue}, pas g4-standard-48."
    label="$(instance_field 'labels.project')" || return 1
    if [[ "${label}" == "generalized-frangi" ]]; then
        return 0
    fi
    if [[ -z "${label}" && \
        "${PROJECT_ID}" == "${DEFAULT_PROJECT_ID}" && \
        "${ZONE}" == "europe-west4-a" && \
        "${INSTANCE_NAME}" == "frangi-blackwell-spot" ]]; then
        printf '[ATTENTION] Mode legacy borné au triplet historique sans label.\n' >&2
        return 0
    fi
    die "Cible sans labels.project=generalized-frangi et hors triplet legacy."
}

emergency_stop() {
    local exit_code=$?
    trap - EXIT INT TERM
    if ((SESSION_STARTED == 1)); then
        if [[ -z "${START_GENERATION}" ]]; then
            printf '%s\n' \
                '[URGENCE] Démarrage non certifié et génération lastStartTimestamp inconnue.' \
                '[URGENCE] Aucun arrêt automatique non versionné n’est autorisé; vérifiez immédiatement la cible :' \
                "  ${PROJECT_ID}/${ZONE}/${INSTANCE_NAME}" >&2
        else
            printf '[SÉCURITÉ] Session non certifiée; arrêt ciblé de la génération %s.\n' \
                "${START_GENERATION}" >&2
            GCP_PROJECT_ID="${PROJECT_ID}" \
            GCP_REGION="${REGION}" \
            GCP_ZONE="${ZONE}" \
            GCP_INSTANCE_NAME="${INSTANCE_NAME}" \
                "$(dirname "${BASH_SOURCE[0]}")/stop_and_verify.sh" \
                --yes \
                --expected-last-start-timestamp "${START_GENERATION}" || \
                printf '[URGENCE] Arrêt non certifié; vérifiez immédiatement la console GCP.\n' >&2
        fi
    fi
    exit "${exit_code}"
}
trap emergency_stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

configured_project="$(gcloud_read config get-value project 2>/dev/null || true)"
[[ "${configured_project}" == "${PROJECT_ID}" ]] || \
    die "Projet actif « ${configured_project:-non configuré} » différent de « ${PROJECT_ID} »."

verify_identity
status="$(instance_field status)" || die "État de la cible illisible."
[[ "${status}" == "TERMINATED" ]] || \
    die "La cible doit être TERMINATED avant ce point d'entrée; état actuel ${status}."
PRE_START_GENERATION="$(instance_field lastStartTimestamp 2>/dev/null || true)"

provisioning="$(instance_field 'scheduling.provisioningModel')"
termination_action="$(instance_field 'scheduling.instanceTerminationAction')"
maintenance="$(instance_field 'scheduling.onHostMaintenance')"
automatic_restart="$(instance_field 'scheduling.automaticRestart')"
max_run_seconds="$(instance_field 'scheduling.maxRunDuration.seconds')"

[[ "${provisioning}" == "SPOT" ]] || die "Provisioning ${provisioning:-inconnu}, SPOT attendu."
[[ "${termination_action}" == "STOP" ]] || die "Action ${termination_action:-inconnue}, STOP attendu."
[[ "${maintenance}" == "TERMINATE" ]] || die "Maintenance ${maintenance:-inconnue}, TERMINATE attendu."
[[ "${automatic_restart}" == "False" || "${automatic_restart}" == "false" ]] || \
    die "Le redémarrage automatique doit être désactivé."
[[ "${max_run_seconds}" =~ ^[0-9]+$ ]] || die "maxRunDuration illisible."
((max_run_seconds >= GUEST_SHUTDOWN_MINUTES * 60 + 300)) || \
    die "Le shutdown invité doit précéder maxRunDuration d'au moins cinq minutes."
((max_run_seconds <= 28800)) || die "maxRunDuration dépasse huit heures."

printf '%s\n' \
    '[START] Session facturable demandée :' \
    "  cible       : ${PROJECT_ID}/${ZONE}/${INSTANCE_NAME}" \
    '  machine     : g4-standard-48 Spot, action STOP' \
    "  garde GCE   : ${max_run_seconds} secondes" \
    "  garde invité: ${GUEST_SHUTDOWN_MINUTES} minutes"

if ((ASSUME_YES == 0)); then
    [[ -t 0 ]] || die "Confirmation interactive requise."
    expected_confirmation="DEMARRER ${INSTANCE_NAME} DANS ${ZONE}"
    read -r -p "Tapez exactement « ${expected_confirmation} » : " confirmation
    [[ "${confirmation}" == "${expected_confirmation}" ]] || die "Démarrage annulé."
fi

SESSION_STARTED=1
gcloud_mutation compute instances start "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet

deadline=$((SECONDS + START_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
    observed_generation="$(instance_field lastStartTimestamp 2>/dev/null || true)"
    if [[ -n "${observed_generation}" && \
        "${observed_generation}" != "${PRE_START_GENERATION}" ]]; then
        if [[ -n "${START_GENERATION}" && \
            "${observed_generation}" != "${START_GENERATION}" ]]; then
            die "La génération a changé pendant la certification; arrêt refusé."
        fi
        START_GENERATION="${observed_generation}"
    fi
    status="$(instance_field status 2>/dev/null || true)"
    if [[ "${status}" == "RUNNING" && -n "${START_GENERATION}" ]]; then
        break
    fi
    if [[ "${status}" == "TERMINATED" || "${status}" == "STOPPING" ]]; then
        die "La VM a été préemptée ou arrêtée pendant le démarrage (${status})."
    fi
    printf '[ATTENTE] État GCE : %s\n' "${status:-illisible}"
    sleep 5
done
[[ "${status}" == "RUNNING" ]] || die "Délai de démarrage dépassé."

guest_command="sudo -n shutdown -P +${GUEST_SHUTDOWN_MINUTES} 'Coupe-circuit CrackSAM G4' && proof=\$(sudo -n shutdown --show 2>/dev/null || sudo -n cat /run/systemd/shutdown/scheduled) && test -n \"\${proof}\" && printf '%s\\n' \"\${proof}\""
guard_deadline=$((SECONDS + 180))
guard_output=""
while ((SECONDS < guard_deadline)); do
    if guard_output="$(timeout --kill-after=10s 45s gcloud compute ssh "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --quiet \
        --command="${guest_command}" 2>&1)"; then
        [[ -n "${guard_output}" ]] || die "La garde invitée n'a produit aucune preuve."
        break
    fi
    printf '[ATTENTE] SSH ou systemd non prêt; nouvel essai dans 10 s.\n'
    sleep 10
done
[[ -n "${guard_output}" ]] || die "Impossible d'armer et relire le shutdown invité."

printf '%s\n' "${guard_output}"
printf '[OK] Génération %s démarrée; garde invitée certifiée.\n' "${START_GENERATION}"
printf '%s\n' \
    '[SUITE] Connectez-vous puis lancez :' \
    '  ./gcp-migration/blackwell_preflight.sh' \
    '[FERMETURE] À la fin :' \
    "  GCP_ZONE=${ZONE} GCP_INSTANCE_NAME=${INSTANCE_NAME} ./gcp-migration/stop_and_verify.sh"

SESSION_STARTED=0
trap - EXIT INT TERM

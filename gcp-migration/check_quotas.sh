#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly REGION="europe-west4"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"

die() {
    printf '[ÉCHEC] %s\n' "$*" >&2
    exit 1
}

quota_values() {
    local quota_json="$1"
    local metric="$2"

    jq -r --arg metric "${metric}" '
        [.quotas[]? | select(.metric == $metric)] | first |
        if . == null then
            "UNKNOWN UNKNOWN"
        else
            "\(.limit // 0) \(.usage // 0)"
        end
    ' <<<"${quota_json}"
}

quota_available() {
    local limit="$1"
    local usage="$2"

    if [[ "${limit}" == "UNKNOWN" || "${usage}" == "UNKNOWN" ]]; then
        printf 'UNKNOWN'
    else
        jq -n --arg limit "${limit}" --arg usage "${usage}" \
            '($limit | tonumber) - ($usage | tonumber)'
    fi
}

quota_is_sufficient() {
    local available="$1"
    local required="$2"

    [[ "${available}" != "UNKNOWN" ]] &&
        jq -e -n --arg available "${available}" --arg required "${required}" \
            '($available | tonumber) >= ($required | tonumber)' >/dev/null
}

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."
command -v jq >/dev/null 2>&1 || die "jq est introuvable."

CONFIGURED_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ "${CONFIGURED_PROJECT}" != "${PROJECT_ID}" ]]; then
    die "Projet actif « ${CONFIGURED_PROJECT:-non configuré} » différent de « ${PROJECT_ID} »."
fi

printf '[CHECK] Quotas disponibles pour g4-standard-48 dans %s/%s...\n' \
    "${PROJECT_ID}" "${REGION}"

PROJECT_DATA="$(gcloud compute project-info describe \
    --project="${PROJECT_ID}" \
    --format=json)"
REGIONAL_DATA="$(gcloud compute regions describe "${REGION}" \
    --project="${PROJECT_ID}" \
    --format=json)"

read -r GLOBAL_GPU_LIMIT GLOBAL_GPU_USAGE < <(
    quota_values "${PROJECT_DATA}" "GPUS_ALL_REGIONS"
)
read -r CPUS_SPOT_LIMIT CPUS_SPOT_USAGE < <(
    quota_values "${REGIONAL_DATA}" "PREEMPTIBLE_CPUS"
)
read -r RTX_SPOT_LIMIT RTX_SPOT_USAGE < <(
    quota_values "${REGIONAL_DATA}" "PREEMPTIBLE_NVIDIA_RTX_PRO_6000_GPUS"
)

GLOBAL_GPU_AVAILABLE="$(quota_available "${GLOBAL_GPU_LIMIT}" "${GLOBAL_GPU_USAGE}")"
CPUS_SPOT_AVAILABLE="$(quota_available "${CPUS_SPOT_LIMIT}" "${CPUS_SPOT_USAGE}")"
RTX_SPOT_AVAILABLE="$(quota_available "${RTX_SPOT_LIMIT}" "${RTX_SPOT_USAGE}")"

printf '%-37s %10s %10s %12s %10s\n' \
    "MÉTRIQUE" "LIMITE" "UTILISÉ" "DISPONIBLE" "REQUIS"
printf '%-37s %10s %10s %12s %10s\n' \
    "GPUS_ALL_REGIONS" "${GLOBAL_GPU_LIMIT}" "${GLOBAL_GPU_USAGE}" "${GLOBAL_GPU_AVAILABLE}" ">= 1"
printf '%-37s %10s %10s %12s %10s\n' \
    "PREEMPTIBLE_CPUS" "${CPUS_SPOT_LIMIT}" "${CPUS_SPOT_USAGE}" "${CPUS_SPOT_AVAILABLE}" ">= 48"
printf '%-37s %10s %10s %12s %10s\n' \
    "PREEMPTIBLE_NVIDIA_RTX_PRO_6000" "${RTX_SPOT_LIMIT}" "${RTX_SPOT_USAGE}" "${RTX_SPOT_AVAILABLE}" ">= 1"

FAILED=false
quota_is_sufficient "${GLOBAL_GPU_AVAILABLE}" "1" || FAILED=true
quota_is_sufficient "${CPUS_SPOT_AVAILABLE}" "48" || FAILED=true
quota_is_sufficient "${RTX_SPOT_AVAILABLE}" "1" || FAILED=true

if [[ "${RTX_SPOT_AVAILABLE}" == "UNKNOWN" ]]; then
    printf '%s\n' \
        "[ÉCHEC] La métrique PREEMPTIBLE_NVIDIA_RTX_PRO_6000_GPUS est absente." \
        "        Son absence n'est pas assimilée à un quota disponible ; vérifiez Cloud Quotas." >&2
fi

if [[ "${FAILED}" == true ]]; then
    die "Quotas disponibles insuffisants ou inconnus pour démarrer une g4-standard-48."
fi

printf '[SUCCÈS] Quotas disponibles : au moins 48 vCPU Spot et une RTX PRO 6000.\n'

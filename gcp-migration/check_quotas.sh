#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PROJECT_ID="devpod-gpu-exploration"
readonly REGION="${GCP_REGION:-europe-west4}"
readonly ZONE="${GCP_ZONE:-${REGION}-a}"
readonly PROJECT_ID="${GCP_PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
readonly REQUIRE_EXTERNAL_ADDRESS="${GCP_REQUIRE_EXTERNAL_ADDRESS:-1}"
readonly QUOTA_SERVICE="compute.googleapis.com"
readonly GLOBAL_GPU_QUOTA_ID="GPUS-ALL-REGIONS-per-project"
readonly RTX_SPOT_QUOTA_ID="PREEMPTIBLE-NVIDIA-RTX-PRO-6000-GPUS-per-project-region"
readonly HDB_QUOTA_ID="HDB-TOTAL-GB-per-project-region"
readonly HDB_IOPS_QUOTA_ID="HDB-TOTAL-IOPS-per-project-zone"
readonly HDB_THROUGHPUT_QUOTA_ID="HDB-TOTAL-THROUGHPUT-per-project-zone"
readonly INSTANCES_QUOTA_ID="INSTANCES-per-project-region"
readonly ADDRESSES_QUOTA_ID="IN-USE-ADDRESSES-per-project-region"
readonly PREEMPTIBLE_CPUS_QUOTA_ID="PREEMPTIBLE-CPUS-per-project-region"
readonly REQUIRED_HDB_GB=100
readonly REQUIRED_HDB_IOPS=600
readonly REQUIRED_HDB_THROUGHPUT=150
readonly HDB_BASELINE_IOPS=3000
readonly HDB_BASELINE_THROUGHPUT=140

die() {
    printf '[ÉCHEC] %s\n' "$*" >&2
    exit 1
}

AI_PARENT_ZONE=""
AI_PARENT_SUFFIX=""
case "${ZONE}" in
    europe-west4-ai1a)
        [[ "${REGION}" == "europe-west4" ]] || \
        die "La zone IA ${ZONE} n'appartient pas à la région déclarée ${REGION}."
        AI_PARENT_ZONE="europe-west4-a"
        AI_PARENT_SUFFIX="a"
        ;;
    *-ai[0-9]*)
        die "Aucune correspondance de quota parent n'est documentée pour ${ZONE}."
        ;;
esac
readonly AI_PARENT_ZONE AI_PARENT_SUFFIX

quota_values() {
    local quota_json="$1"
    local metric="$2"

    jq -r --arg metric "${metric}" '
        [.quotas[]? | select(.metric == $metric)] | first |
        if . == null or .limit == null or .usage == null then
            "UNKNOWN UNKNOWN"
        else
            "\(.limit) \(.usage)"
        end
    ' <<<"${quota_json}"
}

cloud_quota_limit() {
    local quota_json="$1"
    local location="$2"
    local dimension="${3:-}"

    jq -r --arg location "${location}" --arg dimension "${dimension}" '
        [
            .dimensionsInfos[]? |
            select(
                if $dimension == "" then
                    ((.applicableLocations // []) | index($location)) != null
                else
                    (.dimensions[$dimension]? == $location) or
                    (
                        (.dimensions[$dimension]? == null) and
                        (((.applicableLocations // []) | index($location)) != null)
                    )
                end
            ) |
            {
                explicit: (
                    if $dimension != "" and .dimensions[$dimension]? == $location
                    then 0 else 1 end
                ),
                value: .details.value?
            }
        ] |
        map(select(.value != null)) |
        sort_by(.explicit) |
        (first.value // "UNKNOWN")
    ' <<<"${quota_json}"
}

describe_quota() {
    local quota_id="$1"

    gcloud beta quotas info describe "${quota_id}" \
        --service="${QUOTA_SERVICE}" \
        --project="${PROJECT_ID}" \
        --format=json
}

quota_available() {
    local limit="$1"
    local usage="$2"

    if [[ "${limit}" == "UNKNOWN" || "${usage}" == "UNKNOWN" ]]; then
        printf 'UNKNOWN'
    elif [[ "${limit}" == "-1" || "${limit}" == "-1.0" ]]; then
        printf 'UNLIMITED'
    else
        jq -n --arg limit "${limit}" --arg usage "${usage}" \
            '($limit | tonumber) - ($usage | tonumber)'
    fi
}

quota_is_sufficient() {
    local available="$1"
    local required="$2"

    if [[ "${available}" == "UNLIMITED" ]]; then
        return 0
    fi
    [[ "${available}" != "UNKNOWN" ]] &&
        jq -e -n --arg available "${available}" --arg required "${required}" \
            '($available | tonumber) >= ($required | tonumber)' >/dev/null
}

command -v gcloud >/dev/null 2>&1 || die "gcloud est introuvable."
command -v jq >/dev/null 2>&1 || die "jq est introuvable."
[[ "${REQUIRE_EXTERNAL_ADDRESS}" == "0" || "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]] || \
    die "GCP_REQUIRE_EXTERNAL_ADDRESS doit valoir 0 ou 1."

CONFIGURED_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ "${CONFIGURED_PROJECT}" != "${PROJECT_ID}" ]]; then
    die "Projet actif « ${CONFIGURED_PROJECT:-non configuré} » différent de « ${PROJECT_ID} »."
fi

printf '[CHECK] Quotas disponibles pour g4-standard-48 dans %s/%s...\n' \
    "${PROJECT_ID}" "${REGION}"

GLOBAL_GPU_QUOTA_DATA="$(describe_quota "${GLOBAL_GPU_QUOTA_ID}")" || \
    die "Impossible de lire ${GLOBAL_GPU_QUOTA_ID} dans Cloud Quotas."
RTX_SPOT_QUOTA_DATA="$(describe_quota "${RTX_SPOT_QUOTA_ID}")" || \
    die "Impossible de lire ${RTX_SPOT_QUOTA_ID} dans Cloud Quotas."
HDB_QUOTA_DATA="$(describe_quota "${HDB_QUOTA_ID}")" || \
    die "Impossible de lire ${HDB_QUOTA_ID} dans Cloud Quotas."
HDB_IOPS_QUOTA_DATA="$(describe_quota "${HDB_IOPS_QUOTA_ID}")" || \
    die "Impossible de lire ${HDB_IOPS_QUOTA_ID} dans Cloud Quotas."
HDB_THROUGHPUT_QUOTA_DATA="$(describe_quota "${HDB_THROUGHPUT_QUOTA_ID}")" || \
    die "Impossible de lire ${HDB_THROUGHPUT_QUOTA_ID} dans Cloud Quotas."
INSTANCES_QUOTA_DATA="$(describe_quota "${INSTANCES_QUOTA_ID}")" || \
    die "Impossible de lire ${INSTANCES_QUOTA_ID} dans Cloud Quotas."
if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    ADDRESSES_QUOTA_DATA="$(describe_quota "${ADDRESSES_QUOTA_ID}")" || \
        die "Impossible de lire ${ADDRESSES_QUOTA_ID} dans Cloud Quotas."
fi
PREEMPTIBLE_CPUS_QUOTA_DATA=""
if ! PREEMPTIBLE_CPUS_QUOTA_DATA="$(describe_quota \
    "${PREEMPTIBLE_CPUS_QUOTA_ID}" 2>/dev/null)"; then
    PREEMPTIBLE_CPUS_QUOTA_DATA=""
fi

PROJECT_DATA="$(gcloud compute project-info describe \
    --project="${PROJECT_ID}" \
    --format=json)"
REGIONAL_DATA="$(gcloud compute regions describe "${REGION}" \
    --project="${PROJECT_ID}" \
    --format=json)"
DISK_DATA="$(gcloud compute disks list \
    --project="${PROJECT_ID}" \
    --format=json)"

GLOBAL_GPU_LIMIT="$(cloud_quota_limit "${GLOBAL_GPU_QUOTA_DATA}" "global")"
RTX_SPOT_LIMIT="$(cloud_quota_limit "${RTX_SPOT_QUOTA_DATA}" "${REGION}" "region")"
HDB_LIMIT="$(cloud_quota_limit "${HDB_QUOTA_DATA}" "${REGION}" "region")"
HDB_IOPS_QUOTA_ZONE="${ZONE}"
HDB_IOPS_PARENT_DERIVED=0
HDB_IOPS_LIMIT="$(cloud_quota_limit "${HDB_IOPS_QUOTA_DATA}" "${HDB_IOPS_QUOTA_ZONE}" "zone")"
if [[ "${HDB_IOPS_LIMIT}" == "UNKNOWN" && -n "${AI_PARENT_ZONE}" ]]; then
    HDB_IOPS_QUOTA_ZONE="${AI_PARENT_ZONE}"
    HDB_IOPS_PARENT_DERIVED=1
    HDB_IOPS_LIMIT="$(cloud_quota_limit \
        "${HDB_IOPS_QUOTA_DATA}" "${HDB_IOPS_QUOTA_ZONE}" "zone")"
fi
HDB_THROUGHPUT_QUOTA_ZONE="${ZONE}"
HDB_THROUGHPUT_PARENT_DERIVED=0
HDB_THROUGHPUT_LIMIT="$(cloud_quota_limit \
    "${HDB_THROUGHPUT_QUOTA_DATA}" "${HDB_THROUGHPUT_QUOTA_ZONE}" "zone")"
if [[ "${HDB_THROUGHPUT_LIMIT}" == "UNKNOWN" && -n "${AI_PARENT_ZONE}" ]]; then
    HDB_THROUGHPUT_QUOTA_ZONE="${AI_PARENT_ZONE}"
    HDB_THROUGHPUT_PARENT_DERIVED=1
    HDB_THROUGHPUT_LIMIT="$(cloud_quota_limit \
        "${HDB_THROUGHPUT_QUOTA_DATA}" "${HDB_THROUGHPUT_QUOTA_ZONE}" "zone")"
fi
INSTANCES_LIMIT="$(cloud_quota_limit "${INSTANCES_QUOTA_DATA}" "${REGION}" "region")"
if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    ADDRESSES_LIMIT="$(cloud_quota_limit "${ADDRESSES_QUOTA_DATA}" "${REGION}" "region")"
else
    ADDRESSES_LIMIT="N/A"
fi
if [[ -n "${PREEMPTIBLE_CPUS_QUOTA_DATA}" ]]; then
    CPUS_SPOT_LIMIT="$(cloud_quota_limit "${PREEMPTIBLE_CPUS_QUOTA_DATA}" "${REGION}" "region")"
else
    CPUS_SPOT_LIMIT="UNKNOWN"
fi

read -r _ GLOBAL_GPU_USAGE < <(
    quota_values "${PROJECT_DATA}" "GPUS_ALL_REGIONS"
)
read -r _ CPUS_SPOT_USAGE < <(
    quota_values "${REGIONAL_DATA}" "PREEMPTIBLE_CPUS"
)
read -r _ INSTANCES_USAGE < <(
    quota_values "${REGIONAL_DATA}" "INSTANCES"
)
if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    read -r _ ADDRESSES_USAGE < <(
        quota_values "${REGIONAL_DATA}" "IN_USE_ADDRESSES"
    )
else
    ADDRESSES_USAGE="N/A"
fi
HDB_USAGE="$(jq -r --arg region "${REGION}" '
    [
        .[]? |
        ((.type // "") | split("/") | last) as $type |
        if $type == "hyperdisk-balanced" and
            (((.zone // "") | split("/") | last) | startswith($region + "-"))
        then (.sizeGb | tonumber)
        elif $type == "hyperdisk-balanced-high-availability" and
            (((.region // "") | split("/") | last) == $region)
        then ((.sizeGb | tonumber) * 2)
        else empty
        end
    ] | add // 0
' <<<"${DISK_DATA}")"
HDB_IOPS_USAGE="$(jq -r \
    --arg target_zone "${ZONE}" \
    --arg quota_zone "${HDB_IOPS_QUOTA_ZONE}" \
    --arg region "${REGION}" \
    --arg ai_parent_suffix "${AI_PARENT_SUFFIX}" \
    --arg use_parent_scope "${HDB_IOPS_PARENT_DERIVED}" \
    --argjson baseline "${HDB_BASELINE_IOPS}" '
    def in_quota_scope:
        (. // "") as $candidate |
        $candidate == $target_zone or
        ($use_parent_scope == "1" and
         ($candidate == $quota_zone or
          ($ai_parent_suffix != "" and
           ($candidate | test("^" + $region + "-ai[0-9]+" + $ai_parent_suffix + "$")))));
    [
        .[]? |
        ((.type // "") | split("/") | last) as $type |
        select(
            ($type == "hyperdisk-balanced" or
             $type == "hyperdisk-balanced-high-availability") and
            (
                (((.zone // "") | split("/") | last) | in_quota_scope) or
                ([.replicaZones[]? | split("/") | last | select(in_quota_scope)] |
                    length > 0)
            )
        ) |
        [((.provisionedIops | tonumber) - $baseline), 0] | max
    ] | add // 0
' <<<"${DISK_DATA}")"
HDB_THROUGHPUT_USAGE="$(jq -r \
    --arg target_zone "${ZONE}" \
    --arg quota_zone "${HDB_THROUGHPUT_QUOTA_ZONE}" \
    --arg region "${REGION}" \
    --arg ai_parent_suffix "${AI_PARENT_SUFFIX}" \
    --arg use_parent_scope "${HDB_THROUGHPUT_PARENT_DERIVED}" \
    --argjson baseline "${HDB_BASELINE_THROUGHPUT}" '
    def in_quota_scope:
        (. // "") as $candidate |
        $candidate == $target_zone or
        ($use_parent_scope == "1" and
         ($candidate == $quota_zone or
          ($ai_parent_suffix != "" and
           ($candidate | test("^" + $region + "-ai[0-9]+" + $ai_parent_suffix + "$")))));
    [
        .[]? |
        ((.type // "") | split("/") | last) as $type |
        select(
            ($type == "hyperdisk-balanced" or
             $type == "hyperdisk-balanced-high-availability") and
            (
                (((.zone // "") | split("/") | last) | in_quota_scope) or
                ([.replicaZones[]? | split("/") | last | select(in_quota_scope)] |
                    length > 0)
            )
        ) |
        [((.provisionedThroughput | tonumber) - $baseline), 0] | max
    ] | add // 0
' <<<"${DISK_DATA}")"

GLOBAL_GPU_AVAILABLE="$(quota_available "${GLOBAL_GPU_LIMIT}" "${GLOBAL_GPU_USAGE}")"
CPUS_SPOT_AVAILABLE="$(quota_available "${CPUS_SPOT_LIMIT}" "${CPUS_SPOT_USAGE}")"
HDB_AVAILABLE="$(quota_available "${HDB_LIMIT}" "${HDB_USAGE}")"
HDB_IOPS_AVAILABLE="$(quota_available "${HDB_IOPS_LIMIT}" "${HDB_IOPS_USAGE}")"
HDB_THROUGHPUT_AVAILABLE="$(quota_available "${HDB_THROUGHPUT_LIMIT}" "${HDB_THROUGHPUT_USAGE}")"
INSTANCES_AVAILABLE="$(quota_available "${INSTANCES_LIMIT}" "${INSTANCES_USAGE}")"
if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    ADDRESSES_AVAILABLE="$(quota_available "${ADDRESSES_LIMIT}" "${ADDRESSES_USAGE}")"
else
    ADDRESSES_AVAILABLE="N/A"
fi
if [[ "${RTX_SPOT_LIMIT}" == "UNKNOWN" || "${GLOBAL_GPU_AVAILABLE}" == "UNKNOWN" ]]; then
    RTX_SPOT_AVAILABLE="UNKNOWN"
elif [[ "${RTX_SPOT_LIMIT}" == "-1" || "${RTX_SPOT_LIMIT}" == "-1.0" ]]; then
    RTX_SPOT_AVAILABLE="${GLOBAL_GPU_AVAILABLE}"
else
    RTX_SPOT_AVAILABLE="$(jq -n \
        --arg regional "${RTX_SPOT_LIMIT}" \
        --arg global "${GLOBAL_GPU_AVAILABLE}" \
        '[($regional | tonumber), ($global | tonumber)] | min')"
fi

printf '%-37s %10s %10s %12s %10s\n' \
    "MÉTRIQUE" "LIMITE" "UTILISÉ" "DISPONIBLE" "REQUIS"
printf '%-37s %10s %10s %12s %10s\n' \
    "GPUS_ALL_REGIONS" "${GLOBAL_GPU_LIMIT}" "${GLOBAL_GPU_USAGE}" "${GLOBAL_GPU_AVAILABLE}" ">= 1"
printf '%-37s %10s %10s %12s %10s\n' \
    "PREEMPTIBLE_NVIDIA_RTX_PRO_6000" "${RTX_SPOT_LIMIT}" "${GLOBAL_GPU_USAGE}" "${RTX_SPOT_AVAILABLE}" ">= 1"
printf '%-37s %10s %10s %12s %10s\n' \
    "HDB_TOTAL_GB" "${HDB_LIMIT}" "${HDB_USAGE}" "${HDB_AVAILABLE}" ">= ${REQUIRED_HDB_GB}"
printf '%-37s %10s %10s %12s %10s\n' \
    "HDB_TOTAL_IOPS (${HDB_IOPS_QUOTA_ZONE})" "${HDB_IOPS_LIMIT}" "${HDB_IOPS_USAGE}" "${HDB_IOPS_AVAILABLE}" ">= ${REQUIRED_HDB_IOPS}"
printf '%-37s %10s %10s %12s %10s\n' \
    "HDB_TOTAL_THROUGHPUT (${HDB_THROUGHPUT_QUOTA_ZONE})" "${HDB_THROUGHPUT_LIMIT}" "${HDB_THROUGHPUT_USAGE}" "${HDB_THROUGHPUT_AVAILABLE}" ">= ${REQUIRED_HDB_THROUGHPUT}"
printf '%-37s %10s %10s %12s %10s\n' \
    "INSTANCES" "${INSTANCES_LIMIT}" "${INSTANCES_USAGE}" "${INSTANCES_AVAILABLE}" ">= 1"
printf '%-37s %10s %10s %12s %10s\n' \
    "IN_USE_ADDRESSES" "${ADDRESSES_LIMIT}" "${ADDRESSES_USAGE}" "${ADDRESSES_AVAILABLE}" \
    "$([[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]] && printf '>= 1' || printf 'N/A')"
printf '%-37s %10s %10s %12s %10s\n' \
    "PREEMPTIBLE_CPUS (informatif)" "${CPUS_SPOT_LIMIT}" "${CPUS_SPOT_USAGE}" "${CPUS_SPOT_AVAILABLE}" "N/A G4"

FAILED=false
quota_is_sufficient "${GLOBAL_GPU_AVAILABLE}" "1" || FAILED=true
quota_is_sufficient "${RTX_SPOT_AVAILABLE}" "1" || FAILED=true
quota_is_sufficient "${HDB_AVAILABLE}" "${REQUIRED_HDB_GB}" || FAILED=true
quota_is_sufficient "${HDB_IOPS_AVAILABLE}" "${REQUIRED_HDB_IOPS}" || FAILED=true
quota_is_sufficient "${HDB_THROUGHPUT_AVAILABLE}" "${REQUIRED_HDB_THROUGHPUT}" || FAILED=true
quota_is_sufficient "${INSTANCES_AVAILABLE}" "1" || FAILED=true
if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    quota_is_sufficient "${ADDRESSES_AVAILABLE}" "1" || FAILED=true
fi
if [[ "${GLOBAL_GPU_USAGE}" == "UNKNOWN" ]] || \
    ! jq -e -n --arg usage "${GLOBAL_GPU_USAGE}" \
        '($usage | tonumber) == 0' >/dev/null 2>&1; then
    printf '%s\n' \
        "[ÉCHEC] Une autre consommation GPU globale est active ou illisible." \
        "        La qualification reste bornée à une seule G4 Spot concurrente." >&2
    FAILED=true
fi

if [[ "${RTX_SPOT_AVAILABLE}" == "UNKNOWN" ]]; then
    printf '%s\n' \
        "[ÉCHEC] Le quota Cloud Quotas exact ${RTX_SPOT_QUOTA_ID} est absent ou illisible." \
        "        Son absence n'est jamais assimilée à une capacité Spot disponible." >&2
fi

if [[ -n "${AI_PARENT_ZONE}" && \
    ("${HDB_IOPS_QUOTA_ZONE}" != "${ZONE}" || \
     "${HDB_THROUGHPUT_QUOTA_ZONE}" != "${ZONE}") ]]; then
    printf '%s\n' \
        "[INFO] Zone IA ${ZONE} : limite Hyperdisk dérivée de la zone parente ${AI_PARENT_ZONE}." \
        '       Cloud Quotas ne publie pas de dimension IA; la création GCE reste l’arbitre.' \
        '       Le calcul additionne conservativement le parent et ses zones IA associées.'
fi

if [[ "${FAILED}" == true ]]; then
    die "Quotas disponibles insuffisants ou inconnus pour démarrer une g4-standard-48."
fi

if [[ "${REQUIRE_EXTERNAL_ADDRESS}" == "1" ]]; then
    NETWORK_QUOTA_SUMMARY='         une instance et une adresse externe éphémère.'
else
    NETWORK_QUOTA_SUMMARY='         une instance; aucun quota d’adresse externe n’est requis.'
fi
printf '%s\n' \
    '[SUCCÈS] Quotas disponibles pour au plus une g4-standard-48 Spot :' \
    '         une RTX PRO 6000 préemptible, un GPU global, un Hyperdisk Balanced,' \
    "${NETWORK_QUOTA_SUMMARY}" \
    '[INFO] G4 ne consomme aucun quota CPU; PREEMPTIBLE_CPUS est affiché sans être exigé.'

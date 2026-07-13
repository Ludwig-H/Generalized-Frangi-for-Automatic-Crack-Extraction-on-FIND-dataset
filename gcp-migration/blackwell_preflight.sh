#!/usr/bin/env bash
set -euo pipefail

readonly OPEN_MODULE_MESSAGE="requires use of the NVIDIA open kernel modules"
readonly OPEN_MODULE_PACKAGE="linux-modules-nvidia-580-server-open-gcp"
readonly OPEN_DRIVER_PACKAGE="nvidia-driver-580-server-open"

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.9.1-base-ubuntu22.04}"
ARM_SHUTDOWN_MINUTES=""
INSTALL_OPEN_DRIVER=false
RUN_DOCKER_SMOKE=true
FAILURES=0
OPEN_MODULE_REQUIRED=false
SHUTDOWN_GUARD_ARMED=false
NVIDIA_SMI_OUTPUT=""

usage() {
    cat <<'EOF'
Usage: ./blackwell_preflight.sh [options]

Diagnostic non destructif d'une VM G4 Blackwell. Par défaut, le script vérifie
le GPU, CUDA 12.9 et lance un conteneur CUDA éphémère.

Options :
  --arm-shutdown MINUTES   Programmer un arrêt de sécurité après confirmation.
  --install-open-driver    Installer les paquets 580-server-open GCP, uniquement
                           si le message « requires ... open kernel modules »
                           est détecté, et après une seconde confirmation.
  --skip-docker            Ne pas lancer le smoke test Docker GPU.
  -h, --help               Afficher cette aide.

Le script ne purge aucun paquet, ne redémarre pas et n'arrête pas immédiatement
la VM. CUDA_IMAGE permet de remplacer l'image du smoke test.
EOF
}

die() {
    printf '[ERREUR] %s\n' "$*" >&2
    exit 1
}

warn() {
    printf '[ATTENTION] %s\n' "$*" >&2
}

failure() {
    printf '[ÉCHEC] %s\n' "$*" >&2
    FAILURES=$((FAILURES + 1))
}

success() {
    printf '[OK] %s\n' "$*"
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

run_as_root() {
    if ((EUID == 0)); then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo est requis pour cette opération."
        sudo "$@"
    fi
}

confirm_exactly() {
    local expected="$1"
    local answer=""
    [[ -t 0 ]] || die "Confirmation interactive requise : ${expected}"
    read -r -p "Tapez exactement « ${expected} » : " answer
    [[ "${answer}" == "${expected}" ]] || die "Opération annulée."
}

collect_kernel_log() {
    local log=""

    if ((EUID == 0)); then
        log="$(journalctl -k -b --no-pager 2>&1 || true)"
        log+=$'\n'
        log+="$(dmesg 2>&1 || true)"
    else
        log="$(journalctl -k -b --no-pager 2>&1 || true)"
        if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
            log+=$'\n'
            log+="$(sudo -n journalctl -k -b --no-pager 2>&1 || true)"
            log+=$'\n'
            log+="$(sudo -n dmesg 2>&1 || true)"
        else
            log+=$'\n'
            log+="$(dmesg 2>&1 || true)"
        fi
    fi

    printf '%s' "${log}"
}

arm_shutdown_guard() {
    [[ -n "${ARM_SHUTDOWN_MINUTES}" ]] || return 0

    confirm_exactly "ARMER ARRET ${ARM_SHUTDOWN_MINUTES} MINUTES"
    run_as_root shutdown -P "+${ARM_SHUTDOWN_MINUTES}" \
        "Coupe-circuit de sécurité de la session GPU Generalized-Frangi"
    SHUTDOWN_GUARD_ARMED=true
    success "Arrêt de sécurité programmé dans ${ARM_SHUTDOWN_MINUTES} minutes."

    if shutdown --show >/dev/null 2>&1; then
        shutdown --show
    elif [[ -r /run/systemd/shutdown/scheduled ]]; then
        cat /run/systemd/shutdown/scheduled
    else
        warn "Impossible d'afficher l'échéance ; vérifiez-la avec : shutdown --show"
    fi
}

inspect_driver_diagnostics() {
    local kernel_log=""
    local diagnostic_text=""
    local loaded_driver=""
    local module_license=""
    local module_filename=""
    local open_module_loaded=false
    local package=""
    local package_status=""

    kernel_log="$(collect_kernel_log)"
    diagnostic_text="${NVIDIA_SMI_OUTPUT}"$'\n'"${kernel_log}"

    if grep -Fqi "${OPEN_MODULE_MESSAGE}" <<<"${diagnostic_text}"; then
        OPEN_MODULE_REQUIRED=true
        failure "Le noyau indique que ce GPU exige les modules NVIDIA ouverts."
        grep -Fi "${OPEN_MODULE_MESSAGE}" <<<"${diagnostic_text}" | tail -n 3 || true
    else
        success "Le message d'incompatibilité avec les modules fermés n'est pas présent."
    fi

    if [[ -r /proc/driver/nvidia/version ]]; then
        loaded_driver="$(</proc/driver/nvidia/version)"
        printf '[INFO] Module chargé : %s\n' "${loaded_driver//$'\n'/ }"
        if [[ "${loaded_driver}" == *"Open Kernel Module"* ]]; then
            open_module_loaded=true
        fi
    fi

    if command -v modinfo >/dev/null 2>&1 && modinfo nvidia >/dev/null 2>&1; then
        module_license="$(modinfo -F license nvidia 2>/dev/null || true)"
        module_filename="$(modinfo -F filename nvidia 2>/dev/null || true)"
        printf '[INFO] Module nvidia : %s (licence : %s)\n' \
            "${module_filename:-inconnu}" "${module_license:-inconnue}"
        if [[ "${module_license}" == *"MIT/GPL"* ]]; then
            open_module_loaded=true
        fi
    fi

    if [[ "${open_module_loaded}" == true ]]; then
        success "Le module NVIDIA ouvert est chargé."
    else
        failure "Impossible de confirmer que le module NVIDIA ouvert est chargé."
    fi

    if command -v dpkg-query >/dev/null 2>&1; then
        for package in "${OPEN_MODULE_PACKAGE}" "${OPEN_DRIVER_PACKAGE}"; do
            package_status="$(dpkg-query -W -f='${Status}\t${Version}' "${package}" 2>/dev/null || true)"
            if [[ "${package_status}" == "install ok installed"$'\t'* ]]; then
                printf '[INFO] Paquet : %s\t%s\n' "${package}" "${package_status}"
            else
                warn "Paquet absent ou incomplet : ${package}."
            fi
        done
    fi
}

inspect_gpu() {
    local gpu_csv=""
    local gpu_count=0
    local index=""
    local name=""
    local compute_cap=""
    local memory_mib=""
    local driver_version=""
    local memory_integer=""

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_SMI_OUTPUT="nvidia-smi est introuvable"
        failure "nvidia-smi est introuvable."
        return
    fi

    if NVIDIA_SMI_OUTPUT="$(nvidia-smi 2>&1)"; then
        success "nvidia-smi communique avec le pilote."
    else
        failure "nvidia-smi ne communique pas avec le pilote."
        printf '%s\n' "${NVIDIA_SMI_OUTPUT}" >&2
        return
    fi

    if ! gpu_csv="$(nvidia-smi \
        --query-gpu=index,name,compute_cap,memory.total,driver_version \
        --format=csv,noheader,nounits 2>&1)"; then
        failure "Impossible d'interroger CC, VRAM et version du pilote : ${gpu_csv}"
        return
    fi

    while IFS=',' read -r index name compute_cap memory_mib driver_version; do
        [[ -n "${index}${name}${compute_cap}${memory_mib}${driver_version}" ]] || continue
        index="$(trim "${index}")"
        name="$(trim "${name}")"
        compute_cap="$(trim "${compute_cap}")"
        memory_mib="$(trim "${memory_mib}")"
        driver_version="$(trim "${driver_version}")"
        memory_integer="${memory_mib%%.*}"
        gpu_count=$((gpu_count + 1))

        printf '[INFO] GPU %s : %s | CC %s | %s MiB | pilote %s\n' \
            "${index}" "${name}" "${compute_cap}" "${memory_mib}" "${driver_version}"

        [[ "${name}" == *"RTX PRO 6000"* ]] || \
            failure "Le GPU ${index} n'est pas une RTX PRO 6000 Blackwell."
        [[ "${compute_cap}" == "12.0" || "${compute_cap}" == "12" ]] || \
            failure "Le GPU ${index} expose CC ${compute_cap}, CC 12.0 attendu."
        if [[ ! "${memory_integer}" =~ ^[0-9]+$ ]] || ((memory_integer < 95000)); then
            failure "Le GPU ${index} expose ${memory_mib} MiB, moins que le seuil de 95 000 MiB d'une carte 96 Go."
        fi
        [[ "${driver_version}" == 580.* ]] || \
            failure "Le pilote ${driver_version} n'appartient pas à la branche 580 attendue."
    done <<<"${gpu_csv}"

    [[ ${gpu_count} -eq 1 ]] || failure "g4-standard-48 doit exposer exactement un GPU ; ${gpu_count} détecté(s)."
}

inspect_cuda_toolkit() {
    local nvcc_output=""

    if ! command -v nvcc >/dev/null 2>&1; then
        failure "nvcc est introuvable ; l'image CUDA 12.9 n'est pas complètement disponible dans le PATH."
        return
    fi

    nvcc_output="$(nvcc --version 2>&1 || true)"
    printf '%s\n' "${nvcc_output}"
    if grep -Eq 'release 12\.9([,[:space:]]|$)' <<<"${nvcc_output}"; then
        success "Toolkit CUDA 12.9 détecté."
    else
        failure "nvcc n'annonce pas CUDA 12.9."
    fi
}

docker_gpu_smoke() {
    local -a docker_command=(docker)
    local smoke_output=""

    [[ "${RUN_DOCKER_SMOKE}" == true ]] || return 0

    if ! command -v docker >/dev/null 2>&1; then
        failure "Docker est introuvable ; smoke test GPU non exécuté."
        return
    fi

    if ! docker info >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
            docker_command=(sudo docker)
        else
            failure "Le daemon Docker est inaccessible pour cet utilisateur."
            return
        fi
    fi

    printf '[INFO] Smoke test Docker avec %s (pull possible, conteneur --rm).\n' "${CUDA_IMAGE}"
    if smoke_output="$("${docker_command[@]}" run --rm --gpus all "${CUDA_IMAGE}" \
        nvidia-smi --query-gpu=name,compute_cap,memory.total \
        --format=csv,noheader 2>&1)"; then
        printf '%s\n' "${smoke_output}"
        success "Docker accède au GPU via NVIDIA Container Toolkit."
    else
        failure "Le smoke test Docker GPU a échoué : ${smoke_output}"
    fi
}

install_open_driver() {
    [[ "${INSTALL_OPEN_DRIVER}" == true ]] || return 0
    [[ "${OPEN_MODULE_REQUIRED}" == true ]] || \
        die "Installation refusée : le message exact « ${OPEN_MODULE_MESSAGE} » n'a pas été détecté."
    [[ "$(uname -r)" == *-gcp ]] || \
        die "Installation refusée : le noyau $(uname -r) n'est pas de saveur GCP."
    command -v apt-get >/dev/null 2>&1 || die "apt-get est requis pour la réparation documentée."

    printf '%s\n' \
        "[RÉPARATION PROPOSÉE] Installation sans purge de :" \
        "  ${OPEN_MODULE_PACKAGE}" \
        "  ${OPEN_DRIVER_PACKAGE}" \
        "Aucun reboot ne sera lancé automatiquement."
    confirm_exactly "INSTALLER 580-SERVER-OPEN"

    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-remove \
        "${OPEN_MODULE_PACKAGE}" "${OPEN_DRIVER_PACKAGE}"

    success "Paquets ouverts installés."
    printf '%s\n' \
        "[ACTION MANUELLE REQUISE] Redémarrez explicitement la VM, puis relancez ce preflight." \
        "Le script s'arrête avec le code 20 tant que cette validation après reboot n'a pas eu lieu."
    exit 20
}

while (($# > 0)); do
    case "$1" in
        --arm-shutdown)
            (($# >= 2)) || die "--arm-shutdown exige un nombre de minutes."
            ARM_SHUTDOWN_MINUTES="$2"
            shift 2
            ;;
        --install-open-driver)
            INSTALL_OPEN_DRIVER=true
            shift
            ;;
        --skip-docker)
            RUN_DOCKER_SMOKE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "Option inconnue : $1"
            ;;
    esac
done

if [[ -n "${ARM_SHUTDOWN_MINUTES}" ]]; then
    if [[ ! "${ARM_SHUTDOWN_MINUTES}" =~ ^[1-9][0-9]*$ ]] || \
        ((${#ARM_SHUTDOWN_MINUTES} > 4)) || ((ARM_SHUTDOWN_MINUTES > 1440)); then
        die "--arm-shutdown attend un entier entre 1 et 1440 minutes."
    fi
fi

printf '[PREFLIGHT] Hôte %s, noyau %s\n' "$(hostname)" "$(uname -r)"
arm_shutdown_guard
inspect_gpu
inspect_driver_diagnostics
inspect_cuda_toolkit
docker_gpu_smoke
install_open_driver

if [[ "${SHUTDOWN_GUARD_ARMED}" != true ]]; then
    warn "Aucun coupe-circuit invité n'a été armé par ce script. Utilisez --arm-shutdown MINUTES."
fi

if ((FAILURES > 0)); then
    printf '[BILAN] %d contrôle(s) en échec. Ne lancez pas les benchmarks lourds.\n' "${FAILURES}" >&2
    exit 1
fi

if [[ "${RUN_DOCKER_SMOKE}" == true ]]; then
    success "Preflight Blackwell complet : CC 12.0, VRAM 96 Go, CUDA 12.9 et Docker GPU opérationnels."
else
    success "Preflight Blackwell partiel : CC 12.0, VRAM 96 Go et CUDA 12.9 opérationnels ; Docker non testé."
fi

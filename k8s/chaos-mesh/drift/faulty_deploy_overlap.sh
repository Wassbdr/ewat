#!/usr/bin/env bash
# ============================================================================
# faulty_deploy_overlap — θ_{drift ∩ anomaly}
#
# A rolling deploy of the recommendation service that happens to trigger
# a real failure (OOMKill caused by reduced memory limit). This is the
# scenario that distinguishes EWAT from MMD-only drift detectors:
# MMD² will flag drift, but the post-drift test must remain positive for
# anomaly, so the look-through mechanism forwards the signal with
# DRIFT flag instead of recalibrating.
#
# We implement it by patching the memory limit down, triggering a
# rollout, and letting the pods OOM. Cleanup restores the original limit.
# ============================================================================

set -euo pipefail

NAMESPACE="${NAMESPACE:-ewat}"
DEPLOYMENT="${DEPLOYMENT:-recommendation}"
CONTAINER="${CONTAINER:-}"
FAULTY_LIMIT="${FAULTY_LIMIT:-32Mi}"
STATE_FILE="/tmp/ewat_faulty_deploy_${NAMESPACE}_${DEPLOYMENT}.state"

_detect_container() {
    if [[ -n "${CONTAINER}" ]]; then
        echo "${CONTAINER}"
        return 0
    fi
    kubectl -n "${NAMESPACE}" get deploy "${DEPLOYMENT}" \
        -o jsonpath='{.spec.template.spec.containers[0].name}'
}

inject() {
    local container
    container="$(_detect_container)"
    local original
    original="$(kubectl -n "${NAMESPACE}" get deploy "${DEPLOYMENT}" \
        -o jsonpath="{.spec.template.spec.containers[?(@.name=='${container}')].resources.limits.memory}")"
    if [[ -z "${original}" ]]; then original="none"; fi
    echo "${container} ${original}" > "${STATE_FILE}"

    echo "[faulty_deploy_overlap] patching ${DEPLOYMENT}/${container} memory limit to ${FAULTY_LIMIT}"
    kubectl -n "${NAMESPACE}" patch deploy "${DEPLOYMENT}" --type=json \
        -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/limits/memory\",\"value\":\"${FAULTY_LIMIT}\"}]"
    # Do not wait for rollout to succeed — the faulty limit will make pods OOM.
}

cleanup() {
    if [[ ! -f "${STATE_FILE}" ]]; then
        echo "[faulty_deploy_overlap] no saved state, skipping cleanup"
        return 0
    fi
    local container original
    read -r container original < "${STATE_FILE}"

    if [[ "${original}" == "none" ]]; then
        echo "[faulty_deploy_overlap] removing memory limit on ${DEPLOYMENT}/${container}"
        kubectl -n "${NAMESPACE}" patch deploy "${DEPLOYMENT}" --type=json \
            -p '[{"op":"remove","path":"/spec/template/spec/containers/0/resources/limits/memory"}]' || true
    else
        echo "[faulty_deploy_overlap] restoring memory limit on ${DEPLOYMENT}/${container} -> ${original}"
        kubectl -n "${NAMESPACE}" patch deploy "${DEPLOYMENT}" --type=json \
            -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/limits/memory\",\"value\":\"${original}\"}]" || true
    fi
    kubectl -n "${NAMESPACE}" rollout status deploy "${DEPLOYMENT}" --timeout=180s || true
    rm -f "${STATE_FILE}"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {inject|cleanup}"; exit 2 ;;
esac

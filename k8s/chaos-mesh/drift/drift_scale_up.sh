#!/usr/bin/env bash
# ============================================================================
# drift_scale_up — benign drift (θ_drift)
#
# Scales the frontend deployment from R → R+1 replicas, waits for the new
# pod to become Ready, then during cleanup scales back to the original R.
# Produces an autoscaling-like distribution shift (graph size, resource
# share per pod) with no application-level failure.
# ============================================================================

set -euo pipefail

NAMESPACE="${NAMESPACE:-ewat}"
DEPLOYMENT="${DEPLOYMENT:-frontend}"
STATE_FILE="/tmp/ewat_drift_scale_up_${NAMESPACE}_${DEPLOYMENT}.state"

inject() {
    local current
    current="$(kubectl -n "${NAMESPACE}" get deploy "${DEPLOYMENT}" \
        -o jsonpath='{.spec.replicas}')"
    echo "${current}" > "${STATE_FILE}"
    local target=$(( current + 1 ))
    echo "[drift_scale_up] scaling ${DEPLOYMENT}: ${current} -> ${target}"
    kubectl -n "${NAMESPACE}" scale deploy "${DEPLOYMENT}" --replicas="${target}"
    kubectl -n "${NAMESPACE}" rollout status deploy "${DEPLOYMENT}" --timeout=120s
}

cleanup() {
    if [[ ! -f "${STATE_FILE}" ]]; then
        echo "[drift_scale_up] no saved state, skipping cleanup"
        return 0
    fi
    local original
    original="$(cat "${STATE_FILE}")"
    echo "[drift_scale_up] restoring ${DEPLOYMENT} -> ${original} replicas"
    kubectl -n "${NAMESPACE}" scale deploy "${DEPLOYMENT}" --replicas="${original}" || true
    kubectl -n "${NAMESPACE}" rollout status deploy "${DEPLOYMENT}" --timeout=120s || true
    rm -f "${STATE_FILE}"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {inject|cleanup}"; exit 2 ;;
esac

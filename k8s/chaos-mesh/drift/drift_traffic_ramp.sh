#!/usr/bin/env bash
# ============================================================================
# drift_traffic_ramp — benign drift (θ_drift)
#
# Scales the load-generator deployment up (2×) to simulate an organic
# traffic increase, then scales back on cleanup. No failure is injected;
# the target services are in their normal operating envelope. This is
# the canonical "drift bénin" EWAT must learn to ignore (H2 calibration
# of ε_drift).
# ============================================================================

set -euo pipefail

NAMESPACE="${NAMESPACE:-ewat}"
DEPLOYMENT="${DEPLOYMENT:-load-generator}"
MULTIPLIER="${MULTIPLIER:-2}"
STATE_FILE="/tmp/ewat_drift_traffic_ramp_${NAMESPACE}_${DEPLOYMENT}.state"

inject() {
    local current
    current="$(kubectl -n "${NAMESPACE}" get deploy "${DEPLOYMENT}" \
        -o jsonpath='{.spec.replicas}')"
    echo "${current}" > "${STATE_FILE}"
    local target=$(( current * MULTIPLIER ))
    if [[ "${target}" -lt 2 ]]; then target=2; fi
    echo "[drift_traffic_ramp] scaling ${DEPLOYMENT}: ${current} -> ${target}"
    kubectl -n "${NAMESPACE}" scale deploy "${DEPLOYMENT}" --replicas="${target}"
    kubectl -n "${NAMESPACE}" rollout status deploy "${DEPLOYMENT}" --timeout=120s
}

cleanup() {
    if [[ ! -f "${STATE_FILE}" ]]; then
        echo "[drift_traffic_ramp] no saved state, skipping cleanup"
        return 0
    fi
    local original
    original="$(cat "${STATE_FILE}")"
    echo "[drift_traffic_ramp] restoring ${DEPLOYMENT} -> ${original} replicas"
    kubectl -n "${NAMESPACE}" scale deploy "${DEPLOYMENT}" --replicas="${original}" || true
    kubectl -n "${NAMESPACE}" rollout status deploy "${DEPLOYMENT}" --timeout=120s || true
    rm -f "${STATE_FILE}"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {inject|cleanup}"; exit 2 ;;
esac

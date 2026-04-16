#!/usr/bin/env bash
# ============================================================================
# drift_config_change — benign drift (θ_drift)
#
# Adds (then removes) an environment variable on the cart deployment.
# This forces a pod restart and a small behavioural shift (the env var
# has no effect on logic, but warm-up after restart is observable).
# Idempotent: we remove our own variable on cleanup.
# ============================================================================

set -euo pipefail

NAMESPACE="${NAMESPACE:-ewat}"
DEPLOYMENT="${DEPLOYMENT:-cart}"
VAR_NAME="${VAR_NAME:-EWAT_DRIFT_MARKER}"

inject() {
    local value
    value="v$(date +%s)"
    echo "[drift_config_change] set env ${VAR_NAME}=${value} on ${DEPLOYMENT}"
    kubectl -n "${NAMESPACE}" set env deploy/"${DEPLOYMENT}" "${VAR_NAME}=${value}"
    kubectl -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=180s
}

cleanup() {
    echo "[drift_config_change] unset env ${VAR_NAME} on ${DEPLOYMENT}"
    kubectl -n "${NAMESPACE}" set env deploy/"${DEPLOYMENT}" "${VAR_NAME}-" || true
    kubectl -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=180s || true
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {inject|cleanup}"; exit 2 ;;
esac

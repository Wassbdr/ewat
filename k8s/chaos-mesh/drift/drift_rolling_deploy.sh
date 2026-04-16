#!/usr/bin/env bash
# ============================================================================
# drift_rolling_deploy — benign drift (θ_drift)
#
# Triggers a rolling restart of the recommendation deployment (as would
# happen during a normal code deploy). No failure injected. The signal
# shift comes from pod churn (new IPs, warm-up, short GC/cache rebuilds).
# Cleanup waits for rollout to finish; no state to restore.
# ============================================================================

set -euo pipefail

NAMESPACE="${NAMESPACE:-ewat}"
DEPLOYMENT="${DEPLOYMENT:-recommendation}"

inject() {
    echo "[drift_rolling_deploy] rollout restart deploy/${DEPLOYMENT}"
    kubectl -n "${NAMESPACE}" rollout restart deploy/"${DEPLOYMENT}"
    kubectl -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=180s
}

cleanup() {
    echo "[drift_rolling_deploy] ensuring rollout is complete"
    kubectl -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=180s || true
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {inject|cleanup}"; exit 2 ;;
esac

#!/usr/bin/env bash
# EWAT Chaos — probe_misconfig
# Misconfigures the readiness probe causing rolling restarts.
# Usage: ./probe_misconfig.sh inject|cleanup

set -euo pipefail
NAMESPACE="ewat"
COMPONENT="cart"

inject() {
    echo "[EWAT] Saving original readiness probe for ${COMPONENT}..."
    kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        -o json > /tmp/ewat_probe_backup_${COMPONENT}.json

    echo "[EWAT] Setting invalid readiness probe on ${COMPONENT}..."
    kubectl patch deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe","value":{"httpGet":{"path":"/nonexistent-health","port":9999},"periodSeconds":5,"failureThreshold":2}}]' \
        2>/dev/null || \
    kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        -o name | xargs -I{} kubectl patch {} -n "${NAMESPACE}" \
        --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe","value":{"httpGet":{"path":"/nonexistent-health","port":9999},"periodSeconds":5,"failureThreshold":2}}]'

    echo "[EWAT] Probe misconfig injected on ${COMPONENT}"
}

cleanup() {
    if [ ! -f /tmp/ewat_probe_backup_${COMPONENT}.json ]; then
        echo "[EWAT] ERROR: No backup found. Manual cleanup required."
        exit 1
    fi

    echo "[EWAT] Restoring original deployment for ${COMPONENT}..."
    kubectl apply -f /tmp/ewat_probe_backup_${COMPONENT}.json -n "${NAMESPACE}"
    rm -f /tmp/ewat_probe_backup_${COMPONENT}.json

    echo "[EWAT] Cleanup complete for ${COMPONENT}"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

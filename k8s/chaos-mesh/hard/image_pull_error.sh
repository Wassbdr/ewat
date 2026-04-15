#!/usr/bin/env bash
# EWAT Chaos — image_pull_error
# Injects ImagePullBackOff by setting an invalid image tag.
# Usage: ./image_pull_error.sh inject|cleanup

set -euo pipefail
NAMESPACE="ewat"
COMPONENT="ad"
ORIGINAL_IMAGE=""

inject() {
    echo "[EWAT] Saving original image for ${COMPONENT}..."
    ORIGINAL_IMAGE=$(kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        -o jsonpath='{.items[0].spec.template.spec.containers[0].image}')
    echo "${ORIGINAL_IMAGE}" > /tmp/ewat_original_image_${COMPONENT}.txt

    echo "[EWAT] Setting invalid image tag on ${COMPONENT}..."
    kubectl set image deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        "*=invalid-registry.example.com/nonexistent:v999"

    echo "[EWAT] ImagePullError injected on ${COMPONENT}"
}

cleanup() {
    ORIGINAL_IMAGE=$(cat /tmp/ewat_original_image_${COMPONENT}.txt 2>/dev/null || echo "")
    if [ -z "${ORIGINAL_IMAGE}" ]; then
        echo "[EWAT] ERROR: No saved original image found. Manual cleanup required."
        exit 1
    fi

    echo "[EWAT] Restoring original image for ${COMPONENT}: ${ORIGINAL_IMAGE}"
    kubectl set image deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
        "*=${ORIGINAL_IMAGE}"
    rm -f /tmp/ewat_original_image_${COMPONENT}.txt

    echo "[EWAT] Cleanup complete for ${COMPONENT}"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

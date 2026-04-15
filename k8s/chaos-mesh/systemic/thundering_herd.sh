#!/usr/bin/env bash
# EWAT Chaos — thundering_herd
# Spikes Locust concurrent users 10× to simulate thundering herd.
# Usage: ./thundering_herd.sh inject|cleanup

set -euo pipefail
NAMESPACE="ewat"

inject() {
    echo "[EWAT] Getting current load-generator user count..."
    ORIGINAL_USERS=$(kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=loadgenerator" \
        -o jsonpath='{.items[0].spec.template.spec.containers[0].env[?(@.name=="LOCUST_USERS")].value}' 2>/dev/null || echo "5")
    echo "${ORIGINAL_USERS}" > /tmp/ewat_original_locust_users.txt

    NEW_USERS=$((ORIGINAL_USERS * 10))
    echo "[EWAT] Scaling load-generator from ${ORIGINAL_USERS} to ${NEW_USERS} users..."
    kubectl set env deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=loadgenerator" \
        LOCUST_USERS="${NEW_USERS}" LOCUST_SPAWN_RATE="10"

    echo "[EWAT] Thundering herd injected (${NEW_USERS} concurrent users)"
}

cleanup() {
    ORIGINAL_USERS=$(cat /tmp/ewat_original_locust_users.txt 2>/dev/null || echo "5")

    echo "[EWAT] Restoring load-generator to ${ORIGINAL_USERS} users..."
    kubectl set env deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=loadgenerator" \
        LOCUST_USERS="${ORIGINAL_USERS}" LOCUST_SPAWN_RATE="1"
    rm -f /tmp/ewat_original_locust_users.txt

    echo "[EWAT] Thundering herd cleaned up"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

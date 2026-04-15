#!/usr/bin/env bash
# EWAT Chaos — node_drain
# Drains a worker node, forcing pod evictions and rescheduling.
# Usage: ./node_drain.sh inject|cleanup

set -euo pipefail
NAMESPACE="ewat"

inject() {
    # Find a node running ewat pods (pick the one with the most pods)
    TARGET_NODE=$(kubectl get pods -n "${NAMESPACE}" -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' \
        | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')

    if [ -z "${TARGET_NODE}" ]; then
        echo "[EWAT] ERROR: No nodes found running ewat pods"
        exit 1
    fi

    echo "${TARGET_NODE}" > /tmp/ewat_drained_node.txt
    echo "[EWAT] Draining node: ${TARGET_NODE}"
    kubectl drain "${TARGET_NODE}" --ignore-daemonsets --delete-emptydir-data --force --timeout=60s || true
    echo "[EWAT] Node ${TARGET_NODE} drained"
}

cleanup() {
    TARGET_NODE=$(cat /tmp/ewat_drained_node.txt 2>/dev/null || echo "")
    if [ -z "${TARGET_NODE}" ]; then
        echo "[EWAT] ERROR: No saved drained node found"
        exit 1
    fi

    echo "[EWAT] Uncordoning node: ${TARGET_NODE}"
    kubectl uncordon "${TARGET_NODE}"
    rm -f /tmp/ewat_drained_node.txt
    echo "[EWAT] Node ${TARGET_NODE} restored"
}

case "${1:-}" in
    inject)  inject ;;
    cleanup) cleanup ;;
    *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

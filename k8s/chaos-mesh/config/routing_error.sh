#!/usr/bin/env bash
# EWAT Chaos — routing_error
# Injects a service routing error by breaking checkout Service selectors.
# Usage: ./routing_error.sh inject|cleanup

set -euo pipefail

NAMESPACE="ewat"
COMPONENT="checkout"
STATE_DIR="/tmp/ewat_config_failures"
BACKUP_FILE="${STATE_DIR}/routing_error_checkout_service.json"

mkdir -p "${STATE_DIR}"

get_service_name() {
  kubectl get svc -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
    -o jsonpath='{.items[0].metadata.name}'
}

inject() {
  SERVICE_NAME=$(get_service_name)
  if [[ -z "${SERVICE_NAME}" ]]; then
    echo "[EWAT] ERROR: checkout service not found"
    exit 1
  fi

  kubectl get svc "${SERVICE_NAME}" -n "${NAMESPACE}" -o json > "${BACKUP_FILE}"

  echo "[EWAT] Injecting routing error on service ${SERVICE_NAME}"
  kubectl patch svc "${SERVICE_NAME}" -n "${NAMESPACE}" --type=merge \
    -p '{"spec":{"selector":{"app.kubernetes.io/component":"nonexistent-checkout"}}}'
}

cleanup() {
  if [[ ! -f "${BACKUP_FILE}" ]]; then
    echo "[EWAT] ERROR: backup not found at ${BACKUP_FILE}"
    exit 1
  fi

  echo "[EWAT] Restoring checkout service selector"
  kubectl apply -n "${NAMESPACE}" -f "${BACKUP_FILE}"
  rm -f "${BACKUP_FILE}"
}

case "${1:-}" in
  inject) inject ;;
  cleanup) cleanup ;;
  *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

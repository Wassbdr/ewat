#!/usr/bin/env bash
# EWAT Chaos — bad_env_config
# Injects a critical env misconfiguration on payment deployment.
# Usage: ./bad_env_config.sh inject|cleanup

set -euo pipefail

NAMESPACE="ewat"
COMPONENT="payment"
STATE_DIR="/tmp/ewat_config_failures"
BACKUP_FILE="${STATE_DIR}/bad_env_payment_deploy.json"

mkdir -p "${STATE_DIR}"

get_deploy_name() {
  kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
    -o jsonpath='{.items[0].metadata.name}'
}

inject() {
  DEPLOY_NAME=$(get_deploy_name)
  if [[ -z "${DEPLOY_NAME}" ]]; then
    echo "[EWAT] ERROR: payment deployment not found"
    exit 1
  fi

  kubectl get deploy "${DEPLOY_NAME}" -n "${NAMESPACE}" -o json > "${BACKUP_FILE}"

  echo "[EWAT] Injecting invalid env configuration on ${DEPLOY_NAME}"
  kubectl set env deploy "${DEPLOY_NAME}" -n "${NAMESPACE}" \
    PORT=invalid FLAGD_HOST=missing-flagd.invalid CRITICAL_SECRET=
}

cleanup() {
  if [[ ! -f "${BACKUP_FILE}" ]]; then
    echo "[EWAT] ERROR: backup not found at ${BACKUP_FILE}"
    exit 1
  fi

  echo "[EWAT] Restoring payment deployment env configuration"
  kubectl apply -n "${NAMESPACE}" -f "${BACKUP_FILE}"
  rm -f "${BACKUP_FILE}"
}

case "${1:-}" in
  inject) inject ;;
  cleanup) cleanup ;;
  *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

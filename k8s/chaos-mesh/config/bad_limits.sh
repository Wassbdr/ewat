#!/usr/bin/env bash
# EWAT Chaos — bad_limits
# Injects extreme resource limits on checkout deployment.
# Usage: ./bad_limits.sh inject|cleanup

set -euo pipefail

NAMESPACE="ewat"
COMPONENT="checkout"
STATE_DIR="/tmp/ewat_config_failures"
BACKUP_FILE="${STATE_DIR}/bad_limits_checkout_deploy.json"

mkdir -p "${STATE_DIR}"

get_deploy_name() {
  kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
    -o jsonpath='{.items[0].metadata.name}'
}

inject() {
  DEPLOY_NAME=$(get_deploy_name)
  if [[ -z "${DEPLOY_NAME}" ]]; then
    echo "[EWAT] ERROR: checkout deployment not found"
    exit 1
  fi

  kubectl get deploy "${DEPLOY_NAME}" -n "${NAMESPACE}" -o json > "${BACKUP_FILE}"

  echo "[EWAT] Injecting bad resource limits on ${DEPLOY_NAME}"
  kubectl set resources deploy "${DEPLOY_NAME}" -n "${NAMESPACE}" \
    --containers='*' --limits=cpu=10m,memory=64Mi --requests=cpu=10m,memory=64Mi
}

cleanup() {
  if [[ ! -f "${BACKUP_FILE}" ]]; then
    echo "[EWAT] ERROR: backup not found at ${BACKUP_FILE}"
    exit 1
  fi

  echo "[EWAT] Restoring checkout deployment resources"
  kubectl apply -n "${NAMESPACE}" -f "${BACKUP_FILE}"
  rm -f "${BACKUP_FILE}"
}

case "${1:-}" in
  inject) inject ;;
  cleanup) cleanup ;;
  *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

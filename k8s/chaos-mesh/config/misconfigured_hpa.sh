#!/usr/bin/env bash
# EWAT Chaos — misconfigured_hpa
# Forces HPA min=max=1 on frontend under stress.
# Usage: ./misconfigured_hpa.sh inject|cleanup

set -euo pipefail

NAMESPACE="ewat"
COMPONENT="frontend"
STATE_DIR="/tmp/ewat_config_failures"
BACKUP_FILE="${STATE_DIR}/misconfigured_hpa_frontend.json"
CREATED_MARKER="${STATE_DIR}/misconfigured_hpa_created.txt"

mkdir -p "${STATE_DIR}"

get_deploy_name() {
  kubectl get deploy -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
    -o jsonpath='{.items[0].metadata.name}'
}

get_hpa_name() {
  kubectl get hpa -n "${NAMESPACE}" -l "app.kubernetes.io/component=${COMPONENT}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

inject() {
  DEPLOY_NAME=$(get_deploy_name)
  if [[ -z "${DEPLOY_NAME}" ]]; then
    echo "[EWAT] ERROR: frontend deployment not found"
    exit 1
  fi

  HPA_NAME=$(get_hpa_name)

  if [[ -n "${HPA_NAME}" ]]; then
    kubectl get hpa "${HPA_NAME}" -n "${NAMESPACE}" -o json > "${BACKUP_FILE}"
    echo "[EWAT] Patching existing HPA ${HPA_NAME} with min=max=1"
    kubectl patch hpa "${HPA_NAME}" -n "${NAMESPACE}" --type=merge \
      -p '{"spec":{"minReplicas":1,"maxReplicas":1}}'
  else
    HPA_NAME="${DEPLOY_NAME}"
    echo "${HPA_NAME}" > "${CREATED_MARKER}"
    echo "[EWAT] Creating constrained HPA for ${DEPLOY_NAME}"
    kubectl autoscale deploy "${DEPLOY_NAME}" -n "${NAMESPACE}" \
      --cpu-percent=95 --min=1 --max=1
  fi
}

cleanup() {
  if [[ -f "${BACKUP_FILE}" ]]; then
    echo "[EWAT] Restoring original frontend HPA"
    kubectl apply -n "${NAMESPACE}" -f "${BACKUP_FILE}"
    rm -f "${BACKUP_FILE}"
  elif [[ -f "${CREATED_MARKER}" ]]; then
    HPA_NAME=$(cat "${CREATED_MARKER}")
    echo "[EWAT] Deleting created constrained HPA ${HPA_NAME}"
    kubectl delete hpa "${HPA_NAME}" -n "${NAMESPACE}" --ignore-not-found=true
    rm -f "${CREATED_MARKER}"
  else
    echo "[EWAT] No HPA state to restore"
  fi
}

case "${1:-}" in
  inject) inject ;;
  cleanup) cleanup ;;
  *) echo "Usage: $0 inject|cleanup"; exit 1 ;;
esac

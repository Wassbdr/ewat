#!/bin/bash
# EWAT v5 — déploie un runner Train Ticket dans un namespace donné.
# Capitalise la procédure éprouvée sur `tt` :
#   manifests k8s-with-jaeger + fixes mongo:4.4 + jaeger:1.53 + jaeger ClusterIP stable
#   + NodePorts paramétrables (évite conflit entre runners) + rollout JVM.
#
# Usage :
#   bash v5/deploy/deploy_runner.sh <namespace> <ui_nodeport> <jaeger_nodeport>
#   ex : bash v5/deploy/deploy_runner.sh tt-b 32679 32690
set +e
NS=${1:?namespace requis}
UI_NP=${2:-32679}
JAEGER_NP=${3:-32690}
TT_MANIFESTS=${TT_MANIFESTS:-$HOME/repos/train-ticket/deployment/kubernetes-manifests/k8s-with-jaeger}
V5=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "=== déploiement runner $NS (UI NodePort $UI_NP, Jaeger NodePort $JAEGER_NP) ==="
kubectl create namespace "$NS" 2>/dev/null

echo "--- application des manifests TT ---"
for part in part1 part2 part3; do
  kubectl apply -n "$NS" -f "$TT_MANIFESTS/ts-deployment-${part}.yml" >/dev/null 2>&1
done

echo "--- fix MongoDB 4.4 (la v7 supprime OP_QUERY, casse le driver Java TT) ---"
for d in $(kubectl get deploy -n "$NS" --no-headers 2>/dev/null | awk '{print $1}' | grep -- '-mongo$'); do
  kubectl set image -n "$NS" "deploy/$d" "$d=mongo:4.4" >/dev/null 2>&1
done

echo "--- fix Jaeger 1.53 (la v2 supprime l'agent UDP 6831) + service ClusterIP stable ---"
kubectl set image -n "$NS" deploy/jaeger jaeger=jaegertracing/all-in-one:1.53 >/dev/null 2>&1
kubectl delete svc -n "$NS" jaeger >/dev/null 2>&1
cat <<EOF | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Service
metadata: {name: jaeger, namespace: $NS, labels: {app: jaeger, app.kubernetes.io/name: jaeger}}
spec:
  type: ClusterIP
  selector: {app.kubernetes.io/component: all-in-one, app.kubernetes.io/name: jaeger}
  ports:
  - {name: agent-zipkin-thrift, port: 5775, protocol: UDP, targetPort: 5775}
  - {name: agent-compact, port: 6831, protocol: UDP, targetPort: 6831}
  - {name: agent-binary, port: 6832, protocol: UDP, targetPort: 6832}
  - {name: agent-config, port: 5778, protocol: TCP, targetPort: 5778}
EOF

echo "--- services UI + jaeger-query avec NodePorts distincts ---"
# Les manifests codent en dur 32677/32688 → conflit cluster-global avec un autre
# runner → la création ÉCHOUE silencieusement. On (re)crée explicitement ces 2
# services avec des NodePorts dédiés.
kubectl delete svc -n "$NS" ts-ui-dashboard jaeger-query >/dev/null 2>&1
cat <<EOF | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Service
metadata: {name: ts-ui-dashboard, namespace: $NS}
spec:
  type: NodePort
  selector: {app: ts-ui-dashboard}
  ports: [{name: http, port: 8080, targetPort: 8080, nodePort: $UI_NP}]
---
apiVersion: v1
kind: Service
metadata: {name: jaeger-query, namespace: $NS, labels: {app.kubernetes.io/name: jaeger}}
spec:
  type: NodePort
  selector: {app.kubernetes.io/component: all-in-one, app.kubernetes.io/name: jaeger}
  ports: [{name: query, port: 16686, targetPort: 16686, nodePort: $JAEGER_NP}]
EOF

echo "--- attente readiness (peut prendre ~10-15 min sous pression CPU) ---"
sleep 30
# purge d'éventuels pods zombies (ContainerStatusUnknown) puis attente
kubectl get pods -n "$NS" --no-headers 2>/dev/null | awk '$3=="ContainerStatusUnknown"{print $1}' \
  | xargs -r kubectl delete pod -n "$NS" --grace-period=0 >/dev/null 2>&1

echo "--- rollout JVM (javaagent + annotations prometheus.io) ---"
bash "$V5/collect/enable_jvm_metrics.sh" "$NS"

echo "=== runner $NS déployé. Vérifs : ==="
echo "  kubectl get pods -n $NS | grep -c 1/1   # attendu 64"
echo "  curl -s -X POST http://<node-ip>:$UI_NP/api/v1/users/login -d '{\"username\":\"fdse_microservice\",\"password\":\"111111\"}'"
echo "  PromQL : count(jvm_threads_state{namespace=\"$NS\",state=\"BLOCKED\"})"

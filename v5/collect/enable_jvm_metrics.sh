#!/bin/bash
# EWAT v5 — active les métriques JVM sur tous les services TT SANS rebuild.
# Pour chaque deploy ts-*-service : initContainer télécharge jmx_prometheus_javaagent,
# JAVA_TOOL_OPTIONS ajoute -javaagent (expose :9404), annotations prometheus.io
# pour que monitoring-metrics/prometheus-server scrape automatiquement.
# Rollout batché maxSurge=0 (pas de doublement de pods) pour ménager le cluster.
set +e
NS=tt
JAR_URL="https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.17.2/jmx_prometheus_javaagent-0.17.2.jar"
INIT_IMG="codewisdom/ts-order-service-with-jaeger:v1"  # image avec wget+java, déjà sur les nœuds
LOG=/tmp/v5_jvm_rollout.log
echo "START $(date)" > $LOG

# services applicatifs (exclut mongo/mysql) = ceux qui portent une JVM
mapfile -t SVCS < <(kubectl get deploy -n $NS --no-headers | awk '{print $1}' | grep '^ts-' | grep -vE 'mongo$|mysql$' | sort)
echo "${#SVCS[@]} services à instrumenter" >> $LOG

patch_one() {
  local svc=$1
  cat > /tmp/jvmp_$svc.yaml <<EOF
spec:
  strategy: {type: RollingUpdate, rollingUpdate: {maxSurge: 0, maxUnavailable: 1}}
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9404"
        prometheus.io/path: "/metrics"
    spec:
      volumes:
      - name: jmxagent
        emptyDir: {}
      initContainers:
      - name: fetch-jmx
        image: $INIT_IMG
        imagePullPolicy: IfNotPresent
        command: ["sh","-c"]
        args:
        - |
          # retry du téléchargement (résilience egress sur campagne multi-jours ;
          # la ConfigMap est bloquée par la limite de taille de requête du cluster)
          for i in 1 2 3 4 5; do
            wget -qO /jmxagent/jmx.jar $JAR_URL && break
            echo "retry \$i dl jmx jar"; sleep 10
          done
          printf 'lowercaseOutputName: true\nrules:\n- pattern: ".*"\n' > /jmxagent/config.yaml
        volumeMounts:
        - {name: jmxagent, mountPath: /jmxagent}
      containers:
      - name: $svc
        env:
        - name: JAVA_TOOL_OPTIONS
          value: "-javaagent:/jmxagent/jmx.jar=9404:/jmxagent/config.yaml"
        volumeMounts:
        - {name: jmxagent, mountPath: /jmxagent}
EOF
  kubectl patch deploy -n $NS $svc --patch-file /tmp/jvmp_$svc.yaml >> $LOG 2>&1
}

BATCH=6
i=0
for svc in "${SVCS[@]}"; do
  patch_one "$svc"
  echo "patched $svc" >> $LOG
  i=$((i+1))
  if (( i % BATCH == 0 )); then
    echo "=== batch jusqu'à $svc : attente readiness ($(date))" >> $LOG
    for s in "${SVCS[@]:i-BATCH:BATCH}"; do
      kubectl rollout status deploy -n $NS "$s" --timeout=600s >> $LOG 2>&1
    done
  fi
done
# dernier batch partiel
echo "=== attente finale ($(date))" >> $LOG
for svc in "${SVCS[@]}"; do
  kubectl rollout status deploy -n $NS "$svc" --timeout=600s >> $LOG 2>&1
done
echo "ALL JVM INSTRUMENTED $(date)" >> $LOG
echo "ready: $(kubectl get pods -n $NS --no-headers | grep -c '1/1') / $(kubectl get pods -n $NS --no-headers | wc -l)" >> $LOG

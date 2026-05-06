# EWAT v4 — Runbook de collecte

Cible de qualité (issue audit, P2) :

* `0%` NaN sur `disk_io` ;
* traces complètes (Jaeger ou OTel SDK) ;
* épisodes ≥ **40 timesteps** (cf. `configs/collection_v4.yaml`) ;
* données suffisantes pour réhabiliter H2 (drift / anomaly separability).

Ce document décrit le **runbook minimal** pour produire un dataset `ewat_v4`
conforme. Il suppose que l'opérateur dispose des droits **cluster-admin**
(les étapes 1, 2 et 4 nécessitent la création de ressources hors namespace).

## 1. Pré-requis (cluster-admin)

```bash
# vérifier qu'on est cluster-admin sur observit-cluster1
kubectl auth can-i create namespace --all-namespaces

# vérifier les composants existants
kubectl get pods -A | grep -iE "prometheus|otel|jaeger|loki"
```

## 2. Déployer le collecteur OTel central

Le manifest existe déjà (`k8s/otel-collector-gateway.yaml`) et prend en charge
les pipelines OTLP gRPC + HTTP. L'appliquer :

```bash
kubectl apply -f k8s/otel-collector-gateway.yaml
kubectl rollout status -n ewat deploy/otel-collector
```

Le `Service` `otel-collector.ewat.svc.cluster.local` expose :

* `4317` : OTLP gRPC (consommé par les SDK applicatifs) ;
* `4318` : OTLP HTTP ;
* `9464` : exporter Prometheus (scrap par `monitoring-metrics`).

## 3. Instrumenter les services applicatifs (cluster-admin)

Le dataset v3 utilise le `otel-demo` Helm chart. Pour v4 il faut s'assurer
que **tous les services canoniques** (`frontend`, `recommendation`, `cart`,
`ad`, `product-catalog`, `load-generator`) :

1. embarquent un OTel SDK avec exporter OTLP pointant sur l'endpoint
   ci-dessus ;
2. exposent une métrique `container_fs_writes_bytes_total` (ou équivalent)
   visible par Prometheus — c'est la métrique qui alimentait `disk_io` et
   qui était massivement NaN en v3.

Patches type :

```yaml
# k8s/apps/otel-demo-values.yaml — ajout dans chaque deployment
env:
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://otel-collector.ewat.svc.cluster.local:4317"
  - name: OTEL_SERVICE_NAME
    value: "{{ .Release.Name }}-{{ .Chart.Name }}"
  - name: OTEL_TRACES_EXPORTER
    value: "otlp"
  - name: OTEL_METRICS_EXPORTER
    value: "otlp"
```

Pour `disk_io`, vérifier que `kube-state-metrics` ou cAdvisor exportent bien
les compteurs disque, sinon déployer `node-exporter` sur les workers.

## 4. Vérification rapide

Avant lancement de la campagne, smoke test :

```bash
# trace bien reçue côté gateway ?
kubectl logs -n ewat deploy/otel-collector --tail=200 | grep -i trace

# disk_io disponible côté Prometheus ?
kubectl exec -n cattle-monitoring-system deploy/rancher-monitoring-prometheus -- \
  promtool query instant http://localhost:9090 \
    'sum(rate(container_fs_writes_bytes_total{namespace="ewat"}[2m])) by (pod)'
```

## 5. Lancement campagne v4

```bash
# Phase 1 — record (peut tourner pendant ~6.4 jours selon configs)
python -m scripts.record_episode --config configs/collection_v4.yaml --endpoint-mode nodeport

# Phase 2 — features avec graine déterministe
python -m scripts.build_features \
    --config configs/collection_v4.yaml \
    --raw-root data/raw_v4 \
    --feature-set v4

# Phase 3 — assemblage avec stratification cluster-aware
python -m scripts.assemble_dataset \
    --feature-set v4 \
    --output-dir data/datasets/ewat_v4 \
    --min-test-per-cluster 2 \
    --min-val-per-cluster 1

# Quality gate v4
python -m scripts.validate_v4 \
    --features-root data/features/v4 \
    --output reports/v4_gate.md \
    --strict
```

Si `validate_v4 --strict` échoue, l'épisode fautif est documenté dans
`reports/v4_gate.md` (raison du fail : T trop court, NaN disk_io résiduel,
absence de trace, etc.). Corriger l'instrumentation correspondante puis
re-collecter uniquement les scenarios concernés (`record_episode --only ...`).

## 6. Re-validation H2 (post-collecte)

Une fois `ewat_v4` assemblé :

```bash
python -m experiments.h2_lookthrough.run --dataset data/datasets/ewat_v4
```

Le résultat doit montrer :

* MMD-RFF AUC **drift vs anomaly** ≥ 0.7 (vs 0.60 sur v3) ;
* fenêtre look-through stable sans absorption d'anomalies ;
* taux de faux positifs après filtre **< 5%** sur le test split.

Sans ces seuils, H2 reste falsifié : c'est exactement le verdict que cette
collecte v4 cherche à inverser.

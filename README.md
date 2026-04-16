# EWAT — Early Warning and Anomaly Typing

Stage de recherche Devoteam — Wassim Badraoui

Détection précoce et typage automatique des anomalies dans les architectures
microservices Kubernetes.

## Cluster

- **observit-cluster1** (Rancher, RKE2 v1.32.7)
- 9 nœuds (8 Ready, 1 NotReady)
- Observabilité : Prometheus + Grafana + OTel Collector (déjà en place)
- Accès : namespace-admin sur `ewat`

## Hypothèses

- **H1** Structurabilité : silhouette > 0.3 en held-out
- **H2** Séparabilité : réduction FPR significative (p < 0.05)
- **H3** Prédictibilité : AUROC typé > baseline générique

## Services canoniques (6)

Scope réduit aux services effectivement observables sur les 3 modalités
(Prometheus + Jaeger + Loki) du cluster :
`frontend`, `recommendation`, `cart`, `ad`, `product-catalog`, `load-generator`.

## Pipeline dataset — Record → Build → Assemble

Le pipeline de construction du dataset est découplé en trois phases
indépendantes et rejouables. Une phase ne dépend jamais de la suivante, et
seules les phases 1 touche au cluster.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Phase 1     │     │  Phase 2         │     │  Phase 3         │
│  record      │ --> │  build_features  │ --> │  assemble_dataset│
│  (online)    │     │  (offline)       │     │  (offline)       │
└──────────────┘     └──────────────────┘     └──────────────────┘
 data/raw/<ep>      data/features/<set>       data/datasets/<name>
```

### Pré-requis : port-forwards locaux

Depuis la machine hors cluster, les noms `*.svc.cluster.local` ne résolvent
pas. Ouvrir trois forwards :

```bash
kubectl -n monitoring-metrics port-forward svc/prometheus-server 19090:80
kubectl -n rca-sandbox      port-forward svc/rca-jaeger          16686:16686
kubectl -n monitoring-logs  port-forward svc/loki-gateway        13100:80
```

### Phase 1 — record

Orchestre Chaos Mesh (baseline → pre → injection → recovery → cool-down) puis
dumpe les réponses brutes de Prometheus, Jaeger et Loki pour l'épisode. Aucun
feature engineering à cette étape.

```bash
python -m scripts.record_episode \
    --config configs/collection.yaml \
    --base-config configs/default.yaml \
    --endpoint-mode local-portforward
```

Sortie : `data/raw/episode_<scenario>_<rep>_<ts>/{episode.json,prometheus_range.json.gz,jaeger_spans.json.gz,loki_logs.json.gz,manifest.json}`

### Phase 2 — build_features

Rejouable à volonté, hors cluster. Construit `S(t) ∈ ℝ^{N×17}`,
`A(t) ∈ ℝ^{N×N×3}` et `labels.parquet` à partir des dumps bruts.

```bash
python -m scripts.build_features \
    --raw-root data/raw \
    --base-config configs/default.yaml \
    --config configs/collection.yaml \
    --feature-set v1 \
    --grid-step-s 30 \
    --trace-window-s 120
```

Sortie : `data/features/v1/<episode_id>/{signal.npz,signal_mask.npz,adjacency.npz,labels.parquet,services.json,graph_stats.csv,metadata.json,feature_provenance.json}`

### Phase 3 — assemble_dataset

Applique les filtres qualité, vérifie la stabilité de l'ensemble V des
services, puis produit un split **strictement temporel** (train/val/test) des
épisodes.

```bash
python -m scripts.assemble_dataset \
    --features-root data/features/v1 \
    --output data/datasets/ewat_v1 \
    --train-ratio 0.70 \
    --val-ratio   0.15
```

Sortie : `data/datasets/ewat_v1/{episodes/,index.parquet,split.json,services.json,summary.csv,dataset.json}`

### Validation / quality gates

Contrôles par épisode (shape, NaN par modalité, labels, graphe non vide)
et au niveau dataset (stabilité de V, intégrité du split temporel) :

```bash
# Un épisode
python -m scripts.validate_dataset --episode data/features/v1/<episode_id>

# Tous les épisodes d'un feature-set
python -m scripts.validate_dataset --features-root data/features/v1

# Un dataset assemblé
python -m scripts.validate_dataset --dataset data/datasets/ewat_v1 --strict
```

## Scénarios

Définis dans `k8s/chaos-mesh/registry.yaml` et `configs/collection.yaml` :

- **θ_drift (benign)** : `drift_scale_up`, `drift_rolling_deploy`,
  `drift_config_change`, `drift_traffic_ramp`
- **θ_anomaly — hard** : `crash`, `oom`, `network_loss`
- **θ_anomaly — gray** : `intermittent_error`, `fail_slow_latency`, `fail_slow_cpu`
- **θ_anomaly — contention** : `cpu_starvation`, `memory_pressure`,
  `noisy_neighbor`, `resource_leak`
- **θ_{drift ∩ anomaly}** : `faulty_deploy_overlap`

Les scénarios de type `drift` ne sont **pas** des pannes : ils injectent un
décalage de distribution bénin (scaling, rolling deploy, rampe de trafic) et
servent à calibrer ε_drift (étape 0 MMD-RFF) et à falsifier H2.

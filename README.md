# EWAT — Early Warning and Anomaly Typing

Stage de recherche Devoteam — Wassim Badraoui

Détection précoce et typage automatique des anomalies dans les architectures
microservices Kubernetes. Projet de recherche : séparer explicitement drift
bénin et anomalie réelle avant d'apprendre une ontologie empirique des types
de pannes.

EWAT n'est pas du RCA : le RCA est post-mortem (Où, Pourquoi), EWAT est de
l'early warning (Quoi, Dans combien de temps, avant la panne).

## Cluster

- **observit-cluster1** (Rancher, RKE2 v1.32.7)
- 9 nœuds (8 Ready, 1 NotReady)
- Observabilité : Prometheus + Grafana + OTel Collector + Jaeger + Loki (en place)
- Accès : namespace-admin sur `ewat`

## Hypothèses

- **H1** Structurabilité : silhouette > 0.3 en held-out
- **H2** Séparabilité : réduction FPR significative (p < 0.05) grâce à la séparation drift/anomalie
- **H3** Prédictibilité : AUROC typé > baseline générique

## Services canoniques (|V| = 6)

Scope réduit aux services effectivement observables sur les 3 modalités
(Prometheus + Jaeger + Loki) du cluster :
`frontend`, `recommendation`, `cart`, `ad`, `product-catalog`, `load-generator`.

## Pipeline dataset — Record → Build → Assemble

Le pipeline de construction du dataset est découplé en trois phases
indépendantes et rejouables. Une phase ne dépend jamais de la suivante, et
seule la phase 1 touche au cluster.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Phase 1     │     │  Phase 2         │     │  Phase 3         │
│  record      │ --> │  build_features  │ --> │  assemble_dataset│
│  (online)    │     │  (offline)       │     │  (offline)       │
└──────────────┘     └──────────────────┘     └──────────────────┘
 data/raw/<ep>      data/features/<set>       data/datasets/<name>
```

### Phase 1 — record

Orchestre Chaos Mesh (baseline → pre → injection → recovery → cool-down) puis
dumpe les réponses brutes de Prometheus, Jaeger et Loki pour chaque épisode.
Pas de feature engineering à cette étape — uniquement un échantillonnage
fidèle de la télémétrie.

Trois modes d'accès aux backends (`--endpoint-mode`) :

- **`nodeport`** — recommandé pour toute campagne > 1 h. Accès direct via
  NodePort sur un worker Ready. Évite la dégradation SPDY des port-forwards
  longs (ConnectionReset côté Jaeger observé empiriquement). Voir
  `collection.nodeport.*` dans `configs/collection.yaml`.
- **`local-portforward`** — les forwards sont ouverts manuellement, le script
  les consomme sur `127.0.0.1:<port>`. Pratique pour debug.
- **`in-cluster`** — exécution depuis un pod du namespace `ewat`.

Option `--manage-port-forwards` : ouvre et **renouvelle** un port-forward
dédié avant chaque dump d'épisode, puis le ferme. Robuste aux tunnels SPDY
qui se dégradent après 1-2 h.

Robustesse :
- **Checkpoint** append-only (`checkpoint.jsonl`) : reprise idempotente après
  crash ou SIGINT, les épisodes déjà validés sont skip.
- **Graceful shutdown** : SIGINT/SIGTERM laisse l'épisode en cours finir
  (delete Chaos Mesh + dump) avant exit.
- **Quality gate post-dump** : chaque épisode est validé (Prometheus
  queries_ok non vide, Jaeger n_traces > 0, Loki n_lines > 0). Les échecs
  sont marqués `.quality_failed` et non checkpointés — donc rejoués.
- **Timeouts chaos** : 60 s apply / 30 s delete, fallback `--ignore-not-found`
  pour éviter qu'un delete bloqué ne fige la campagne.

```bash
python -m scripts.record_episode \
    --config configs/collection.yaml \
    --base-config configs/default.yaml \
    --endpoint-mode nodeport       # ou local-portforward / in-cluster
```

Sortie : `data/raw/episode_<scenario>_<rep>_<ts>/` contenant
`episode.json`, `manifest.json`, `prometheus_range.json.gz`,
`jaeger_spans.json.gz`, `loki_logs.json.gz`.

Contrôle rapide de la complétude (lit uniquement les manifests, pas de
gunzip) :

```bash
python -m scripts.validate_raw --raw-root data/raw
python -m scripts.validate_raw --episode data/raw/episode_crash_000_... --strict
```

### Phase 2 — build_features

Rejouable à volonté, hors cluster. Construit `S(t) ∈ ℝ^{N×17}`,
`A(t) ∈ ℝ^{N×N×3}` et `labels.parquet` à partir des dumps bruts via les
extracteurs fichier (`src/telemetry/extractors/`) qui réutilisent la logique
des collecteurs online.

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

```bash
# Un épisode feature-isé
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

Les scénarios `drift_*` ne sont **pas** des pannes : ils injectent un
décalage de distribution bénin (scaling, rolling deploy, rampe de trafic) et
servent à calibrer ε_drift (étape 0 MMD-RFF) et à falsifier H2.

## État actuel

- Pipeline 3-phase opérationnel, extracteurs offline séparés des collecteurs online.
- Campagne de validation (2 rep × 14 scénarios = 28 épisodes) collectée
  dans `data/raw_new/` ; les dumps Jaeger y montrent la dégradation SPDY
  qui a motivé la bascule NodePort + port-forwards renouvelés.
- Tests unitaires pour collecteurs, signal builder, validation dataset.
- Scripts chaos complets (contention, gray, drift, systemic, drift∩anomaly).

Voir `docs/notes/synthese_collecte_dataset.md` pour le détail des évolutions
et des décisions de conception.

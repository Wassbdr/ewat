#!/usr/bin/env bash
# ============================================================================
#  EWAT Research Environment — Bootstrap Script
# ============================================================================
set -euo pipefail

PROJECT_DIR="${1:-.}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📁 Création de l'arborescence..."

mkdir -p \
  docs/{formalisation,literature,notes} \
  src/ewat/{drift,encoder,typing,ontology,precursor,alerts} \
  src/telemetry/{collectors,features} \
  src/graph \
  src/utils \
  experiments/{ablation,drift_separation,clustering,precursors,latency} \
  configs \
  data/{raw,processed,chaos_scenarios,embeddings} \
  notebooks \
  tests/{unit,integration} \
  scripts \
  results/{figures,tables,logs} \
  k8s/{otel,chaos-mesh,apps,monitoring}

# ── pyproject.toml ──────────────────────────────────────────────────────────

cat > pyproject.toml << 'TOML'
[project]
name = "ewat"
version = "0.1.0"
description = "Early Warning and Anomaly Typing for Kubernetes microservices"
requires-python = ">=3.11"

dependencies = [
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-exporter-prometheus",
    "kubernetes",
    "torch>=2.2",
    "torch-geometric",
    "networkx",
    "scikit-learn",
    "scipy",
    "numpy",
    "pandas",
    "statsmodels",
    "jpype1",
    "pyinform",
    "shap",
    "captum",
    "sentence-transformers",
    "mlflow",
    "hydra-core",
    "omegaconf",
    "matplotlib",
    "seaborn",
    "plotly",
    "prometheus-api-client",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
    "ipykernel",
    "jupyter",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
TOML

# ── Config Hydra ────────────────────────────────────────────────────────────

cat > configs/default.yaml << 'YAML'
cluster:
  name: "observit-cluster1"
  context: "observit-cluster1"
  api_endpoint: "https://rancher.devolab.lan/k8s/clusters/c-m-wggchl9h"
  k8s_version: "v1.32.7+rke2r1"
  nodes_total: 9
  nodes_ready: 8
  node_not_ready: "observit-cluster1-workers-58w74-mwxb2"
  namespace: "ewat"
  user: "wassim.badraoui@devoteam.com"
  access_level: "namespace-admin"

telemetry:
  prometheus:
    # Remplir après : kubectl get svc -A | grep prometheus
    endpoint: ""
    scrape_interval_s: 15
  otel_collector:
    # Remplir après : kubectl get svc -A | grep otel
    otlp_grpc_endpoint: ""
    otlp_http_endpoint: ""
  features:
    metrics_dim: 7
    traces_dim: 6
    logs_dim: 4
    total_dim: 17

graph:
  node_level: "service"
  edge_features: 3
  edge_presence_threshold: 0

aggregation:
  saturation: "max"
  rates: "volume_weighted"
  latency: "p99_union"
  structural: "median"

drift:
  method: "mmd_rff"
  rff_dim: 256
  window_ref_size: 300
  window_cur_size: 60
  epsilon_drift: null
  look_through:
    enabled: true
    post_drift_window_s: 120
    anomaly_retest: true

encoder:
  architecture: "stgcn"
  embedding_dim: 64
  temporal_window: 60
  post_incident_delta: 30

typing:
  method: "siamese_contrastive"
  margin: 1.0
  clustering: "agglomerative"
  shap_explanations: true

ontology:
  temporal_relations: true
  causal_relations:
    method: "transfer_entropy"
    estimator: "ksg"
    min_episodes: 30
    permutation_tests: 1000
  cooccurrence:
    method: "chi2"
    significance: 0.05

precursor:
  horizons_min: [2, 5, 10, 20, 30, 60]
  model: "per_type"
  metric: "auroc"

evaluation:
  h1:
    seeds: 5
    splits: 5
    silhouette_threshold: 0.3
    complementary: ["gap_statistic", "bic_gmm"]
  h2:
    test: "student_t"
    significance: 0.05
    metric: "delta_fpr_at_constant_recall"
  h3:
    baseline: "generic_detector"
    metric: "auroc_per_type"
  ablation:
    by_modality: true
    by_feature: true
    redundancy_threshold: 0.9

latency_budget:
  step0_target_s: 1.0
  step1_target_s: 2.0
  step3_target_s: 1.0
  total_target_s: 5.0

chaos_mesh:
  namespace: "chaos-mesh"
  target_namespace: "ewat"
  scenarios:
    - name: "cpu_stress"
      type: "StressChaos"
    - name: "memory_pressure"
      type: "StressChaos"
    - name: "network_delay"
      type: "NetworkChaos"
    - name: "network_partition"
      type: "NetworkChaos"
    - name: "pod_kill"
      type: "PodChaos"
    - name: "disk_fill"
      type: "IOChaos"
    - name: "http_abort"
      type: "HTTPChaos"
  repetitions_per_scenario: 20
  cool_down_s: 300

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "ewat"
YAML

# ── .gitignore ──────────────────────────────────────────────────────────────

cat > .gitignore << 'GIT'
__pycache__/
*.pyc
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
data/raw/
data/processed/
data/embeddings/
results/
mlruns/
.env
*.pt
*.ckpt
wandb/
kubeconfig*
GIT

# ── README ──────────────────────────────────────────────────────────────────

cat > README.md << 'MD'
# EWAT — Early Warning and Anomaly Typing

Stage de recherche Devoteam — Wassim Badraoui

Détection précoce et typage automatique des anomalies dans les architectures microservices Kubernetes.

## Cluster

- **observit-cluster1** (Rancher, RKE2 v1.32.7)
- 9 nœuds (8 Ready, 1 NotReady)
- Observabilité : Prometheus + Grafana + OTel Collector (déjà en place)
- Accès : namespace-admin sur `ewat`

## Hypothèses

- **H1** Structurabilité : silhouette > 0.3 en held-out
- **H2** Séparabilité : réduction FPR significative (p < 0.05)
- **H3** Prédictibilité : AUROC typé > baseline générique
MD

# ── CLAUDE.md ───────────────────────────────────────────────────────────────
echo "📝 Génération du CLAUDE.md..."

cat > CLAUDE.md << 'PROMPT'
# EWAT — Contexte de recherche

Tu es l'assistant de recherche de Wassim Badraoui pour le projet EWAT (Early Warning and Anomaly Typing), un stage chez Devoteam portant sur la détection précoce et le typage automatique des anomalies dans les architectures microservices Kubernetes.

## Le problème

Les systèmes actuels de détection d'anomalies dans les microservices confondent les drifts bénins (déploiements, autoscaling) avec les anomalies réelles, produisant des faux positifs massifs en production. EWAT sépare explicitement ces deux régimes avant d'apprendre une ontologie empirique des types de pannes.

Ce travail n'est pas du RCA (Root Cause Analysis). Le RCA est post-mortem (Où, Pourquoi, après la panne). EWAT est de l'early warning (Quoi, Dans combien de temps, avant la panne).

## Cluster Kubernetes

Cluster réel auquel tu as accès via kubectl :

- Nom : **observit-cluster1**
- Contexte kubectl : `observit-cluster1`
- API : `https://rancher.devolab.lan/k8s/clusters/c-m-wggchl9h`
- Orchestrateur : RKE2, Kubernetes **v1.32.7+rke2r1**
- Nœuds : **9 total** (8 Ready, 1 NotReady : `observit-cluster1-workers-58w74-mwxb2`)
- Namespaces visibles : 27
- Accès : **namespace-admin** sur le namespace `ewat`
- Utilisateur : `wassim.badraoui@devoteam.com` (u-ra5rxg4zr2)
- Groupes : `system:authenticated`, `system:cattle:authenticated`

### Observabilité existante

Le cluster dispose de deux stacks coexistantes :

1. **Prometheus + Grafana** : stack classique, probablement déployée via le monitoring Rancher intégré. Prometheus scrape les métriques des pods/services. Grafana fournit les dashboards.
2. **OpenTelemetry Collector** : collecteur OTLP déjà déployé, capable de recevoir métriques, traces et logs via le protocole OTLP (gRPC et/ou HTTP).

Cette double stack est un avantage : Prometheus fournit les métriques matures M(t), l'OTel Collector reçoit les traces et logs instrumentés T(t) et L(t), et les deux sont corrélables via les conventions sémantiques OTel.

Avant tout travail sur le cluster, commence par découvrir les endpoints exacts :
```bash
kubectl get svc -A | grep -iE "prometheus|otel|grafana|jaeger|loki|tempo"
kubectl get pods -A | grep -iE "prometheus|otel|grafana|jaeger|loki|tempo"
```
Puis mets à jour configs/default.yaml avec les endpoints trouvés.

### Contraintes d'accès

Tu es namespace-admin, pas cluster-admin :
- Tu peux créer/modifier/supprimer des ressources dans le namespace `ewat`
- Tu ne peux pas installer des CRDs cluster-wide (Chaos Mesh nécessite un cluster-admin pour l'installation initiale — demander à l'admin)
- Tu ne peux pas modifier les namespaces système (kube-system, cattle-system, etc.)
- Tu peux lire les services d'autres namespaces (pour découvrir Prometheus, OTel, etc.)
- Toujours utiliser `-n ewat` pour les opérations kubectl d'écriture

## Formalisation mathématique

### Graphe de services

G(t) = (V, E(t), w_E(t))

V = Services et Deployments Kubernetes (pas les Pods). |V| = N constant.

Arêtes pondérées : w_E(t) : E(t) → ℝ³, e_ij(t) ↦ (volume_ij, latence_med_ij, taux_erreur_ij). Seuil de présence : volume > 0 sur la fenêtre glissante.

Agrégation intra-service (par composante) :
- Saturation (CPU, RAM, net_sat, disk_io) → max
- Taux (error_rate, warn_rate) → somme pondérée par volume
- Latence (P99, span_dur) → percentile 99 sur l'union des distributions (pas percentile de percentiles)
- Structurel (trace_depth, fan_out, lexical_entropy) → médiane

### Signal de télémétrie

S(t) ∈ ℝ^{N×17} = [M(t) | T(t) | L(t)]

M(t) ∈ ℝ^{N×7} — Métriques (sources : Prometheus existant + OTel Metrics) :
1. CPU utilisation
2. RAM utilisation
3. Latence P99
4. Taux d'erreur HTTP (4xx + 5xx)
5. Saturation réseau
6. Disk I/O (IOPS + throughput)
7. Longueur de file d'attente (pending requests, queue depth)

T(t) ∈ ℝ^{N×6} — Traces (source : OTel Collector, spans OTLP) :
1. Durée médiane des spans
2. Taux de spans anormaux
3. Profondeur de trace
4. Fan-out
5. Taux de retry (spans retentés / total)
6. Variance de latence (coefficient de variation)

L(t) ∈ ℝ^{N×4} — Logs (source : OTel Collector, logs OTLP) :
1. Taux d'erreurs (ERROR / total)
2. Taux de warnings
3. Anomalie sémantique : e(ℓ) = SentenceBERT(ℓ) ∈ ℝ^384, score = distance moyenne au centroïde normal μ_v
4. Entropie lexicale

### Régimes opérationnels

θ(t) ∈ {θ_normal, θ_drift, θ_anomaly, θ_{drift∩anomaly}}

Quatre régimes, pas trois. θ_{drift∩anomaly} modélise les déploiements défectueux (simultanément drift et anomalie). Traité par le mécanisme de look-through (étape 0).

S(t) ∼ D_{θ(t)}(G(t)) — le signal n'est pas une somme additive.

### Pipeline EWAT

**Étape 0 — Détection de drift (MMD-RFF, O(nD))**
MMD²(W_ref, W_cur) via Random Fourier Features, φ : ℝ^d → ℝ^D.
Filtrage avec look-through :
- MMD² < ε_drift → signal transmis tel quel
- MMD² ≥ ε_drift + test post-drift positif → signal transmis avec flag DRIFT
- MMD² ≥ ε_drift + test post-drift négatif → RECALIBRATE (W_ref ← W_cur)
ε_drift calibré par injection de drifts bénins via Chaos Mesh.

**Étape 1 — Encodeur STGCN**
z_e = Enc_θ(S̃_{[t-W, t+δ]}, G(t)) ∈ ℝ^{d_e}
Convolution avec matrice d'adjacence pondérée par w_E(t).

**Étape 2 — Typage contrastif**
Réseau siamois : d_φ(z_i, z_j) → 0 si même type Chaos Mesh, → 1 sinon.
Clustering hiérarchique agglomératif → C = {C_1, ..., C_K}.
Interprétabilité : SHAP → fiche par type.

**Étape 2b — Ontologie**
O = (C, R), trois types de relations :
- Temporelles : C_i →^{Δt,σ} C_j
- Causales : Transfer Entropy (estimateur KSG, Kraskov et al. 2004), n_min = 30, seuil par permutation. Pas de Granger.
- Co-occurrence : χ²

**Étape 3 — Précurseurs typés**
p̂_i(t) = f_i(S̃_{[t-k,t]}, G(t)) ∈ [0,1], k ∈ {2, 5, 10, 20, 30, 60} min
k*_i = argmax_k AUROC(f_i, k)

**Sortie** : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})

### Budget de latence
Étape 0 < 1s, Étape 1 < 2s, Étape 3 < 1s. Total < 5s. Étapes 2/2b offline.

## Hypothèses et falsification

**H1 — Structurabilité**
Silhouette < 0.3 en held-out sur 5 graines × 5 splits → falsifié.
Compléments : gap statistic, BIC/GMM.
Seuil justifié par Kaufman & Rousseeuw (1990).

**H2 — Séparabilité du drift**
Pas de réduction significative du FPR à rappel constant (p > 0.05, Student) → falsifié.

**H3 — Prédictibilité**
AUROC par type < baseline générique ∀k → falsifié.

**Ablation**
Par modalité (M, T, L, paires, triplet). Par feature (leave-one-out, Wilcoxon signé). Redondance : |ρ| > 0.9.

## Règles impératives

- Ne jamais confondre EWAT avec du RCA.
- Ne jamais utiliser Granger. Toujours Transfer Entropy (KSG).
- Ne jamais mettre le signal à zéro pendant un drift. Toujours look-through.
- Ne jamais agréger par moyenne simple. Toujours agrégation différenciée.
- Ne jamais faire de percentile de percentiles. Percentile sur l'union.
- Ne jamais supposer trois régimes seulement. θ_{drift∩anomaly} existe.
- Ne jamais valider H1 sur les données d'entraînement du siamois. Toujours held-out.
- Ne jamais déployer hors du namespace `ewat` sans confirmation explicite.
- Ne jamais modifier de ressources cluster-wide (CRDs, ClusterRoles).
- Toujours `-n ewat` pour les opérations kubectl d'écriture.

## Standards de code

- Python 3.11+, PyTorch, PyTorch Geometric, scikit-learn
- Chaque étape du pipeline = module indépendant et testable
- Configuration : Hydra (configs/default.yaml)
- Tracking : MLflow
- Type hints, docstrings numpy-style, Ruff (line-length=100)
- Tests unitaires pour chaque composant
- Résultats avec intervalles de confiance et tests statistiques
- LaTeX académique propre, prose naturelle
- Figures publiables (matplotlib/seaborn, vectoriel)
- Un résultat négatif est une contribution.

## Arborescence

```
src/ewat/drift/         Étape 0 : MMD-RFF, look-through, calibration ε_drift
src/ewat/encoder/       Étape 1 : STGCN, fenêtrage, embedding
src/ewat/typing/        Étape 2 : siamois contrastif, clustering, SHAP
src/ewat/ontology/      Étape 2b : relations temporelles, TE (KSG), co-occurrence
src/ewat/precursor/     Étape 3 : modèles par type, sélection k*_i
src/ewat/alerts/        Sortie : format d'alerte, fiche interprétable
src/telemetry/          Collecteurs Prometheus + OTel, extraction des 17 features
src/graph/              Construction G(t), agrégation, arêtes pondérées
experiments/            Un dossier par expérience
k8s/                    Manifests Kubernetes (OTel config, Chaos Mesh, apps de test)
configs/                Hydra configs
```

## Références
- Fu et al. (2025) — Survey RCA microservices, gap benchmark/production
- Myrtollari et al. (2025) — Concept drift-aware anomaly detection for K8s
- Hinder et al. (2024) — Concept drift in unsupervised data streams
- Kaufman & Rousseeuw (1990) — Justification seuil silhouette
- Kraskov et al. (2004) — Estimateur KSG pour Transfer Entropy
- Tibshirani et al. (2001) — Gap statistic
- Gregg (2013) — Méthodologie USE, queue depth comme leading indicator
PROMPT

echo ""
echo "✅ Environnement EWAT initialisé dans $(pwd)"
echo ""
echo "Prochaines étapes :"
echo "  1. Copier la formalisation dans docs/formalisation/"
echo "  2. pip install -e '.[dev]'"
echo "  3. Découvrir les endpoints :"
echo "     kubectl get svc -A | grep -iE 'prometheus|otel|grafana'"
echo "  4. Mettre à jour configs/default.yaml avec les endpoints"
echo "  5. Ouvrir dans Claude Code (CLAUDE.md lu automatiquement)"
echo ""
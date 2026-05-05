# EWAT — État courant du projet

_Mis à jour : 2026-05-06_

## Dataset

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | ✅ 300 épisodes collectés | 15 scénarios × 20 rép. (1 exclu : `network_loss_018`, Loki 100% NaN) |
| Phase 2 — build_features | ✅ 300 épisodes | `data/features/v1/` — signal.npz, adjacency.npz, labels.parquet |
| Phase 2b — patch error_rate + latency | ✅ Fait | `data/features/v1p/` — error_rate_http et latency_p99 patchés depuis spans |
| Phase 2c — imputation complète | ✅ Fait | `data/features/v3/` — 15/17 features à 0% NaN, disk_io 17%, log 0.4% |
| Phase 3 — assemble | ✅ Trois datasets produits | `ewat_v1_strat` + `ewat_v2` + `ewat_v3` (stratifié, 209/45/45) |

### Dataset recommandé : `ewat_v3`

| Feature | NaN | État |
|---|---|---|
| cpu_util, ram_util, net_sat, queue_depth | 0% | ✅ |
| latency_p99 | 0% | ✅ (Prometheus pour frontend ; spans P99 pour cart/load-gen ; ffill pour recommendation) |
| error_rate_http | 0% | ✅ (HTTP spans pour frontend/cart/load-gen ; gRPC client pour ad/product-catalog/recommendation) |
| disk_io | **16.7%** | ⚠️ product-catalog toujours NaN (pod sur nœud NotReady) |
| span features (7-12) | 0% | ✅ |
| log features (13-16) | 0.4% | ✅ (résiduel irréductible) |

**NaN total global : ~1.5%** (uniquement disk_io product-catalog + bord d'épisode)

Split stratifié : chaque scénario → ~14 train / 3 val / 3 test. Tous les 15 scénarios (dont 4 drifts) dans le test set.

### Pipeline de reconstruction après nouvelle collecte

```bash
# 1. Build features (inclut error_rate + latency depuis spans dès la construction)
TRANSFORMERS_OFFLINE=1 python -m scripts.build_features \
    --raw-root data/raw --feature-set v1 --workers 4

# 2. Patch (si build_features sur ancienne version sans le fix intégré)
python -m scripts.patch_error_rate  --features-root data/features/v1 --raw-root data/raw
python -m scripts.patch_latency_p99 --features-root data/features/v1 --raw-root data/raw

# 3. Impute + assemble
python -m scripts.impute_features \
    --features-root data/features/v1 --split-json data/datasets/ewat_v1_strat/split.json \
    --output data/features/v3
python -m scripts.assemble_dataset \
    --features-root data/features/v3 --output data/datasets/ewat_v3 --stratified
```

## Infrastructure code

| Module | État |
|---|---|
| `src/telemetry/` | ✅ Complet — collecteurs Prometheus + OTel + extracteurs fichier |
| `src/graph/` | ✅ Complet — builder, adjacency, serialization, validation |
| `src/ewat/drift/` | ✅ Implémenté (Étape 0 : MMD-RFF + look-through) |
| `src/ewat/encoder/` | ✅ Implémenté (Étape 1 : STGCN — spatial GCN + TCN, 13 tests) |
| `src/ewat/encoder/dataset.py` | ✅ Implémenté — EpisodeDataset + collate_fn + StandardScaler |
| `src/ewat/typing/` | ✅ Implémenté (Étape 2 : siamois + clustering + gradient attribution) |
| `src/ewat/ontology/` | ✅ Implémenté (Étape 2b : temporal + TE-KSG + χ²) |
| `src/ewat/precursor/` | ✅ Implémenté (Étape 3 : one-vs-rest LR + AUROC par k, 21 tests) |
| `src/ewat/alerts/` | ❌ À implémenter (sortie) |

### src/ewat/drift/ (Étape 0)

- `mmd.py` — `RFFKernel` : phi(), mmd_squared(), fit_sigma() (médiane des distances)
- `detector.py` — `DriftDetector` : ring buffer, state machine NORMAL/DRIFT/RECALIBRATE, look-through
- `calibration.py` — `calibrate_epsilon()` : percentile 95 du MMD² sur épisodes drift train
- Tests : `tests/unit/drift/` — 34 tests, 34 passent

## Scripts pipeline

| Script | État |
|---|---|
| `record_episode.py` | ✅ Opérationnel (graceful shutdown, checkpointing) |
| `build_features.py` | ✅ Opérationnel (intègre error_rate + latency depuis spans, --workers N) |
| `assemble_dataset.py` | ✅ Opérationnel (--stratified) |
| `impute_features.py` | ✅ Opérationnel (ffill 10 dims + médiane intra-service pour 3 dims) |
| `patch_error_rate.py` | ✅ Opérationnel (patch rétroactif ~5 min pour 300 épisodes) |
| `patch_latency_p99.py` | ✅ Opérationnel (patch rétroactif ~5 min pour 300 épisodes) |
| `validate_dataset.py` | ✅ Opérationnel |
| `chaos_injector.py` | ✅ Opérationnel |

## Cluster

- 300 épisodes enregistrés (15 scénarios × 20 rép.)
- OTel Gateway déployé dans `ewat`, internalTrafficPolicy:Cluster appliqué
- SentenceBERT all-MiniLM-L6-v2 : singleton partagé, cache d'embeddings par instance
- Nœud `observit-cluster1-workers-58w74-mwxb2` NotReady → disk_io product-catalog NaN

### Intervention infra planifiée (pour v4)

Ajouter OTel SDK sur les pods sans instrumentation directe :
- `ad`, `product-catalog`, `recommendation` → actuellement sans spans propres dans Jaeger
- Après déploiement : latency_p99 à 0% NaN pour ces 3 services
- Après collecte + rebuild pipeline → `ewat_v4` sans NaN structurel

## Calibration drift (Étape 0)

- `experiments/drift_separation/calibrate.py` ✅ exécuté sur `ewat_v3`
- **ε_drift = 0.5226** (Youden-optimal, ROC-AUC = 0.60, TPR=0.55, FPR=0.33)
- AUC = 0.60 → séparabilité partielle par MMD² seul (attendu : anomalies causent aussi des shifts)
- H2 complet : simulation look-through temporelle sur test set (après Étape 1)
- `configs/default.yaml` mis à jour : `epsilon_drift: 0.5226`

### src/ewat/encoder/ (Étape 1)

- `stgcn.py` — `STGCNEncoder` : spatial GCN (multi-canal A(t), 2 couches) + TCN causal (2 blocs dilatés) + MLP head
- Adjacency : 3 canaux combinés par poids appris (softmax), normalisation D^{-1/2}AD^{-1/2}
- Residual connections GCN + TCN, LayerNorm, GELU
- Tests : `tests/unit/encoder/` — 13 tests, 13 passent

### src/ewat/typing/ (Étape 2)

- `siamese.py` — `ProjectionHead` (MLP + L2 norm), `SiameseTyper` (encodeur + head), `ContrastiveLoss` (hinge, margin=1.0)
- `pairs.py` — `EpisodePairSampler` : paires positives (même scénario) / négatives (scénarios différents)
- `clustering.py` — `cluster_embeddings()` : AgglomerativeClustering (Ward) + silhouette + gap statistic
- `shap_explainer.py` — `compute_cluster_shap()` : gradient×input attribution par cluster, `write_cluster_fiches()` JSON
- Tests : `tests/unit/typing/` — 32 tests, 32 passent

### Scripts expériences

- `experiments/encoder/train.py` — pré-entraînement reconstruction (L1 sur signal moyen-temporel, MLflow local)
- `experiments/typing/train.py` — fine-tuning siamois + clustering + fiches (MLflow local)

### Commandes pour entraîner

```bash
# 1. Pré-entraîner l'encodeur (~100 epochs, ~30 min CPU)
python -m experiments.encoder.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --output experiments/encoder --epochs 100

# 2. Fine-tuner le typage siamois + clustering + fiches (~50 epochs, ~15 min CPU)
python -m experiments.typing.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --encoder-checkpoint experiments/encoder/checkpoints/best_encoder.pt \
    --output experiments/typing --epochs 50
```

### Résultat dry-run (2 epochs encodeur + 2 epochs siamois)

- Silhouette train=0.793, val=0.767, **test=0.698** → H1 ✓ PASS (seuil 0.3)
- K optimal = 2 (sur 2 epochs — nombre réel de types sera déterminé sur run complet)

### src/ewat/ontology/ (Étape 2b)

- `graph.py` — `OntologyRelation` + `OntologyGraph` : structure O = (C, R), save/load JSON
- `temporal.py` — `compute_temporal_relations()` : transitions C_i →^{Δt,σ} C_j sur épisodes consécutifs
- `causal.py` — `compute_causal_relations()` : TE-KSG (estimateur Kraskov 2004) + test permutation
- `cooccurrence.py` — `compute_cooccurrence_relations()` : χ² avec correction Yates par scénario
- Tests : `tests/unit/ontology/` — 41 tests, 41 passent

### Résultat build ontologie (dry-run, 20 permutations)

- 22 relations temporelles (dont 10 self-loops C_i→C_i, 12 transitions cross-cluster)
- 2 relations causales : C6→C8 (TE=0.195, p<0.001), C2→C8 (TE=0.155, p<0.001)
- 0 relations co-occurrence (p>0.05 pour tous les pairs au seuil 5%)

### Scripts expériences

- `experiments/encoder/train.py` — pré-entraînement reconstruction (L1, MLflow local)
- `experiments/typing/train.py` — fine-tuning siamois + clustering + SHAP + artifacts
- `experiments/ontology/build.py` — build O = (C, R) depuis cluster_artifacts/

### Commandes pour entraîner

```bash
# 1. Pré-entraîner l'encodeur (~100 epochs, ~30 min CPU)
python -m experiments.encoder.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --output experiments/encoder --epochs 100

# 2. Fine-tuner le typage siamois + clustering + fiches (~50 epochs, ~15 min CPU)
python -m experiments.typing.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --encoder-checkpoint experiments/encoder/checkpoints/best_encoder.pt \
    --output experiments/typing --epochs 50

# 2b. Build ontologie TE-KSG (100 permutations, ~15 min CPU)
python -m experiments.ontology.build \
    --typing-dir experiments/typing --features-root data/features/v3 \
    --output experiments/ontology --n-permutations 100
```

### Résultats run complet (encodeur 100 epochs + siamois 50 epochs)

- Silhouette train=0.577, val=0.601, **test=0.615** → H1 ✓ PASS (seuil 0.3)
- K optimal = 10

### src/ewat/precursor/ (Étape 3)

- `dataset.py` — `PrecursorDataset` : fenêtres pré-injection de longueur k (gauche-paddées si warmup < k)
- `model.py` — `PrecursorClassifier` : one-vs-rest LogisticRegression par type + AUROC, `find_optimal_k()`
- Tests : `tests/unit/precursor/` — 21 tests, 21 passent

### Résultats précurseurs (k ∈ {2,4,6,8,10,12} steps = {1—6 min}, test set)

| Type | AUROC(k*) | k* | Note |
|---|---|---|---|
| C6 | **1.000** | 2 | Signal pré-injection parfaitement distinctif |
| C3 | **0.706** | 12 | Meilleur avec plus de contexte |
| C2 | **0.611** | 2 | Signal précoce suffisant |
| C8 | **0.530** | 2 | Légèrement au-dessus du hasard |
| C0,1,4,5 | < 0.5 | — | Non prédictibles depuis la fenêtre pré-injection |
| C7,C9 | NaN | — | Pas assez d'exemples test |

**H3 ✓ PASS** : 4/10 types AUROC > 0.5 (baseline = 0.5)

### Commandes complètes

```bash
# 3. Précurseurs typés
python -m experiments.precursor.train \
    --typing-dir experiments/typing --features-root data/features/v3 \
    --output experiments/precursor --k-values 2 4 6 8 10 12
```

## Prochaine priorité

**Implémenter `src/ewat/alerts/`** (sortie du pipeline EWAT) :
`Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})` — assemblage de la sortie finale.

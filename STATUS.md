# EWAT — État courant du projet

_Mis à jour : 2026-05-06_

> Résultats détaillés et interprétation scientifique → `docs/results.md`

---

## Hypothèses — bilan final

| Hypothèse | Résultat | Valeur clé |
|---|---|---|
| **H1** — Structurabilité des embeddings | ✅ PASS | Silhouette test = 0.615 (seuil 0.3) |
| **H2** — Séparabilité drift par look-through MMD² | ❌ FAIL | FPR_lt=0.67, p=0.27 — épisodes trop courts pour la confirmation temporelle |
| **H3** — Prédictibilité des précurseurs | ✅ PASS | 4/10 types AUROC > 0.5 (C6=1.000, C3=0.706) |

---

## Pipeline EWAT — complet

```
S(t) ∈ ℝ^{N×17}
    ↓ Étape 0 : DriftDetector (MMD-RFF, ε=0.5226, look-through)
    ↓ Étape 1 : STGCNEncoder → z_e ∈ ℝ^64
    ↓ Étape 2 : SiameseTyper → cluster C_i (K=10)
    ↓ Étape 2b : OntologyGraph (temporal + TE-KSG + χ²)
    ↓ Étape 3 : PrecursorClassifier → p̂_i(t), k*_i
    ↓ Sortie : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})
```

295 tests unitaires, lint propre. Toutes les étapes implémentées et évaluées sur ewat_v3.

---

## Dataset

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | ✅ 300 épisodes | 15 scénarios × 20 rép. (1 exclu : `network_loss_018`) |
| Phase 2 — build_features | ✅ 299 épisodes | `data/features/v3/` — 15/17 features à 0% NaN |
| Phase 3 — assemble | ✅ | `ewat_v3` — split stratifié 209/45/45 |

**NaN restant** : disk_io 16.7% (product-catalog, nœud NotReady) — résolu en ewat_v4.

---

## Infrastructure code

| Module | État | Tests |
|---|---|---|
| `src/ewat/drift/` | ✅ MMD-RFF + look-through | 34 |
| `src/ewat/encoder/` | ✅ STGCN + EpisodeDataset | 13 |
| `src/ewat/typing/` | ✅ Siamois + clustering + SHAP | 32 |
| `src/ewat/ontology/` | ✅ Temporal + TE-KSG + χ² | 41 |
| `src/ewat/precursor/` | ✅ One-vs-rest LR + AUROC/k | 21 |
| `src/ewat/alerts/` | ✅ Alert + AlertAssembler (+ scaler) | 27 |

---

## Expériences — résultats

### Étape 0 — Calibration drift

- ε_drift = **0.5226** (Youden-optimal, AUC=0.60 sur train)
- `configs/default.yaml` mis à jour

### H2 — Look-through temporel (test set)

- TPR drift : lt=0.42 vs baseline=0.67 — ❌ look-through moins bon que le seuil simple
- FPR anomalie : lt=0.67 vs baseline=0.73 — réduction non significative (p=0.27)
- **Cause** : épisodes ~21 steps trop courts pour la confirmation temporelle (post=3–6 steps)
- Résultat négatif exploitable : séparabilité drift/anomalie requiert les embeddings STGCN, pas le MMD² brut

### Étape 1+2 — Encodeur + Typage (47 epochs + 50 epochs)

- Silhouette train=0.577 / val=0.601 / **test=0.615** → **H1 ✓ PASS**
- K optimal = **10** clusters

### Étape 2b — Ontologie (dry-run, 20 permutations)

- 22 relations temporelles, **2 causales** (C6→C8, C2→C8), 0 co-occurrence
- Chaîne causale : C6 (AUROC=1.000) → C2 → C8

### Étape 3 — Précurseurs (k ∈ {2,4,6,8,10,12} steps)

| Type | AUROC(k*) | k* |
|---|---|---|
| C6 | **1.000** | 2 steps (1 min) |
| C3 | **0.706** | 12 steps (6 min) |
| C2 | **0.611** | 2 steps |
| C8 | **0.530** | 2 steps |
| C0,1,4,5 | < 0.5 | — |

**H3 ✓ PASS** — 4/10 types prédictibles

### Simulation en ligne — AlertAssembler (test set, 45 épisodes)

| Seuil | Détection | FA drift | Lead |
|---|---|---|---|
| 0.3 | **90.9%** | 66.7% | 4.1 min |
| 0.4 | 81.8% | 58.3% | 3.9 min |
| **0.5** | **72.7%** | **58.3%** | **3.6 min** |
| 0.7 | 51.5% | 16.7% | 2.2 min |

Cluster correct ≈ 0% en online (voir `docs/results.md` §7 pour l'explication).

### Ablation modalités + features (test set, masquage à l'inférence)

**Par modalité** : M (métriques) porte l'essentiel. T et L seuls → silhouette négative.

| Condition | Silhouette | Δ | Sig. |
|---|---|---|---|
| full | 0.519 | — | — |
| M+L | 0.475 | −0.044 | ✗ |
| M_only | 0.442 | −0.077 | ✓ |
| T_only | −0.115 | −0.634 | ✓ |

**Features critiques** (leave-one-out, p<0.05) : `net_sat` (Δ−0.169), `disk_io` (Δ−0.143), `lexical_entropy` (Δ−0.142), `cpu_util` (Δ−0.104)

**Paires redondantes** : `latency_p99`↔`span_dur_median` (ρ=0.936), `error_rate_http`↔`abnormal_span_rate` (ρ=0.927)

---

## Commandes — pipeline complet

```bash
# Encodeur (100 epochs, ~30 min CPU)
python -m experiments.encoder.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --output experiments/encoder --epochs 100

# Typage siamois (50 epochs, ~15 min CPU)
python -m experiments.typing.train \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --encoder-checkpoint experiments/encoder/checkpoints/best_encoder.pt \
    --output experiments/typing --epochs 50

# Ontologie TE-KSG (100 permutations, ~15 min CPU)
python -m experiments.ontology.build \
    --typing-dir experiments/typing --features-root data/features/v3 \
    --output experiments/ontology --n-permutations 100

# Précurseurs (k ∈ {2,4,6,8,10,12})
python -m experiments.precursor.train \
    --typing-dir experiments/typing --features-root data/features/v3 \
    --output experiments/precursor --k-values 2 4 6 8 10 12

# Évaluation alertes (test set, ~2 min)
python -m experiments.alerts.eval \
    --typing-dir experiments/typing --encoder-dir experiments/encoder \
    --precursor-dir experiments/precursor --features-root data/features/v3 \
    --output experiments/alerts

# H2 look-through (test set, ~1 min)
python -m experiments.h2_lookthrough.eval \
    --features-root data/features/v3 --typing-dir experiments/typing \
    --output experiments/h2_lookthrough

# Ablation modalités + features (~5 min)
python -m experiments.ablation.run \
    --typing-dir experiments/typing --encoder-dir experiments/encoder \
    --features-root data/features/v3 --output experiments/ablation
```

---

## Cluster

- 299 épisodes (15 scénarios × ~20 rép.)
- Nœud `observit-cluster1-workers-58w74-mwxb2` NotReady → disk_io product-catalog NaN

---

## Prochaines pistes

### Court terme (sans nouvelle collecte)

1. **Inhiber les FA drift dans l'assembleur** : si `DriftDetector.flag==True` → suspendre les alertes précurseurs pendant la transition
2. **H2 bis avec embeddings** : remplacer MMD² brut par MMD²(z_ref, z_cur) dans l'espace STGCN
3. **Réduction feature space** : supprimer les 2 paires redondantes + features non significatives (17→~10), réentraîner

### Moyen terme

4. **ewat_v4** : OTel SDK sur `ad`/`product-catalog`/`recommendation` → disk_io 0% NaN, spans complets pour 3 services (nécessite cluster-admin pour les sidecars)
5. **Ablation rigoureuse** : réentraînement complet par condition de masquage (~5h CPU)

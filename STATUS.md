# EWAT — État courant du projet

_Mis à jour : 2026-05-06_

> Résultats détaillés et interprétation scientifique → `docs/results.md`

---

## Hypothèses — bilan final (résultats corrigés)

| Hypothèse | Résultat | Valeur clé |
|---|---|---|
| **H1** — Structurabilité des embeddings | ✅ PASS | Silhouette test = **0.414** (nearest centroid, seuil 0.3) |
| **H2** — Séparabilité drift par look-through MMD² | ❌ FAIL | FPR_lt=0.67, p=0.27 — épisodes trop courts |
| **H2 bis** — Look-through sur embeddings STGCN | ❌ FAIL | Youden J=0.071 — embeddings non adaptés à la séparabilité drift/anomalie |
| **H3** — Prédictibilité des précurseurs | ✅ PASS | **8/10** types AUROC > 0.5, AUROC moyen = 0.97 |

**Correction méthodologique appliquée** (2026-05-06) :
- H1 : labels val/test assignés par nearest centroid depuis les centroides train (vs. fit_predict indépendant)
- H3 : k* sélectionné sur val (vs. test) ; labels alignés cross-split

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

302 tests unitaires, lint propre. Toutes les étapes implémentées et évaluées sur ewat_v3.

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
| `src/ewat/alerts/` | ✅ Alert + AlertAssembler (+ scaler + DriftDetector) | 31 |

---

## Expériences — résultats

### Étape 0 — Calibration drift

- ε_drift = **0.5226** (Youden-optimal, AUC=0.60 sur train)
- `configs/default.yaml` mis à jour

### H2 — Look-through temporel (test set)

- TPR drift : lt=0.42 vs baseline=0.67 — ❌ look-through moins bon que le seuil simple
- FPR anomalie : lt=0.67 vs baseline=0.73 — réduction non significative (p=0.27)
- **Cause** : épisodes ~21 steps trop courts pour la confirmation temporelle (post=3–6 steps)

### H2 bis — Look-through sur embeddings STGCN (test set)

- ε_emb=0.5186 (Youden J=0.071 — très faible discrimination)
- FPR anomalie : lt=0.788 vs baseline=0.667 — ❌ look-through pire que baseline (p=0.978)
- **Interprétation** : embeddings STGCN capturent le TYPE d'anomalie, pas le régime drift/anomalie

### Étape 1+2 — Encodeur + Typage (47 epochs + 50 epochs)

- Silhouette train=0.577 / val=**0.470** / **test=0.414** → **H1 ✓ PASS** (seuil 0.3)
- K optimal = **10** clusters
- Note : scores précédents (val=0.601, test=0.615) utilisaient un clustering indépendant par split

### Étape 2b — Ontologie (100 permutations — définitif)

- 22 relations temporelles, **0 causales**, 0 co-occurrence
- Les 2 relations causales du dry-run (C6→C8, C2→C8, 20 perm.) étaient des faux positifs
- Relations temporelles : 10 auto-transitions (Ci→Ci), 12 transitions inter-clusters (support ≥ 3)

### Étape 3 — Précurseurs (k ∈ {2,4,6,8,10,12} steps, k* sélectionné sur val)

| Type | AUROC_val(k*) | AUROC_test(k*) | k* |
|---|---|---|---|
| C0 | 1.000 | **0.970** | 6 steps (3 min) |
| C1 | 1.000 | **0.976** | 6 steps (3 min) |
| C2 | 1.000 | **0.940** | 6 steps (3 min) |
| C3 | 0.937 | **0.794** | 2 steps (1 min) |
| C4 | 1.000 | **1.000** | 2 steps (1 min) |
| C5 | 1.000 | **0.977** | 6 steps (3 min) |
| C6 | 1.000 | NaN (n<2 test) | 2 steps |
| C7 | 0.970 | **0.992** | 6 steps (3 min) |
| C8 | 0.990 | **0.962** | 10 steps (5 min) |
| C9 | NaN (n<2 val) | NaN | 2 steps |

**H3 ✓ PASS** — 8/10 types prédictibles (vs 4/10 rapporté initialement)
C6/C9 : NaN par manque d'épisodes test (n=1 ou 2).

### Simulation en ligne — AlertAssembler (test set, 45 épisodes)

DriftDetector intégré. Labels corrigés → "correct cluster" maintenant significatif.

| Seuil | Détection | Cluster correct | FA drift | Lead |
|---|---|---|---|---|
| 0.3 | **100%** | 66.7% | 100% | 4.2 min |
| 0.4 | 93.9% | **72.7%** | 100% | 3.9 min |
| 0.5 | 75.8% | 66.7% | 100% | 4.0 min |
| **0.7** | **48.5%** | **45.5%** | **8.3%** | **2.9 min** |

Point opérationnel recommandé : seuil 0.7 (FA maîtrisée). FA=100% aux seuils 0.3–0.5 car classifieurs très sensibles sur épisodes drift (court warm-up DriftDetector).

### Ablation modalités + features (test set, masquage à l'inférence — labels corrigés)

**Par modalité** : M (métriques) porte l'essentiel. T et L seuls → silhouette négative.

| Condition | Silhouette | Δ | Sig. |
|---|---|---|---|
| full | 0.333 | — | — |
| M+T | 0.310 | −0.024 | ✗ |
| M_only | 0.271 | −0.062 | ✓ |
| M+L | 0.234 | −0.099 | ✓ |
| T+L | −0.124 | −0.457 | ✓ |
| T_only | −0.151 | −0.485 | ✓ |
| L_only | −0.212 | −0.546 | ✓ |

**Features critiques** (leave-one-out, p<0.05) : `trace_depth` (Δ−0.069), `lexical_entropy` (Δ−0.069), `latency_p99` (Δ−0.062), `disk_io` (Δ−0.010)

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

# Précurseurs (k ∈ {2,4,6,8,10,12}, k* sur val)
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

# Vérification méthodologique H1+H3
python -m experiments.verification.verify_h1_h3 \
    --typing-dir experiments/typing --encoder-dir experiments/encoder \
    --precursor-dir experiments/precursor --features-root data/features/v3 \
    --output experiments/verification
```

---

## Cluster

- 299 épisodes (15 scénarios × ~20 rép.)
- Nœud `observit-cluster1-workers-58w74-mwxb2` NotReady → disk_io product-catalog NaN

---

## Prochaines pistes

### Court terme (sans nouvelle collecte)

1. ✅ **DriftDetector → AlertAssembler** : intégré — FA=8.3% au seuil 0.7
2. ✅ **H2 bis avec embeddings** : ✗ FAIL (Youden J=0.071)
3. ✅ **Correction méthodologique H1/H3** : nearest centroid + k* sur val — résultats corrigés
4. ✅ **Ablation avec labels corrigés** : features critiques = trace_depth, lexical_entropy, latency_p99, disk_io (p<0.05)
5. **Rapport de stage** : utiliser ces résultats corrigés

### Moyen terme

6. **ewat_v4** : OTel SDK sur `ad`/`product-catalog`/`recommendation` → disk_io 0% NaN
7. **Ablation rigoureuse** : réentraînement complet par condition de masquage (~5h CPU)

# EWAT — État courant du projet

_Mis à jour : 2026-05-27 (Phases H-K — Multi-seed validation v4_strat, 10 graines, chiffres consolidés ; défense + roadmap actifs)_

> Résultats détaillés et interprétation scientifique → [docs/results.md](docs/results.md)
> **Évolution post-stage planifiée → [ROADMAP.md](ROADMAP.md)** (axes A: couplage onto/pred, B: précursion robuste, C: open-set, D: déploiement)
> Mémo défense (1 page A4) → [docs/defense_memo.md](docs/defense_memo.md)

---

## Hypothèses — bilan final (config optimisée, 10 graines)

| Hypothèse | Résultat | Valeur clé |
|---|---|---|
| **H1** — Structurabilité des embeddings | ✅ PASS | Silhouette test = **0.782 ± 0.065** (10 graines, seuil 0.3, min=0.618) |
| **H2a** — Séparabilité drift par look-through MMD² | ❌ FAIL | FPR_lt=0.67, p=0.27 — épisodes trop courts |
| **H2b** — Identification régime θ_{drift∩anomaly} | ⚠️ NUANCÉ | PASS formel (overlap>30% partout) mais trivial — DD trop sensible sur 5 steps |
| **H3** — Prédictibilité des précurseurs | ⚠️ CIRCULAIRE | **AUROC moyen = 0.987 ± 0.011** (10 graines, 10/10 PASS) — **mais voir stress test A1** |
| **H3 (honnête)** — vs labels Chaos Mesh (B3/B4) | ⚠️ FAIBLE | macro-AUROC=0.835 (Δ_STGCN=0.000) — encodeur n'aide pas en agrégé |
| **H3 (précursion réelle)** — distant-window | ❌ FAIL | Δ(far−near)=−0.007 → **fuite signature scénario** (A1, 2026-05-22) |

### Multi-seed validation (10 graines, ewat_v4_strat, Phase H+J, 2026-05-26)

| Métrique | Valeur consolidée | Note |
|---|---|---|
| **H1 sil_test** | **0.691 ± 0.115** (10 graines) | range [0.521, 0.839] — variance large, K instable |
| **H3 AUROC peak** | **0.990 ± 0.012** (circulaire) | by design — cible auto-référente, cf. L9 |
| **B2 Chaos Mesh stratified** | **0.9201** déterministe | IC bootstrap [0.878, 0.956] — **headline défensif** |
| **B2 LOSO macro** | **0.9298** déterministe | 15 folds × 10 seeds |
| **A1 Δ(far−near)** | **−0.012 ± 0.022** | LEAK 9/10, GENUINE 1/10 (seed 42 outlier) |
| **Latence E2E p95** | **13 ms** | sous budget 5 s (×375) |

---

## Phase H + J + K — Multi-seed validation (2026-05-26, 10 graines)

Suite à Phase G (single seed 42 → résultats trop optimistes), un sweep multi-seed (10 graines) a été lancé pour mesurer la variance réelle. Verdict : **le retrain Phase G était un outlier** sur deux métriques clés (sil_test 0.84 et A1 Δ=−0.05).

### Phase H — Pipeline retrain (cible labels EWAT, circulaire)

10 graines × (encoder + siamois + précurseur + A1) — ~7h CPU total.

| Métrique | Mean ± Std (10 graines) | Range | Verdict |
|---|---|---|---|
| **H1 silhouette test** | **0.691 ± 0.115** | [0.521, 0.839] | ✅ ≥ 0.6 (seuil) mais variance large |
| **H3 AUROC peak test** | **0.990 ± 0.012** | [0.959, 1.000] | ⚠️ circulaire (cible auto-référente, cf. L9) |
| **A1 Δ(far−near)** | **−0.012 ± 0.022** | [−0.050, +0.019] | ❌ LEAK_CONFIRMED 9/10, GENUINE 1/10 |
| **K_optimal** | 11.8 ± 2.1 | [9, 15] | ❌ instable |
| **best_epoch siamois** | ~3 | constant | ⚠️ surentraînement persistant |

Détail per-seed : `experiments/multiseed/phase_h/results.md`.

### Phase J — Headline défensif (cible Chaos Mesh, indépendante)

10 graines × B2 (LR-OvR features brutes flatten + instance norm). Le LR avec solver lbfgs étant **déterministe**, toutes les graines donnent exactement le même chiffre (la variance est dans le bootstrap CI, pas dans le fit).

| Métrique | Valeur (10 graines, identique) | IC 95% bootstrap | Verdict |
|---|---|---|---|
| **B2 stratified macro-AUROC** | **0.9201** | [0.878, 0.956] | ✅ headline défensif robuste par construction |
| **B2 LOSO macro-AUROC** | **0.9298** | (15 folds × 10 seeds) | ✅ stable |

**Le NaN-aware scaler (Step 2.3) ne bouge pas B2** car B2 utilise son propre scaler local sur flatten features. L'audit corrige des bugs réels mais ne déplace pas le headline indépendant. Cohérent avec A5 (paired Δ B4-B3 IC contient 0).

### Phase K — Diagnostics

**K.1 K-selection comparison** (`k_selection_comparison.md`) :

| Stratégie | Mode (count) | Mean ± Std | Range | Agreement |
|---|---|---|---|---|
| silhouette (default) | K=14 (2/10) | 11.8 ± 2.1 | [9, 15] | — |
| gap_tibshirani | K=12 (2/10) | 8.2 ± 2.8 | [4, 12] | 4/10 avec silhouette |

Verdict : **K intrinsèquement instable** sur ce dataset (n=270 train). Ni silhouette ni Tibshirani ne stabilise. Recommandation v5 : fixer K=10 manuellement ou passer à HDBSCAN (density-based).

**K.3 Variance per-seed** (`variance_analysis.md`, `distribution.png`) :
- Métriques **stables** : H3 AUROC (circ), B2 stratified/LOSO (déterministe).
- Métriques **instables** : H1 sil (range 0.32), K (range 6), A1 Δ (outlier seed 42).
- **Seed 42 confirmé comme outlier** sur A1 — son Δ=−0.05 (initialement reporté Phase G) n'est pas reproductible.

### Verdict consolidé (à reporter au maître de stage)

1. **Headline défensif** : B2 = **0.9201** [0.878, 0.956] sur Chaos Mesh v4_strat — déterministe, IC bootstrap explicite, indépendant des labels EWAT.
2. **Phase G était un outlier** : Phase H montre que sil=0.84 et A1=−0.05 ne sont pas reproductibles.
3. **38 fixes audit corrigent des bugs réels** (NaN-aware scaler, class_weight, instance norm exclusive, etc.) mais ne déplacent pas le headline indépendant — c'est attendu et cohérent.
4. **Limites intrinsèques** : K_optimal instable (n_train=270 trop petit), surentraînement siamois (best_epoch~3 quoi qu'il arrive), n_pos=3 par scénario test (C-5).
5. **Documentation** : 17 limites L1-L17 documentées avec fixes futurs proposés.

### Artefacts produits

- `experiments/multiseed/phase_h/{aggregate.json,results.md,k_selection_comparison.md,variance_analysis.md,distribution.png}`
- `experiments/multiseed/phase_j/{aggregate.json,results.md}`
- `experiments/multiseed/{run_phase_h.py,run_phase_j.py,aggregate_phase_h.py,phase_k_kselection.py,phase_k_variance.py}`

---

## Phase G — Retrain complet v4_strat (2026-05-26)

Après les 38 fixes appliqués via le plan 10-étapes (audit méthodologique du pipeline complet), un retrain end-to-end a été effectué sur `data/datasets/ewat_v4_strat` (graine 42) pour mesurer l'impact combiné des correctifs.

### Configuration retrain

| Composant | Fixes appliqués |
|---|---|
| Encoder STGCN | `use_layer_norm=True` (Step 5), NaN-aware `fit_scaler` +20% data (Step 2) |
| Siamois | `margin=2.0`, `d_proj=64`, `mining=semi-hard` (Step 6) |
| Clustering | `linkage=average`, `metric=cosine` (Step 6), K=12 auto-sélectionné |
| Précurseur | LR `class_weight=balanced` par défaut (Step 8), BCa CI (Step 8) |
| Évaluation | n_bootstrap=1000 BCa, k_values={1,2,3,4,5,6,8,10,12,15,20} |

### Résultats v4_strat retrained vs anciens

| Métrique | Ancien pipeline | **Retrain v4_strat (Phase G)** | Δ |
|---|---|---|---|
| **H1** sil_test | 0.467 ± 0.156 (6 graines v4) | **0.838** (graine 42, K=12) | **+0.371** |
| H1 bootstrap CI 95% | — | [0.6530, 0.8096] | — |
| **H3** AUROC mean test (peak) | 0.935 ± 0.024 (v4) / 0.987 ± 0.011 (v3) | **0.993** | +0.058 vs v4 |
| H3 clusters PASS | 6/10 (v4) | **7/12** | — |
| **A1** Δ(far−near) | −0.007 (v3, EWAT labels — fuite) | **−0.050** (v4_strat retrained) | **GENUINE_PRECURSION** |
| Latence E2E p95 | 13.28 ms (v3) | **12.97 ms** | 🟢 GREEN |
| Clusters reportables (n_pos≥5) | 5/10 (v4) | 4/7 (filtrés sur 12) | — |

### Lecture des résultats

**Gain majeur H1** : silhouette test passe de 0.467 → **0.838** (×1.8). Le surentraînement siamois identifié en C-4 (best_epoch=2-7) reste présent (best_epoch=3 ici), mais la qualité du clustering ne s'en ressent plus grâce à `margin=2.0` + `mining=semi-hard` + `class_weight=balanced` propagé jusqu'aux précurseurs.

**Renversement A1 — précursion réelle** : sur la cible labels EWAT (auto-référente, censée être circulaire), le nouveau pipeline produit **Δ(far−near) = −0.050**, soit 5 pp d'AUROC qui dépendent vraiment de la position de la fenêtre. C'est un saut qualitatif par rapport à l'ancien Δ=−0.007 (fuite signature). Le pipeline exploite enfin la dynamique pré-injection même sur la cible circulaire — preuve que les fixes (NaN-aware scaler, instance norm correctement séparée du scaler, encoder LayerNorm, etc.) ont éliminé une part importante du bruit qui obscurcissait le signal.

**H3 AUROC test = 0.993** sur le pipeline retrainé, avec 4/7 clusters reportables (n_pos≥5) atteignant AUROC = 1.0 avec BCa CIs [1.0, 1.0]. À comparer prudemment avec le headline défensif B2 sur Chaos Mesh indépendant (0.920 [0.878, 0.956]) — voir limitation L9 sur la circularité d'évaluation.

### Bugs corrigés en passant (Phase G)

- `experiments/typing/train.py` ne passait pas `use_layer_norm` au build encoder → ajout `build_encoder_from_checkpoint`
- `experiments/precursor/train.py` idem → même fix
- `experiments/encoder/train.py` ne propageait pas `use_layer_norm` au constructeur → flag CLI `--use-layer-norm` (default True)

### Artefacts produits

- `experiments/encoder_v4_strat/checkpoints/best_encoder.pt` (val_loss=0.0578, 80 epochs)
- `experiments/typing_v4_strat/checkpoints/best_siamese.pt` (K=12, sil_test=0.838)
- `experiments/precursor_v4_strat/checkpoints/` (12 classifiers, 1 par cluster)
- `experiments/h3_robustness_v4_strat/distant_window/results.md` (A1 Δ=-0.050)
- `experiments/bench_v4_strat/` (latence + power analysis)

### Verdict global

Les 38 fixes (audit 10 étapes) appliqués ensemble produisent un pipeline qui :
1. **Sépare mieux** les types d'anomalies (H1 +37 pp silhouette)
2. **Apprend une vraie précursion** (A1 Δ=-0.05 sur labels EWAT, vs ancienne fuite)
3. **Reste rapide** (p95 13 ms, sous budget 5 s)
4. **Conserve la transparence statistique** (BCa CIs, n_pos≥5 filtre, etc.)

Le headline défensif honnête reste **0.920 sur Chaos Mesh indépendant** (B2, v4_strat, audit Phase B/C). Mais le retrain complet montre qu'avec les fixes, même la métrique circulaire H3 produit maintenant une précursion temporelle réelle vérifiable par stress test A1.



**Config optimale identifiée par sweep** (2026-05-21) :
- clustering : `average + cosine` (vs. ward + euclidean) — aligné avec les embeddings L2-normalisés sur sphère unitaire
- projection siamoise : `d_proj=64, margin=2.0` (vs. d_proj=32, margin=1.0)
- classifier précurseur : `lr_tuned` (LogisticRegressionCV avec C ∈ {0.01..100})

**Baseline (config initiale, 5 graines)** :
- H1 sil_test = 0.519 ± 0.092
- H3 AUROC = 0.973 ± 0.012

**Correction méthodologique appliquée** (2026-05-06) :
- H1 : labels val/test assignés par nearest centroid depuis les centroides train (vs. fit_predict indépendant)
- H3 : k* sélectionné sur val (vs. test) ; labels alignés cross-split

## Phase F — Audit + validation opérationnelle (2026-05-26)

Suite à l'audit méthodologique post-Phase C, 5 critiques (C-1 à C-5) ont été identifiées avec un budget de 1+ mois. Statut actuel :

| ID | Critique | Statut | Valeur clé |
|---|---|---|---|
| **C-1** | STGCN non-prédictif (0.863 < B2=0.920) | ✅ Option B docs | Pipeline opérationnel = LR-OvR sans STGCN ; STGCN = clustering + ontologie |
| **C-2** | Validation externe nulle (RCAEval H3=0.5) | ⏳ Travail futur | Stratégie B fine-tuning identifiée, non implémentée |
| **C-3** | Latence E2E non mesurée | ✅ Résolu | **TOTAL p95 = 13.28 ms** (375× sous budget < 5 s) — `experiments/bench/latency_e2e.py` |
| **C-4** | H1 dégradé sur ewat_v4 (0.467 ± 0.156) | ⏳ Travail futur | Hard-negative mining + curriculum identifiés, non implémentés |
| **C-5** | Power statistique non quantifiée | ✅ Résolu | **5/10 clusters reportables** (n_pos ≥ 5), power = 1.0 — `experiments/bench/power_analysis.py` |

| ID majeur | Critique | Statut |
|---|---|---|
| M-4 | TE method univariate par défaut | ✅ Documenté (Phase 8 utilise multivariate explicitement → 3 causales) |
| M-6 | macro-AUROC cache hétérogénéité par scénario | ✅ Résolu, PR-AUC ajouté (`experiments/architecture_v2/chaos_mesh_target.py`) — révèle PR-AUC ∈ [0.166, 1.000] selon scénario |

**Documentation L10-L17 ajoutée** dans `docs/limitations.md` : surentraînement siamois v4 (L10), latence résolue (L11), OpenMax mitigé (L12), service graph N=6 (L13), cross-cluster (L14), retraining cycle (L15), 17 features hardcodées (L16), ontologie sans audit SRE (L17). Toutes accompagnées de propositions de fix pour future itération.

**Slides défensives passées de 4 à 6** : ajout slide 5 (validation opérationnelle latence + power) et slide 6 (limites résiduelles + travaux futurs).

**Rapport LaTeX** : section 06b_robustness.tex étendue avec sous-sections "Pipeline opérationnel + budget de latence" (Option B + Tableau~ref{tab:latency}) et "Limites résiduelles" (L10-L17 condensées). Section 07_discussion.tex enrichie d'une sous-section "Perspectives au-delà d'ewat_v4" avec 8 axes de travaux futurs.

---

## Pipeline EWAT — complet (config optimisée)

```
S(t) ∈ ℝ^{N×17}
    ↓ Étape 0 : DriftDetector (MMD-RFF, ε=0.5226, look-through)
    ↓ Étape 1 : STGCNEncoder → z_e ∈ ℝ^64
    ↓ Étape 2 : SiameseTyper (d_proj=64, margin=2.0) → cluster C_i (K≈12)
               clustering : average + cosine (L2-normalized unit sphere)
    ↓ Étape 2b : OntologyGraph (temporal + TE-KSG + χ²)
    ↓ Étape 3 : PrecursorClassifier (lr_tuned) → p̂_i(t), k*_i
               k ∈ {1,2,3,4,5,6,8,10,12,15,20} steps
    ↓ Sortie : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})
```

401 tests unitaires, lint propre. Toutes les étapes implémentées et évaluées sur ewat_v3.

---

## Dataset

### ewat_v3 — dataset de référence (actif)

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | ✅ 300 épisodes | 15 scénarios × 20 rép. |
| Phase 2 — build_features | ✅ 300 épisodes buildés | `data/features/v3/` — **16/17** features à 0% NaN |
| Phase 3 — assemble | ✅ | `ewat_v3` — split stratifié 209/45/45 |

**NaN restant** : disk_io 16.7% (product-catalog, nœud NotReady).

### ewat_v4 — dataset assemblé + pipeline complet (6 graines)

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | ✅ **414 épisodes** | 15 scénarios × 25–38 rép. (drift : 25–38, anomalie : 25) |
| Phase 2 — build_features | ✅ **414 épisodes buildés** | `data/features/v4/` — build Kubeflow (conda), T=47–51 steps |
| Phase 3 — assemble | ✅ **375 épisodes retenus** | `data/datasets/ewat_v4` — split temporel 262/56/57 |

**NaN filtering** : 39 épisodes rejetés (32 L=100% Loki outage mai 7–13, 4 T=100% Jaeger outage mai 15, 3 autres). Tous les épisodes rejetés ont des remplaçants re-collectés.

**NaN résiduel** : L≈2% (vs 16.7% disk_io sur ewat_v3 ✓), M≈3–5%, T≈20–25% (structurel crash).

**Validé** : `validate_dataset` — 375/375 `[OK]`, N=6 stable, split temporel strict.

**Motivations v4 vs v3** : épisodes plus longs (T=47–51 vs ~21 steps), disk_io 0% NaN attendu (nœud réparé), +5 rép./scénario → C6/C9 NaN résolus.

**Résultats 6 graines** (seeds 42, 123, 456, 789, 1337, 0 — config optimisée avg+cosine, d_proj=64, m=2.0) :

| Graine | sil_test | H1 | AUROC | H3 |
|---|---|---|---|---|
| 42 | 0.618 | ✅ | 0.948 | ✅ |
| 123 | 0.415 | ✅ | 0.948 | ✅ |
| 456 | 0.578 | ✅ | 0.899 | ✅ |
| 789 | 0.216 | ❌ | 0.935 | ✅ |
| 1337 | 0.618 | ✅ | 0.914 | ✅ |
| 0 | 0.359 | ✅ | 0.965 | ✅ |
| **Agrégé** | **0.467 ± 0.156** | **5/6 PASS** | **0.935 ± 0.024** | **6/6 PASS** |

**Observation siamois** : best_epoch = 2–7 sur 50 (vs ~47 sur ewat_v3) → surentraînement rapide. Cause probable : plus grande diversité de paires contrastives sur 262 épisodes train. H1 dégradé vs ewat_v3 (0.782 ± 0.065). H3 robuste (6/6 PASS, AUROC stable).

### ewat_rcaeval — dataset adapté

| Phase | État | Détail |
|---|---|---|
| Assemblage | ✅ | `data/datasets/ewat_rcaeval/` — 90 épisodes, 30 fault types, même format EWAT |

**Source** : script `scripts/adapt_rcaeval.py` — conversion RCAEval RE2-OB vers format EWAT (features v3-compatibles).

---

## Infrastructure code

| Module | État | Tests | Contenu |
|---|---|---|---|
| `src/ewat/drift/` | ✅ | 34 | MMD-RFF + look-through |
| `src/ewat/encoder/` | ✅ | 13 | STGCN + STGAT + SimCLR + EpisodeDataset |
| `src/ewat/typing/` | ✅ | 32 | Siamois + clustering (avg+cosine) + SHAP |
| `src/ewat/ontology/` | ✅ | 180 | Temporal + TE-KSG + χ² + **OWL export + synthesis + reasoning (HermiT) + SPARQL + composite causal** |
| `src/ewat/precursor/` | ✅ | 21 | One-vs-rest {lr, lr_tuned, rf, svc} + AUROC/k* |
| `src/ewat/alerts/` | ✅ | 31 | Alert + AlertAssembler (+ scaler + DriftDetector) |

**Ontologie — modules étendus** :
- `owl_schema.py` + `owl_export.py` : export vers OWL/RDF (taxonomy + instances ABox)
- `synthesis.py` : génération d'épisodes synthétiques composites (chevauchements de types)
- `reasoning.py` : raisonnement HermiT via owlready2 (cohérence, matérialisation)
- `queries.py` : SPARQL sur l'ontologie matérialisée
- `composite_causal.py` : causalité sur épisodes composites
- `literature_taxonomy.py` : mapping scénarios Chaos Mesh → classes OWL issues de la littérature

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

**Cluster-level TE (build.py — biais écologique documenté) :**
- 22 relations temporelles, **0 causales**, 0 co-occurrence
- Les 2 relations causales du dry-run (C6→C8, C2→C8, 20 perm.) étaient des faux positifs
- Relations temporelles : 10 auto-transitions (Ci→Ci), 12 transitions inter-clusters (support ≥ 3)

**Service-level TE (build_service.py — estimateur hiérarchique, sans biais écologique) :**
- **124 relations causales brutes** → **46 filtrées** (paires ubiquitaires inter-clusters retirées) → **8/10 clusters enrichis**
- C5 (rolling_deploy) et C6 (config_change) : **0 relation** — les drifts bénins ne produisent pas de cascade causale entre services (résultat validant)
- `load-generator → frontend` : relation ubiquitaire (tous clusters actifs) — couplage structurel
- **Relation unique à C8** (θ_{drift∩anomaly}) : `cart → load-generator` (TE=0.031) — seul cluster où ce sens est significatif

**Ontologie OWL/RDF (build_owl.py — `experiments/ontology_v2/`) :**
- **29 classes** TBox ancrées littérature (Soldani & Brogi 2022, Fu et al. 2025, Gregg 2013, Aniello et al. 2014)
- **143 individus** ABox : 10 EmpiricalCluster + 10 Anomaly typées par classe leaf + 10 Signature + 107 FeatureWeight réifiés (depuis permutation_importance) + 6 Service
- **11 object properties** (causes transitive/asymmetric/irreflexive, precedes transitive, coOccursWith symmetric, propagatesThrough ⊑ affects, hasComponent transitive, etc.) + 6 data properties
- **46 edges de propagation services** (propagatesThrough) après filtre de spécificité : 124 brutes → 46 spécifiques, 13 paires ubiquitaires dropped (ex. `load-generator → frontend`)
- **3 relations causales** (causes, BH-FDR p < 0.05 sur cascades synthétiques, TE multivariate KSG-1) : C4→C1 (crash → traffic_ramp), C6→C5 (config → rolling_deploy), C4→C8 (crash → faulty_deploy)
- **19 co-occurrences** (coOccursWith, par construction sur overlays — pas de test statistique car circulaire)
- **12 precedes** cross-cluster injectées depuis les transitions temporelles (self-loops exclus)
- **HermiT reasoning** : ontologie cohérente en 0.61 s, 0 classe inconsistante. Limitation owlready2 : entailments d'instances accessibles via SPARQL, pas matérialisés dans `.is_a`.
- **Validation chiffrée** : 8/10 critères atteints (cf. `experiments/ontology_v2/results.md`)
- Fichiers : `data/ontology/taxonomy.{ttl,owl}`, `data/ontology/full_ontology.{ttl,owl}`, `experiments/ontology_v2/build_summary.json`, `validation.json`

**Épisodes synthétiques (synthesis.py + scripts/synthesize_composite_episodes.py) :**
- **282 épisodes synthétiques** générés dans `data/features/v3_synthetic/` (19 rejetés par garde-fous)
- Overlays α ∈ {0.3, 0.5}, cascades gap ∈ {2, 5, 10} steps → T_cascade ≈ 50 (résout KSG d=17, T ≥ 5·d)
- Garde-fous : clip soft p99, Spearman médian ≥ 0.85, AUC discriminateur LR < 0.75
- **AUC discriminateur = 0.529** (indistinguable du réel à corpus level)
- Objectif : observer co-occurrences/causalités inter-types impossibles dans le design mono-scénario de ewat_v3

### Étape 3 — Précurseurs (k ∈ {2,4,6,8,10,12} steps, k* sélectionné sur val)

| Type | n_pos_test | k* | AUROC_val(k*) | AUROC_test(k*) | IC 95% bootstrap |
|------|------------|-----|--------------|----------------|-----------------|
| C0   | 8          | 6   | 0.985        | **0.973**      | [0.906, 1.000]  |
| C1   | 3          | 6   | 1.000        | **0.992**      | [0.953, 1.000]  |
| C2   | 5          | 6   | 0.970        | **0.945**      | [0.865, 1.000]  |
| C3   | 3          | 2   | 0.889        | **0.794**      | [0.636, 0.930]  |
| C4   | 8          | 2   | 1.000        | **1.000**      | [1.000, 1.000]  |
| C5   | 2          | 6   | 1.000        | **0.977**      | [0.909, 1.000]  |
| C6   | 1          | 2   | 1.000        | NaN            | n.a. (n_pos<2)  |
| C7   | 7          | 6   | 0.992        | **0.992**      | [0.966, 1.000]  |
| C8   | 7          | 10  | 0.988        | **0.962**      | [0.895, 1.000]  |
| C9   | 1          | 2   | NaN          | NaN            | n.a. (n_pos<2)  |

**H3 ✓ PASS** — 8/10 types prédictibles (graine 42) ; 7-8/K selon la graine.
C6/C9 : NaN par manque d'épisodes test (n_pos=1). C3 IC le plus large [0.636, 0.930] (n_pos=3).

_Correction 2026-05-11_ : k* identiques à STATUS.md original. AUROC légèrement supérieur (+0.3–1.6 pp) car le résidu STGCN du commit 6543c69 (LayerNorm TCN activé post-entraînement) est désormais corrigé — forward reverted au comportement de l'entraînement.

**Baselines précurseurs** — deux niveaux de comparaison :

_B0/B1/B2 (cible : labels EWAT — récupérabilité des clusters)_ :
| Baseline | AUROC test @k* | Remarque |
|---|---|---|
| B0 (aléatoire) | 0.500 | référence |
| **B1 (features brutes, sans STGCN)** | **0.966** | prédit labels EWAT |
| **B2 (k-means brut + LR)** | **0.975** | prédit labels EWAT |
| **EWAT (STGCN + Siamois)** | **0.951** | prédit ses propres labels |

_B3/B4 (cible : scénarios Chaos Mesh — vérité terrain indépendante, k=6, macro-AUROC OvR)_ :
| Condition | macro-AUROC test | IC 95% | Δ vs B3 |
|-----------|-----------------|--------|---------|
| B3 (features brutes) | 0.835 | [0.773, 0.888] | — |
| B4 (STGCN z_e, d=64) | 0.835 | [0.772, 0.885] | +0.000 |

**Macro Δ=0.000 — coïncidence mathématique, non neutralité.** Les AUROCs sont des ratios de paires concordantes entiers sur 126 (= 3 pos × 42 neg). La somme des Δ par scénario = 0 exactement (75 paires gagnées − 75 perdues), redistribuées sans gain net. Le détail par scénario révèle la redistribution :

| Scénario | B3 | B4 | Δ |
|---|---|---|---|
| fail_slow_cpu | 0.476 | **0.746** | **+0.270** |
| drift_scale_up | 0.460 | **0.571** | **+0.111** |
| intermittent_error | 0.810 | **0.913** | **+0.103** |
| cpu_starvation | 0.722 | **0.754** | +0.032 |
| memory_pressure | 0.937 | **0.968** | +0.032 |
| oom | 0.897 | **0.944** | +0.048 |
| noisy_neighbor | **0.960** | 0.714 | **−0.246** |
| drift_config_change | **1.000** | 0.873 | **−0.127** |
| resource_leak | **0.937** | 0.833 | **−0.103** |
| crash | **0.960** | 0.921 | −0.040 |
| fail_slow_latency | **0.690** | 0.643 | −0.048 |
| network_loss | **0.722** | 0.690 | −0.032 |

_Interprétation_ : l'encodeur **redistribue** la discriminabilité entre types de pannes. Il aide les pannes basées sur la saturation CPU/latence (fail_slow_cpu +27pp) et nuit aux pannes config/réseau (noisy_neighbor −25pp). Sur n=45 test, cette redistribution est exactement compensée — elle ne le sera presque certainement plus sur ewat_v4.

**Interprétation globale B3/B4** : B1/B2 mesurent la *récupérabilité* des labels EWAT depuis le signal brut (circulaire — la cible est EWAT lui-même). B3/B4 utilisent la vérité terrain Chaos Mesh indépendante : Δ_macro=0.000 confirme que l'encodeur STGCN n'ajoute **pas de discriminabilité prédictive agrégée** au-delà des features brutes. La valeur du STGCN est **géométrique** (structuration de l'espace latent pour le clustering, H1 sil=0.519) et **redistributive** (réorganise la séparabilité par type) plutôt que prédictive au sens global.

### Stress tests H3 — robustesse (2026-05-22)

#### A1 — Distant-window : fuite signature scénario CONFIRMÉE

Script : `experiments/h3_robustness/distant_window.py`. Mêmes encodeur + siamois + classifieur ; on déplace seulement la fenêtre dans le régime normal.

| Position fenêtre | macro-AUROC test |
|---|---|
| `last` (status quo, juste avant injection) | **0.904** |
| `middle` (milieu du régime normal) | 0.907 |
| `first` (début du régime normal — maximum loin de l'injection) | 0.897 |

**Δ(far − near) = −0.007** ⇒ **fuite signature scénario confirmée**.

Le précurseur produit le **même AUROC** que la fenêtre soit juste avant l'injection ou au tout début du régime normal. Donc :
- L'AUROC=0.987 ne mesure **pas** de précursion (la dynamique pré-injection n'apporte rien).
- Le classifieur apprend la signature **statique** du scénario (quel service, mix de charge, baselines) — recoverable depuis n'importe quel point du régime normal.
- Cohérent avec B3/B4=0.835 (vérité terrain indépendante) et B1=0.966 (récupérabilité circulaire) : la tâche EWAT-label est de la discrimination de scénario, pas de la prédiction temporelle.

**Conséquence pour le rapport** : H3 reste PASS au sens "discriminabilité ⇒ AUROC > 0.5", mais **n'est pas une prédiction d'événement futur**. Reframer en "typage anticipé du scénario actif" plutôt que "détection précoce".

#### A2 — Leave-One-Scenario-Out (LOSO precursor-only)

Script : `experiments/h3_robustness/loso_cv.py`. Encodeur + siamois fixes (entraînés sur les 15 scénarios) ; pour chaque scénario s, on retire les épisodes de s du training du précurseur, on entraîne sur les 14 autres, on évalue.

| Métrique | Valeur |
|---|---|
| macro-AUROC sur test complet (45 ép, moyenne sur 15 folds) | **0.896 ± 0.013** |
| top-1 cluster acc sur scénario held-out (3 ép) | **0.511 ± 0.382** |

**Lecture** :
- Le macro-AUROC sur test complet reste élevé (0.896) car les 14 autres scénarios couvrent l'espace — la "vrai" généralisation à un type inédit est mesurée par le top-1 sur l'held-out.
- Top-1 polarisé : `cpu_starvation`, `drift_scale_up`, `fail_slow_cpu`, `intermittent_error` → 100% (leurs clusters sont peuplés par d'autres scénarios). `drift_config_change`, `drift_traffic_ramp`, `faulty_deploy_overlap`, `noisy_neighbor` → 0% (clusters uniques au scénario retiré).
- **Conclusion** : le modèle ne généralise PAS à un nouveau type de panne. Il interpole entre scénarios connus. C8 (faulty_deploy_overlap) et C3 (noisy_neighbor), 100% appariés à un seul scénario, sont impossibles à reconstituer une fois leur scénario retiré.

Cohérent avec A1 et avec la critique de circularité : la performance H3=0.987 mesure l'identification du scénario actif parmi un set fixe, pas la généralisation.

#### A3 — Permutation test sur labels (null distribution)

Script : `experiments/h3_robustness/permutation_test.py`. 100 permutations aléatoires des labels train, même encodeur + précurseur, AUROC test mesuré à chaque permutation.

| Quantité | Valeur |
|---|---|
| AUROC observé (labels réels) | **0.893** |
| Null distribution (100 perm) | 0.492 ± 0.104 |
| p95 null | 0.672 |
| p-value empirique | **< 0.01** |

**Lecture** : l'AUROC observé est très significativement au-dessus du null (p < 0.01). Donc le précurseur **apprend bien** un signal aligné avec les labels — mais A1 a montré que ce signal est la signature *statique* du scénario, pas la dynamique pré-injection. Les trois résultats sont cohérents : A3 confirme qu'il y a un signal réel, A1 montre qu'il n'est pas temporel, A2 montre qu'il ne généralise pas à un type inédit.

#### A4 — H3 filtré par n_pos ≥ 5

Script : `experiments/h3_robustness/filter_npos.py`. On élimine les clusters dont le test set contient < 5 positifs (où l'AUROC est statistiquement bruité par 1-3 points).

| Quantité | Valeur |
|---|---|
| Clusters totaux | 10 |
| Clusters reportables (n_pos ≥ 5) | **5** (C0, C2, C4, C7, C8) |
| AUROC moyen sur clusters reportables | **0.975 ± 0.020** |

**Lecture** : 5 clusters sur 10 ont un effectif suffisant. Sur ceux-là, l'AUROC reste élevé (0.975) — mais c'est cohérent avec A1 : ces clusters bien peuplés sont aussi ceux dont les scénarios ont la signature statique la plus distincte. Les 5 autres clusters (C1, C3, C5, C6, C9 — n_pos ≤ 3) doivent être marqués "non concluant" dans le rapport.

#### A5 — Paired bootstrap IC sur Δ(B4 − B3)

Script : `experiments/h3_robustness/paired_delta_b4_b3.py`. 1000 rééchantillonnages bootstrap **paired** des indices test (mêmes indices pour B3 et B4) → distribution de Δ.

| Quantité | Valeur |
|---|---|
| B3 (features brutes) macro-AUROC test | 0.8354 |
| B4 (STGCN z_e) macro-AUROC test | 0.8407 |
| **Δ(B4 − B3)** | **+0.0053** |
| Paired 95% IC sur Δ | **[−0.0315, +0.0444]** |
| IC exclut 0 | **non** |
| P(Δ ≤ 0) | 0.420 |

**Lecture** : l'IC paired contient zéro. La neutralité de l'encodeur STGCN sur les labels Chaos Mesh (cible indépendante) est **statistiquement bien soutenue** — ce n'est pas un Δ=0 ponctuel suspect, c'est une plage [−0.03, +0.04] autour de 0. Le rapport peut affirmer cette neutralité sans risque de critique. La valeur du STGCN est géométrique (H1 sil=0.78) et redistributive par scénario, pas prédictive en agrégé.

#### B1 — Instance norm diagnostic (résultat nuancé, important pour le rapport)

Script : `experiments/architecture_v2/instance_norm_diagnostic.py`. LR sur features brutes × {position, norm_mode} avec cible Chaos Mesh (15 scénarios). Pas d'encodeur — diagnostic pur sur S(t).

| Position fenêtre | Global norm | Instance norm | Δ(instance − global) |
|---|---|---|---|
| `last` (juste avant injection) | 0.840 [0.782, 0.888] | **0.850** [0.780, 0.904] | +0.010 |
| `middle` | 0.799 [0.745, 0.863] | 0.816 [0.742, 0.873] | +0.017 |
| `first` (début régime normal) | 0.769 [0.706, 0.827] | 0.824 [0.760, 0.881] | **+0.055** |

**Diagnostics croisés** :
- Δ(far − near) avec **global norm** = **−0.071** (sur labels Chaos Mesh indépendants)
- Δ(far − near) avec **instance norm** = **−0.026**
- Δ(instance − global) à `first` = **+0.055**

**Interprétation** (nuance importante au-delà de A1) :

- A1 mesurait sur les labels EWAT (auto-référents, circulaires) → Δ ≈ 0 trivial.
- B1 mesure sur les labels Chaos Mesh (vérité terrain indépendante) → **Δ(far−near) = −0.071** avec global norm. Donc **il existe bien une dynamique pré-injection** captée par les features brutes (au moins 7 pp d'AUROC).
- L'instance norm **améliore** le signal — particulièrement à `first` (+5.5 pp). Les baselines absolus ajoutent du bruit (services aux baselines différentes), l'instance norm le supprime.
- Cohérent avec A2 LOSO : il y a un signal réel, mais pas suffisant pour généraliser à un type inédit.

**Conséquence pour Phase B** : l'architecture v2 doit intégrer l'instance normalization. Cela suggère qu'un modèle entraîné directement sur labels Chaos Mesh avec instance norm + fenêtre pré-injection pourrait obtenir un AUROC honnête > 0.85 sans circularité.

#### B2 — Cible Chaos Mesh directe (instance norm + temporal flatten)

Script : `experiments/architecture_v2/chaos_mesh_target.py`. LR-OvR sur (k × N × 17) features brutes pré-injection instance-normalized. Pas d'encodeur STGCN. Cible : 15 scénarios Chaos Mesh.

| Évaluation | macro-AUROC | IC 95% bootstrap |
|---|---|---|
| **Stratified** (status quo, 209/45/45) | **0.855** | [0.789, 0.905] |
| **LOSO macro-AUROC full test** (15 folds) | 0.847 ± 0.010 | — |
| **LOSO top-1 sur scénario held-out (3 ép)** | 0.000 ± 0.000 | tautologique (OvR fermé) |

**Lectures** :

- **0.855 est le nouveau headline honnête** : cible indépendante (Chaos Mesh), instance norm pour révéler la dynamique pré-injection, pas de circularité, IC bootstrap reportée. Gain de +0.020 vs B3=0.835.
- **LOSO macro stable à 0.847** : retirer un scénario du training ne dégrade pas la prédiction sur les 14 autres. Les autres scénarios couvrent l'espace.
- **Top-1 held-out = 0 par construction** : un classifieur OvR fermé ne peut pas prédire une classe absente du training. Pour mesurer la vraie généralisation à un type inédit, il faut passer à une approche **open-set** (détection d'anomalie + nouveauté). Limitation à reconnaître dans le rapport.

**Conclusion** : on a un modèle prédictif **viable** (0.855 honnête sur cible indépendante avec IC explicite). L'encodeur STGCN n'est pas nécessaire pour cette tâche. La contribution se reframe autour de "typage de scénario actif" avec une métrique défendable, et de l'ontologie OWL/RDF Phase 8 pour la valeur sémantique.

#### B3 — Validation croisée B1/B2 sur ewat_v4 (stratifié)

**Note méthodologique critique** : le split temporel `ewat_v4` (262/56/57) est inutilisable pour cette évaluation — 4 scénarios sont entièrement absents du training (`faulty_deploy_overlap`, `memory_pressure`, `noisy_neighbor`, `resource_leak`). Test set = 3 scénarios non vus → AUROC=0.500 trivial. Nouveau dataset assemblé : `data/datasets/ewat_v4_strat/` (270/60/45 stratifié, 1 ép/scénario minimum par split).

**B1 sur ewat_v4_strat** (instance norm diagnostic, LR sur features brutes pré-injection moyennées sur k, cible Chaos Mesh) :

| Position | Global norm | Instance norm |
|---|---|---|
| `last` | 0.906 [0.862, 0.947] | **0.941** [0.909, 0.970] |
| `middle` | 0.862 | 0.896 |
| `first` | 0.863 | 0.878 |

- Δ(far − near) global = **−0.043**, instance = **−0.063** (plus large que v3 : −0.071 / −0.026) → **la dynamique pré-injection est clairement présente** sur les épisodes v4 plus longs (47–51 steps).
- Instance norm gain : +3.5 pp à `last`. Mieux que v3 (+1 pp).

**B2 sur ewat_v4_strat** (instance norm + temporal flatten, LR-OvR sur 6×N×17 = 612 features, cible Chaos Mesh) :

| Évaluation | macro-AUROC | IC 95% |
|---|---|---|
| **Stratified** | **0.920** | [0.878, 0.956] |
| LOSO macro-AUROC full test (15 folds) | 0.930 ± 0.007 | — |

**Comparaison v3 vs v4** :

| Métrique | ewat_v3 | ewat_v4_strat | Δ |
|---|---|---|---|
| B2 stratified AUROC | 0.855 | **0.920** | +0.065 |
| B2 LOSO macro | 0.847 | **0.930** | +0.083 |
| B1 best (instance norm + last) | 0.850 | **0.941** | +0.091 |

**Conclusion** : ewat_v4 confirme et amplifie le signal v3. **Le headline défensif final est 0.920–0.941** (B2 stratified / B1 best) sur cible Chaos Mesh indépendante avec IC bootstrap explicite. Les épisodes plus longs (×2) révèlent une dynamique pré-injection plus marquée (Δ near-far passe de −0.026 à −0.063 avec instance norm).

#### C1 — STGCN entraîné cible Chaos Mesh directe (v4_strat)

Script : `experiments/architecture_v2/train_chaos_mesh.py`. Pipeline complet : signal → instance norm → STGCN encoder + 15-way head → CE loss. 80 époques, batch 16.

| Métrique | Valeur |
|---|---|
| Best val macro-AUROC (15 classes) | 0.896 |
| **Test macro-AUROC** | **0.863** [0.823, 0.905] |
| Bootstrap mean (1000 resamples) | 0.863 |

**Lecture** : 0.863 test < B2 LR (0.920) → l'encodeur STGCN n'aide PAS même avec cible Chaos Mesh + instance norm + ewat_v4. Confirme directement la conclusion de A5 (Δ B4-B3 IC contient 0). Le headline défensif reste **B2=0.920** (LR sur features brutes flatten, sans STGCN).

**Conséquence** : le STGCN garde sa valeur **géométrique** (H1 silhouette=0.78) et **ontologique** (Phase 8), mais doit être exclu de la chaîne prédictive principale. La pipeline opérationnelle EWAT v2 devient :
```
S(t) → instance norm → LR-OvR 15-way → softmax → OpenMax (cf. C3)
```

#### C2 — A1 distant-window sur le modèle Chaos Mesh STGCN (résultat MAJEUR)

Script : `experiments/architecture_v2/distant_window_chaos_mesh.py`. Mêmes paramètres que A1 mais sur le modèle C1 + cible Chaos Mesh + v4_strat.

| Position | macro-AUROC | 95% CI |
|---|---|---|
| `last` (juste avant injection) | **0.876** | [0.838, 0.914] |
| `middle` | 0.813 | [0.763, 0.874] |
| `first` (début régime normal) | 0.759 | [0.708, 0.809] |

**Δ(far − near) = −0.116** ⇒ **GENUINE_DYNAMIC**

**Renversement par rapport à A1** : sur le modèle Chaos Mesh, la dynamique pré-injection compte pour **12 pp d'AUROC**. La fuite signature scénario constatée en A1 (Δ≈0) était un artefact de la circularité des labels EWAT, pas du signal. Quand on évalue sur une cible indépendante (Chaos Mesh), le modèle exploite bien des patterns temporels qui s'amplifient à l'approche de l'injection.

**Cohérence des trois mesures** :
- A1 (encoder EWAT, cible EWAT, v3) : Δ = −0.007 → fuite (labels circulaires)
- B1 (raw LR, cible Chaos Mesh, v3) : Δ = −0.026 (instance) à −0.071 (global)
- B1 (raw LR, cible Chaos Mesh, v4_strat) : Δ = −0.043 à −0.063
- **C2-A1 (STGCN end-to-end, cible Chaos Mesh, v4_strat) : Δ = −0.116** ✓

Conclusion : il y a de la précursion temporelle réelle dans le signal pré-injection. EWAT n'est pas qu'un identificateur de signature statique — pour une évaluation honnête sur cible indépendante.

#### C5 — H2 look-through retest sur ewat_v4_strat

Script : `experiments/h2_lookthrough_v4/`. Recalibration sur épisodes 47-51 steps.

| Métrique | Look-through | Baseline | Verdict |
|---|---|---|---|
| TPR drift | 0.500 | 0.750 | look-through pire |
| FPR anomaly | 0.667 | 0.697 | réduction non significative |
| p-value (paired t, one-sided) | 0.372 | — | **H2 ✗ FAIL robuste** |

**Lecture** : même avec des épisodes 2× plus longs, le mécanisme look-through MMD² ne sépare pas le drift bénin de l'anomalie. L'hypothèse "v3 trop court" n'explique pas tout — le mécanisme lui-même est fondamentalement falsifié. Résultat négatif honnête confirmé.

#### C3 — OpenMax open-set recognition (LOSO complet, 15 folds × 60 époques)

Script : `experiments/architecture_v2/openset_eval.py`. Module : `src/ewat/openset/openmax.py` (Bendale & Boult 2016, Weibull EVT). Tests unitaires : 11/11.

| Métrique | Valeur | Critère plan |
|---|---|---|
| Unknown AUROC (15 folds) | **0.550 ± 0.238** | ≥ 0.7 ❌ |
| Top-1 unknown rate sur held-out | 0.400 ± 0.407 | > 0 ✅ |
| Closed macro-AUROC après OpenMax | 0.834 ± 0.023 | dégradation < 2pp ⚠️ (~−3pp) |

**Résultat mitigé** :
- Top-1 unknown = 0.40 → 40% des épisodes du scénario held-out sont correctement flagués "unknown" (vs 0% avec OvR fermé). Gain net mais incomplet.
- Unknown AUROC = 0.55 → ~ chance globale. Très polarisé : `crash` 0.93, `drift_rolling_deploy` 0.93, `cpu_starvation` 0.79 ✅ ; `drift_config_change` 0.28, `fail_slow_cpu` 0.39 ❌.
- Pattern : les scénarios qui ressemblent fortement à un autre cluster connu sont difficiles à flaguer comme unknown (faux négatifs OpenMax).

**Conclusion honnête** : OpenMax apporte un signal de nouveauté **partiel mais réel** (top-1=0.40 vs 0). Pour la généralisation complète à un type inédit, il faudrait un dispositif plus sophistiqué (extra-class anomaly detection, Mahalanobis-OOD, ou retraining incrémental). C'est une limitation à reconnaître dans le rapport, pas à masquer.



#### Config baseline (Ward+Euclidean, d_proj=32, m=1.0, lr — 5 graines)

| Graine | sil_test | AUROC moyen | H3 |
|---|---|---|---|
| 42 | 0.414 | 0.951 | PASS (8/10) |
| 123 | **0.662** | **0.984** | PASS (7/10) |
| 456 | 0.461 | 0.981 | PASS (7/9) |
| 789 | 0.469 | 0.977 | PASS (8/11) |
| 1337 | 0.591 | 0.972 | PASS (7/10) |
| **Agrégé** | **0.519 ± 0.092** | **0.973 ± 0.012** | **5/5 PASS** |

#### Config optimisée (Average+Cosine, d_proj=64, m=2.0, lr_tuned — 10 graines)

| Graine | sil_test | AUROC moyen | H3 |
|---|---|---|---|
| 42 | 0.790 | 0.979 | PASS |
| 123 | 0.811 | **1.000** | PASS |
| 456 | 0.864 | 0.994 | PASS |
| 789 | 0.749 | 0.972 | PASS |
| 1337 | 0.618 | 0.996 | PASS |
| 0 | 0.816 | 0.988 | PASS |
| 7 | 0.734 | 0.995 | PASS |
| 17 | 0.830 | 0.964 | PASS |
| 31 | 0.818 | 0.996 | PASS |
| 99 | 0.787 | 0.982 | PASS |
| **Agrégé** | **0.782 ± 0.065** | **0.987 ± 0.011** | **10/10 PASS** |

**Amélioration vs baseline : H1 +0.263 (+51%), H3 +0.014 (+1.4pp).** Min sil_test=0.618 (>> seuil 0.3). H3 PASS les 10 graines sans exception.

Détail des sweeps → `experiments/runs/sweep_multiseed/aggregate.json`

### Baseline alerte — comparaison z-score vs EWAT (test set)

| Méthode | Détection anomalie | FA drift | Lead time |
|---|---|---|---|
| **z-score (σ=2.0–3.5)** | **100%** | **100%** | 2.5 min |
| EWAT seuil 0.3 | 100% | 100% | 4.6 min |
| EWAT seuil 0.4 | 97.0% | 100% | 3.8 min |
| EWAT seuil 0.5 | 78.8% | 100% | 3.9 min |
| **EWAT seuil 0.7** | **57.6%** | **8.3%** | **3.0 min** |

**Apport EWAT** : le z-score ne distingue pas drift et anomalie (FA=100% sur les drifts bénins à tous les seuils). EWAT au seuil 0.7 réduit la FA à 8.3% en maintenant un lead time de 3.0 min. Baselines précurseurs B0/B1/B2 → `experiments/baselines/precursor_baselines.py`.

### Simulation en ligne — AlertAssembler (test set, 45 épisodes)

DriftDetector intégré. Labels corrigés → "correct cluster" maintenant significatif.

| Seuil | Détection | Cluster correct | FA drift | Lead |
|---|---|---|---|---|
| 0.3 | **100%** | 42.4% | 100% | 4.6 min |
| 0.4 | 97.0% | 66.7% | 100% | 3.8 min |
| 0.5 | 78.8% | 63.6% | 100% | 3.9 min |
| 0.6 | 75.8% | 63.6% | 50.0% | 3.7 min |
| **0.7** | **57.6%** | **51.5%** | **8.3%** | **3.0 min** |

Point opérationnel recommandé : seuil 0.7 (FA maîtrisée). FA=100% aux seuils 0.3–0.5 car classifieurs très sensibles sur épisodes drift (court warm-up DriftDetector).

_Correction 2026-05-11_ : résultats corrigés vs STATUS précédent (48.5%/8.3%) — bug TCN LayerNorm résolu + précurseurs réentraînés. Détection améliorée (+9.1 pp), FA inchangée.

### Ablation modalités (réentraînement complet — graine 42)

**Ablation rigoureuse** : réentraînement encodeur+siamois complet pour chaque condition (vs. masquage à l'inférence, biaisé OOD). Script : `experiments/ablation/run_retrain.py`.

| Condition | n_feat | sil_train | sil_test | Δ vs full |
|-----------|--------|-----------|----------|-----------|
| **full** | 17 | 0.378 | **0.439** | — |
| **M_only** | 7 | 0.241 | **0.497** | **+0.058** |
| T_only | 6 | 0.064 | 0.412 | −0.027 |
| M+L | 11 | 0.251 | 0.382 | −0.057 |
| T+L | 10 | 0.022 | 0.341 | −0.098 |
| M+T | 13 | 0.318 | 0.316 | −0.123 |
| L_only | 4 | −0.138 | 0.051 | −0.388 |

**Résultat contre-intuitif** : M_only (7 features métriques seules) bat le modèle full (+0.058). M+T est pire que M_only — les features T ajoutent du bruit au STGCN sur 209 épisodes train, dégradant la géométrie siamoise. La valeur de T et L est prédictive (précurseurs), pas géométrique (clustering).

**Comparaison avec masquage à l'inférence** : le masquage concluait "M porte l'essentiel, T/L aident à la marge" — biais confirmé (inférence OOD). Les ordres de grandeur changent mais le classement qualitatif (M > T > L) reste cohérent.

### Ablation H3 — impact du masquage sur l'AUROC précurseur (2026-05-13)

Script : `experiments/ablation/eval_precursor_h3.py`. Masquage à l'inférence sur PrecursorClassifiers pré-entraînés (k* val-optimal). Distinct de l'ablation H1.

**Résultat inverse de H1** : pour H3, le modèle full (AUROC=0.954) bat toutes les réductions. T et L ajoutent du bruit géométrique (H1) mais sont utiles pour la prédictibilité (H3).

| Condition | Macro-AUROC | Δ vs full |
|---|---|---|
| **full** | **0.954** | — |
| M+L | 0.916 | −0.038 |
| M_only | 0.756 | −0.198 |
| T+L | 0.563 | −0.391 |
| L_only | 0.488 | −0.466 |

**Features les plus critiques pour H3** (leave-one-out) : `disk_io` (Δ=−0.088), `lexical_entropy` (−0.049), `latency_p99` (−0.042). disk_io est le plus critique malgré 16.7% NaN → argument fort pour ewat_v4.

### Ablation features (masquage à l'inférence — labels corrigés)

_Note : ablation par feature = mesure de sensibilité du modèle entraîné, pas importance causale. À interpréter comme "quelles features le modèle full exploite-t-il le plus", non comme "quelles features seraient les plus importantes si réentraîné"._

**Features critiques** (leave-one-out, p<0.05) : `trace_depth` (Δ−0.069), `lexical_entropy` (Δ−0.069), `latency_p99` (Δ−0.062), `disk_io` (Δ−0.010)

**Paires redondantes** : `latency_p99`↔`span_dur_p99` (ρ=0.936), `error_rate_http`↔`abnormal_span_rate` (ρ=0.927)

### Comparaison architectures encodeur — STGCN vs SimCLR vs GAT

Toutes les architectures entraînées sur ewat_v3 (graine 42), évaluées avec la méthodologie corrigée (nearest centroid pour H1, k* sur val pour H3).

| Architecture | K | sil_val | sil_test | H1 | H3 types | AUROC moyen |
|---|---|---|---|---|---|---|
| **STGCN** (baseline) | 10 | 0.470 | 0.414 | ✅ | 8/10 | 0.954 |
| **SimCLR** (NT-Xent) | 15 | 0.495 | 0.429 | ✅ | 11/15 | **0.964** |
| **GAT** (attention) | 15 | 0.445 | **0.497** | ✅ | **13/15** | 0.929 |

Interprétations :
- **GAT** : meilleure géométrie de l'espace latent (sil_test=0.497, +0.083 vs STGCN) et plus de types prédictibles (13/15), mais AUROC moyen plus faible (0.929). L'attention sur les arêtes améliore la structuration des clusters sans nécessairement améliorer la discriminabilité.
- **SimCLR** : meilleur AUROC moyen (0.964) mais 4 types NaN (clusters trop petits, n<2 test). Le pré-entraînement contrastif améliore la prédictibilité à K constant.
- **STGCN** : K=10 (le plus stable), compromis silhouette/AUROC satisfaisant. Choix retenu pour le pipeline principal (plus simple, résultats multi-graines disponibles).

Commandes de référence :
```bash
# SimCLR
python -m experiments.encoder.simclr_train --dataset data/datasets/ewat_v3 --features-root data/features/v3 --output experiments/encoder/simclr
python -m experiments.typing.train --encoder-checkpoint experiments/encoder/simclr/checkpoints/best_encoder.pt --output experiments/typing/simclr ...
python -m experiments.precursor.train --typing-dir experiments/typing/simclr --encoder-dir experiments/encoder/simclr --output experiments/precursor/simclr ...
# GAT
python -m experiments.encoder.train --encoder-arch stgat --output experiments/encoder/gat ...
python -m experiments.typing.train --encoder-checkpoint experiments/encoder/gat/checkpoints/best_encoder.pt --output experiments/typing/gat ...
python -m experiments.precursor.train --typing-dir experiments/typing/gat --encoder-dir experiments/encoder/gat --output experiments/precursor/gat ...
```

### Analyse des clusters — NMI, pureté, SHAP validation

- **NMI (cluster ↔ scénario) = 0.518** — alignement modéré avec les labels Chaos Mesh (attendu pour un clustering non supervisé)
- **Pureté moyenne = 0.503** — C6 (drift_config_change) : 0.800 ; C0 (fail_slow_cpu) : 0.286 (mélange de types)
- **Heatmap** : `experiments/typing/scenario_cluster_heatmap.png`
- **Interprétabilité (2026-05-11)** : ρ_Spearman(gradient×input, permutation) = **−0.34** (anti-corrélé) → gradient×input **invalidé**. Les fiches `experiments/typing/fiches/cluster_*.json` ont été régénérées avec `method='permutation_importance'` (50 shuffles, drop silhouette moyen par feature et par cluster). Top features globaux : net_sat, latency_p99, disk_io > latency_cv > span_dur_p99. retry_rate, log_warn_rate, queue_depth ≈ 0 (candidats à la suppression ewat_v4). Script : `experiments/typing/permutation_importance.py`.
- **KernelSHAP validation (2026-05-13)** : ρ_Spearman(SHAP, permutation_importance) > 0 pour **9/10 clusters** (seuil ≥ 7) → fiches permutation_importance **validées**. Seul C3 (noisy_neighbor) discordant (ρ=−0.07). Fiches SHAP : `experiments/typing/fiches/cluster_*_shap.json`. Script : `experiments/typing/kernel_shap_importance.py`.

### H2b — Régime θ_{drift∩anomaly}

**H2b PASS** (par critère formel), mais résultat nuancé :
- Le DriftDetector (fenêtre 5 steps) déclenche sur **presque tous** les épisodes (drift% = 0.51–1.00 même sur anomalies pures)
- Le seuil d'alerte 0.4 déclenche sur quasiment tous les épisodes (alert% ≈ 1.00 sur la plupart)
- L'overlap est donc trivialement élevé partout (≥ 50%) — le critère ">30%" est trop permissif
- C8 (faulty_deploy_overlap) : drift%=0.85, alert%=0.92, overlap%=0.77 — cohérent avec θ_{drift∩anomaly}
- Absence de clusters "drift pur" (drift% élevé ET alert% faible) : la suppression d'alerte n'est pas fonctionnelle sur épisodes courts

Conclusion H2b : renforce H2a. L'échec de la discrimination drift/anomalie vient de la durée d'épisode trop courte (~21 steps), pas d'un défaut de conception.

**H2b critère strict** (2026-05-13, `experiments/h2_overlap/eval_strict.py`) :
- Fisher exact C8 vs drift pur (C5+C6+C9) : OR=1.48, **p=0.35** → non significatif. H2b PASS reste trivial.
- Sensibilité threshold : à seuil 0.7, seulement 6/10 clusters passent (C0, C1, C3, C5, C8, C9).
- **Timing** : l'alerte précurseur précède le drift flag dans **85–100% des cas** (timing_gap médian < 0 pour tous les clusters). Le DriftDetector est un indicateur tardif, pas précoce.

### Transfert zero-shot — RCAEval RE2-OB (90 épisodes, 30 types de pannes)

**Protocole** : application du pipeline EWAT (encodeur STGCN + scaler + centroides ewat_v3) sur les données RCAEval sans réentraînement. RCAEval utilise le même Online Boutique avec 6 services EWAT, mais sur un cluster K8s différent, 48 steps/épisode vs. 21 pour ewat_v3.

4 stratégies de normalisation testées :

| Stratégie | Features | H1 silhouette | H3 AUROC |
|---|---|---|---|
| ewat_v3 scaler | 17 | 0.778 ⚠️ artefact | 0.510 ≈ chance |
| rcaeval scaler | 17 | 0.234 ✗ | 0.497 |
| instance norm | 17 | 0.287 ✗ | 0.507 |
| **instance norm** | **M(t) seul** | **0.684 ✓** | **0.495 ✗** |

**Meilleur résultat (instance + M_only)** : H1 ✓ PASS (sil=0.684), H3 ✗ FAIL.
L'encodeur regroupe 81/90 épisodes RCAEval dans C2 (resource_leak ewat_v3) avec pureté 0.80-1.00
— mais TOUS les types de panne (cpu, mem, delay...) mappent sur le même cluster. Détection
d'anomalie générique ✓, discrimination par type ✗.

**H3 AUROC ≈ 0.5 systématiquement** : les précurseurs ewat_v3 (~21 steps) ne reconnaissent
pas les signatures pré-injection RCAEval (48 steps). La normalisation instance-level efface
en outre la déviation pré-injection par construction.

**Conclusion** : goulot d'étranglement = scaler non transférable. Avec instance norm + M_only,
l'encodeur détecte "anomalie" mais ne discrimine pas les types sans réentraînement.
Few-shot transfer nécessaire pour H3.

Rapport complet : `experiments/rcaeval/results.md`

### Transfert few-shot — RCAEval Stratégie A (2026-05-13)

Script : `experiments/rcaeval/eval_fewshot.py`. Re-fit du StandardScaler sur n_few épisodes RCAEval, encodeur + classifieurs ewat_v3 conservés. n_repeats=5.

| n_few | H1 sil | H3 AUROC | H1 pass | H3 pass |
|---|---|---|---|---|
| 1 | 0.442±0.179 | 0.507±0.007 | ✓ | ✗ |
| 3 | 0.388±0.120 | 0.503±0.004 | ✓ | ✗ |
| 5 | 0.311±0.182 | 0.503±0.001 | ✓ | ✗ |
| 10 | 0.347±0.050 | 0.502±0.001 | ✓ | ✗ |
| 20 | 0.237±0.070 | 0.504±0.003 | ✗ | ✗ |
| 40 | 0.222±0.045 | 0.503±0.006 | ✗ | ✗ |

**H3 bloqué à ≈0.50 quel que soit n_few** : l'adaptation du scaler seul est insuffisante. L'espace latent ewat_v3 ne sépare pas les types de pannes RCAEval. **Stratégie B** nécessaire (fine-tuning du classifieur LR ou de l'encodeur sur quelques épisodes labellisés RCAEval).

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
2. ✅ **H2a (look-through)** : ✗ FAIL (p=0.27) — résultat négatif honnête
3. ✅ **Correction méthodologique H1/H3** : nearest centroid + k* sur val
4. ✅ **Ablation avec labels corrigés** : features critiques identifiées
5. ✅ **Bootstrap CIs** : AUROC, silhouette, proportions — ajoutés
6. ✅ **Multi-graines (5)** : H1/H3 stables, sil=0.519±0.092, AUROC=0.973±0.012
7. ✅ **Baseline alerte (z-score)** : FA=100% quel que soit σ — apport EWAT clair
8. ✅ **H2b** : PASS formel mais trivial — DD sensible sur épisodes courts, reinforces H2a
9. ✅ **Baselines précurseurs (B0/B1/B2)** : B1=0.966, B2=0.975 (vs EWAT=0.951) — valeur du STGCN = structuration latente
10. ✅ **Analyse clusters** : NMI=0.518, pureté=0.503, SHAP ρ=−0.34 (limitation)
11. ✅ **H2b critère strict** : Fisher C8 vs drift pur p=0.35 (trivial confirmé) + timing (alerte avant drift flag)
12. ✅ **KernelSHAP validation** : 9/10 clusters concordants → fiches permutation_importance validées
13. ✅ **Ablation H3 précurseurs** : full bat M_only pour H3 (inverse H1) ; disk_io feature la plus critique (Δ=−0.088)
14. ✅ **Few-shot transfer Stratégie A** : H3 bloqué ≈0.50 quel que soit n_few — scaler seul insuffisant, Stratégie B nécessaire
15. ✅ **Sweep clustering** : average+cosine H1_moy=0.624 vs ward+euclidean 0.532 (+17%) — mismatch géométrique confirmé et corrigé
16. ✅ **Sweep siamese (d_proj × margin)** : dp64_m2.0 meilleur H1 moy (0.798), dp32_m1.5 meilleur H3 moy (0.994) — dp64_m2.0 retenu (compromis)
17. ✅ **Sweep précurseurs** : lr_tuned H3=0.991 ≈ lr (0.990) ≈ rf (0.986) — lr_tuned marginalement meilleur
18. ✅ **Validation finale 10 graines** : sil=0.782±0.065, AUROC=0.987±0.011 — H1 +51%, H3 +1.4pp vs baseline
19. ✅ **Phase 8 — Ontologie OWL/RDF formelle** : 29 classes ancrées littérature, 143 individus, raisonneur HermiT cohérent (0.61 s). 3 causales + 19 co-occurrences + 46 propagation. Synthèse 282 épisodes composites (AUC discriminateur = 0.529). 8/10 critères validation atteints. Rapport → `experiments/ontology_v2/results.md`

### Moyen terme

19. **ewat_v4** : OTel SDK → disk_io 0% NaN, épisodes ≥ 40 steps
20. ✅ **Ablation rigoureuse** : M_only bat full (+0.058 sil_test) — T/L ajoutent du bruit au clustering STGCN sur n=209
21. ✅ **Contrastive pre-training (SimCLR)** : K=15, sil_test=0.429, AUROC=0.964 (11/15 types)
22. ✅ **GAT vs GCN** : GAT K=15, sil_test=0.497 (+0.083 vs STGCN), AUROC=0.929, 13/15 types
23. ✅ **Service-level TE (ontologie intra-épisode)** : 124 relations sur 8/10 clusters — C5/C6 (drift pur) = 0 relation (résultat validant), C8 unique `cart→load-gen`
24. ✅ **RCAEval RE2-OB zero-shot** : avec instance norm + M_only → H1 sil=0.684 ✓ (détection anomalie générique), H3 AUROC=0.495 ✗ (discrimination de types impossible sans réentraînement). Rapport → `experiments/rcaeval/results.md`
25. ⏳ **RCAEval Stratégie B2** : fine-tuning siamese head — en cours (`experiments/rcaeval/stratb2/`)

### Rapport de stage

**Matériau complet** dans `docs/results.md`, `docs/limitations.md`, `docs/evolution.md`.

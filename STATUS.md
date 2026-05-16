# EWAT — État courant du projet

_Mis à jour : 2026-05-13 (H2b critère strict, KernelSHAP validation, ablation H3, few-shot transfer RCAEval)_

> Résultats détaillés et interprétation scientifique → `docs/results.md`

---

## Hypothèses — bilan final (résultats corrigés)

| Hypothèse | Résultat | Valeur clé |
|---|---|---|
| **H1** — Structurabilité des embeddings | ✅ PASS | Silhouette test = **0.519 ± 0.092** (5 graines, seuil 0.3, min=0.414) |
| **H2a** — Séparabilité drift par look-through MMD² | ❌ FAIL | FPR_lt=0.67, p=0.27 — épisodes trop courts |
| **H2b** — Identification régime θ_{drift∩anomaly} | ⚠️ NUANCÉ | PASS formel (overlap>30% partout) mais trivial — DD trop sensible sur 5 steps |
| **H3** — Prédictibilité des précurseurs | ✅ PASS | **AUROC moyen = 0.973 ± 0.012** (5 graines, 7-8/10 types) |

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

401 tests unitaires, lint propre. Toutes les étapes implémentées et évaluées sur ewat_v3.

---

## Dataset

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | ✅ 300 épisodes | 15 scénarios × 20 rép. (`data/raw/` contient aussi `collection.log`, `rcaeval/`, `run_20260416_112413/` — non comptés) |
| Phase 2 — build_features | ✅ 300 épisodes buildés | `data/features/v3/` — **16/17** features à 0% NaN dans ewat_v3 (`network_loss_018` buildé mais exclu du split Phase 3) |
| Phase 3 — assemble | ✅ | `ewat_v3` — split stratifié 209/45/45 |

**NaN restant** : disk_io 16.7% (product-catalog, nœud NotReady) — résolu en ewat_v4. Les 4 features log (log_error_rate, log_warn_rate, semantic_anomaly, lexical_entropy) ont 0.33% NaN **uniquement dans network_loss_018** (exclu du split) ; dans ewat_v3 elles sont à 0%.

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

**Cluster-level TE (build.py — biais écologique documenté) :**
- 22 relations temporelles, **0 causales**, 0 co-occurrence
- Les 2 relations causales du dry-run (C6→C8, C2→C8, 20 perm.) étaient des faux positifs
- Relations temporelles : 10 auto-transitions (Ci→Ci), 12 transitions inter-clusters (support ≥ 3)

**Service-level TE (build_service.py — estimateur hiérarchique, sans biais écologique) :**
- **124 relations causales significatives** (p < 0.05, BH-FDR) sur 8/10 clusters
- C5 (rolling_deploy) et C6 (config_change) : **0 relation** — les drifts bénins ne produisent pas de cascade causale entre services (résultat scientifiquement attendu et validant)
- `load-generator → frontend` : relation ubiquitaire (présente dans les 8 clusters actifs) — couplage structurel de trafic, TE variable de 0.047 (C4, crash) à 0.187 (C2, resource_leak)
- **Relation unique à C8** (θ_{drift∩anomaly}) : `cart → load-generator` (TE=0.031) — seul cluster où ce sens de causalité est significatif, cohérent avec les dynamiques de retry/backoff pendant un déploiement défectueux
- **C2 (resource_leak) amplitude maximale** : TE load-gen→frontend = 0.187 (vs 0.047–0.108 ailleurs) — l'épuisement progressif des ressources amplifie les couplages inter-services
- Nb relations par cluster : C0=23, C4=20, C8=20, C1=19, C7=15, C3=10, C9=9, C2=8

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

### Évaluation multi-graines — robustesse H1 et H3 (5 graines)

| Graine | sil_val | sil_test | AUROC moyen | H3 |
|---|---|---|---|---|
| 42 | 0.470 | 0.414 | 0.951 | PASS (8/10) |
| 123 | 0.624 | **0.662** | **0.984** | PASS (7/10) |
| 456 | 0.574 | 0.461 | 0.981 | PASS (7/9) |
| 789 | 0.466 | 0.469 | 0.977 | PASS (8/11) |
| 1337 | 0.545 | 0.591 | 0.972 | PASS (7/10) |
| **Agrégé** | **0.536 ± 0.061** | **0.519 ± 0.092** | **0.973 ± 0.012** | **5/5 PASS** |

H1 PASS toutes graines (min sil_test=0.414 >> seuil 0.3). H3 PASS toutes graines.
K optimal stable : {9, 10, 10, 10, 11} selon la graine.

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

### Moyen terme

15. **ewat_v4** : OTel SDK → disk_io 0% NaN, épisodes ≥ 40 steps
16. ✅ **Ablation rigoureuse** : M_only bat full (+0.058 sil_test) — T/L ajoutent du bruit au clustering STGCN sur n=209
17. ✅ **Contrastive pre-training (SimCLR)** : K=15, sil_test=0.429, AUROC=0.964 (11/15 types)
18. ✅ **GAT vs GCN** : GAT K=15, sil_test=0.497 (+0.083 vs STGCN), AUROC=0.929, 13/15 types
19. ✅ **Service-level TE (ontologie intra-épisode)** : 124 relations sur 8/10 clusters — C5/C6 (drift pur) = 0 relation (résultat validant), C8 unique `cart→load-gen`
20. ✅ **RCAEval RE2-OB zero-shot** : avec instance norm + M_only → H1 sil=0.684 ✓ (détection anomalie générique), H3 AUROC=0.495 ✗ (discrimination de types impossible sans réentraînement). Rapport → `experiments/rcaeval/results.md`

### Rapport de stage

**En cours** — matériau complet dans `docs/results.md` et `docs/limitations.md`.

# Feuille de chiffres — Rapport de stage EWAT (source unique de vérité)

> Toute valeur citée dans `squelette_rapport_stage.md` doit provenir de cette feuille.
> Ne jamais écrire un chiffre de mémoire. `‹FLAG›` = divergence entre sources, à trancher.
> `‹À RECALCULER›` = absent des artefacts. Tag **D** = défendable (cible indépendante Chaos Mesh,
> avec IC) ; tag **C** = circulaire (cible auto-référente labels EWAT).

Sources principales : `STATUS.md` (le plus récent, 2026-06-03), `docs/results.md` (2026-05-21),
`experiments/*/results.md`, `experiments/multiseed/phase_{h,j}/`.

---

## 0. Métadonnées projet
| Clé | Valeur | Source |
|---|---|---|
| Cluster | observit-cluster1, 9 nœuds (8 Ready, 1 NotReady), RKE2 v1.32.7+rke2r1 | CLAUDE.md |
| Tests unitaires | **586** (comptage réel `grep -rE "def test_" tests/`, 48 fichiers, 2026-06-03) ; dont 154 ontologie, 97 télémétrie, 63 encodeur, 61 typing, 38 alerts, 37 graph, 34 drift, 30 precursor, 25 scripts, 23 openset, 21 utils. ‹RÉSOLU : STATUS=401 et results.md=672 obsolètes ; citer 586› | comptage dépôt |
| Latence E2E p95 | **13 ms** consolidé (v3=13.28 ms ; v4_strat Phase G=12.97 ms) ; budget < 5 s → ~375× sous budget | STATUS.md (C-3), results.md §11 |
| Budget latence par étape | étape 0 < 1 s, étape 1 < 2 s, étape 3 < 1 s ; étapes 2/2b offline | docs/formalisation.md |

## 1. Datasets (anti-fusion — ne pas mélanger)
| Version | #ép. | Split | N | T | NaN | Source |
|---|---|---|---|---|---|---|
| **ewat_v3** | 300 collectés, **299 retenus** (1 rejeté `network_loss_018`) ; 15 scén. × ~20 rép. | stratifié **209/45/45** | 6 | ~21 steps | disk_io **16.7%** (product-catalog, nœud NotReady), logs 0.4%, global ~1.5% | results.md §1, STATUS |
| **ewat_v4** | 414 collectés, **375 retenus** (39 rejetés : Loki/Jaeger outages) | temporel **262/56/57** | 6 | L≈2%, M≈3–5%, T≈20–25% | STATUS |
| **ewat_v4_strat** | 375 | stratifié **270/60/45** (≥1 ép./scén./split) | 6 | idem v4 | STATUS |
| **ewat_rcaeval** | **90** | adapté | 6 | format EWAT v3 | STATUS |
| **ewat_v5** | cible **~720 ép.** (3 runners) | held-out enforced | **41** | schéma ℝ^{T×41×18}, T=60 | STATUS (v5) |
| v5 scénarios | **22** (15 mono + 4 compo + 3 held-out) + bugs F1/F3 | — | — | — | STATUS (v5) |
| v3 services (6) | frontend, cart, load-gen, recommendation, ad, product-catalog | — | — | — | results.md §1 |
| v3 scénarios drift (4) | drift_config_change, drift_rolling_deploy, drift_scale_up, drift_traffic_ramp | — | — | — | results.md §1 |

## 2. Étape 0 — Drift (calibration)
| Clé | Valeur | Source |
|---|---|---|
| ε_drift (signal brut) | **0.5226** (Youden), ROC-AUC=0.60, TPR=0.55, FPR=0.33 (train) | results.md §2, STATUS |
| ε_emb (embeddings STGCN) | 0.5186 (Youden J=0.071) | results.md §2 |

## 3. H2a — Séparabilité drift par look-through (NÉGATIF)
| Sous-cas | Look-through | Baseline seuil | p-value | Source |
|---|---|---|---|---|
| Signal brut (v3 test) | TPR=0.42 ; FPR=0.67 | TPR=0.67 ; FPR=0.73 | **p=0.27** (Student unilatéral) | results.md §2, STATUS |
| Embeddings STGCN (v3) | FPR=0.788 | FPR=0.667 | **p=0.978** | results.md §2, STATUS |
| Retest v4_strat (C5) | TPR=0.500 ; FPR=0.667 | TPR=0.750 ; FPR=0.697 | **p=0.372** | STATUS (C5) |
Verdict : **H2a ✗ FAIL robuste** (double confirmation v3 + v4_strat).

## 4. H1 — Structurabilité (silhouette held-out, seuil 0.3)
| Config | Valeur | Source | Tag |
|---|---|---|---|
| v3 single (graine 42) | sil train 0.577 / val 0.470 / **test 0.414** ; K=10 | results.md §4 | C |
| v3 baseline 5 graines (ward+eucl, dp32, m1.0) | **0.519 ± 0.092** | STATUS, results.md §10 | C |
| v3 **config optimisée** 10 graines (avg+cos, dp64, m2.0) | **0.782 ± 0.065** ; min 0.618 | STATUS, results.md §10 | C |
| v4 6 graines | **0.467 ± 0.156** ; 5/6 PASS | STATUS | C |
| v4_strat **Phase H** 10 graines | **0.691 ± 0.115** ; range [0.521, 0.839] | STATUS, results.md §11 | C |
| v4_strat **Phase G** single 42 | 0.838 (K=12) ; bootstrap CI [0.6530, 0.8096] — **outlier non reproductible** | STATUS | C |
| Accord nearest-centroid / clustering indép. (train) | 97.6% | results.md §4 | — |

## 5. H2b — Régime θ_{drift∩anomaly} (NUANCÉ)
| Clé | Valeur | Source |
|---|---|---|
| Critère formel | PASS trivial (overlap > 30% partout) | STATUS, results.md §2 |
| Fisher C8 vs drift pur (C5+C6+C9) | OR=1.48, **p=0.35** (non significatif) | results.md §2, STATUS |
| Timing alerte précurseur avant drift flag | **85–100%** des cas (DD = indicateur tardif) | results.md §2, STATUS |
| C8 (faulty_deploy_overlap) | drift%=0.85, alert%=0.92, overlap%=0.77 | STATUS |

## 6. H3 — Prédictibilité précurseurs (cible labels EWAT = CIRCULAIRE)
> ⚠ Tag **C**. Encadré d'avertissement obligatoire dans le rapport.

Table v3 corrigée (k* sur val, AUROC test, IC95% bootstrap) :
| Type | n_pos_test | k* | AUROC_test | IC 95% | Source |
|---|---|---|---|---|---|
| C0 | 8 | 6 | 0.973 | [0.906, 1.000] | STATUS |
| C1 | 3 | 6 | 0.992 | [0.953, 1.000] | STATUS |
| C2 | 5 | 6 | 0.945 | [0.865, 1.000] | STATUS |
| C3 | 3 | 2 | 0.794 | [0.636, 0.930] | STATUS |
| C4 | 8 | 2 | 1.000 | [1.000, 1.000] | STATUS |
| C5 | 2 | 6 | 0.977 | [0.909, 1.000] | STATUS |
| C6 | 1 | 2 | NaN (n_pos<2) | — | STATUS |
| C7 | 7 | 6 | 0.992 | [0.966, 1.000] | STATUS |
| C8 | 7 | 10 | 0.962 | [0.895, 1.000] | STATUS |
| C9 | 1 | 2 | NaN | — | STATUS |
‹FLAG AUROC_test C0=0.973 (STATUS) vs 0.970 (results.md §6) — écart 0.3pp, citer STATUS›
| Agrégat | Valeur | Source | Tag |
|---|---|---|---|
| H3 v3 | **8/10 types PASS** ; AUROC moyen ≈ 0.95 | STATUS, results.md §6 | C |
| H3 config optimisée 10 graines | **0.987 ± 0.011** ; 10/10 PASS | STATUS | C |
| H3 Phase H 10 graines | **0.990 ± 0.012** ; range [0.959, 1.000] | STATUS, results.md §11 | C |

## 7. Headline DÉFENDABLE — cible Chaos Mesh indépendante (tag D)
| Clé | Valeur | IC 95% | Source |
|---|---|---|---|
| **B2 stratified v4_strat** (LR-OvR flatten, sans STGCN) | **0.9201** | **[0.878, 0.956]** | STATUS Phase J, results.md §11 |
| **B2 LOSO v4_strat** | **0.9298** | (15 folds × 10 seeds, déterministe) | STATUS Phase J |
| B1 best v4_strat (instance norm + last) | **0.941** | [0.909, 0.970] | results.md §10/§11 |
| B2 stratified v3 | 0.855 | [0.789, 0.905] | STATUS |
| C1 STGCN end-to-end Chaos Mesh v4_strat | 0.863 | [0.823, 0.905] | STATUS, results.md §10 |
| B1 v4_strat global norm (last) | 0.906 | [0.862, 0.947] | STATUS |

## 8. Neutralité STGCN sur cible indépendante (B3/B4, A5, C1)
| Clé | Valeur | Source |
|---|---|---|
| B3 (features brutes, Chaos Mesh, k=6, v3) | macro-AUROC **0.835** [0.773, 0.888] | STATUS |
| B4 (STGCN z_e, v3) | macro-AUROC **0.835** [0.772, 0.885] ; **Δ_macro = 0.000** | STATUS |
| A5 paired Δ(B4−B3) | B3=0.8354, B4=0.8407, **Δ=+0.0053**, IC [−0.0315, +0.0444] (contient 0), P(Δ≤0)=0.420 | STATUS |
| B3/B4 par scénario (redistribution) | ex. fail_slow_cpu +0.270 ; noisy_neighbor −0.246 ; somme Δ = 0 exact | STATUS |
| C1 vs B2 | 0.863 < 0.920 → STGCN n'aide pas en prédictif agrégé | STATUS, results.md §10 |

## 9. Baselines précurseurs (B0–B4)
| Baseline | Cible | AUROC | Source |
|---|---|---|---|
| B0 (aléatoire) | — | 0.500 | STATUS |
| B1 (features brutes) | labels EWAT (C) | 0.966 | STATUS |
| B2 (k-means brut + LR) | labels EWAT (C) | 0.975 | STATUS |
| EWAT (STGCN+Siamois) | labels EWAT (C) | 0.951 | STATUS |
| B3 (features brutes) | Chaos Mesh (D) | 0.835 [0.773, 0.888] | STATUS |
| B4 (STGCN z_e) | Chaos Mesh (D) | 0.835 [0.772, 0.885] | STATUS |
‹NB : B1/B2 « précurseurs » (cible EWAT) ≠ B1/B2 « architecture_v2 » (cible Chaos Mesh). Le squelette
réutilise les labels B1/B2 dans les deux contextes — bien distinguer §8.9 (cible EWAT) de §8.7 (cible Chaos Mesh).›

## 10. Stress tests robustesse (A1–A5, C2)
| Test | Valeur | Source | Tag |
|---|---|---|---|
| A1 distant-window v3 (EWAT) | last 0.904 / middle 0.907 / first 0.897 ; **Δ(far−near)=−0.007** (fuite signature) | STATUS | C |
| A1 Phase H 10 graines (EWAT) | **Δ=−0.012 ± 0.022** ; LEAK 9/10, GENUINE 1/10 (seed 42 outlier) | STATUS, results.md §11 | C |
| A2 LOSO | macro 0.896 ± 0.013 ; top-1 held-out **0.511 ± 0.382** (4×100%, 4×0%) | STATUS | — |
| A3 permutation | observé **0.893** ; null 0.492 ± 0.104 ; p95=0.672 ; **p<0.01** | STATUS | — |
| A4 n_pos≥5 | **5/10 clusters reportables** (C0,C2,C4,C7,C8) ; AUROC 0.975 ± 0.020 | STATUS | — |
| A5 | (cf. §8 ci-dessus) | STATUS | D |
| B1 diagnostic v3 | Δ(far−near) global **−0.071**, instance **−0.026** ; Δ(instance−global)@first +0.055 | STATUS | D |
| B1 diagnostic v4_strat | Δ(far−near) global **−0.043**, instance **−0.063** ; instance@last +3.5pp | STATUS | D |
| **C2-A1** distant-window STGCN+Chaos Mesh v4_strat | last 0.876 [0.838,0.914] / middle 0.813 / first 0.759 ; **Δ(far−near)=−0.116 ⇒ GENUINE** | STATUS, results.md §10 | D |
| C3 OpenMax | unknown AUROC **0.550 ± 0.238** ; top-1 unknown **0.400 ± 0.407** ; closed macro 0.834 ± 0.023 | STATUS | — |

## 11. Multi-seed Phases H / J / K
| Métrique | Valeur | Source |
|---|---|---|
| Phase H — H1 sil_test | 0.691 ± 0.115 [0.521, 0.839] | STATUS, results.md §11 |
| Phase H — H3 AUROC peak (C) | 0.990 ± 0.012 [0.959, 1.000] | STATUS |
| Phase H — A1 Δ (C) | −0.012 ± 0.022 | STATUS |
| Phase H — K_optimal | 11.8 ± 2.1 [9, 15] (instable) | STATUS |
| Phase J — B2 stratified (D) | 0.9201 [0.878, 0.956] (déterministe) | STATUS |
| Phase J — B2 LOSO (D) | 0.9298 (15 folds) | STATUS |
| Phase K.1 — silhouette | mode K=14 (2/10), range [9,15] | STATUS |
| Phase K.1 — Tibshirani gap | mode K=12 (2/10), range [4,12] ; agreement 4/10 | STATUS |

## 12. Ablations
### Modalités H1 (réentraînement complet, graine 42, 100+50 epochs) — RÉSOLU par artefact
Source de vérité : `experiments/ablation/retrain/summary.md` (masquage *après* StandardScaler,
réentraînement encodeur+typer complet). Le tableau de `results.md §8` (full=0.333) est **périmé** —
ne pas l'utiliser.
| Condition | n_feat | sil_train | sil_test | Δ vs full |
|---|---|---|---|---|
| **full** | 17 | 0.3777 | **0.4389** | — |
| **M_only** | 7 | 0.2414 | **0.4969** | **+0.0581** |
| T_only | 6 | 0.0635 | 0.4121 | −0.0268 |
| M+L | 11 | 0.2507 | 0.3819 | −0.0570 |
| T+L | 10 | 0.0216 | 0.3410 | −0.0979 |
| M+T | 13 | 0.3184 | 0.3159 | −0.1230 |
| L_only | 4 | −0.1375 | 0.0505 | −0.3884 |
Conclusion : **M_only (7 features métriques) bat le modèle full** → T/L ajoutent du bruit géométrique
au clustering STGCN sur n=209. (Leur valeur est prédictive/H3, pas géométrique/H1 — cf. §12 ablation H3.)

### Modalités H3 (masquage inférence) — concordant
| Condition | Macro-AUROC | Source |
|---|---|---|
| full | **0.954** | results.md §8, STATUS |
| M+L | 0.916 | idem |
| M_only | 0.756 | idem |
| T+L | 0.563 | idem |
| L_only | 0.488 | idem |
Feature H3 la plus critique : **disk_io (Δ=−0.088)**, puis lexical_entropy, latency_p99. | STATUS, results.md §8 |

### Leave-one-out H1 (Wilcoxon, p<0.05)
trace_depth −0.069 ; lexical_entropy −0.069 ; latency_p99 −0.062 ; disk_io −0.010.
Non significatifs : net_sat p=0.090, cpu_util p=0.246, ram_util p=0.074. | results.md §8 |

### Paires redondantes (|ρ|≥0.9 Spearman)
latency_p99 ↔ span_dur_p99 = **0.936** ; error_rate_http ↔ abnormal_span_rate = **0.927**. | results.md §8 |

## 13. Comparaison encodeurs (v3, graine 42)
| Archi | K | sil_val | sil_test | H3 types | AUROC moyen | Source |
|---|---|---|---|---|---|---|
| STGCN (baseline) | 10 | 0.470 | 0.414 | 8/10 | 0.954 | results.md §10, STATUS |
| SimCLR | 15 | 0.495 | 0.429 | 11/15 | **0.964** | results.md §10, STATUS |
| GAT | 15 | 0.445 | **0.497** | **13/15** | 0.929 | results.md §10, STATUS |

## 14. Baseline alerte & AlertAssembler (test set, 45 ép.) — RÉSOLU par artefact
Source de vérité : `experiments/alerts/results.md` (= STATUS). `results.md §7` périmé (ne pas utiliser).
z-score : détection 100%, FA drift 100% (tous σ), lead 2.5 min. | STATUS |
| Seuil | Détection | Cluster correct | FA drift | Lead (min) |
|---|---|---|---|---|
| 0.30 | 1.000 | 0.424 | 1.000 | 4.6 |
| 0.40 | 0.970 | 0.667 | 1.000 | 3.8 |
| 0.50 | 0.788 | 0.636 | 1.000 | 3.9 |
| 0.60 | 0.758 | 0.636 | 0.500 | 3.7 |
| **0.70** | **0.576** | **0.515** | **0.083** | **3.0** |
Point opérationnel recommandé : seuil **0.70** (FA drift maîtrisée à 8.3%, lead 3.0 min). | experiments/alerts/results.md |

## 15. Ontologie
| Clé | Valeur | Source |
|---|---|---|
| Itération 1 (mono-scénario) | 22 temporelles (10 self + 12 cross), **0 causales**, 0 co-occurrence | results.md §5.1 |
| Service-level TE | 124 relations brutes → **46 filtrées** ; 8/10 clusters ; C5/C6 = 0 | STATUS, results.md §5.2 |
| OWL TBox | **29 classes** ancrées littérature ; 11 object + 6 data properties ; 2 axiomes équivalence | results.md §5.2, STATUS |
| OWL ABox | **143 individus** (10 cluster + 10 anomaly + 10 signature + 107 featureweight + 6 service) | results.md §5.2 |
| Propagation | **46 edges** (propagatesThrough) après filtre (124→46, 13 paires ubiquitaires dropped) | STATUS |
| Causales (multivariate KSG-1, BH-FDR) | **3** : C4→C1 (TE=0.182, p_adj=0.015) ; C6→C5 (0.067, 0.015) ; C4→C8 (0.141, 0.030) | results.md §5.2 |
| Co-occurrences | **19** (par construction overlays, pas de test stat) | STATUS, results.md §5.2 |
| precedes cross-cluster | 12 (self-loops exclus) | STATUS |
| HermiT | cohérente en **0.61 s**, 0 classe inconsistante | results.md §5.2, STATUS |
| Synthèse composite | **282 ép.** (19 rejetés) ; α∈{0.3,0.5} ; gap∈{2,5,10} ; T≈50 ; AUC discriminateur **0.529** | results.md §5.2 |
| Validation | **8/10 critères** atteints (échecs : ≥15 causales, ≥30 inférences matérialisées) | results.md §5.2, STATUS |

## 16. Analyse clusters & interprétabilité
| Clé | Valeur | Source |
|---|---|---|
| NMI (cluster ↔ scénario) | **0.518** | STATUS |
| Pureté moyenne | **0.503** (C6=0.800 ; C0=0.286) | STATUS |
| gradient×input (validité) | ρ_Spearman vs permutation = **−0.34** → invalidé | STATUS |
| KernelSHAP vs permutation | concordant **9/10 clusters** (seul C3 discordant ρ=−0.07) | STATUS |
| Top features (permutation) | net_sat, latency_p99, disk_io > latency_cv > span_dur_p99 | STATUS |

## 17. Transfert externe RCAEval
| Clé | Valeur | Source |
|---|---|---|
| Zero-shot meilleur (instance norm + M_only) | H1 sil **0.684** ✓ / H3 AUROC **0.495** ✗ | STATUS, experiments/rcaeval/results.md |
| Few-shot Stratégie A | H3 bloqué ≈ **0.50** quel que soit n_few (1→40) | STATUS |
| Conclusion | goulot = scaler non transférable ; Stratégie B (fine-tuning) requise | STATUS |

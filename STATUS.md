# EWAT — État courant du projet

_Mis à jour : 2026-07-02 (Phase V5 — résultats multi-graines ewat_v5, Train Ticket 41 services)_

> **Nouveau sur le projet → [HANDOVER.md](HANDOVER.md)** · Index de la doc → [docs/README.md](docs/README.md)
> Résultats détaillés et interprétation scientifique → [docs/results.md](docs/results.md)
> **Évolution post-stage planifiée → [HANDOVER.md § Suites possibles](HANDOVER.md#7-suites-possibles)** (axes A: couplage onto/pred, B: précursion robuste, C: open-set, D: déploiement)
> Nomenclature des datasets → [docs/datasets.md](docs/datasets.md) · Protocole v5 figé → [docs/evaluation_protocol_v5.md](docs/evaluation_protocol_v5.md)
> **Phases historiques (L, H/J/K, G, F, détail expériences v3/v4) → [docs/status_archive.md](docs/status_archive.md)**

---

## Phase V5 — ewat_v5 Train Ticket : résultats multi-graines (2026-07-02)

### Dataset ewat_v5

| Propriété | Valeur |
|---|---|
| Topologie | Train Ticket (FudanSELab, 41 services Spring Cloud) |
| Épisodes collectés | 611 (Phase 1) |
| Épisodes retenus | 409 (202 rejetés NaN > 50%) |
| Split | train=224 / val=47 / test=138 |
| Scénarios train+val | 19 (15 mono + 4 compo) |
| Scénarios held-out (test only) | 5 (F1, F3, held_io_latency, held_kernel_fault, held_net_bandwidth) |
| Features | S(t) ∈ ℝ^{T×41×18} — **schéma v5.1** (M[0-9] + T[10-13] + L[14-17]). Le builder produit v5.2 depuis `c4b599b` : refeaturiser ce dataset changerait M[9] ([COLLECTE.md](docs/COLLECTE.md) §2) |
| Fuite held-out | **AUCUNE** ✅ |

### Résultats pipeline (5 graines, K=10 fixe)

`experiments/multiseed/phase_v5/` — encodeur 80 epochs + siamois 50 epochs + précurseurs BCa CI.

> ⚠️ **Ces sorties ne sont pas dans le dépôt.** Le run a eu lieu sur la VM de
> campagne (`dataset.json` porte `features_root = /home/jovyan/ewat/data/raw_v5`) ;
> seul le dataset assemblé est revenu, `data/raw_v5/` est vide localement. Les
> chiffres du **dataset** ci-dessus sont vérifiables dans
> `data/datasets/ewat_v5/{dataset,split}.json` ; ceux des **résultats** ne le sont
> pas. Pour les régénérer : `python -m experiments.multiseed.run_phase_v5`
> (script versionné) sur un dataset v5 complet, ou récupérer l'archive de la VM.

| Métrique | **ewat_v5 (5 graines)** | ewat_v4 Phase H-bis (10 graines) |
|---|---|---|
| **H1 sil_test** | **0.779 ± 0.042** | 0.843 ± 0.063 |
| range sil_test | [0.705, 0.833] | [0.708, 0.920] |
| **H3 AUROC peak** | **0.927 ± 0.025** | 0.999 ± 0.003 (circulaire) |
| range AUROC | [0.880, 0.949] | — |
| best_epoch siamois | 3.0 ± 1.7 | 1.5 ± 0.7 |
| H1 PASS | **5/5** | 10/10 |
| H3 PASS | **5/5** | 10/10 |

**Par graine :**

| Graine | sil_test | AUROC | best_epoch | H1 | H3 |
|---|---|---|---|---|---|
| 42 | 0.778 | 0.949 | 2 | ✅ | ✅ |
| 123 | 0.705 | 0.939 | 6 | ✅ | ✅ |
| 456 | 0.833 | 0.932 | 3 | ✅ | ✅ |
| 789 | 0.796 | 0.880 | 1 | ✅ | ✅ |
| 1337 | 0.784 | 0.937 | 3 | ✅ | ✅ |

**Lecture :**
- H1 **0.779 ± 0.042** — variance ÷1.5 vs Phase H (±0.115 sur v4). Minimum 0.705 >> seuil 0.3. Clustering stable malgré ×7 services.
- H3 **0.927** sur cible clusters v5 (non circulaire — évaluation sur test indépendant du training). Plus honnête que la cible auto-référente v4.
- best_epoch ~3 = surentraînement siamois rapide, structurel (limitation L10, identique v4).
- USAD (baseline publiée KDD 2020) = 0.878 sur v4 → EWAT v5 **0.927** sur topologie ×7 plus grande.

### Stress test A1 — distant-window (graine 42, 2026-07-02)

`experiments/a1_v5/` — même encodeur + siamois + classifieur ; fenêtre déplacée dans le régime normal.
**Sorties absentes du dépôt**, comme pour `phase_v5/` ci-dessus.

| Position fenêtre | macro-AUROC test |
|---|---|
| `last` (juste avant injection) | **0.914** |
| `middle` (milieu du régime normal) | 0.874 |
| `first` (début du régime normal) | **0.868** |

**Δ(far − near) = −0.046** ⇒ **GENUINE_DYNAMIC**

Comparaison historique :

| Dataset / modèle | Δ(far−near) | Verdict |
|---|---|---|
| v3 (labels EWAT circulaires) | −0.007 | LEAK |
| v4 STGCN cible Chaos Mesh (C2-A1) | −0.116 | GENUINE_DYNAMIC |
| **v5 Train Ticket (labels EWAT, graine 42)** | **−0.046** | **GENUINE_DYNAMIC** |

**Lecture** : le pipeline v5 exploite une vraie dynamique pré-injection (−4.6 pp AUROC quand la fenêtre
est loin de l'injection). Contrairement à v3 (Δ≈0, fuite statique), les épisodes Train Ticket plus longs
laissent apparaître un gradient temporel réel. H3 v5 est donc non circulaire à la fois par construction
(évaluation sur test tenu hors training) et par stress test (Δ < 0).

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

## Pipeline EWAT — complet (config optimisée)

_Configuration v3/v4 (6 services, 17 features). La collecte courante est v5 —
41 services, 18 features — cf. § Phase V5 ci-dessus._

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

**770 tests unitaires** (773 avec l'intégration), lint propre. Toutes les étapes
implémentées et évaluées sur ewat_v3.

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

**Source** : script `scripts/dev/adapt_rcaeval.py` — conversion RCAEval RE2-OB vers format EWAT (features v3-compatibles).

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

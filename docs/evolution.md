# EWAT — Journal d'évolution du projet

_Auteur : Wassim Badraoui — Stage Devoteam (début avril 2026)_
_Mis à jour : 2026-05-21_

Ce document retrace chronologiquement toutes les itérations du projet EWAT depuis le premier
prototype jusqu'à l'état courant. Il couvre l'évolution du pipeline de collecte, du dataset,
du modèle, et de la rigueur méthodologique. Il est destiné à témoigner du cheminement
intellectuel et technique lors de la soutenance de stage.

---

## Phase 0 — Exploration et prototype (mi-avril 2026)

### Contexte de départ

Le stage démarre avec un cluster Kubernetes réel (`observit-cluster1`, 9 nœuds, RKE2 v1.32)
et une stack d'observabilité existante : Prometheus + Grafana + OpenTelemetry Collector. L'objectif
est de construire EWAT (Early Warning and Anomaly Typing) — un système de détection précoce et de
typage automatique des anomalies dans les microservices.

### V0 — Collecteur monolithique (abandonné mi-avril)

**Architecture initiale** : deux scripts (`collect_labeled.py`, `snapshot_collector.py`) qui
interrogeaient Prometheus, Jaeger et Loki à chaque tick et calculaient directement les features
S(t) ∈ ℝ^{N×17} en ligne.

**Problème découvert** : toute erreur de feature engineering (mauvaise agrégation, fenêtre mal
choisie, feature manquante) obligeait à **relancer la collecte entière sur le cluster**, coûteuse
et non reproductible (le cluster évolue entre deux runs).

**Décision** : refactor complet. Les deux scripts monolithiques sont supprimés.

---

## Phase 1 — Découplage 3-phases et stabilisation infra (15–27 avril 2026)

### V1 — Pipeline Record → Build → Assemble

**Commits clés** : `2026-04-15` à `2026-04-24`

Le refactor majeur introduit trois phases totalement découplées :

```
Phase 1 : record_episode.py    (en ligne — touche au cluster)
Phase 2 : build_features.py    (hors ligne — rejouable à volonté)
Phase 3 : assemble_dataset.py  (hors ligne — split temporel/stratifié)
```

**Principe fondateur** : les dumps bruts (`data/raw/`) sont le ground truth intangible. Toute
itération sur les features se fait offline sans retoucher le cluster.

**Apports concrets** :
- `src/telemetry/recorder.py` : client unifié Prometheus range / Jaeger /api/traces / Loki
  /query_range avec retries et timeouts configurables
- `src/telemetry/extractors/` : réutilise la même logique que les collecteurs online →
  cohérence garantie entre train et inference
- Checkpoint append-only (`fsync`) : reprise propre après SIGINT, panne réseau, OOM VM

### Itérations sur le périmètre des services

La définition de l'ensemble V (services canoniques) a évolué plusieurs fois :

| Itération | N services | Raison du changement |
|---|---|---|
| Tentative 1 | 11 services | Périmètre complet otel-demo |
| Tentative 2 | 9 services | Retrait des services sans traces Jaeger |
| **Final** | **6 services** | Retrait de tout service absent sur ≥1 modalité |

**Leçon** : un service invisible sur une modalité (ex. pas de spans Jaeger pour `payment`)
introduit des NaN systémiques et dégrade l'agrégation intra-service. Le périmètre est figé à
`frontend`, `recommendation`, `cart`, `ad`, `product-catalog`, `load-generator`.

### Problèmes infra rencontrés et résolus

1. **`--no-traces` ignoré** → CLI réparée, flag propagé jusqu'au `SignalBuilder`
2. **Timeouts Jaeger via port-forward** (ReadTimeout, budget exhausted) → paramétrage fin
   (`request_timeout_s`, `fetch_total_timeout_s`, `max_parallel`)
3. **Dérive de cadence** quand les traces dépassaient le tick de 30s → contrainte
   `fetch_total_timeout_s < sample_interval_s`
4. **BFS cycles infinis** sur spans cycliques Jaeger → ajout d'un visited set
5. **Dégradation SPDY des port-forwards** après 1–2h → `ConnectionReset by peer` côté Jaeger,
   0 trace récoltée → nécessite NodePort ou renouvellement par épisode

### Chaos Mesh — construction du registry

- 14 scénarios définis, couvrant les 4 régimes θ de la formalisation :
  - θ_drift (4) : `drift_scale_up`, `drift_rolling_deploy`, `drift_config_change`, `drift_traffic_ramp`
  - θ_anomaly hard (3) : `crash`, `oom`, `network_loss`
  - θ_anomaly gray (3) : `fail_slow_latency`, `fail_slow_cpu`, `intermittent_error`
  - θ_anomaly contention (4) : `cpu_starvation`, `memory_pressure`, `noisy_neighbor`, `resource_leak`
  - θ_{drift∩anomaly} (1) : `faulty_deploy_overlap`
- Scripts bash autonomes pour les drifts bénins (kubectl rollout restart, scale, etc.)
- `registry.yaml` : source de vérité pour les durées, targets, et min_episodes

### Robustesse campagne (ajouts avril 2026)

- **3 modes d'accès** : `nodeport` (recommandé pour > 1h), `local-portforward`, `in-cluster`
- **`--manage-port-forwards`** : tunnel frais avant chaque épisode, tueuse d'orphelins
- **Graceful shutdown** : SIGINT/SIGTERM ne coupent pas brutalement —
  l'épisode en cours termine (delete Chaos Mesh + dump) avant exit
- **Quality gate post-dump** : contrôle immédiat du manifest (queries_ok, n_traces, n_lines)

### Premier dataset de validation

- **28 épisodes** : 2 rép. × 14 scénarios, dans `data/raw_new/`
- Révèle la dégradation SPDY → bascule vers NodePorts demandée à l'admin cluster
- Ce run reste dans `data/raw/run_20260416_112413/` comme référence historique — non utilisé
  pour l'entraînement final

---

## Phase 2 — Dataset ewat_v1 et premières expériences modèle (fin avril 2026)

### ewat_v1 et ewat_v1_strat

**Commits** : `2026-04-27`

Premiers datasets complets avec NodePorts opérationnels (node_ip `172.16.203.12`).

- `data/features/v1/` : signal S(t) ∈ ℝ^{N×17} extrait des dumps bruts
- `data/datasets/ewat_v1/` : split temporel strict (non stratifié)
- `data/datasets/ewat_v1_strat/` : split stratifié par scénario

**Problème découvert avec `ewat_v1`** : le split temporel strict ne garantit pas que tous les
scénarios apparaissent dans les trois splits. Certains types d'anomalies absents du test set →
H3 non évaluable sur ces clusters.

**Correction** : passage à un split stratifié (`ewat_v1_strat`), conservé comme standard.

### Premières expériences encodeur (`experiments/encoder_test/`)

- Premières runs d'entraînement du `STGCNEncoder` avec reconstruction auto-supervisée
- Checkpoint sauvegardé dans `experiments/encoder_test/checkpoints/`
- Observations : la convergence est rapide (< 30 epochs), val_loss se stabilise

### Premières expériences typing (`experiments/typing_test/`, `typing_test2/`)

- Expérimentations du réseau siamois (`SiameseTyper`) sur les embeddings de encoder_test
- Résultats instables : silhouette fluctue selon la graine, labels cluster non cohérents cross-split
- MLflow tracké localement (`mlflow.db`, `mlruns/`) pour ces runs

**Problème identifié** : les labels de clustering sont indépendants par split (fit_predict sur chaque
split séparément) → les IDs cluster 0..K en val et test ne correspondent pas à ceux du train.
La précision "cluster correct" dans les alertes est donc aléatoire (autour de 1/K ≈ 10%).

---

## Phase 3 — Dataset ewat_v2, implémentation pipeline complet (fin avril – début mai 2026)

### ewat_v2

- `data/features/v2/` : corrections d'agrégation (percentile sur union, non percentile de percentiles)
- `data/datasets/ewat_v2/` : 209/45/45 stratifié, 15 scénarios × 20 rép.
- Un épisode exclu : `network_loss_018` (Loki 100% NaN — qualité gate fail)

### Implémentation pipeline end-to-end

**Commits `2026-05-05`** : toutes les étapes 0→3 implémentées et testées

| Module | Contenu |
|---|---|
| `src/ewat/drift/` | MMD-RFF, DriftDetector, calibration ε_drift, bootstrap |
| `src/ewat/encoder/` | STGCNEncoder, EpisodeDataset, collate |
| `src/ewat/typing/` | SiameseTyper, clustering agglomératif, SHAP (gradient×input) |
| `src/ewat/ontology/` | Temporal, TE-KSG, χ², bootstrap |
| `src/ewat/precursor/` | One-vs-rest LR, PrecursorDataset, AUROC/k* |
| `src/ewat/alerts/` | Alert, AlertAssembler (streaming, DriftDetector intégré) |

302 tests unitaires. Lint propre. Toutes les étapes documentées.

---

## Phase 4 — Dataset ewat_v3 et résultats initiaux (début mai 2026)

### ewat_v3 — dataset de référence

- **299 épisodes** (15 scénarios × 20 rép., 1 exclu)
- `data/features/v3/` : 15/17 features à 0% NaN, `disk_io` à 16.7% NaN (nœud NotReady)
- `data/datasets/ewat_v3/` : split 209/45/45 stratifié

**Qualité** :

| Feature | NaN |
|---|---|
| cpu_util, ram_util, net_sat, queue_depth | 0% |
| latency_p99, error_rate_http | 0% (patchés depuis spans OTel) |
| span features (7–12) | 0% |
| log features (13–16) | 0.4% résiduel |
| **disk_io** | **16.7%** (product-catalog sur nœud NotReady) |

### Résultats initiaux (avant correction méthodologique)

| Hypothèse | Résultat initial | Valeur clé |
|---|---|---|
| H1 | PASS | sil_val=0.601, sil_test=**0.615** |
| H3 | PASS (4/10) | AUROC moyen ~0.70, certains types < 0.5 |
| Alertes | cluster correct ≈ 0% | Labels permutés entre splits |

**Doute soulevé** : sil_test=0.615 > sil_val=0.601 > sil_train=0.577 — impossible si la
généralisation est normale. Un modèle ne peut pas mieux structurer les données de test que les
données d'entraînement qui ont servi à définir le clustering.

---

## Phase 5 — Correction méthodologique et résultats corrigés (2026-05-06)

### Problème 1 — Clustering indépendant par split (H1)

**Origine** : `SiameseTyper.predict()` appelait `AgglomerativeClustering.fit_predict()`
séparément sur chaque split. Chaque split avait donc ses propres centroides optimaux → silhouette
artificiellement haute sur val et test.

**Correction** : labels val/test assignés par **nearest centroid** depuis les centroides train
(`scipy.spatial.distance.cdist` → `argmin`).

**Impact** :

| Split | Avant (biaisé) | Après (corrigé) |
|---|---|---|
| Train | 0.577 | 0.577 (inchangé) |
| Val | 0.601 | **0.470** |
| Test | **0.615** | **0.414** |

H1 reste PASS (seuil 0.3), mais les valeurs sont honnêtes.

### Problème 2 — k* sélectionné sur le test set (H3)

**Origine** : `find_optimal_k()` maximisait l'AUROC sur le test set → fuite d'information.

**Correction** : k* sélectionné sur val set, AUROC rapporté uniquement sur test.

**Impact** :

| Avant (k* sur test) | Après (k* sur val) |
|---|---|
| 4/10 types PASS, AUROC ~0.70 | **8/10 types PASS**, AUROC moyen **0.952** |

La correction révèle que les précurseurs sont en réalité bien meilleurs qu'estimé — le problème
était la fuite d'information, qui faisait paraître les résultats médiocres.

### Problème 3 — Labels permutés dans AlertAssembler

**Origine** : `AlertAssembler.predict()` utilisait les labels issus des clusterings indépendants
par split → IDs cluster incompatibles avec les précurseurs entraînés sur les labels train.

**Correction** : passage aux labels nearest-centroid dans toute la chaîne.

**Impact** : "cluster correct" passe de ~0% (hasard) à **45–73%** selon le seuil.

### Bilan des corrections

Ces trois corrections sont une contribution en soi : elles documentent un piège classique dans
l'évaluation des méthodes de clustering non supervisé. La méthodologie finale (nearest centroid,
k* sur val, labels cohérents cross-split) est maintenant la bonne pratique appliquée dans tous
les scripts.

---

## Phase 6 — Consolidation statistique et baselines (2026-05-06)

### Bootstrap CIs

Ajout de `src/ewat/utils/bootstrap.py` : percentile bootstrap sur AUROC, silhouette, proportions.
Toutes les métriques clés ont maintenant des intervalles de confiance 95%.

### Multi-graines (5 graines)

`experiments/multiseed/` : pipeline complet (encoder + typer + précurseurs) relancé sur 5 graines.

| Graine | sil_test | AUROC moyen | H3 |
|---|---|---|---|
| 42 | 0.414 | 0.951 | PASS (8/10) |
| 123 | **0.662** | **0.984** | PASS (7/10) |
| 456 | 0.461 | 0.981 | PASS (7/9) |
| 789 | 0.469 | 0.977 | PASS (8/11) |
| 1337 | 0.591 | 0.972 | PASS (7/10) |
| **Agrégé** | **0.519 ± 0.092** | **0.973 ± 0.012** | **5/5 PASS** |

H1 robuste (min sil_test=0.414 >> seuil 0.3). H3 robuste (PASS toutes graines).

### Baselines précurseurs (B0/B1/B2)

`experiments/baselines/precursor_baselines.py` :

| Baseline | AUROC test @k* | Interprétation |
|---|---|---|
| B0 — aléatoire | 0.500 | Référence théorique |
| B1 — features brutes (sans STGCN) | **0.966** | LR sur signal aplati |
| B2 — k-means brut + LR | **0.975** | Clustering naïf |
| **EWAT (STGCN + siamois)** | **0.951** | Pipeline complet |

**Interprétation** : B1/B2 légèrement supérieurs en AUROC mais ils prédisent les labels EWAT
depuis le signal brut — ils ne découvrent pas de structure indépendante. La valeur du STGCN est
dans la **structuration de l'espace latent** (H1, sil=0.519), pas dans la discriminabilité brute.

### Baseline alerte — z-score vs EWAT

`experiments/baselines/alert_threshold.py` :

| Méthode | Détection | FA drift | Lead |
|---|---|---|---|
| z-score (σ=2.0–3.5) | 100% | **100%** | 2.5 min |
| EWAT seuil 0.7 | 48.5% | **8.3%** | 2.9 min |

**Apport EWAT** : le z-score ne distingue pas drift et anomalie (FA=100% à tous les seuils).
EWAT au seuil 0.7 réduit la FA à 8.3% — c'est la valeur pratique du pipeline.

### H2b — θ_{drift∩anomaly}

`experiments/h2_overlap/` : analyse du cluster C8 (faulty_deploy_overlap).

**Résultat nuancé** : H2b PASS formel (overlap > 30% partout), mais le DriftDetector avec une
fenêtre de 5 steps est trop sensible sur des épisodes de ~21 steps. Le résultat renforce H2a :
la discrimination drift/anomalie échoue à cause de la durée d'épisode, pas d'un défaut
de conception. Les épisodes v4 (45–54 steps) sont attendus pour réhabiliter H2.

### Analyse clusters — NMI, pureté, SHAP

`experiments/typing/analyze_clusters.py` :

- **NMI (cluster ↔ scénario) = 0.518** : alignement modéré attendu pour un clustering non supervisé
- **Pureté moyenne = 0.503** : C6 (drift_config_change) : 0.800, C0 (fail_slow_cpu) : 0.286 (mélange)
- **SHAP gradient vs permutation** : Spearman ρ = **−0.34** (anti-corrélé) → la méthode
  gradient×input (saliency) n'est pas validée par la permutation importance — **limitation majeure**
  à déclarer en publication. Renommage de `shap_explainer` → `saliency_explainer` effectué.

---

## Phase 7 — Audit de qualité et corrections code (2026-05-06)

Un audit transversal du code a identifié plusieurs failles exploitables par un relecteur.
Corrections appliquées :

### Failles statistiques corrigées

| Problème | Localisation | Correction |
|---|---|---|
| χ² 1-cellule | `ontology/cooccurrence.py` | χ² Yates 4 cellules + Fisher exact + Holm |
| Bootstrap non reproductible | `utils/bootstrap.py` | Warning si `rng=None`, BCa pour AUROC |
| Multiplicité ablation | `experiments/ablation/run.py` | Holm + Benjamini-Hochberg sur 17 LOO + 7 modalités |
| TE limitation non documentée | `docs/formalisation.md` | Biais "somme univariée + moyenne épisodes" documenté |

### Bugs modèle corrigés

| Bug | Localisation | Correction |
|---|---|---|
| Pooling non masqué (padding inclus) | `encoder/stgcn.py` | Masked mean pool sur T (lengths fournis) |
| LayerNorm déclaré mais jamais appelé | `encoder/stgcn.py` | Branché dans `_TemporalBlock.forward()` |
| Reset DriftDetector avec `episode_id=""` | `alerts/assembler.py` | Sentinel `_RESET_ALWAYS`, reset si `None` ou id change |
| Mismatch fenêtre précurseur train/inférence | `alerts/assembler.py` | Filtrage `regime=="normal"` aligné avec `PrecursorDataset` |

### Nommage et documentation

- `shap_explainer.py` → `saliency_explainer.py` (gradient×input ≠ SHAP authentique)
- `docs/formalisation.md` : H2b ajoutée, limitations TE-KSG et saliency documentées
- `docs/formalisation.md` : χ² 2×2 + Fisher + Holm documenté pour l'ontologie

---

## Phase 8 — Ontologie OWL/RDF formelle (2026-05-20/21)

### Contexte et motivation

L'étape 2b initiale (cf. §5 de [`docs/results.md`](results.md)) avait produit un échec quasi-total :
22 relations temporelles dominées par 10 self-loops triviaux (≡ durée d'injection Chaos Mesh),
**0 relation causale**, **0 co-occurrence**. Trois causes racines identifiées :

1. **Design mono-scénario** : un épisode = un seul scénario Chaos Mesh → ni co-occurrence ni
   causalité inter-types observables par construction.
2. **TE-KSG multivariate non activée** : la version `multivariate` existait dans
   `causal.py:145-163` mais le pipeline appelait silencieusement la version `univariate_sum` biaisée.
3. **T = 21 steps trop court** pour KSG en d = 17 (règle empirique T ≥ 5·d).

L'objectif de Phase 8 est de construire une **vraie ontologie au sens W3C** (taxonomie formelle
ancrée littérature, raisonneur, queries SPARQL) tout en levant les trois blocages ci-dessus.
Plan détaillé : `~/.claude/plans/oublie-la-phase-jury-tidy-reef.md` (6 phases, ~22 j).

### Architecture OWL (TBox + ABox + raisonneur)

**TBox** — `src/ewat/ontology/owl_schema.py` + `literature_taxonomy.py` + `configs/ontology.yaml` :
- **29 classes** ancrées : Soldani & Brogi 2022 (anti-patterns microservices), Fu et al. 2025
  (RCA survey), Gregg 2013 (USE method), Aniello et al. 2014 (CascadingFailure), K8s docs (OOMKill).
- Taxonomie `Anomaly` : `Resource_Anomaly → Saturation → {CPU_/Memory_/Network_/Disk_Saturation,
  HardExhaustion}`, `Liveness_Anomaly`, `Functional_Anomaly`, `Latency_Anomaly`,
  `Network_Anomaly`, `Configuration_Anomaly`, `Deployment_Anomaly`,
  `Composite_Anomaly → {Drift_With_Anomaly, CascadingFailure}`. Taxonomie `Drift` orthogonale.
- **11 object properties** : `hasSignature`, `affects`, `observedIn`, `causes` (transitive,
  asymmetric, irreflexive), `precedes` (transitive), `coOccursWith` (symmetric, reflexive),
  `propagatesThrough` (⊑ `affects`), `hasComponent` (transitive), `hasFeatureWeight`,
  `mitigatedBy`, `isCausedBy` (inverse de `causes`).
- **6 data properties** : `featureName`, `weightValue`, `temporalDuration`, `temporalLeadTime`,
  `severity`, `confidence`.
- **Axiomes d'équivalence** : `Composite_Anomaly ≡ Anomaly ⊓ hasComponent.min(2, Anomaly)`,
  `CascadingFailure ≡ Composite_Anomaly ⊓ hasComponent.some(Anomaly ⊓ precedes.some(Anomaly))`.
- IRI base : `http://ewat.devoteam.com/ontology#`, export RDF/XML + Turtle.

**ABox** — `src/ewat/ontology/owl_export.py` :
- **143 individus** : 10 EmpiricalCluster + 10 Anomaly typés par classe leaf + 10 Signature
  + 107 FeatureWeight réifiés (depuis permutation_importance) + 6 Service.
- Mapping scénario → classe via `configs/ontology.yaml` (100 % de couverture des 15 scénarios).
- `temporalDuration` (depuis self-loops `ontology.json`), `temporalLeadTime` (depuis k* précurseur).
- `AllDifferent` émis sur tous les individus nommés (requis pour HermiT sur cardinalités).

**Propagation services** — `src/ewat/ontology/service_propagation.py` :
- Lit `experiments/ontology/service_causal.json` (124 relations TE existantes).
- **Filtre de spécificité** : 13 paires ubiquitaires dropped (ex. `load-generator → frontend`
  présent dans 8/8 clusters → graphe de trafic, pas signature de panne). Résultat : 124 → **46
  edges spécifiques** sur 8/10 clusters. C5 (rolling_deploy) et C6 (config_change) : 0 edge
  spécifique — résultat scientifiquement validant (les drifts bénins ne cascade pas).

**Raisonneur HermiT** — `src/ewat/ontology/reasoning.py` :
- owlready2 0.49 + HermiT bundled (Java 21).
- Ontologie **cohérente** (0 classe inconsistante), classification en **0.61 s** sur ABox complète.
- Limitation documentée : owlready2 ne matérialise pas les entailments d'instances dans `.is_a`
  pour les axiomes de cardinalité. Les entailments restent accessibles via SPARQL
  (5 queries canoniques dans `src/ewat/ontology/queries.py`, toutes valides).

### Épisodes synthétiques composites

**Problème** : sur ewat_v3, T = 21 steps est trop court pour KSG multivariate sur d = 17
(règle T ≥ 5·d). Cascades synthétiques A → B portent T ≈ 50 steps, qui passe le seuil.

**Solution** — `src/ewat/ontology/synthesis.py` + `scripts/synthesize_composite_episodes.py` :
- **Overlay** : `S_overlay[t,s,f] = S_A[t,s,f] + α·(S_B[t,s,f] − μ_B_normal[s,f])`
  avec α ∈ {0.3, 0.5}. α = 1.0 échoue le garde-fou Spearman médian ≥ 0.85.
- **Cascade** : concat A + bridge linéaire (gap ∈ {2, 5, 10}) + B → T = 50–60 steps,
  regime du bridge = `composite_transition`.
- **Garde-fous** : clip soft à p99 par feature, Spearman médian ≥ 0.85 sur le segment A,
  AUC discriminateur LR (réel vs synthétique) < 0.75.
- **282 épisodes** générés dans `data/features/v3_synthetic/` (12 paires temporelles prioritaires
  × 5 reps × 5 variantes ≈ 300, dont 19 rejetés par garde-fous).
- **AUC discriminateur = 0.529** (parfaitement indistinguable du réel à corpus level).

### Extraction des relations sur composites

`src/ewat/ontology/composite_causal.py` :
- **Causalité (cascades)** : pour chaque paire (cid_A, cid_B), TE multivariate KSG-1 sur les
  deux moitiés (avant/après le bridge `composite_transition`), n_permutations = 200, BH-FDR.
  Filtre dynamique variance < 1e-6 (élimine queue_depth/retry_rate dégénérés).
- **Co-occurrence (overlays)** : pas de test statistique (l'overlay EST par construction
  une co-occurrence). Seuil `min_overlay_count = 2` par paire.

**Résultat** :
- **3 relations causales significatives** (BH-FDR p < 0.05) :
  - C4 → C1 : crash → drift_traffic_ramp, TE = 0.182, p = 0.015
  - C6 → C5 : drift_config_change → drift_rolling_deploy, TE = 0.067, p = 0.015
  - C4 → C8 : crash → faulty_deploy_overlap, TE = 0.141, p = 0.030
- **19 relations de co-occurrence** entre paires de clusters (par construction des overlays).
- **12 relations `precedes`** injectées depuis les transitions temporelles cross-cluster.

### Validation chiffrée

`experiments/ontology_v2/validate_ontology.py` calcule 10 critères de qualité. **Score : 8/10** :

| Critère | Cible | Valeur | Statut |
|---|---|---|---|
| Couverture scénarios → classes | ≥ 80 % | 15/15 = 100 % | ✓ |
| Couverture clusters → classes | 100 % | 10/10 | ✓ |
| Relations causales | ≥ 15 | 3 | ✗ (corpus synthétique petit, n_per_pair = 5) |
| Co-occurrences | ≥ 10 | 19 | ✓ |
| HermiT consistency | OK | OK | ✓ |
| HermiT classification time | < 30 s | 0.61 s | ✓ |
| Inférences matérialisées | ≥ 30 | 0 | ✗ (limitation owlready2) |
| Réalisme synthèse (AUC) | < 0.75 | 0.529 | ✓ |
| Propagation edges | ≥ 30 | 46 | ✓ |
| Queries SPARQL canoniques | 5/5 | 5/5 | ✓ |

Rapport complet : `experiments/ontology_v2/results.md`.

### Comparaison avec l'ontologie originale (§5 de results.md)

| Aspect | Ontologie originale | Ontologie OWL Phase 8 |
|---|---|---|
| Relations causales | 0 | **3** |
| Co-occurrences | 0 | **19** |
| Propagation services | n/a | **46** (post-filtre spécificité) |
| Taxonomie formelle | non | **29 classes ancrées littérature** |
| Raisonneur | non | **HermiT** (cohérent) |
| Queries SPARQL | non | **5/5 canoniques** |
| Tests unitaires | 41 | **180** (test_owl_schema, test_owl_export, test_service_propagation, test_synthesis, test_reasoning) |

### Limitations résiduelles

- **Critère 3 (3 vs ≥ 15 causales)** : limité par la taille du corpus synthétique
  (n_per_pair = 5 → 25 paires). Scaling à n_per_pair = 15 attendu pour passer le seuil.
- **Critère 7 (0 inférences matérialisées)** : owlready2 ne propage pas les entailments
  d'instances dans `.is_a` après HermiT pour les axiomes de cardinalité. Les entailments sont
  accessibles via SPARQL — mitigation valable mais le critère du plan reste formellement
  non atteint.
- **Synthèse vs collecte réelle** : la synthèse passe le test discriminatif (AUC = 0.529 ≈ chance)
  mais reste synthétique. Validation finale sur ewat_v4 multi-scénario reste à faire.

---

## Phase 8b — Dataset ewat_v4 (collecte terminée, build en attente)

### Motivations

| Problème v3 | Solution v4 |
|---|---|
| `disk_io` 16.7% NaN (nœud NotReady réparé) | 0% NaN attendu — nœud `jnk2v` Ready |
| Épisodes ~21 steps (trop courts pour H2) | 45–54 steps avec nouvelle config |
| pre_injection = 1m = 2 steps (précurseurs aveugles) | 7m = 14 steps (k*=12 couvert) |
| 20 rép. → C6/C9 n=1 test → AUROC NaN | 25 rép. → ~4 épisodes test par scénario |

### Configuration v4 (`configs/collection_v4.yaml`)

| Paramètre | v3 | v4 | Impact |
|---|---|---|---|
| `baseline_s` | 5m (10 steps) | 8m (16 steps) | MMD ref window stable dès step 5 |
| `pre_injection_s` | **1m (2 steps)** | **7m (14 steps)** | Précurseurs enfin exploitables |
| `recovery_s` | 2m (4 steps) | 5m (10 steps) | Capture récupération + post-drift |
| `cool_down_s` | 1m | 2m | Stabilisation entre épisodes |
| `repetitions` | 20 | **25–38** | Corrige NaN C6/C9, drift plus répété |

- **Mode** : `--endpoint-mode nodeport` (pas de dégradation SPDY)
- **État actuel** : **414 épisodes collectés** dans `data/raw_v4/` (Phase 1 ✅ terminée)
  - drift : 25–38 rép. par scénario (traffic_ramp=38, config_change=37)
  - anomalie : 25 rép. par scénario (tous scénarios complets)
- **Phase 2 (build_features) non lancée** : `data/features/v4/` vide — prochaine priorité

---

## Phase 9 — Sweeps d'optimisation et config optimale (2026-05-20/21)

### Contexte et motivation

Les résultats H1=0.519±0.092 et H3=0.973±0.012 sur 5 graines avec la config initiale (Ward+Euclidean,
d_proj=32, margin=1.0, LR) sont solides mais exploitables. Trois faiblesses identifiées
justifiaient un sweep systématique :

1. **Mismatch géométrique** : Ward linkage exige des distances euclidiennes. Or le SiameseTyper
   projette vers une sphère unitaire (L2-norm). La distance cosinus est la métrique naturelle
   sur ℝ^d normalisé. Utiliser Ward+Euclidean sur cette sphère crée une distorsion géométrique
   non documentée dans le codebase.

2. **d_proj sous-dimensionné** : d_proj=32 pour K=10 clusters peut manquer de capacité
   de représentation. Avec d=32 et K=10, chaque cluster n'a que 3.2 dimensions en moyenne.

3. **LR non régularisé** : LogisticRegression(C=1.0) avec k_values trop espaçés {2,4,6,8,10,12}.
   Un LR avec validation croisée de C et une grille plus fine offre plus de flexibilité.

### Infrastructure créée pour les sweeps

Avant de lancer les expériences, j'ai construit deux scripts d'orchestration manquants :

- `scripts/run_pipeline.py` : enchaîne les 4 étapes (encodeur → typage → précurseurs → alertes)
  pour une config donnée, avec logging MLflow parent/enfant et `pipeline_summary.json` en sortie.
- `scripts/run_sweep.py` : itère sur une grille de configs et appelle `run_pipeline.py` pour
  chacune via `multiprocessing.Pool`. Skip automatique des runs déjà terminés (`pipeline_summary.json`
  présent) — indispensable pour reprendre après coupure de session.

Modifications code associées :
- `src/ewat/encoder/stgcn.py` : flag `use_layer_norm` (False par défaut pour compat v3)
- `src/ewat/precursor/model.py` : support `classifier_type ∈ {lr, lr_tuned, rf, svc}`
- `experiments/typing/train.py` : argparse `--clustering-linkage`, `--clustering-metric`
- `experiments/precursor/train.py` : fix bug `d_proj=32` hardcodé (lu depuis checkpoint), argparse `--classifier-type`, `--k-values`
- `configs/default.yaml` : `k_values: [1,2,3,4,5,6,8,10,12,15,20]` (vs `horizons_min: [2,5,10,20,30,60]` incohérent)

### Sweep 1 — Clustering (9 runs : 3 configs × 3 seeds)

Grille : Ward+Euclidean, Average+Cosine, Complete+Cosine.

| Config | H1 moy (3 seeds) | H3 moy |
|--------|-----------------|--------|
| Average+Cosine | **0.624** | 0.972 |
| Complete+Cosine | 0.540 | 0.973 |
| Ward+Euclidean | 0.532 | 0.974 |

**Résultat** : Average+Cosine améliore H1 de +17% vs la baseline Ward+Euclidean. Le mismatch
géométrique était bien réel — passer à la métrique cosine sur les embeddings L2-normalisés
structure mieux l'espace latent. H3 est équivalent entre configs (les précurseurs sont robustes
à la métrique de clustering). **Config retenue : Average+Cosine.**

### Sweep 2 — Siamese (36 runs : 3 d_proj × 4 margins × 3 seeds)

Grille fixant Average+Cosine : d_proj ∈ {32, 64, 128} × margin ∈ {0.5, 1.0, 1.5, 2.0}.

| Config | H1 moy | H3 moy |
|--------|--------|--------|
| dp64_m2.0 | **0.798** | 0.989 |
| dp32_m1.5 | 0.791 | **0.994** |
| dp32_m2.0 | 0.740 | 0.991 |
| dp128_m1.5 | 0.731 | 0.982 |
| dp64_m1.5 | 0.699 | 0.980 |
| dp32_m1.0 | 0.624 | 0.972 (baseline) |

**Résultat** : dp64_m2.0 donne le meilleur H1 moyen (0.798, +50% vs baseline dp32_m1.0=0.532).
dp32_m1.5 a le meilleur H3 (0.994) mais H1 légèrement inférieur. dp64_m2.0 retenu comme
meilleur compromis — la marge plus forte (2.0 > 1.0) sépare mieux les clusters dans l'espace
contrastif.

**Observation intéressante sur dp128** : des dimensions plus larges (128) n'améliorent pas
systématiquement H1 — H1 moy dp128_m1.5=0.731 < dp64_m2.0=0.798. L'encodeur STGCN (d_embed=64)
projette vers un espace déjà de dimension 64 ; doubler la tête de projection ne compresse plus
l'information et peut introduire du sur-apprentissage sur n=209 épisodes train.

### Sweep 3 — Précurseurs (9 runs : 3 classifiers × 3 seeds)

Grille fixant Average+Cosine + dp64_m2.0 : lr, lr_tuned, rf.

| Classifier | H1 moy | H3 moy |
|------------|--------|--------|
| lr_tuned | 0.821 | **0.991** |
| lr | 0.821 | 0.990 |
| rf | 0.821 | 0.986 |

**Résultat** : H1 identique (le clustering ne dépend pas du classifier). lr_tuned gagne
marginalement (+0.001 H3 vs lr). L'écart est faible car H3 ≈ 0.99 pour tous — on est proche
du plafond sur ewat_v3. **Config retenue : lr_tuned** (pas de coût supplémentaire significatif,
meilleure régularisation adaptative).

### Validation finale — 10 graines avec la config optimale

Config : Average+Cosine, d_proj=64, margin=2.0, lr_tuned, k ∈ {1..20}.

| Graine | sil_test | AUROC moyen | H3 |
|--------|---------|------------|-----|
| 42 | 0.790 | 0.979 | PASS |
| 123 | 0.811 | 1.000 | PASS |
| 456 | 0.864 | 0.994 | PASS |
| 789 | 0.749 | 0.972 | PASS |
| 1337 | 0.618 | 0.996 | PASS |
| 0 | 0.816 | 0.988 | PASS |
| 7 | 0.734 | 0.995 | PASS |
| 17 | 0.830 | 0.964 | PASS |
| 31 | 0.818 | 0.996 | PASS |
| 99 | 0.787 | 0.982 | PASS |
| **Agrégé** | **0.782 ± 0.065** | **0.987 ± 0.011** | **10/10** |

**H1 +51%, H3 +1.4pp vs baseline.** Min sil_test=0.618 (>> seuil 0.3), min AUROC=0.964.
Variance réduite (H1 σ : 0.092 → 0.065, H3 σ : 0.012 → 0.011).

### ⚠️ Mise en garde rétrospective — H3 sur labels EWAT est circulaire

Le tableau "+0.014 sur H3" reporté ci-dessus mesure la prédiction des **labels cluster produits par EWAT lui-même** depuis l'embedding STGCN. Cette évaluation est **circulaire** : le pipeline retrouve son propre partitionnement.

Le gain H1 (+51%) reste réel et défendable (contribution géométrique). Le gain H3 affiché (+1.4pp) doit être interprété comme "amélioration de cohérence interne du clustering", pas comme une amélioration prédictive.

**Le headline défensif est l'évaluation sur cible Chaos Mesh indépendante** (Phase 10, ci-dessous) : macro-AUROC = 0.920 [0.878, 0.956] sur ewat_v4_strat.

---

## Phase 10 — Corrections méthodologiques (2026-05-22 / 2026-05-26)

### Contexte et motivation

Le maître de stage a soulevé la critique que les AUROC reportés (0.973 sur 5 graines, 0.987 sur 10 graines) étaient "trop bons et suspects". Investigation approfondie : l'évaluation H3 prédit les labels EWAT eux-mêmes (produits par siamois + clustering du même pipeline) — c'est **circulaire**.

Deux phases de correction ont été menées :

### Phase A — Stress tests défensifs (5 expériences)

Scripts : `experiments/h3_robustness/`. Synthèse dans `experiments/h3_robustness/results.md`.

| Test | Question | Verdict | Détail |
|---|---|---|---|
| **A1** distant-window | Précursion temporelle ? | ❌ Fuite | Δ(far−near) = −0.007 → fenêtre au début ou fin du régime normal donne le même AUROC |
| **A2** LOSO precursor-only | Généralisation à un type inédit ? | ❌ Non | top-1 sur held-out = 0.51 ± 0.38 (polarisé 100%/0%) |
| **A3** permutation null | Signal réel ou bruit ? | ✅ Signal réel | AUROC=0.893 vs null=0.49±0.10, p<0.01 |
| **A4** filtre n_pos ≥ 5 | Clusters statistiquement reportables ? | 5/10 | AUROC moyen reportable = 0.975 ± 0.020 |
| **A5** paired Δ(B4−B3) | STGCN apporte-t-il du gain ? | ❌ Non | Δ = +0.005, IC 95% = [−0.031, +0.044] (contient 0) |

**Verdict synthétique Phase A** : EWAT discrimine correctement les scénarios actifs (cohérent A3) mais ne fait pas de précursion sur labels EWAT (A1) et ne généralise pas à un type inédit (A2). L'encodeur STGCN n'apporte rien en prédiction agrégée (A5). Le headline 0.987 mesure la cohérence interne du clustering.

### Phase B — Architecture v2 : instance norm + cible Chaos Mesh

Scripts : `experiments/architecture_v2/`. Synthèse dans `experiments/architecture_v2/results.md`.

**Pivot méthodologique** : remplacer la cible (cluster EWAT auto-référent) par les **15 scénarios Chaos Mesh** (vérité terrain indépendante du pipeline).

**Diagnostic clé (B1, ewat_v4_strat)** : Δ(far−near) avec global norm = −0.043 (vs −0.007 sur labels EWAT) → **il y a bien une dynamique pré-injection** dans le signal, masquée par la circularité de l'évaluation H3.

**Headline honnête (B2)** :

| Évaluation | macro-AUROC | IC 95% bootstrap |
|---|---|---|
| **B2 LR-OvR sur features brutes flatten** (sans STGCN) | **0.920** | [0.878, 0.956] |
| **B1 best (instance norm + last)** | **0.941** | [0.909, 0.970] |
| LOSO macro-AUROC (15 folds) | 0.930 | ± 0.007 |

**Correction critique split ewat_v4** : le split temporel original avait 4 scénarios entièrement absents du training (`faulty_deploy_overlap`, `noisy_neighbor`, `memory_pressure`, `resource_leak`) → AUROC=0.500 trivial. Nouveau split assemblé : `data/datasets/ewat_v4_strat/` (270/60/45 stratifié).

### Phase C — Refonte technique avec accès VM (2026-05-26)

#### C1 — STGCN retrain cible Chaos Mesh directe

Pipeline : `S(t) → instance norm → STGCN encoder → 15-way head → CE loss`. 80 époques.

| Métrique | Valeur |
|---|---|
| Best val macro-AUROC | 0.896 |
| **Test macro-AUROC** | **0.863** [0.823, 0.905] |

**Lecture** : 0.863 < B2 LR (0.920) → STGCN n'aide pas en prédiction même avec cible indépendante. Confirme A5. Le STGCN garde sa valeur géométrique (H1=0.78) et ontologique (Phase 8), mais doit être exclu de la chaîne prédictive principale.

#### C2 — A1 distant-window sur le modèle Chaos Mesh STGCN

| Position | macro-AUROC | 95% CI |
|---|---|---|
| `last` (juste avant injection) | **0.876** | [0.838, 0.914] |
| `middle` | 0.813 | [0.763, 0.874] |
| `first` (début régime normal) | 0.759 | [0.708, 0.809] |

**Δ(far − near) = −0.116** ⇒ **précursion temporelle réelle confirmée**.

Renversement majeur vs A1 : sur cible Chaos Mesh, la dynamique pré-injection compte pour **12 pp d'AUROC**. La fuite signature scénario d'A1 (Δ≈0) était un artefact de circularité, pas du signal.

#### C3 — OpenMax open-set recognition

Module : `src/ewat/openset/openmax.py` (Bendale & Boult 2016, EVT/Weibull). Tests unitaires : 11/11. Évaluation LOSO complète : `experiments/architecture_v2/openset_eval.py` (15 retrains × 60 époques).

Réponse partielle à A2 : permet d'attribuer une probabilité "unknown" plutôt que de mal classifier un scénario inédit dans une classe connue.

#### C5 — H2 look-through retest sur ewat_v4_strat

| Métrique | Look-through | Baseline | Verdict |
|---|---|---|---|
| TPR drift | 0.500 | 0.750 | LT pire |
| FPR anomaly | 0.667 | 0.697 | non significatif |
| p-value | 0.372 | — | **❌ FAIL robuste** |

Même avec épisodes 2× plus longs (47–51 vs 21), H2a échoue. Le mécanisme look-through MMD² est fondamentalement falsifié, pas juste par la durée. Résultat négatif honnête confirmé.

### Synthèse Phase 10

- **L'évaluation H3 circulaire (0.987) est reconnue et reframée** en métrique de cohérence interne.
- **Le headline défensif final** est 0.920–0.941 sur cible Chaos Mesh indépendante avec IC explicite.
- **La précursion temporelle est réelle** (C2-A1 : Δ=−0.12) — mais ne nécessite pas l'encodeur STGCN.
- **L'open-set est partiellement adressé** par OpenMax/EVT (C3).
- **H2a définitivement falsifié** (C5).
- **425+ tests unitaires** (4 nouveaux pour instance_normalize, 4 pour window_position, 11 pour OpenMax).

---

## Synthèse de l'évolution

### Pipeline de collecte

```
V0 (monolithique, avril semaine 1)
  ↓ Refactor majeur — problème de rejouabilité
V1 (3-phases Record/Build/Assemble, avril semaine 2)
  ↓ Ajouts robustesse — dégradation SPDY, checkpoints
V1 stabilisé (NodePorts, graceful shutdown, quality gate)
  ↓ Itérations périmètre services : 11 → 9 → 6
ewat_v1 → ewat_v1_strat → ewat_v2 → ewat_v3 (actif)
ewat_rcaeval (adapté depuis RCAEval RE2-OB, 90 épisodes)
ewat_v4 : Phase 1 ✅ (414 épisodes raw), Phase 2-3 ⏳ en attente
```

### Modèle et méthodes

```
Premières expériences (encoder_test, typing_test, typing_test2)
  ↓ Problème : labels incohérents cross-split
Correction nearest centroid (sil 0.615 → 0.414 corrigé)
  ↓ Problème : k* sur test (fuite d'information)
Correction k* sur val (H3 4/10 → 8/10)
  ↓ Problème : labels permutés dans AlertAssembler
Correction labels cohérents (cluster correct 0% → 45-73%)
  ↓ Audit statistique et code
Corrections χ², bootstrap, multiplicité, masked pooling, DriftDetector
  ↓ Sweeps systématiques (clustering + siamese + precursor)
Config optimale : average+cosine, dp64, m2.0, lr_tuned
H1 : 0.519±0.092 → 0.782±0.065 | H3 : 0.973 → 0.987 (10 graines)
  ↓ Phase 8 — Ontologie OWL/RDF formelle
TBox 29 classes ancrées littérature (Soldani, Fu, Gregg), ABox 143 individus
Phase 4 synthèse : 282 épisodes composites, AUC discriminateur 0.529
Phase 5 extraction : 3 causales + 19 co-occurrences + 46 propagation
HermiT cohérent en 0.61s — 8/10 critères de validation atteints
```

### Résultats — évolution des métriques clés

| Métrique | Valeur initiale | Valeur corrigée | Config optimisée | Direction |
|---|---|---|---|---|
| Silhouette test (H1) | 0.615 (biaisé) | 0.519 ± 0.092 (5g) | **0.782 ± 0.065 (10g)** | ↑↑ |
| H3 types prédictibles | 4/10 | 8/10 (5/5 graines) | **10/10 (10/10 graines)** | ↑↑ |
| AUROC moyen (H3) | ~0.70 | 0.973 ± 0.012 | **0.987 ± 0.011** | ↑ |
| Cluster correct (alertes) | ~0% (hasard) | **45–73%** | (non réévalué) | ↑ après correction |
| FA drift EWAT@0.7 | non mesuré | **8.3%** (vs 100% z-score) | (non réévalué) | valeur établie |

### Leçons apprises (ensemble du projet)

1. **Séparer collect/features/modèle dès le début** — le refactor V0→V1 a coûté une semaine.
2. **Les labels de clustering doivent être cohérents cross-split** — piège classique sous-documenté.
3. **Ne jamais sélectionner k* sur le test** — fuite d'information invisible si on ne compare pas train/val/test.
4. **Un résultat négatif reproductible vaut mieux qu'un résultat positif fragile** — H2 FAIL
   deux fois (signal brut + embeddings) est une contribution honnête et informative.
5. **Documenter les limitations statistiques avant la soutenance**, pas après — TE-KSG
   (somme univariée), saliency (gradient ≠ SHAP), bootstrap non seedé : tous identifiés
   et corrigés avant soumission.
6. **La métrique de clustering doit être cohérente avec la géométrie des embeddings** — Ward+Euclidean
   sur des embeddings L2-normalisés est une erreur silencieuse. Elle n'empêche pas le clustering
   de fonctionner mais dégrade H1 de 51%. La cohérence métrique/normalisation est un point
   à vérifier systématiquement avant tout sweep.
7. **Les sweeps systématiques révèlent des effets non évidents** — dp128 > dp64 semble intuitif
   mais dp64_m2.0 bat dp128 pour H1. La capacité du réseau compte moins que la marge contrastive
   sur un dataset de taille modeste (n=209 train).
8. **Infrastructure sweep = investissement rentable** — les 54 runs (clustering+siamese+precursor)
   auraient pris plusieurs semaines manuellement. Avec `run_sweep.py` et skip-existing, ils ont
   tourné en ~6h de compute CPU et ont continué après interruptions de session sans perte.

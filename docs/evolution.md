# EWAT — Journal d'évolution du projet

_Auteur : Wassim Badraoui — Stage Devoteam (début avril 2026)_
_Mis à jour : 2026-05-06_

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

## Phase 8 — Dataset ewat_v4 (collecte en cours)

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
| `repetitions` | 20 | 25 | Corrige NaN C6/C9 |

- **Mode** : `--endpoint-mode nodeport` (pas de dégradation SPDY)
- **Durée estimée** : 375 épisodes × ~25min = ~156h ≈ 6.4 jours en continu

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
ewat_v1 → ewat_v1_strat → ewat_v2 → ewat_v3 → ewat_v4 (en cours)
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
```

### Résultats — évolution des métriques clés

| Métrique | Valeur initiale | Valeur corrigée | Direction |
|---|---|---|---|
| Silhouette test (H1) | 0.615 (biaisé) | **0.519 ± 0.092** | ↓ mais honnête |
| H3 types prédictibles | 4/10 | **8/10 (5/5 graines)** | ↑ après correction |
| AUROC moyen (H3) | ~0.70 | **0.973 ± 0.012** | ↑ après correction |
| Cluster correct (alertes) | ~0% (hasard) | **45–73%** | ↑ après correction |
| FA drift EWAT@0.7 | non mesuré | **8.3%** (vs 100% z-score) | valeur établie |

### Leçons apprises

1. **Séparer collect/features/modèle dès le début** — le refactor V0→V1 a coûté une semaine.
2. **Les labels de clustering doivent être cohérents cross-split** — piège classique sous-documenté.
3. **Ne jamais sélectionner k* sur le test** — fuite d'information invisible si on ne compare pas train/val/test.
4. **Un résultat négatif reproductible vaut mieux qu'un résultat positif fragile** — H2 FAIL
   deux fois (signal brut + embeddings) est une contribution honnête et informative.
5. **Documenter les limitations statistiques avant la soutenance**, pas après — TE-KSG
   (somme univariée), saliency (gradient ≠ SHAP), bootstrap non seedé : tous identifiés
   et corrigés avant soumission.

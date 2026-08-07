# EWAT — Architecture du code

Où se trouve quoi, et pourquoi c'est découpé ainsi. Pour la formalisation
mathématique, voir [`formalisation.md`](formalisation.md) ; pour produire un
dataset, [`COLLECTE.md`](COLLECTE.md) ; pour reprendre le projet,
[`../HANDOVER.md`](../HANDOVER.md).

---

## 1. Les quatre étages du dépôt

Le dépôt sépare quatre choses qui évoluent à des rythmes différents. C'est le
découpage à comprendre avant tout le reste.

| Étage | Répertoire | Nature | Versionné |
|---|---|---|---|
| **Bibliothèque** | `src/` | Logique réutilisable, testée unitairement | code |
| **Pipeline dataset** | `scripts/` | Points d'entrée des trois phases + validation | code |
| **Collecte v5** | `v5/` | Infrastructure Train Ticket : loadgen, chaos, collect, deploy | code |
| **Recherche** | `experiments/` | Un script = une question scientifique | code seul, sorties ignorées |

`src/` ne doit jamais importer `experiments/` ni `v5/`. L'inverse est la règle :
quand un script d'expérience produit de la logique réutilisable, elle remonte
dans `src/ewat/<étape>/` avec ses tests.

## 2. `src/` — la bibliothèque

### Le pipeline EWAT, étapes 0 → 3

```
S(t) ∈ ℝ^{T×N×F}
    │
    ├─ Étape 0   DriftDetector — MMD-RFF, look-through          src/ewat/drift/
    ├─ Étape 1   STGCNEncoder → z_e ∈ ℝ^64                      src/ewat/encoder/
    ├─ Étape 2   SiameseTyper + clustering → cluster C_i        src/ewat/typing/
    ├─ Étape 2b  OntologyGraph — temporel + TE-KSG + χ²         src/ewat/ontology/
    ├─ Étape 3   PrecursorClassifier → p̂_i(t), k*_i             src/ewat/precursor/
    └─ Sortie    Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})    src/ewat/alerts/
```

| Module | Étape | Tests | Contenu |
|---|---|---:|---|
| `src/ewat/drift/` | 0 | 39 | MMD-RFF (`mmd.py`), détecteur avec look-through (`detector.py`), calibration d'ε_drift |
| `src/ewat/encoder/` | 1 | 80 | STGCN, STGAT, pré-entraînement SimCLR, `EpisodeDataset`, fabrique de modèles |
| `src/ewat/typing/` | 2 | 68 | Réseau siamois, clustering agglomératif, explicabilité (saliency, SHAP) |
| `src/ewat/ontology/` | 2b | 244 | Relations temporelles, causales (TE-KSG), co-occurrence χ², export OWL/RDF, raisonnement HermiT, requêtes SPARQL |
| `src/ewat/precursor/` | 3 | 37 | Classifieurs one-vs-rest `{lr, lr_tuned, rf, svc}`, sélection de `k*` |
| `src/ewat/alerts/` | sortie | 38 | `Alert`, `AlertAssembler` (intègre scaler + DriftDetector) |
| `src/ewat/openset/` | — | 23 | Mahalanobis, OpenMax — rejet des types inconnus |
| `src/ewat/baselines/` | — | — | USAD (baseline publiée, KDD 2020) |

### Télémétrie : la distinction collecteurs / extracteurs

C'est le point de conception le plus important de `src/`, et le plus facile à
casser par accident.

| | `src/telemetry/collectors/` | `src/telemetry/extractors/` |
|---|---|---|
| Quand | **En ligne**, pendant la collecte | **Hors ligne**, sur les dumps |
| Entrée | Prometheus / Jaeger / Loki en direct | `prometheus_range.json.gz`, `jaeger_spans.json.gz`, `loki_logs.json.gz` |
| Rôle | Porte la logique de features | **Rejoue la même logique** sur les dumps |

Conséquence directe : **un bug de featurisation ne coûte jamais une
recollecte**. On corrige la logique, on relance la phase 2, et les 611 épisodes
sont refeaturisés hors cluster. C'est pour cela que les dumps de `data/raw*/`
sont immuables.

Le reste de `src/telemetry/` (109 tests) : `signal_builder.py` assemble S(t),
`features/{aggregation,lexical,semantic}.py` portent les agrégations
différenciées, `feature_names.py` versionne le schéma.

> `feature_names.py` porte encore le **schéma v4 à 17 features**. Le schéma
> courant est v5.1 à 18 features, et il est versionné **par épisode** dans
> `metadata.signal_feature_names`, qui fait foi. Ne prenez pas la constante
> globale pour la référence v5.

### Graphe et utilitaires

- `src/graph/` — construction de G(t) : adjacence pondérée `w_E(t) ∈ ℝ³`
  (volume, latence médiane, taux d'erreur), sérialisation, validation.
- `src/utils/` — `seeding.py` (déterminisme), `serialization.py`.

## 3. `scripts/` — le pipeline dataset

Trois phases découplées, jamais en boucle. Détail opérationnel dans
[`COLLECTE.md`](COLLECTE.md).

| Phase | Script | Entrée → sortie |
|---|---|---|
| 1. Record | `record_episode.py` | cluster → `data/raw*/<ep>/` |
| 2. Build | `build_features.py` (v1–v4), `v5/collect/build_features_v5.py` (v5) | dumps → `data/features/<set>/` |
| 3. Assemble | `assemble_dataset.py` | features → `data/datasets/<nom>/` |
| Contrôle | `validate_raw.py`, `validate_dataset.py`, `validate_v4.py`, `validate_v5.py` | portes qualité |

Autres points d'entrée : `run_pipeline.py` (enchaîne encodeur → typage →
précurseurs → alertes), `run_sweep.py`, `export_thesis_figures.py` /
`export_paper_figures.py` / `export_report_figures.py`,
`build_release_v5.py` + `audit_leak_v5.py` (kit de publication du dataset,
avec audit de fuite d'infrastructure), `enforce_heldout_v5.py`.

`scripts/dev/` contient des utilitaires ponctuels et datés (`patch_error_rate.py`,
`impute_features.py`, `adapt_rcaeval.py`…). Ce ne sont pas des points d'entrée du
pipeline : ils documentent des correctifs appliqués une fois à un dataset donné.

## 4. `v5/` — l'infrastructure de collecte Train Ticket

Autonome, avec sa propre documentation ([`../v5/README.md`](../v5/README.md),
[`../v5/LAUNCH.md`](../v5/LAUNCH.md), [`../v5/PREFLIGHT.md`](../v5/PREFLIGHT.md)).

| Sous-paquet | Rôle |
|---|---|
| `loadgen/` | Générateur de charge — fork vendorisé de `train-ticket-auto-query`, patché pour un login sans CAPTCHA |
| `chaos/` | Injection : `catalog.yaml` (28 scénarios + 5 bugs réels) et `inject.py` |
| `collect/` | Orchestration de campagne, featurisation v5.1, sondes, reprise, monitoring santé |
| `deploy/` | Déploiement Train Ticket et NodePorts de télémétrie |

Ce code s'exécute **depuis `v5/`** avec `PYTHONPATH=../src`, contrairement au
reste du dépôt.

## 5. `experiments/` — la recherche

Un répertoire par question. Le **code** est versionné (87 scripts) ; les
**sorties** (JSON, npy, pt, PNG, runs MLflow, rapports de run) ne le sont pas.

| Répertoire | Question |
|---|---|
| `encoder/`, `typing/`, `precursor/`, `ontology/`, `alerts/` | Les étapes du pipeline, entraînement et évaluation |
| `h2_lookthrough/`, `h2_overlap/`, `h2_embeddings/` | H2a (séparabilité du drift) et H2b (régime θ drift∩anomaly) |
| `h3_robustness/` | H3 : fenêtre distante, LOSO, test de permutation, delta apparié |
| `verification/` | Vérification méthodologique croisée H1 + H3 |
| `ablation/` | Ablation par modalité et par feature, avec réentraînement complet |
| `baselines/` | Baselines de précurseurs, seuil d'alerte, baseline par scénario |
| `multiseed/` | Agrégation multi-graines (phases H, J, K, V5) |
| `audit2026/` | Correctifs et contre-vérifications de l'audit 2026-06 |
| `rcaeval/`, `sota/` | Transfert vers RCAEval, comparaison à USAD |
| `architecture_v2/`, `drift_separation/`, `data_quality/`, `bench/`, `figures/` | Variantes d'architecture, calibration, qualité du signal, latence, figures |

> **Ce répertoire n'était pas versionné jusqu'au 2026-08.** Le `.gitignore`
> excluait `experiments/` au niveau du répertoire, ce qui neutralisait les
> négations placées ensuite — git ne descend jamais dans un répertoire exclu.
> 57 des 65 scripts manquaient au dépôt. Si vous ajoutez une exception de
> ré-inclusion sous un répertoire exclu, **vérifiez-la avec `git add --dry-run`**.

## 6. Tests

**773 tests** : 770 unitaires et 3 d'intégration (déterminisme bout en bout).

```bash
make test                       # tout
pytest tests/unit -q            # unitaires seuls
pytest tests/integration -q     # déterminisme bout en bout
```

L'arborescence de `tests/unit/` reflète celle de `src/`. La CI
(`.github/workflows/data-pipeline-quality.yml`) exécute lint + unitaires +
intégration à chaque push.

## 7. Dettes connues

Défauts identifiés, non corrigés, à traiter en connaissance de cause.

| Où | Quoi | Effet |
|---|---|---|
| `v5/loadgen/runner.py:92` | `logger` non défini (`F821`) | **`NameError` sur le chemin d'échec du login admin** — la campagne plante là où elle devrait seulement avertir |
| `v5/loadgen/atomic_queries.py` (×9) | `if response.status_code is not 200` (`F632`) | Comparaison d'identité sur un littéral : le comportement dépend de l'internement des petits entiers en CPython. Code vendorisé FudanSELab, laissé aligné sur l'amont |
| `v5/loadgen/queries.py:276` | `!= None` (`E711`) | Sans effet ici, mais fragile |
| `v5/loadgen/atomic_queries.py:8-12` | Cookie de session et JWT en dur | Placeholders de démonstration amont (JWT expiré en 2021), pas des identifiants du cluster — mais présents dans un fichier versionné |
| `src/telemetry/feature_names.py` | Schéma v4 (17 features) | Ne fait plus foi pour v5.1 ; cf. § 2 |
| `src/ewat/typing/saliency_explainer.py` | ρ_Spearman(gradient, permutation) = −0,34 | Méthode gradient non validée ; les fiches de cluster restent indicatives |

Les limites **scientifiques** (par opposition à ces dettes techniques) sont dans
[`limitations.md`](limitations.md).

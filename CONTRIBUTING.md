# Contribuer à EWAT

Nouveau sur le projet ? Lisez [`HANDOVER.md`](HANDOVER.md) d'abord.
Architecture du code : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Environnement

```bash
python3 -m venv .venv && source .venv/bin/activate
make dev            # pip install -e ".[dev]"
make test           # pytest tests/ — 773 tests (770 unitaires + 3 d'intégration)
make lint           # ruff check src scripts tests experiments
```

Python ≥ 3.11 requis (le code utilise `datetime.UTC` et les annotations PEP 604).
`pyproject.toml` est l'unique source des dépendances.

Après `pip install -e .`, les paquets `ewat`, `telemetry`, `graph` et `utils` sont
importables directement : **`PYTHONPATH=src` n'est plus nécessaire**. Seuls les
modules de `v5/` gardent un préfixe, parce qu'ils s'exécutent depuis `v5/` et
importent `src/` en dehors du paquet installé :

```bash
python -m scripts.assemble_dataset ...          # depuis la racine
cd v5 && PYTHONPATH=../src python -m collect.run_campaign ...
```

## Standards de code

- Type hints, docstrings numpy-style.
- Ruff, `line-length = 100`, règles `E, F, I, N, W, UP`.
- Chaque étape du pipeline est un module indépendant et testable.
- Configuration par Hydra (`configs/default.yaml`), tracking par MLflow.
- Tout résultat s'accompagne d'un intervalle de confiance et d'un test
  statistique. **Un résultat négatif est une contribution** — le dépôt en
  contient plusieurs (H2a, transfert RCAEval), ne les cachez pas.
- Figures publiables : matplotlib/seaborn, sortie vectorielle.

Deux répertoires ont un régime de lint allégé, déclaré en `per-file-ignores` :

| Répertoire | Pourquoi |
|---|---|
| `experiments/**` | Scripts de recherche : notation mathématique du papier (`S`, `W`, `K`), longues lignes de tables de résultats. Versionnés pour la reproductibilité, pas tenus au standard de `src/`. |
| `v5/**` | Code de campagne gelé, validé par 7 jours de collecte. On ne le retouche pas pour du style. |

Ce régime allégé ne couvre **pas** les défauts de correction (`F632`, `F821`,
`E711`) : ceux qui restent dans `v5/` sont connus et listés dans
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) § Dettes connues.

## Invariants scientifiques

Ces règles ne sont pas des préférences. Chacune correspond à une erreur
méthodologique identifiée, et l'enfreindre invalide silencieusement un résultat.

- **EWAT n'est pas du RCA.** Le RCA est post-mortem (Où, Pourquoi, après la
  panne) ; EWAT est de l'early warning (Quoi, Dans combien de temps, avant la
  panne). Le projet frère `matrix_simple` traite l'autre moitié du problème.
- **Jamais Granger, toujours Transfer Entropy** (estimateur KSG, Kraskov 2004),
  avec seuil par bootstrap de permutation et correction FDR.
- **Jamais de mise à zéro du signal pendant un drift** — toujours le mécanisme
  de look-through. Couper le signal détruit précisément l'information qu'on
  cherche à évaluer.
- **Jamais de moyenne simple pour agréger** au niveau service : saturation → max,
  taux → somme pondérée par le volume, latence → percentile sur l'**union** des
  distributions, structurel → médiane.
- **Jamais de percentile de percentiles.** Un P99 de P99 n'a pas de sens.
- **Quatre régimes θ, pas trois.** `θ_{drift ∩ anomaly}` existe (déploiement
  défectueux) et c'est celui qui casse les approches naïves.
- **H1 ne se valide jamais sur les données d'entraînement du siamois.**
  Toujours en held-out.
- Le paramètre `k*` des précurseurs se sélectionne sur **validation**, jamais sur
  test.

Les définitions formelles correspondantes sont dans
[`docs/formalisation.md`](docs/formalisation.md).

## Règles de pipeline dataset

- **Ordre strict** : Phase 1 `record` → Phase 2 `build_features` → Phase 3
  `assemble`. Jamais en boucle, jamais de feature de phase 2 réinjectée en
  phase 1.
- **Les dumps de `data/raw*/` sont sacrés** : jamais modifiés sur place,
  toujours réécrits ailleurs. C'est ce qui permet de recalculer tout le dataset
  après un correctif de featurisation sans retourner sur le cluster.
- **`validate_dataset` doit passer** avant d'utiliser un dataset dans une
  expérience, aux trois granularités (épisode / feature-set / dataset assemblé).
- `docs/datasets.md` fait foi sur **quelle version de dataset fait foi**. En cas
  de doute entre STATUS.md et un `results.md`, c'est `datasets.md` qui tranche.

Le mode opératoire complet de la collecte est dans
[`docs/COLLECTE.md`](docs/COLLECTE.md).

## Cluster Kubernetes

Le compte de stage est **namespace-admin sur `ewat`**, pas cluster-admin.

- Toujours `-n ewat` pour toute opération d'écriture `kubectl`.
- Pas de ressource cluster-wide : CRD, ClusterRole, ClusterRoleBinding.
  L'installation de Chaos Mesh en mode cluster-scoped demande un admin.
- Jamais de modification des namespaces système (`kube-system`,
  `cattle-system`, `cattle-*`).
- Jamais de `--force` ni de `--grace-period=0` sans raison explicite.
- Le `kubeconfig` n'est **pas** dans le dépôt (voir `HANDOVER.md` § Accès cluster).

Les permissions d'autonomie accordées à un agent sont détaillées dans
[`agents.md`](agents.md).

## Workflow d'une nouvelle expérience

1. `experiments/<nom>/config.yaml` avec l'override Hydra.
2. Implémenter la logique réutilisable dans `src/ewat/<étape>/`, pas dans le
   script d'expérience.
3. Test unitaire dans `tests/unit/`.
4. Lancer, logger dans MLflow.
5. Résultats avec intervalles de confiance dans `experiments/<nom>/results.md`.

Le **code** de `experiments/` est versionné ; ses **sorties** (JSON, npy, pt,
PNG, runs MLflow) ne le sont pas. Si un `results.md` doit être conservé, ajoutez
une exception explicite dans `.gitignore`, comme pour `ontology_v2` et `rcaeval`.

## Convention de commit

[Conventional Commits](https://www.conventionalcommits.org/) :

```
feat(encoder): ajoute la variante STGAT
fix(drift): la calibration ε_drift ignorait les épisodes masqués
docs: index de la documentation
```

## Ne jamais committer

`data/raw*/`, `data/features/`, `data/datasets/` (16 Go), `*.pt`, `*.npz`,
`*.parquet`, `mlruns/`, les dumps `*.json.gz` (ils contiennent des noms de
nœuds, IP internes, DNS et namespaces), et tout `kubeconfig`.

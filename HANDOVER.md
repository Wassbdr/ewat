# Reprise du projet — EWAT

Porte d'entrée du dépôt. Trois questions dans l'ordre : **qu'est-ce que c'est**,
**comment le faire tourner**, **ce dont vous n'héritez pas**.

| Ensuite | |
|---|---|
| Découpage du code | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Index de la documentation | [`docs/README.md`](docs/README.md) |
| Produire un dataset | [`docs/COLLECTE.md`](docs/COLLECTE.md) |
| Conventions et invariants | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| État courant des résultats | [`STATUS.md`](STATUS.md) |

---

## 1. Ce qu'est ce projet

**EWAT — Early Warning and Anomaly Typing.** Détection précoce et typage
automatique des anomalies dans les architectures microservices Kubernetes.
Stage de recherche Devoteam, Wassim Badraoui.

Le problème : les systèmes de détection d'anomalies confondent les **drifts
bénins** (déploiement, autoscaling, rampe de trafic) avec les **anomalies
réelles**, et noient les équipes sous les faux positifs. EWAT sépare
explicitement ces deux régimes *avant* d'apprendre une ontologie empirique des
types de pannes.

**EWAT n'est pas du RCA.** Le RCA est post-mortem — après la panne, il dit *où*
et *pourquoi*. EWAT est du pré-mortem : il dit *quoi* et *dans combien de temps*,
avant. Le projet frère `matrix_simple` traite l'autre moitié du problème (RCA,
framework STA) ; les deux dépôts sont indépendants.

Trois hypothèses falsifiables structurent le travail :

| | Hypothèse | Critère | Résultat |
|---|---|---|---|
| **H1** | Structurabilité — les embeddings d'anomalies forment des types | silhouette > 0,3 en held-out | ✅ 0,779 ± 0,042 (v5, 5 graines) |
| **H2a** | Séparabilité du drift par look-through MMD² | réduction significative du FPR | ❌ FAIL (p = 0,27) — épisodes trop courts |
| **H3** | Prédictibilité — un précurseur typé bat une baseline générique | AUROC > 0,5 | ✅ 0,927 ± 0,025 (v5) |

**Ne retirez pas H2a des documents.** Un résultat négatif honnête est une
contribution, et celui-ci est documenté avec sa cause.

## 2. Prérequis

| | |
|---|---|
| Python | **≥ 3.11** (le code utilise `datetime.UTC` et les annotations PEP 604) |
| Disque | ~6 Go pour l'environnement virtuel ; ~16 Go de plus pour les datasets si vous les récupérez |
| Calcul | CPU suffisant — l'encodeur tourne en ~30 min, le siamois en ~15 min |
| Optionnel | `kubectl` + accès cluster, **uniquement** pour recollecter des données |

Aucun accès cluster n'est nécessaire pour lire les résultats, faire tourner les
tests, ni rejouer le pipeline sur un dataset existant.

## 3. Installation et vérification

```bash
git clone <url> ewat && cd ewat
python3 -m venv .venv && source .venv/bin/activate
make dev              # pip install -e ".[dev]"
make test             # 773 tests (770 unitaires + 3 d'intégration)
make lint
```

Après `pip install -e .`, les paquets `ewat`, `telemetry`, `graph` et `utils`
sont importables : **`PYTHONPATH=src` n'est pas nécessaire**. Seul le code de
`v5/` s'exécute depuis `v5/` avec `PYTHONPATH=../src`, parce qu'il importe `src/`
en dehors du paquet installé.

```bash
python -m scripts.assemble_dataset --help          # depuis la racine
cd v5 && PYTHONPATH=../src python -m collect.run_campaign --help
```

## 4. Trois parcours

### a. Lire les résultats sans rien exécuter

> Les chiffres du **dataset** `ewat_v5` sont vérifiables dans le dépôt
> (`data/datasets/ewat_v5/{dataset,split}.json`). Ceux des **résultats** v5
> (H1, H3, stress test A1) ne le sont pas : le run a eu lieu sur la VM de
> campagne et ses sorties ne sont pas revenues. Cf. [`STATUS.md`](STATUS.md).

1. [`STATUS.md`](STATUS.md) — tableau de bord : dataset courant, résultats
   multi-graines, bilan des hypothèses.
2. [`docs/results.md`](docs/results.md) — l'interprétation scientifique
   complète, y compris les corrections méthodologiques et les scores d'avant
   correction.
3. [`docs/limitations.md`](docs/limitations.md) — **à lire avant d'extrapoler
   quoi que ce soit.**
4. [`docs/datasets.md`](docs/datasets.md) — quelle version fait foi pour quel
   chiffre. En cas de contradiction entre deux documents, c'est lui qui tranche.

Deux jeux de chiffres coexistent, et il faut savoir lequel citer :

| Jeu | Dataset | Ce qu'il couvre |
|---|---|---|
| **Soutenance** | `ewat_v4_strat` (6 services, 17 features) | Tous les chiffres présentés en soutenance : B2 = 0,920, multiseed phases H/J/K, correctifs de l'audit 2026-06 |
| **v5** | `ewat_v5` (Train Ticket, 41 services, 18 features) | Résultats les plus récents : H1 = 0,779 ± 0,042, H3 = 0,927 ± 0,025, stress test A1 `GENUINE_DYNAMIC` |

`ewat_v4` (split **temporel**, sans `_strat`) est **cassé** — quatre scénarios
absents du train donnaient un AUROC de 0,500 trivial. Il est conservé comme pièce
à conviction ; ne l'utilisez pas.

### b. Rejouer le pipeline sur un dataset existant

Suppose `data/datasets/<nom>` et `data/features/<set>` présents (voir § 5).

```bash
make pipeline          # encodeur → typage siamois → précurseurs → alertes
# équivaut à :
python scripts/run_pipeline.py \
    --dataset data/datasets/ewat_v3 --features-root data/features/v3 \
    --output experiments/thesis_run --seed 42

make figures           # ROC/PR, matrice de confusion, heatmap scénario×cluster
```

Les étapes individuelles et les évaluations complémentaires (H2, H3, ablation,
vérification) sont dans [`STATUS.md`](STATUS.md) § Commandes, et leurs scripts
dans `experiments/`. Protocole détaillé :
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) (v3),
[`docs/evaluation_protocol_v5.md`](docs/evaluation_protocol_v5.md) (v5, figé).

### c. Collecter un nouveau dataset

Tout est dans [`docs/COLLECTE.md`](docs/COLLECTE.md) — prérequis cluster, les
trois phases, la provenance des 18 features, le catalogue des scénarios, les
pièges connus avec leur coût réel, et la marche à suivre pour une campagne v6.
Le runbook opérationnel testé est [`v5/LAUNCH.md`](v5/LAUNCH.md) : suivez-le,
ne le paraphrasez pas.

Ordre de grandeur : ~720 épisodes sur trois runners parallèles ≈ **7 à 9 jours**.

## 5. Ce dont vous n'héritez pas

| Manquant | Pourquoi | Comment l'obtenir |
|---|---|---|
| `data/` (~16 Go) — dumps bruts, features, datasets | Volumineux, et les dumps contiennent des noms de nœuds, IP internes, DNS et namespaces | Recollecter (§ 4c), ou récupérer l'archive auprès de Devoteam |
| `experiments/**` sorties — JSON, npy, checkpoints, runs MLflow | Régénérables | Rejouer les scripts (le **code**, lui, est versionné) |
| Sorties des runs v5 (`experiments/multiseed/phase_v5/`, `experiments/a1_v5/`) | Produites sur la VM de campagne, jamais rapatriées | `python -m experiments.multiseed.run_phase_v5` sur un dataset v5 complet, ou l'archive de la VM |
| `mlruns/`, `release/` | Régénérables | `make pipeline`, `scripts/build_release_v5.py` |
| Accès au cluster `observit-cluster1` | Fin de stage | Demander à Devoteam |
| Manifests Train Ticket amont | Dépendance externe | `git clone https://github.com/FudanSELab/train-ticket` |

### Accès cluster

Le `kubeconfig` **n'est plus dans le dépôt** — il contenait des identifiants du
cluster réel. Il a été déplacé hors du dépôt (poste d'origine :
`~/.kube/ewat-observit-cluster1.yaml`, mode 600). Pour l'utiliser :

```bash
export KUBECONFIG=~/.kube/ewat-observit-cluster1.yaml
kubectl config current-context        # attendu : observit-cluster1
```

Le compte est **namespace-admin sur `ewat`**, pas cluster-admin : pas de CRD ni
de ClusterRole, et toujours `-n ewat` en écriture. L'installation de Chaos Mesh
en mode cluster-scoped demande un admin. Détail dans
[`CONTRIBUTING.md`](CONTRIBUTING.md) § Cluster.

## 6. Ce qu'il faut savoir avant de toucher au code

Trois pièges qui ont réellement coûté du temps sur ce projet.

1. **Les dumps de `data/raw*/` sont immuables.** La featurisation est hors ligne
   et rejouable précisément pour que corriger un bug de features ne coûte pas une
   recollecte de plusieurs jours. Ne modifiez jamais un dump sur place.
2. **Le schéma de features est versionné par épisode**, dans
   `metadata.signal_feature_names`. La constante globale
   `src/telemetry/feature_names.py` porte encore le schéma v4 à 17 features et ne
   fait plus foi pour v5.1.
3. **Une exception `!` dans `.gitignore` sous un répertoire exclu ne marche
   pas.** Git ne descend jamais dans un répertoire exclu. C'est ce qui a laissé
   57 scripts d'expérience hors du dépôt jusqu'en août 2026. Vérifiez toujours
   avec `git add --dry-run`.

Les invariants scientifiques (jamais Granger, jamais de mise à zéro pendant un
drift, quatre régimes θ et pas trois…) sont dans
[`CONTRIBUTING.md`](CONTRIBUTING.md) § Invariants scientifiques. Ce ne sont pas
des préférences de style : chacun correspond à une erreur méthodologique
identifiée.

Les dettes techniques connues et non corrigées sont listées dans
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) § Dettes connues — dont un
`NameError` latent dans `v5/loadgen/runner.py`.

## 7. Suites possibles

Quatre axes d'évolution post-stage, par ordre de maturité :

| Axe | Objet | Point de départ |
|---|---|---|
| **A — Couplage ontologie / prédiction** | L'ontologie (étape 2b) et les précurseurs (étape 3) sont aujourd'hui indépendants. Utiliser les relations causales TE-KSG pour conditionner la prédiction. | `src/ewat/ontology/`, `experiments/ontology_v2/results.md` |
| **B — Précursion robuste** | H3 est validé mais le siamois surapprend en ~3 epochs. Comprendre et corriger cette limitation structurelle. | `experiments/h3_robustness/`, limitation L10 dans `docs/limitations.md` |
| **C — Open-set** | Rejeter les types d'anomalie jamais vus plutôt que de les forcer dans un cluster existant. Mahalanobis et OpenMax sont implémentés mais peu évalués. | `src/ewat/openset/`, `experiments/architecture_v2/openset_eval.py` |
| **D — Déploiement** | Le pipeline tient le budget de latence (p95 = 13 ms contre 5 s de budget) mais n'a jamais été servi en ligne. `matrix_simple` a un stack MLOps complet dont s'inspirer. | `src/ewat/alerts/`, `experiments/bench/latency_e2e.py` |

Deux pistes déjà explorées et **négatives**, à ne pas refaire à l'identique :
le transfert zero-shot vers RCAEval (H3 AUROC = 0,495 — la discrimination de
types n'est pas transférable sans réentraînement,
`experiments/rcaeval/results.md`) et le look-through MMD² pour H2a sur des
épisodes courts.

# Index de la documentation — EWAT

Trois portes d'entrée selon ce que vous cherchez :

| Je veux… | Aller à |
|---|---|
| Reprendre le projet, l'installer, le faire tourner | [`../HANDOVER.md`](../HANDOVER.md) |
| Comprendre le découpage du code | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Produire un dataset | [`COLLECTE.md`](COLLECTE.md) |

Tableau de bord de l'état courant : [`../STATUS.md`](../STATUS.md).
Conventions et invariants : [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

---

## Référence — les documents qui font foi

Ce sont ceux qui tranchent en cas de contradiction avec un autre document.

| Document | Fait foi sur |
|---|---|
| [`datasets.md`](datasets.md) | **Quelle version de dataset fait foi.** Nomenclature v1→v5, statut de chacune, et pourquoi `ewat_v4` est cassé. À consulter avant de citer un chiffre. |
| [`formalisation.md`](formalisation.md) | Définitions mathématiques : G(t), S(t), les 4 régimes θ, les étapes 0→3, H1–H3, plan d'ablation. |
| [`COLLECTE.md`](COLLECTE.md) | Mode opératoire de la collecte : 3 phases, provenance des 18 features, écart v5.1/v5.2, catalogue des scénarios, prérequis cluster, pièges. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Découpage du code, correspondance module ↔ étape ↔ tests, dettes techniques connues. |
| [`evaluation_protocol_v5.md`](evaluation_protocol_v5.md) | Protocole d'évaluation **v5 figé** — celui à appliquer aux travaux post-collecte. |
| [`evaluation_protocol.md`](evaluation_protocol.md) | Protocole d'évaluation v3, utilisé pour la soutenance. |

## Résultats et interprétation

| Document | Contenu |
|---|---|
| [`results.md`](results.md) | **Lecture analytique complète** : chaque résultat, son interprétation scientifique, les corrections méthodologiques appliquées et les scores d'avant correction. Le matériau du rapport de stage. |
| [`limitations.md`](limitations.md) | Limites du travail, leurs causes, et les améliorations envisageables. À lire avant d'extrapoler un résultat. |
| [`cluster_semantics.md`](cluster_semantics.md) | Nommage sémantique des clusters appris : scénario dominant, pureté, features saillantes. Généré par `scripts/build_cluster_semantics.py`. |
| [`point_projet.md`](point_projet.md) | Présentation du projet en prose non technique — utile pour expliquer EWAT à quelqu'un qui n'est ni du domaine ni du code. |

## Historique et traçabilité

| Document | Contenu |
|---|---|
| [`evolution.md`](evolution.md) | Journal chronologique de toutes les itérations, du premier prototype à l'état courant. Témoigne du cheminement, y compris des impasses. |
| [`status_archive.md`](status_archive.md) | Phases historiques détaillées (L, H/J/K, G, F, expériences v3/v4), sorties de `STATUS.md` pour le garder lisible. |
| [`audit_2026_06.md`](audit_2026_06.md) | Audit interne de juin 2026 et ses correctifs (D1–D11). **Archive** : certains chemins qu'il cite décrivent l'arborescence de l'époque et n'existent plus. |
| [`incident_2026-06-06_cluster.md`](incident_2026-06-06_cluster.md) | Post-mortem de l'incident cluster du 6 juin. **Archive.** |
| [`runbook_v4.md`](runbook_v4.md) | Runbook de la campagne v4 (6 services, 17 features). **Archive** — pour v5, voir [`COLLECTE.md`](COLLECTE.md) et [`../v5/LAUNCH.md`](../v5/LAUNCH.md). |
| [`notes/`](notes/) | Notes de travail des premières semaines et PDF de justification (ontologie, modélisation des anomalies). **Archives.** |

> Les documents marqués **Archive** sont conservés pour la traçabilité
> scientifique — ils expliquent *pourquoi* le projet est dans son état actuel.
> Ils ne décrivent pas l'état courant, et les chemins qu'ils citent peuvent
> avoir bougé.

## Rédactions

| Chemin | Contenu |
|---|---|
| [`paper/`](paper/) | Article LaTeX : `main.tex` + `sections/` (00 abstract → 10 conclusion + annexe). |
| [`rapport/`](rapport/) | Rapport de stage LaTeX + slides de soutenance, chiffres consolidés (`chiffres.md`), livrable (`livrable_ewat.md`), delta d'audit. |
| [`formalisation/`](formalisation/) | PDF de la formalisation v2. |

Figures régénérées par `make figures` (`scripts/export_thesis_figures.py`) ;
sorties LaTeX dans `rapport/figures/` et `paper/figures/`.

## Ailleurs dans le dépôt

| Chemin | Contenu |
|---|---|
| [`../v5/README.md`](../v5/README.md) | Infrastructure de collecte Train Ticket : topologie, endpoints, composants. |
| [`../v5/LAUNCH.md`](../v5/LAUNCH.md) | **Runbook opérationnel de campagne** — trois runners parallèles, garde-fou RAM, suivi quotidien. Testé ; à suivre plutôt qu'à paraphraser. |
| [`../v5/PREFLIGHT.md`](../v5/PREFLIGHT.md) | Contrôles avant lancement d'une campagne. |
| [`../mlflow/experiment_registry.md`](../mlflow/experiment_registry.md) | Registre des expériences MLflow. |
| [`../scripts/release_assets/`](../scripts/release_assets/) | Gabarits du kit de publication du dataset (README, DATASHEET, CITATION, LICENSE) assemblés par `scripts/build_release_v5.py`. **Tâche ouverte** : le DATASHEET annonce le schéma v5.1 et `schema.json` doit être régénéré en v5.2 — suivi dans [`../v5/PREFLIGHT.md`](../v5/PREFLIGHT.md) §Packaging. |
| [`../agents.md`](../agents.md) | Permissions d'autonomie accordées à un agent sur ce dépôt. |

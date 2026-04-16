# Synthèse — collecte & construction du dataset EWAT

Date : 2026-04-16 — **aligné sur le pipeline actuel** (Record → Build → Assemble)

## Contexte et objectif

Le projet **EWAT** vise à collecter des données **étiquetées** dans un environnement Kubernetes afin de :
- détecter précocement des anomalies,
- distinguer **drift** et **anomalies** (et le régime **drift ∩ anomalie**),
- constituer un dataset réutilisable pour les expériences sur la formalisation (typage, ontologie, précurseurs).

## Architecture actuelle (trois phases découplées)

| Phase | Rôle | Où ça tourne | Sortie typique |
|-------|------|----------------|----------------|
| **1 — Record** | Orchestration chaos + **dump brut** Prometheus / Jaeger / Loki sur la fenêtre d’un épisode | Cluster + accès observabilité | `data/raw/episode_*/` |
| **2 — Build features** | Grille temporelle uniforme, calcul de \(S(t)\), \(G(t)\), labels à partir des dumps | Machine locale, **sans** cluster | `data/features/<feature_set>/<episode_id>/` |
| **3 — Assemble** | Filtres qualité, cohérence du périmètre de services, **split temporel** train/val/test | Locale | `data/datasets/<name>/` |

Les commandes d’entrée sont documentées dans le **README** racine (`python -m scripts.record_episode`, `build_features`, `assemble_dataset`). La validation opère sur les **artefacts de phase 2 ou 3** (`python -m scripts.validate_dataset`), pas sur du brut seul.

## Périmètre de services (6)

Ensemble **canonique** aligné sur ce qui est réellement observable sur les trois backends (Prometheus, Jaeger, Loki) :  
`frontend`, `recommendation`, `cart`, `ad`, `product-catalog`, `load-generator`.

## Scénarios (régimes)

Référence : `configs/collection.yaml` et `k8s/chaos-mesh/registry.yaml`.

- **θ_drift** (déplacements bénins) : scaling, rolling deploy, changement de config, rampe de trafic.
- **θ_anomalie** : pannes dures, grises, contention (selon les manifests Chaos Mesh / scripts associés).
- **θ_{drift ∩ anomalie}** : scénario de chevauchement (ex. déploiement défectueux).

## Ce qui a été réalisé (historique utile)

Les points ci-dessous restent vrais comme **travail passé**, même si le mécanisme d’exécution a changé :

- stabilisation de l’accès aux backends (port-forward, timeouts, requêtes vides),
- réduction du périmètre de services pour limiter le bruit et les NaN,
- alignement scénarios ↔ cibles mesurées,
- garde-fous qualité (aujourd’hui portés par `validate_dataset` et les métadonnées d’épisode).

## Évolution par rapport à une version antérieure du pipeline

**Ancien modèle (déprécié dans le dépôt)** : une “campagne” collectait des **snapshots** à cadence fixe pendant un run long, avec feature engineering **en ligne** et consolidation de runs.

**Modèle actuel** : un **épisode** = baseline → pré-injection → injection → recovery → cool-down ; la phase 1 ne fait qu’**enregistrer** le brut ; le calcul de \(S(t)\) et \(G(t)\) est **rejouable** hors cluster (phase 2), ce qui réduit le risque de perte de données et permet de retoucher grille d’échantillonnage ou règles sans refaire les injections.

## Difficultés rencontrées (toujours pertinentes)

1. Fragilité des interfaces de lancement et de la résolution des endpoints hors cluster.
2. Traces sensibles aux timeouts et au port-forward (données partielles).
3. Coût temporel si l’on multiplie modalités, scénarios et répétitions.
4. Tension périmètre large vs périmètre stable pour un dataset comparable.
5. Besoin de couvrir explicitement **drift** et **overlap** pour la formalisation (H2, ε_drift, look-through).

## État actuel et suite logique

Le pipeline est structuré en **phases indépendantes** : on peut enregistrer des épisodes, les traiter offline, puis assembler un dataset de référence avec split temporel strict.

Suite logique : accumuler suffisamment d’épisodes par scénario (avec répétitions), valider les feature sets, figer un **dataset** sous `data/datasets/…` pour les expériences EWAT.

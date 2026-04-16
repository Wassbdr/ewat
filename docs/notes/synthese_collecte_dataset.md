# Synthèse — collecte & construction du dataset EWAT

Date : 2026-04-16

## Contexte et objectif

Le projet **EWAT** vise à collecter des données **étiquetées** dans un environnement Kubernetes afin de :
- détecter précocement des anomalies,
- distinguer **drift** et **anomalies**,
- constituer un dataset réutilisable pour les étapes suivantes (typage, ontologie, précurseurs).

Ce document rend compte, à un niveau **synthétique**, de ce qui a été réalisé depuis le début et des principales difficultés rencontrées, **sans détailler** les solutions techniques au niveau des paramètres ou de l’implémentation.

## Fonctionnement de la collecte (vue d’ensemble)

La collecte est organisée en **runs**. Chaque run exécute une série de **scénarios** (avec répétitions) et alterne des phases standardisées (période normale, période d’injection, période de retour à la normale). À intervalles réguliers, un “snapshot” est capturé pour constituer un signal temporel.

À chaque snapshot, plusieurs sources d’observabilité peuvent être agrégées :
- **Métriques** (ex. CPU, mémoire, latences, erreurs),
- **Traces** (ex. interactions entre services, latences par dépendance),
- **Logs** (ex. volume d’erreurs/warnings, signaux sémantiques).

L’objectif est d’obtenir un signal **aligné** dans le temps et **comparable** entre runs, sur un périmètre de services stabilisé.

## Comment le dataset est construit (vue d’ensemble)

À partir des snapshots successifs, le pipeline construit :
- un **signal** \(S(t)\) : matrice temporelle des features par service,
- (optionnellement) une représentation des **relations entre services** (ex. graphe de dépendances), utile pour les étapes orientées graphe,
- des **labels** : associés à chaque fenêtre temporelle/épisode (régime normal vs injection vs recovery, type/catégorie du scénario, services concernés).

Le dataset final est une **consolidation** de plusieurs runs, accompagnée de métadonnées (contexte du run, périmètre, qualité de collecte) et de contrôles de qualité permettant d’écarter les runs non exploitables.

## Ce qui a été réalisé (depuis le début)

### 1) Mise en place et stabilisation de la collecte

- **Outillage de lancement** : stabilisation de la commande de collecte (options/flags cohérents), avec une meilleure traçabilité de ce qui a été activé ou non pendant un run.
- **Fiabilisation “prod-like”** : meilleure robustesse face aux erreurs intermittentes des backends d’observabilité (requêtes lentes, données manquantes, fenêtres vides).
- **Instrumentation de la santé de collecte** : ajout d’indicateurs de run pour distinguer un échec “outil” d’un run réellement inutilisable (ex. collecte partielle de traces).

### 2) Clarification et maîtrise du périmètre observé

- **Périmètre de services maîtrisé** : définition d’un ensemble de services “cibles” stable, aligné sur les scénarios étudiés, pour éviter bruit (trop large) ou manque de signal (trop étroit).
- **Alignement entre scénarios et collecte** : réduction des incohérences entre ce qu’un scénario vise à perturber et ce que la collecte mesure effectivement.

### 3) Structuration des runs et industrialisation de la campagne

- **Orchestration répétable** : exécution des scénarios avec des phases standard (normal → injection → recovery), pour produire des épisodes comparables.
- **Campagnes itératives** : mise en place d’un mode “par vagues” (validation rapide → consolidation → rattrapage), afin de construire un dataset progressivement sans investir d’emblée dans des runs très longs.
- **Validation automatique** : contrôles de qualité automatiques en sortie de run (couverture, cohérence des labels, stabilité du périmètre, présence d’un minimum de signal exploitable).

### 4) Capitalisation et compréhension du codebase

- **Analyse globale** : génération d’une vue “graphe de connaissances” du dépôt pour accélérer la navigation, la compréhension des modules et l’identification des zones critiques liées à la collecte/dataset.

## Difficultés rencontrées (principales)

1) **Fragilité des interfaces de lancement**  
Au début, certaines options de la CLI n’étaient pas prises en compte correctement, ce qui pouvait bloquer un run ou produire un run différent de ce qui était attendu.

2) **Collecte de traces instable (timeouts, fenêtres vides)**  
La collecte de traces s’est révélée sensible aux latences réseau, au port-forward et aux budgets de temps par “tick”, avec des cas fréquents de données partielles.

3) **Contrainte de cadence et coût temporel**  
Plus la collecte est riche (modalités, services, scénarios), plus il est difficile de tenir une cadence régulière et plus la durée totale des campagnes explose.

4) **Périmètre de services difficile à figer**  
Il existe une tension entre : collecter large pour “ne rien rater” et collecter ciblé pour garder un dataset stable, comparable, et exploitable.

5) **Couverture des scénarios et complétude du dataset**  
Obtenir une couverture satisfaisante sur tous les scénarios (avec répétitions et phases) demande une stratégie de campagne, sinon on se retrouve vite avec des trous ou un dataset déséquilibré.

## Comment ces difficultés ont été adressées (vue haute-niveau)

- **Robustesse de l’outillage** : harmonisation du lancement, meilleure propagation des options et traçabilité des runs.
- **Pilotage par indicateurs** : ajout d’indicateurs de santé permettant de qualifier objectivement un run (collecte partielle vs collecté correctement).
- **Réduction de la variabilité** : stabilisation du périmètre “services observés” et alignement explicite avec les scénarios.
- **Approche incrémentale** : campagnes par vagues et validations automatiques pour limiter le risque de “long run” inutile.

## État actuel et suite logique

Le pipeline de collecte est désormais suffisamment structuré pour lancer des campagnes longues de manière **progressive** et **contrôlée**, avec des garde-fous de qualité.

La suite logique consiste à poursuivre les campagnes itératives jusqu’à atteindre une couverture et une stabilité suffisantes, puis à figer un dataset de référence pour les expériences EWAT (séparation drift/anomalie, typage, précurseurs).

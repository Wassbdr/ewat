# EWAT — Évaluation zero-shot sur RCAEval RE2-OB

_Date : 2026-05-12_

## Contexte

Évaluation du transfert zero-shot du pipeline EWAT (entraîné sur ewat_v3) vers le dataset
public **RCAEval RE2-OB** (Zenodo 14590730). L'objectif est de tester si les encodeurs et
classificateurs EWAT généralisent à un nouvel environnement sans réentraînement.

**RCAEval RE2-OB** :
- Application Online Boutique (mêmes 6 services que EWAT : ad, cart, frontend, load-generator,
  product-catalog, recommendation)
- 5 services supplémentaires non couverts (checkout, currency, email, payment, shipping)
- 30 types de pannes × 3 instances = **90 épisodes**
- Durée : ~24 min (48 steps à 30 s), vs. ~10 min (21 steps) pour ewat_v3
- Features : 3/17 structurellement absentes (queue_length, retry_rate, semantic_anomaly)
  → imputées à 0 (NaN résiduel = 31–38%)

## Protocole

1. Application du scaler StandardScaler (entraîné sur ewat_v3 train split)
2. Encodage STGCN → projection SiameseTyper → espace ℝ³² (ewat_v3)
3. Assignation par nearest-centroid aux 10 centroides ewat_v3
4. H1 : silhouette score sur embeddings RCAEval avec labels nearest-centroid
5. H3 : AUROC précurseurs (embedding fenêtre pré-injection vs fenêtre baseline)

## Résultats — comparaison des stratégies de normalisation

| Stratégie | Features | Sil (H1) | H1 | AUROC (H3) | H3 | Distribution |
|---|---|---|---|---|---|---|
| ewat_v3 scaler | 17 | 0.778 | ⚠️ artefact | 0.510 | ✗ | C1 : 99% |
| rcaeval scaler | 17 | 0.234 | ✗ | 0.497 | ✗ | C5/C6 : 89/1 |
| instance norm | 17 | 0.287 | ✗ | 0.507 | ✓ faible | C2/C5 : 58/32 |
| **instance norm** | **M(t) seul** | **0.684** | **✓ PASS** | **0.495** | ✗ | **C2 : 90%** |

### Meilleure configuration : instance normalization + M(t) seul

**H1 silhouette = 0.684 ✓ PASS** — mais résultat nuancé.

Distribution des clusters :

| Cluster ewat_v3 | n épisodes | Interprétation |
|---|---|---|
| **C2 (resource_leak)** | **81** | Cluster "anomalie générique" |
| C5 (rolling_deploy) | 9 | Drift — épisodes limites |

Pureté par type de panne (tous → C2) :

| Fault type | n | Cluster dominant | Pureté |
|---|---|---|---|
| cpu | 15 | C2 | 1.00 |
| delay | 15 | C2 | 1.00 |
| disk | 15 | C2 | 0.80 |
| loss | 15 | C2 | 0.80 |
| mem | 15 | C2 | 0.80 |
| socket | 15 | C2 | 1.00 |

**H3 AUROC = 0.495 ✗ FAIL** — les classificateurs précurseurs ne discriminent pas
la fenêtre pré-injection de la fenêtre baseline.

## Interprétation

**H1 ✓ avec instance+M_only** : l'encodeur STGCN, appliqué avec une normalisation relative
à la baseline intra-épisode et en se limitant aux features métriques, regroupe correctement
les épisodes RCAEval dans le cluster C2 (resource_leak d'ewat_v3). L'encodeur reconnaît
que ces épisodes sont "anomaliques" au sens ewat_v3.

**Mais H1 est un artefact partiel** : la silhouette élevée (0.684) reflète la cohésion
d'un seul cluster (C2), pas une discrimination entre types de panne. Tous les types
(cpu, mem, delay, loss, disk, socket) mappent sur C2 avec pureté 0.80-1.00. L'encodeur
détecte "anomalie sur Online Boutique" mais ne distingue pas les types.

**H3 ✗ systématiquement** : l'AUROC reste ≈ 0.5 quelle que soit la normalisation.
Deux raisons :
1. Les classificateurs précurseurs sont entraînés sur les patterns temporels ewat_v3
   (~21 steps) et ne reconnaissent pas les signatures pré-injection RCAEval (48 steps,
   injection à mi-épisode)
2. La normalisation instance-level efface précisément la déviation pré-injection en
   centrant sur la baseline de l'épisode

## Diagnostic : causes du domain shift

1. **Scaler non transférable** : le StandardScaler est ajusté sur les statistiques d'ewat_v3.
   Les métriques RCAEval ont des distributions différentes (cluster K8s différent, charge
   différente, durée d'épisode ×2.3). Après standardisation, les embeddings RCAEval tombent
   dans une région de l'espace ewat_v3 correspondant à C1 (drift_traffic_ramp).

2. **Longueur d'épisode** : ewat_v3 T≈21 steps vs. RCAEval T=48 steps. La fenêtre de pooling
   temporel du STGCN est calibrée pour des épisodes courts.

3. **Features manquantes** : 3/17 features imputées à 0 (≠ distribution ewat_v3 où elles
   sont mesurées). Ces zéros introduisent un biais systématique dans les embeddings.

4. **Environnement d'injection différent** : les pannes RCAEval sont injectées sur les 11
   services (dont 5 hors scope EWAT), et les signaux propagés aux 6 services EWAT ont
   des signatures différentes de celles entraînées.

## Conclusion scientifique

**Le transfert zero-shot EWAT→RCAEval échoue.** Ce résultat négatif est scientifiquement
utile :

- Il confirme que l'encodeur STGCN capture des caractéristiques spécifiques à l'environnement
  de collecte (ewat_v3), pas des invariants génériques de pannes microservices.
- Le scaler est un point de blocage critique : sans adaptation des statistiques de normalisation,
  les embeddings s'effondrent.
- Le résultat renforce l'argument pour l'ewat_v4 (données propres, mêmes caractéristiques de
  collecte) plutôt que pour une validation externe immédiate.

**Travaux futurs nécessaires pour la généralisation** :
- Adaptation de domaine : fine-tuning du scaler + couche de projection sur quelques épisodes
  RCAEval (few-shot transfer)
- Normalisation instance-level (z-score par épisode au lieu de scaler global) — plus robuste
  au changement d'environnement
- Encodeur contrastif cross-domaine (ewat_v3 + RCAEval en pré-entraînement SimCLR)

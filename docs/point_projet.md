# EWAT — Point de projet

_Rédigé le 2026-05-11_

---

## Contexte

**EWAT** (Early Warning and Anomaly Typing) est un système de détection précoce et de typage automatique des anomalies dans les architectures microservices Kubernetes, réalisé dans le cadre d'un stage chez Devoteam.

**Problème adressé** : les systèmes actuels confondent les drifts bénins (déploiements, autoscaling) avec les vraies anomalies, générant des faux positifs massifs en production. EWAT sépare explicitement ces deux régimes opérationnels, puis apprend une ontologie empirique des types de pannes.

**Ce n'est pas du RCA** (Root Cause Analysis post-mortem). C'est de l'early warning : _Quoi_ et _Dans combien de temps_, avant la panne.

**Signal** : S(t) ∈ ℝ^{N×17} — 17 features par service K8s, combinant métriques Prometheus (×7), traces OpenTelemetry (×6) et logs (×4).

**4 régimes** : θ_normal, θ_drift, θ_anomaly, θ_{drift∩anomaly}.

---

## Ce qui a été fait

### 1. Pipeline complet — 5 étapes implémentées et évaluées

```
S(t) ∈ ℝ^{N×17}
    ↓ Étape 0 : DriftDetector (MMD-RFF, look-through)
    ↓ Étape 1 : STGCNEncoder → embeddings z_e ∈ ℝ^64
    ↓ Étape 2 : SiameseTyper → cluster C_i (K=10 types)
    ↓ Étape 2b : OntologyGraph (relations temporelles + causales TE-KSG + co-occurrence)
    ↓ Étape 3 : PrecursorClassifier → p̂_i(t), k*_i
    ↓ Sortie : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})
```

**302 tests unitaires**, lint propre.

### 2. Dataset ewat_v3

- **300 épisodes** collectés sur cluster Kubernetes réel (observit-cluster1)
- 15 scénarios Chaos Mesh × 20 répétitions
- 299 épisodes feature-isés (`data/features/v3/`), split stratifié 209 / 45 / 45

### 3. Expériences et évaluations

| Expérience | Résultat |
|---|---|
| Calibration drift (ε) | ε = 0.5226, AUC = 0.60 |
| H1 — Structurabilité | ✅ PASS — sil_test = 0.519 ± 0.092 (5 graines) |
| H2a — Look-through MMD² | ❌ FAIL — p = 0.27, épisodes trop courts |
| H2b — Régime θ_{drift∩anomaly} | ⚠️ NUANCÉ — PASS formel mais trivial |
| H3 — Précurseurs | ✅ PASS — AUROC = 0.973 ± 0.012 (5 graines) |
| Ontologie | 22 relations temporelles, 0 causale (100 permutations) |
| Système d'alerte | Seuil 0.7 → 48.5% recall, **8.3% FA**, lead time 2.9 min |

### 4. Rigueur scientifique ajoutée

- **Correction méthodologique** : nearest centroid pour les labels cross-split (vs. fit_predict indépendant) + k* sélectionné sur val (vs. test)
- **Bootstrap 95% CI** sur AUROC, silhouette, taux de détection
- **Multi-graines (5)** : H1 et H3 stables sur toutes les graines
- **Baselines comparatives** :
  - z-score (alerte) : FA = 100% sur tous les drifts bénins → apport EWAT clair
  - B1/B2 (précurseurs) : AUROC légèrement supérieur à EWAT en brut → valeur du STGCN = **structuration latente**, pas discrimination brute
- **Ablation** par modalité et feature : M (métriques) porte l'essentiel ; T et L seuls → silhouette négative
- **Analyse clusters** : NMI = 0.518, pureté = 0.503

---

## Difficultés rencontrées

### Difficultés scientifiques

**H2a FAIL — épisodes trop courts**
Le mécanisme de look-through repose sur une confirmation temporelle post-drift (3–6 steps). Les épisodes v3 (~21 steps) sont trop courts pour que cette confirmation soit significative. Résultat négatif honnête : p = 0.27, FPR non réduit.

**H2b — résultat trivial**
Le DriftDetector (fenêtre 5 steps) se déclenche sur quasiment tous les épisodes, y compris les anomalies pures. L'overlap drift∩alerte est mécaniquement élevé partout → critère ">30%" trop permissif. Même conclusion : les épisodes courts brisent la discrimination.

**Baselines B1/B2 ≥ EWAT sur AUROC brut**
Les features brutes prédisent presque aussi bien les labels EWAT que le STGCN. Cela révèle que la valeur ajoutée du STGCN est dans la **structuration de l'espace latent** (H1) — les clusters sont interprétables et stables — et non dans la prédiction marginale des précurseurs.

**SHAP gradient non validé**
Spearman ρ = −0.34 entre gradient×input et permutation importance → la méthode d'interprétabilité actuelle des fiches n'est pas fiable. Limitation à déclarer dans la publication.

**Ontologie causale vide**
0 relation causale significative sur 100 permutations. Les 2 relations du dry-run (20 perm.) étaient des faux positifs. Cause probable : n_min = 30 épisodes par paire de clusters non atteint avec 299 épisodes et 10 clusters.

### Difficultés techniques

**NaN disk_io (16.7%)**
Le nœud `observit-cluster1-workers-58w74-mwxb2` est NotReady → les métriques disk_io de product-catalog sont absentes. Contourné par imputation dans v3.

**Timeout Loki lors de la collecte v4**
Le timeout HTTP Loki était fixé à 30s. Les épisodes v4 couvrent ~1333s de logs (vs ~600s en v3) → timeout systématique → quality gate `loki-empty` → abort après 3 échecs consécutifs. **Corrigé** : `loki_timeout_s: 30 → 90` dans `configs/collection_v4.yaml`.

**Authentification kubectl expirée**
Le token d'accès au cluster Rancher expire périodiquement. Aucune opération kubectl en dehors du JupyterHub distant.

---

## État actuel

### Pipeline et évaluations

Le pipeline EWAT est **complet et évalué** sur ewat_v3. Toutes les hypothèses ont été testées avec intervalles de confiance et multi-graines.

| Composant | État |
|---|---|
| Code pipeline (`src/ewat/`) | ✅ Complet, 302 tests |
| Dataset ewat_v3 | ✅ 299 épisodes, split 209/45/45 |
| Calibration drift | ✅ ε = 0.5226 |
| Encodeur STGCN | ✅ Entraîné, sil_test = 0.519 |
| Typage siamois | ✅ K = 10 clusters |
| Ontologie | ✅ 22 relations temporelles |
| Précurseurs | ✅ AUROC = 0.973 |
| Système d'alerte | ✅ FA = 8.3% @seuil 0.7 |
| Baselines | ✅ z-score + B0/B1/B2 |
| Multi-graines | ✅ 5 graines |

### Dataset ewat_v4 — en cours de collecte

**Objectif** : corriger les limitations de v3 (épisodes trop courts, disk_io NaN).

| Paramètre | v3 | v4 |
|---|---|---|
| baseline | 5 min | 8 min (16 steps) |
| pre_injection | 1 min | 7 min (14 steps) |
| recovery | 2 min | 5 min (10 steps) |
| Répétitions | 20 | 25 |
| Épisodes totaux | 300 | 375 |
| Durée moy. épisode | ~21 steps | ~45 steps |
| Coût estimé | ~2.4 jours | ~6.4 jours |

**Pourquoi v4 est nécessaire** : les épisodes v3 (~21 steps) sont trop courts pour que le look-through fonctionne (H2a) et pour que les précurseurs opèrent dans leurs conditions nominales (k*=12 steps = 6 min).

**État** : collecte relancée après correction du timeout Loki. Le mécanisme de checkpoint garantit la reprise sans repasser les épisodes déjà validés.

---

## Ce qui reste à faire

### Court terme — publication (sans nouvelle collecte)

| Priorité | Action | Effort estimé |
|---|---|---|
| P1 | **Courbes ROC/PR** pour le système d'alerte (sweep seuil 0.05→0.95, AUC-ROC, AUC-PR) | 0.5 jour |
| P1 | **Matrice de confusion** clusters 10×10 (cluster prédit vs. réel) | 0.5 jour |
| P1 | **Nommage sémantique** des clusters (cluster_semantics.json : nom, scénario dominant, feature dominante) | 0.5 jour |

### Moyen terme — ewat_v4 et architecture

| Priorité | Action | Effort estimé |
|---|---|---|
| M1 | **ewat_v4 → réentraînement complet** : corriger H2a et H2b avec épisodes longs | ~1 semaine CPU |
| M2 | **Ablation rigoureuse** avec réentraînement par condition (~11 conditions × 5 graines) | ~41h CPU |
| M3 | **Pré-entraînement contrastif (SimCLR)** : NT-Xent + augmentations sur épisodes STGCN | 2–3 jours |
| M4 | **GAT vs GCN** : comparaison architectures encodeur (attention apprise vs. adjacence fixe) | 1–2 jours |

### Long terme — perspectives de publication

Cibles : **AIOps@KDD**, **ISSRE**, **QRS**.

Contributions clés :
1. **Formalisation à 4 régimes** θ incluant θ_{drift∩anomaly} — distingue EWAT du RCA et de l'anomaly detection classique
2. **H1 ✅** : clustering non supervisé stable, interprétable, robuste (sil = 0.519 ± 0.092 sur 5 graines)
3. **H3 ✅** : précurseurs typés prédictibles 1–5 min avant la panne (AUROC = 0.973 ± 0.012)
4. **Apport vs. baseline** : z-score FA = 100% sur drifts ; EWAT FA = 8.3% — contribution opérationnelle claire
5. **H2a ❌ comme résultat négatif honnête** : épisodes courts insuffisants pour confirmation temporelle — ewat_v4 valide le mécanisme dans ses conditions nominales

---

## Synthèse en une phrase

Le pipeline EWAT est complet et évalué, avec des résultats solides sur H1 et H3 (précurseurs typés, 8.3% FA). La principale limitation — les épisodes v3 trop courts pour valider H2 — est en cours de correction avec ewat_v4.

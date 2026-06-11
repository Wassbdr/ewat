# Protocole d'évaluation EWAT v5 (Train Ticket) — FIGÉ AVANT LES DONNÉES

_Rédigé le 2026-06-11, pendant la collecte (audit 2026-06). Ce protocole est
figé avant l'arrivée des données pour exclure toute adaptation a posteriori
du protocole aux résultats. Tout écart sera documenté comme amendement daté._

## 0. Décisions structurantes (actées, audit 2026-06)

**T1 — Chaîne prédictive sans STGCN.** La chaîne prédictive officielle v5 est
`S(t) → instance norm → LR-OvR` (C1 : STGCN 0.863 < B2 0.920 ; A5 : IC paired
de Δ contient 0). Le STGCN est évalué uniquement comme module de
*clustering/ontologie* (H1, fiches, TE) — il n'entre plus dans les chiffres
prédictifs headline.

**T2 — Cible primaire = labels indépendants.** Toutes les métriques headline
sont calculées contre les scénarios Chaos Mesh et les bugs F (vérité terrain
indépendante). Les évaluations sur labels EWAT (auto-référentes, L9) sont
reléguées en diagnostic interne du clustering, clairement étiquetées
« circulaire ».

**T3 — K acté.** K = 10 fixe (`cluster_embeddings(fixed_k=10)`) pour toute
comparaison multi-graines ; HDBSCAN (`k_selection_method="hdbscan"`) en
diagnostic de sensibilité. Plus aucune sélection argmax-silhouette dans un
résultat headline (Phase K : K ∈ [9, 15] instable, accord Tibshirani 4/10).

## 1. Splits

- Assemblage : `scripts/assemble_dataset` (stratifié par défaut, D4),
  held-out batch A (3 chaos) + batch B (bugs F) routés **test-only**
  automatiquement via `held_out_flag` (D5).
- **k-fold scénario × répétition (E5)** : 5 folds stratifiés par scénario sur
  les 19 scénarios d'entraînement (`--split-mode shuffled --split-seed 0..4`),
  en PLUS du split temporel officiel. Le headline est reporté sur le split
  temporel ; la variance inter-fold accompagne systématiquement le chiffre.
- Aucun épisode held-out dans aucun fold d'entraînement (vérifié par
  `validate_v5 --dataset` : fuite = échec de la porte).

## 2. Métriques — systématiques pour tout résultat

| Métrique | Obligatoire | Référence |
|---|---|---|
| macro-AUROC + IC bootstrap (≥ 1000, BCa) | oui | protocole B2 |
| **macro-PR-AUC + détail par scénario** | oui (E3) | M-6 : la PR-AUC démasque ce que l'AUROC lisse (B2 v4 : AUROC 0.920 / PR-AUC 0.587) |
| **Calibration : Brier + ECE (10 bins) + reliability** | oui (E2) | tout seuil opérationnel exige ECE < 0.10 ou recalibration isotonique fittée sur val |
| Filtre reportable n_pos ≥ 5 | oui (E8) | les clusters sous le seuil sont « non concluants » |
| Stress tests A1–A5 | oui | [evaluation_protocol.md](evaluation_protocol.md) §Stress tests |
| Robustesse NaN (courbe AUROC vs taux injecté) | oui (E9) | protocole `experiments/audit2026/nan_robustness` |

Tests statistiques : cadre unifié d'[evaluation_protocol.md](evaluation_protocol.md)
§Cadre des tests statistiques (McNemar/Wilcoxon/Fisher/BCa/permutation,
correction Holm ou BH).

## 3. Drift (étape 0) — dernière chance, critères de sortie

ε_drift v3 (0.5226) est mort : recalibrations v4_strat → AUC 0.315 (fenêtres
10/10) et 0.284 (5/5), pires que le hasard. Sur v5 (T ≈ 120) :

1. Recalibrer via `experiments/drift_separation/calibrate` avec fenêtres
   {5, 10, 20} — l'ε retenu va dans `drift_calibration.json` (jamais en dur :
   M1, l'AlertAssembler refuse un détecteur sans calibration).
2. `sigma_policy="keep"` (M2) partout.
3. **Critère de sortie** : si AUC < 0.65 sur les trois tailles de fenêtres,
   l'étape 0 est officiellement abandonnée comme mécanisme de *séparation*
   drift/anomalie (H2a falsifiée trois fois : v3, v4_strat, v5) et le
   DriftDetector est requalifié en simple annotateur de changement de régime.

## 4. Open-set (résultat de premier rang v5)

Les held-out batch A (chaos inédits) + batch B (bugs réels F) sont la valeur
scientifique principale de v5. Protocole comparatif M13 :

- Méthodes : OpenMax (`tail_size_ratio=0.3`, plus le tail fixe 20),
  Mahalanobis (`ewat.openset.mahalanobis`), energy-based (−logsumexp des
  logits LR).
- Évaluation : unknown-AUROC (held-out vs test connu), top-1 unknown rate,
  dégradation closed-set (< 2 pp tolérés) — protocole C3 inchangé, 15 folds.
- Headline open-set : la méthode gagnante sur unknown-AUROC, avec IC.

## 5. Typage / clustering (valeur géométrique du STGCN)

- H1 : silhouette nearest-centroid val/test, **avec modèle nul** (M9 :
  200 permutations, Δ vs null + p empirique) et IC bootstrap.
- Checkpoint siamois sélectionné sur silhouette val (M6). Le constat v4
  (best_epoch = 1 : le contrastif n'améliore pas la géométrie héritée de
  l'encodeur) est à re-tester sur v5 — si reproduit, le siamois passe en
  composant optionnel.
- Encodeur : self-loops actifs (M4), d_feat=18 résolu depuis metadata (D1).

## 6. Précurseurs / alertes

- k* parcimonieux (M12, tol 0.02) ; PR-AUC + AUROC par cluster.
- **Aucune table seuil/FA/lead sans recalibration isotonique préalable**
  (E2 v4 : ECE précurseurs 0.120 ; le point « seuil 0.7 » historique ne se
  transfère pas).
- Lead-time toujours avec IC bootstrap (E4) et courbe lead vs FA.

## 7. Baselines

- **USAD** (KDD 2020, `ewat.baselines.usad`) : volet détection (score
  d'anomalie, fenêtres normales vs injection) + volet typage (LR sur latents)
  — protocole `experiments/sota/usad_eval`, featurisation identique à B2.
- z-score max (naïve), B0 aléatoire.
- Extension souhaitable post-stage : TranAD (ROADMAP axe B).

## 8. Couverture et qualité (avant toute analyse)

- `validate_v5 --features-root` : 100 % `[OK]` requis.
- **Rapport de couverture service × scénario (D10)** : nombre d'épisodes où
  chaque service est cible ; les services jamais ciblés sont listés dans le
  datasheet du dataset (limite de typage explicite).
- `assemble_dataset` → `validate_v5 --dataset` (fuite held-out = bloquant).

## 9. Ordre d'exécution

1. Build (`build_features_v5 --raw-root`) → validate → assemble → validate dataset.
2. Couverture D10 + datasheet.
3. B2 v5 (headline + PR-AUC + calibration + IC) sur split temporel + 5 folds.
4. USAD + z-score (mêmes splits).
5. Drift §3 (critère de sortie).
6. Typage H1 + fiches (STGCN, K=10) ; multiseed 10 graines si budget.
7. Open-set §4 sur held-out A/B.
8. Stress tests A1–A5 + robustesse NaN.
9. STATUS/results/limitations mis à jour ; chiffres figés pour le rapport.

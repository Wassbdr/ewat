# Protocole d'évaluation EWAT (ewat_v3)

Document de référence unique pour la soutenance. Complète [STATUS.md](../STATUS.md) (tableau de bord) et [results.md](results.md) (interprétation).

## Jeux de données

| Split | Épisodes | Usage |
|-------|----------|--------|
| Train | 209 | Entraînement encodeur, siamois, centroides, classifieurs précurseurs |
| Val | 45 | Sélection de K (silhouette), k* précurseurs, calibration intermédiaire |
| Test | 45 | **Toutes** les métriques rapportées (H1, H2, H3, alertes) |

Split **stratifié** par scénario Chaos Mesh (15 scénarios × ~3 épisodes test chacun). Un seul split fixe (seed assemble) — variance liée au split non quantifiée par k-fold (limitation L2.5).

## H1 — Structurabilité

1. Entraîner STGCN + SiameseTyper sur train.
2. Clustering hiérarchique sur embeddings **train** ; fixer K par max silhouette sur **val**.
3. Calculer centroïdes sur **train**.
4. Assigner val/test par **plus proche centroïde** (jamais `fit_predict` indépendant sur val/test).
5. Métrique : silhouette sur embeddings test, seuil PASS = **0.3** (Kaufman & Rousseeuw).
6. Robustesse : 5 graines `[42, 123, 456, 789, 1337]` — script `experiments/verification/verify_h1_h3.py`.

## H2a — Séparabilité drift (look-through MMD²)

1. DriftDetector calibré sur train (ε_drift = 0.5226, Youden).
2. Simulation **streaming** pas à pas sur épisodes **test** (45).
3. Comparer FPR sur épisodes **anomalie** (look-through vs seuil simple).
4. Test : Student unilatéral apparié ; seuil PASS : p < 0.05 et FPR_lt < FPR_baseline.
5. Résultat v3 : **FAIL** (p = 0.27) — épisodes ~21 steps insuffisants pour warm-up + confirmation.

## H2b — Régime θ_{drift∩anomaly}

1. Critère formel : overlap(drift flag, alerte) > 30 % par cluster.
2. Critère strict : Fisher exact C8 vs clusters drift purs (C5, C6, C9) — script `experiments/h2_overlap/eval_strict.py`.
3. Résultat v3 : PASS formel mais **trivial** (DD sensible sur épisodes courts) ; Fisher p = 0.35.
4. Timing : alerte précurseur précède le drift flag dans 85–100 % des cas (early warning, pas indicateur tardif).

## H3 — Prédictibilité des précurseurs

1. Un classifieur LR one-vs-rest par cluster C_i.
2. Pour chaque k ∈ {2, 4, 6, 8, 10, 12} : AUROC sur val et test.
3. **k*_i = argmax_k AUROC_i(k) sur val** ; rapport AUROC sur **test** à k*_i.
4. PASS si ≥ 1 type a AUROC test > 0.5 (baseline aléatoire).
5. Intervalles : bootstrap BCa, n = 1000 sur épisodes test.

## Système d'alerte (AlertAssembler)

1. Simulation en ligne timestep par timestep sur test (anomalie + drift).
2. Seuils p̂ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}.
3. Métriques : détection pré-injection, cluster correct, FA sur drifts bénins, lead time.
4. Courbes ROC/PR : sweep fin (0.05–0.95) — `experiments/alerts/eval.py --roc-sweep`.
5. Point opérationnel recommandé : **seuil 0.7** (FA drift 8.3 %, lead ~3 min).

## Ablations

| Type | Script | Interprétation |
|------|--------|----------------|
| Modalités (H1) | `experiments/ablation/run_retrain.py` | Réentraînement complet par condition |
| Précurseurs (H3) | `experiments/ablation/eval_precursor_h3.py` | Masquage à l'inférence sur classifieurs pré-entraînés |
| Features | `experiments/ablation/run.py` | Leave-one-out sur modèle full |

**Message clé** : M_only bat full en silhouette (H1) ; full bat M_only en AUROC précurseur (H3) — la géométrie latente et la prédictibilité ne dépendent pas des mêmes modalités.

## Reproduction

```bash
python scripts/run_pipeline.py \
  --dataset data/datasets/ewat_v3 \
  --features-root data/features/v3 \
  --output experiments/thesis_run --seed 42

python -m scripts.export_thesis_figures
```

Voir [README.md](../README.md) § Reproduction soutenance pour la séquence complète d'évaluations.

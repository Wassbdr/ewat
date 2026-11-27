# Résultats historiques (données internes)

> **Résultats historiques sur données internes.** Ils ont été obtenus pendant le stage sur un cluster
> Kubernetes interne (dataset `ewat_v3`). Ces données ne sont pas publiques et ne sont pas dans le
> dépôt : ces chiffres ne peuvent pas être rejoués depuis un clone. Les résultats reproductibles
> (données synthétiques, `ewat benchmark`) sont dans le [README](../README.md#résultats-clés).

| Hypothèse | Résultat | Source |
|---|---|---|
| H1 structurabilité des types | silhouette test 0,414 (seuil 0,3), PASS | [`results.md`](results.md) § 4 |
| H2 drift vs panne par look-through | FPR 0,67 avec look-through contre 0,73 au seuil simple, p = 0,27, FAIL (épisodes de ~21 pas) | [`results.md`](results.md) § 2 |
| H3 précurseurs typés | AUROC test moyen 0,952 sur 8 types sur 10, PASS | [`results.md`](results.md) § 6 |

Commande d'origine (nécessite `data/datasets/ewat_v3`, absent du dépôt) :
`python scripts/run_pipeline.py --dataset data/datasets/ewat_v3 --features-root data/features/v3 --output experiments/thesis_run --seed 42`.

Sur des épisodes synthétiques de ~110 pas, le look-through qui échouait en H2 sépare drift et panne :
cela va dans le sens de la limite L1 de [`limitations.md`](limitations.md) (épisodes trop courts),
sans le prouver sur données réelles.

# EWAT — agents.md

## Autonomie totale (sans confirmation)

- `kubectl get/describe/logs/top` sur tout namespace
- `kubectl apply/delete` dans le namespace `ewat` uniquement (toujours `-n ewat`)
- Exécution des scripts `scripts/` en local (record, build, assemble, validate)
- Lecture/écriture dans `src/`, `configs/`, `experiments/`, `tests/`, `data/`
- Commits locaux sur une branche de travail

## Confirmation requise

- `kubectl apply` de manifests non revus par l'utilisateur
- Injection de chaos (ChaosMesh) — même en namespace `ewat`
- Tout `kubectl delete` sauf ConfigMaps/Secrets temporaires préfixés `ewat-tmp-`
- Push git sur `main`
- Modification de `configs/default.yaml` (contient les endpoints de production)
- Lancement d'un `record_episode` long (> 10 min) sur le cluster réel

## Interdit / Jamais

- Ressources cluster-wide : CRDs, ClusterRoles, ClusterRoleBindings
- Namespaces système : `kube-system`, `cattle-system`, `cattle-monitoring-system`, `cattle-logging-system`
- `kubectl --force` ou `--grace-period=0` sans demande explicite
- Modifier les règles impératives de `CLAUDE.md` ou `agents.md`

## Règles de pipeline

- Ordre strict : **Phase 1** (record) → **Phase 2** (build_features) → **Phase 3** (assemble)
- Ne jamais utiliser des features de Phase 2 comme entrée de Phase 1
- Les dumps `data/raw/` sont sacrés — jamais de modification in-place, toujours écriture dans un nouveau dossier
- Valider avec `validate_dataset.py` avant toute utilisation d'un dataset dans une expérience

## Workflow recommandé pour une nouvelle expérience

1. Créer `experiments/<nom>/config.yaml` avec override Hydra
2. Implémenter dans `src/ewat/<étape>/`
3. Test unitaire dans `tests/unit/`
4. Lancer l'expérience, logguer dans MLflow
5. Résultats avec intervalles de confiance dans `experiments/<nom>/results.md`

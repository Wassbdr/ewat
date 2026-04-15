# EWAT — Early Warning and Anomaly Typing

Stage de recherche Devoteam — Wassim Badraoui

Détection précoce et typage automatique des anomalies dans les architectures microservices Kubernetes.

## Cluster

- **observit-cluster1** (Rancher, RKE2 v1.32.7)
- 9 nœuds (8 Ready, 1 NotReady)
- Observabilité : Prometheus + Grafana + OTel Collector (déjà en place)
- Accès : namespace-admin sur `ewat`

## Hypothèses

- **H1** Structurabilité : silhouette > 0.3 en held-out
- **H2** Séparabilité : réduction FPR significative (p < 0.05)
- **H3** Prédictibilité : AUROC typé > baseline générique

## Exécution locale de la collecte

Quand vous lancez la collecte depuis votre machine (hors cluster), les noms
`*.svc.cluster.local` ne sont pas résolus. Utilisez des port-forwards et le
mode d'endpoint local.

1. Ouvrir les forwards:
	- `kubectl -n monitoring-metrics port-forward svc/prometheus-server 19090:80`
	- `kubectl -n rca-sandbox port-forward svc/rca-jaeger 16686:16686`
	- `kubectl -n monitoring-logs port-forward svc/loki-gateway 13100:80`
2. Lancer la collecte:
	- `python scripts/collect_labeled.py --config configs/collection.yaml --base-config configs/default.yaml --endpoint-mode local-portforward`
3. Valider la run en mode strict (data réelle):
	- `python scripts/validate_dataset.py data/raw/<run_id>`

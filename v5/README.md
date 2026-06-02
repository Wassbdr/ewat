# EWAT v5 — Infrastructure de collecte Train Ticket

État au 2026-06-01. Voir `docs/dataset_v5_plan.md` §0 pour le contexte complet.

## Topologie déployée

- **Train Ticket** dans le namespace `tt` (manifests `k8s-with-jaeger`, 64 deployments,
  41 services + 22 DB + Jaeger). Conforme au schéma FudanSELab.
- UI : `http://172.16.203.12:32677` — Jaeger : `http://172.16.203.12:32688`
- Fixes version : `mongo:4.4`, `jaegertracing/all-in-one:1.53`, service `jaeger` en ClusterIP stable.

## Sources de télémétrie S(t) (endpoints confirmés)

| Source | S(t) | Endpoint |
|---|---|---|
| Prometheus (cAdvisor) | M(t) | `monitoring/monitoring-kube-prometheus-prometheus:9090` |
| Jaeger | T(t) | `tt/jaeger-query:16686` (NodePort 32688) |
| Loki (via promtail) | L(t) | `monitoring-metrics/loki:3100` |

Jeu de features : **Lean enrichi, 17 features** (cf. `docs/dataset_v5_plan.md` §0.5).

## Composants v5

### `loadgen/` — générateur de charge
Fork vendorisé de `train-ticket-auto-query` (patché : login sans CAPTCHA).
```bash
# charge nominale (mix pondéré de scénarios métier)
python -m loadgen.runner --address http://172.16.203.12:32677 --users 12 --duration 600
# charge ciblée (1 scénario, pour injection bug)
python -m loadgen.runner --address ... --scenario query_and_cancel --users 20
```
Contrainte : **faible concurrence** (cluster partagé). 12 users = baseline stable, 0 erreur.

### `chaos/` — injection de pannes
Catalogue `catalog.yaml` : **22 scénarios validés** (15 mono + 4 compo + 3 held-out)
+ 5 bugs réels (batch B, swap d'image).
```bash
python -m chaos.inject list
python -m chaos.inject apply cpu_stress --intensity high --duration 600s
python -m chaos.inject delete cpu_stress
python -m chaos.inject apply-bug F1        # swap image fautive (F1 prêt)
```
Intensités `low/med/high` pour le ramp-up. Types : StressChaos, NetworkChaos,
PodChaos, DNSChaos, TimeChaos, IOChaos.

**Bugs F** : seul F1 a une image pré-buildée (`kylinxiang/ts-voucher-service:f1.1`).
F3/F5/F7/F12 à builder depuis les branches `ts-error-*-Fxx` (maven+docker).

### `collect/` — collecte + featurisation
- `probe.py` : pull les 3 sources pour `tt` sur une fenêtre → dumps gzip.
  Prometheus = `monitoring-metrics/prometheus-server` (cAdvisor **et**
  kube-state-metrics → restart_count). Loki paginé par tranches de `step` s
  (évite le plafond 5000 lignes).
- `build_features_tt.py` : dumps → S(t) ∈ ℝ^{41×T×17} (jeu Lean enrichi).
- `run_episode.py` : **orchestrateur** — charge continue + baseline → injection
  → recovery + collecte + features + labels régime, en un appel.

```bash
# un épisode complet (chaos)
python -m collect.run_episode --scenario cpu_stress --out data/raw_v5/ep_001 \
    --baseline 180 --injection 180 --recovery 120 --users 12 --intensity high
# un épisode bug F (swap image)
python -m collect.run_episode --scenario F1 --bug --out data/raw_v5/ep_F1_001 ...
# sortie : prometheus/jaeger/loki.json.gz + features.npz (S, services, regime…) + meta.json
```

### Protocole de collecte massive (boucle)
Pour chaque scénario × répétition : `run_episode` produit un dossier autonome.
Anatomie cible plan §4 : T=120 steps (60 min) — baseline 16 / pre 30 / ramp 10 /
inj 50 / recovery 14. Pour le pilote on utilise des durées réduites.

## Ressources cluster

Contrainte CPU résolue sans demande admin : `ewat/load-generator` et `rca-sandbox`
(benchmark synthétique) scalés à 0 → ~31 vCPU libres. Chaos Mesh control-plane
restauré dans `rca-sandbox` (requis). Voir mémoire `project_v5_tt_deployed`.

## Reste à faire avant collecte massive

1. **Phase 2 `build_features` pour TT** : N=41 + extraction des 17 features depuis
   les dumps (cAdvisor → M, Jaeger spans → T, Loki JSON → L). Le code v4 est
   hardcodé 6 services OB → réécriture ciblée nécessaire.
2. **Orchestrateur d'épisode** : enchaîner baseline → ramp → injection → recovery
   avec la charge + le chaos + la collecte (anatomie §4 du plan, T=120 steps).
3. **Builder F3/F5/F7/F12** (optionnel, batch B peut démarrer avec F1 seul).

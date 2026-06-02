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

### Workflow 2-phases : COLLECTE puis BUILD (Record → Build → Assemble)

La collecte et le build sont **séparés** (les dumps `data/raw_v5/` sont sacrés) :
la boucle de collecte est rapide (~50 min/ép, pas de build), le build tourne
**offline, en parallèle, rejouable** (ré-exécutable si le schéma S(t) change, sans
recollecter — crucial).

**Phase 1 — collecte** (run_episode) : phases + pull 3 sources → dumps gzip +
`episode_meta.json` (boundaries + meta). Gate qualité brut (traces/logs/prom > 0).
```bash
python -m collect.run_episode --scenario cpu_stress --out data/raw_v5/ep_001
python -m collect.run_episode --scenario F1 --bug --held-out --out data/raw_v5/ep_F1_001
# sortie : prometheus/jaeger/loki.json.gz + episode_meta.json  (PAS de features)
```

**Phase 2 — build offline** (build_features_v5) : dumps → contrat v4 complet
(signal/mask/adjacency/labels/metadata). Batch parallèle, idempotent.
```bash
python -m collect.build_features_v5 --raw-root data/raw_v5 --workers 4
python -m collect.build_features_v5 --episode data/raw_v5/ep_001 --force   # un seul
```

**Phase 3 — validation + assemblage**
```bash
python scripts/validate_v5.py --features-root data/raw_v5      # gate features
python -m scripts.assemble_dataset --features-root data/raw_v5 --output data/datasets/ewat_v5 --stratified
python scripts/enforce_heldout_v5.py --dataset data/datasets/ewat_v5   # held-out → test only
python scripts/validate_v5.py --dataset data/datasets/ewat_v5  # check fuite held-out
```

### Collecte massive
`run_campaign` orchestre la Phase 1 (collecte uniquement, gate brut, checkpoint,
reset). Le build (Phase 2) se lance après/en parallèle sur le poste de travail.
Anatomie épisode : 60 steps × 30 s = 30 min (baseline12/pre14/ramp6/inj20/rec8 ;
override test via env `V5_PHASES="b,pre,ramp,inj,rec"`).

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

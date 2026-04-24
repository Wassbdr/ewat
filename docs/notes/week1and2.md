# Synthèse collecte et construction du dataset EWAT

Date : 2026-04-22 (dernière mise à jour)
Auteur : Wassim Badraoui — stage Devoteam

## 1) Objectif

Construire un dataset étiqueté pour EWAT, couvrant les quatre régimes
`θ ∈ {normal, drift, anomaly, drift∩anomaly}`, utilisable pour :

1. calibrer le filtre de drift (étape 0, MMD-RFF),
2. apprendre le typage contrastif (étape 2),
3. ajuster les précurseurs typés k*_i (étape 3),
4. falsifier les trois hypothèses H1/H2/H3.

Ce document résume **ce qui a été fait depuis le début du stage**, en retraçant
l'évolution du pipeline, les blocages rencontrés et les solutions adoptées.

## 2) Évolution du pipeline (du prototype unique au découplage 3-phases)

### 2.1 V0 — collecteur monolithique (abandonné)

- Un unique `scripts/collect_labeled.py` et `scripts/snapshot_collector.py`
  interrogeaient Prometheus, Jaeger, Loki à chaque tick et calculaient
  directement les features S(t).
- Problème : toute erreur de feature engineering (mauvaise agrégation, feature
  manquante, fenêtre mal choisie) obligeait à **relancer la collecte entière**
  sur le cluster, qui est coûteuse et non reproductible (le cluster évolue).

### 2.2 V1 — découplage Record → Build → Assemble (actuel)

Refactor majeur : les deux scripts monolithiques ont été supprimés au profit
de trois phases indépendantes.

```
Phase 1  record_episode         (en ligne, touche au cluster)
Phase 2  build_features          (hors ligne, rejouable à volonté)
Phase 3  assemble_dataset        (hors ligne, split temporel)
```

Conséquences concrètes :

- Les dumps bruts (`data/raw/*.json.gz`) sont le **ground truth** intangible ;
  toute itération sur les features se fait offline.
- Les extracteurs fichier (`src/telemetry/extractors/`) réutilisent la même
  logique que les collecteurs online, garantissant la cohérence.
- Un nouveau module `src/telemetry/recorder.py` centralise l'appel Prometheus
  range / Jaeger /api/traces / Loki /query_range avec retries courts.

### 2.3 Stabilisation CLI et périmètre services

- Flag `--no-traces` réparé et **propagé** jusqu'au `SignalBuilder` (avant,
  il était accepté mais ignoré). État `traces_enabled` persisté dans le
  metadata du run.
- `collection.canonical_services` : ensemble V figé pour tous les épisodes.
  Après itérations (11 → 9 → 6 services), le périmètre final est **6 services**
  effectivement observables sur les trois modalités : `frontend`,
  `recommendation`, `cart`, `ad`, `product-catalog`, `load-generator`.
  Justification : inclure un service invisible sur une modalité dégrade
  l'agrégation intra-service et introduit des NaN systémiques.

### 2.4 Fiabilisation de la collecte traces (Jaeger)

- Paramétrage via config : `request_timeout_s`, `fetch_total_timeout_s`,
  `max_parallel`, `limit_per_service`, `trace_window_s`, `span_cache_ttl_s`.
- Le backend Jaeger expose des stats (services considérés, timeouts, erreurs,
  elapsed) ; `traces_empty_window_ratio` remonté dans le `metadata.json`.
- Fix BFS : ajout d'un `visited set` pour éviter les boucles infinies sur
  spans cycliques.

### 2.5 Chaos Mesh — complétion du registry

- Scripts bash autonomes pour les drifts bénins : `drift_scale_up.sh`,
  `drift_rolling_deploy.sh`, `drift_config_change.sh`, `drift_traffic_ramp.sh`,
  `faulty_deploy_overlap.sh`.
- Manifests YAML pour hard / gray / contention / systemic.
- `k8s/chaos-mesh/registry.yaml` consolide la liste des 14 scénarios actifs.
- `scripts/chaos_injector.py` : timeouts `apply`=60 s / `delete`=30 s + fallback
  `--ignore-not-found` pour éviter qu'un delete bloqué ne fige la campagne.

## 3) Orchestration d'un épisode

Chaque épisode suit la même timeline :

```
baseline (5m) → pre-injection (1m) → INJECTION (scenario.duration)
              → recovery (2m) → cool-down (1m, hors dump)
```

Après la fin de la fenêtre de recovery, le script **dumpe** le range complet
`[t_start, t_end]` pour les trois backends et écrit un `manifest.json`
enregistrant, par modalité : chemin du gzip, sha256, taille, elapsed, erreurs
par service, `queries_ok`, compteurs (`n_traces_total`, `n_lines`).

## 4) Robustesse (ajouts récents — avril 2026)

Ces ajouts sont la réponse aux premières campagnes réelles, qui ont révélé
deux modes d'échec : la dégradation des port-forwards SPDY et les crashes
ponctuels du runner.

### 4.1 Trois modes d'accès aux backends

Au lieu de forcer un seul mode, `--endpoint-mode` expose :

- `nodeport` — accès direct via NodePort (Prometheus, Jaeger, Loki) sur un
  worker Ready. **Mode recommandé** pour toute campagne > 1 h car il évite
  totalement le SPDY.
- `local-portforward` — consomme des forwards ouverts manuellement.
- `in-cluster` — exécution depuis un pod du namespace `ewat` (DNS interne).

### 4.2 Port-forwards gérés par le script (`--manage-port-forwards`)

Une classe `_PortForward` ouvre un tunnel **frais** avant chaque dump
d'épisode puis le ferme. Tueuse d'orphelins (`lsof`), attente de readiness
TCP (`socket.create_connection`), refresh explicite de la session HTTP côté
`TelemetryRecorder` pour purger le pool de sockets morts. Palliatif tant
qu'on n'a pas les NodePorts.

### 4.3 Checkpoint idempotent

- `checkpoint.jsonl` (append-only, fsync) liste les couples
  `(scenario, rep)` déjà dumpés **et** validés.
- Le matching se fait sur `(scenario, rep)`, pas sur `episode_id` (qui
  contient un timestamp donc changerait à chaque relance).
- Permet la reprise propre après SIGINT, panne réseau, OOM de la VM.

### 4.4 Graceful shutdown

Handlers SIGINT/SIGTERM qui **ne coupent pas** brutalement : le flag
`_shutdown_requested` est testé aux points sûrs du loop, l'épisode en cours
termine (delete Chaos Mesh + dump) avant exit. Évite de laisser un
`PodChaos` ou un `NetworkChaos` actif sur le cluster.

### 4.5 Quality gate post-dump

Contrôle immédiat de chaque manifest :
- Prometheus : `queries_ok` non vide,
- Jaeger : `n_traces_total > 0`,
- Loki : `n_lines > 0`.

Un épisode qui échoue reçoit un marker `.quality_failed`, n'est **pas**
checkpointé, et sera rejoué au run suivant. Le contrôle de campagne se fait
via `scripts/validate_raw.py` (nouveau) qui lit uniquement les manifests —
rapide même sur 300 épisodes.

### 4.6 Reproductibilité

`src/utils/seeding.py` exporte désormais `PYTHONHASHSEED`, en plus du seeding
`random`/`numpy`/`torch`, pour stabiliser l'ordre d'itération des dicts/sets
dans le pipeline de features.

## 5) Difficultés rencontrées (chronologique)

1. **`--no-traces` ignoré** → CLI réparée, flag propagé jusqu'à `SignalBuilder`.
2. **Timeouts Jaeger** via port-forward local (ReadTimeout, budget exhausted)
   → paramétrage fin + stats dans le metadata.
3. **Dérive de cadence** quand la collecte traces dépassait le tick
   → `fetch_total_timeout_s < sample_interval_s` imposé.
4. **Périmètre services incohérent** (services cibles absents de la
   télémétrie) → 11 → 9 → 6 canonical services.
5. **Durée totale** trop longue pour couvrir 14 scénarios × N reps
   → campagne A/B/C (validation rapide, consolidation stricte, rattrapage).
6. **Dégradation SPDY des port-forwards** après 1-2 h :
   `ConnectionReset by peer` côté Jaeger, 0 trace récoltée sur la seconde
   moitié des runs → NodePort + port-forwards renouvelés par épisode.
7. **Delete Chaos Mesh bloquant** (CRD en `Terminating`) → timeout + fallback
   `--ignore-not-found`.
8. **Crashes en campagne longue** (SIGKILL OOM, perte SSH) → checkpoint
   idempotent + graceful shutdown.

## 6) État du dataset (22 avril 2026)

- **Campagne de validation** : 2 rep × 14 scénarios = **28 épisodes** dans
  `data/raw_new/`.
- **Modalités** : Prometheus OK systématiquement ; Jaeger compromis sur
  cette campagne à cause de la dégradation SPDY (d'où le travail NodePort) ;
  Loki OK.
- **Premier run historique** : `data/raw/run_20260416_112413/` (monolithique
  V0, conservé pour référence, non exploité pour l'entraînement).
- **Features / datasets** : pas encore construits à partir de `raw_new` — on
  rejoue la campagne en mode NodePort d'abord pour avoir les 3 modalités.

## 7) Recommandations config (VM 4 CPU / 64 Go, NodePort)

Principes :
- `sample_interval_s` = 30 s (cohérent avec les scrape intervals
  Prometheus/OTel).
- `fetch_total_timeout_s` strictement inférieur à `sample_interval_s`.
- Parallélisme traces limité pour ne pas saturer le backend Jaeger.
- NodePort plutôt que port-forward dès qu'une campagne dépasse 1 h.

Configuration cible :
```yaml
sample_interval_s: 30
request_timeout_s: 10
fetch_total_timeout_s: 20
max_parallel: 3
limit_per_service: 10
trace_window_s: 60
span_cache_ttl_s: 120
```

## 8) Suite

1. **Rejouer la campagne de validation** en mode `--endpoint-mode nodeport`
   (dès que l'admin cluster a publié les NodePorts Prometheus/Jaeger/Loki)
   pour récupérer les traces Jaeger manquantes.
2. **Phase 2** (`build_features`) sur `data/raw_new/` une fois les traces
   complètes → `data/features/v1/`.
3. **Phase 3** (`assemble_dataset`) → `data/datasets/ewat_v1/` avec split
   temporel strict.
4. Étendre à ≥ 20 reps par scénario pour satisfaire la contrainte H1
   (structurabilité : il faut assez d'épisodes par type pour que la
   silhouette held-out soit estimable).
5. Calibration ε_drift (étape 0 MMD-RFF) sur les 4 scénarios drift bénins.

## 9) Résumé pour weekly (6 min)

- **Pipeline** : refactor complet en 3 phases découplées
  (Record / Build / Assemble). Les dumps bruts sont immuables, tout le
  feature engineering est rejouable offline.
- **Services canoniques** : figés à 6 services effectivement observables
  sur les 3 modalités.
- **Chaos** : 14 scénarios couvrant les 4 régimes θ, y compris drift∩anomaly.
- **Robustesse** : checkpoint idempotent, graceful shutdown, quality gate,
  timeouts chaos, trois modes d'accès aux backends (NodePort / port-forward /
  in-cluster), port-forwards renouvelés par épisode, script de validation
  dédié aux dumps bruts.
- **Campagne de validation** : 28 épisodes collectés, a révélé la
  dégradation SPDY des port-forwards longs → bascule NodePort.
- **Prochaine étape** : rejouer proprement en NodePort, puis construire
  `features/v1` et `datasets/ewat_v1`.

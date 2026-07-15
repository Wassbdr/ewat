# EWAT v5 — Pré-vol collecte (à valider AVANT la campagne complète)

_But : dérisquer ~2 semaines de cluster. Chaque item est figé à la collecte —
un défaut découvert après = collecte perdue. Rien ne se lance tant que les items
bloquants ne sont pas verts._

## Contexte des changements (2026-07-07)

Améliorations pré-collecte pour maximiser la note dataset (plafond mono-topo ≈ 16/20) :

1. **Coverage traces** — loadgen corrigé (`add_contact`, `full_journey`, session admin).
   Imputation traces **0.53 → 0.146** au pilote (35/41 tracés). ✅ vérifié.
2. **Feature morte** — `jvm_threads_blocked` (BLOCKED = 0 sur 37 pods) remplacé par
   `jvm_threads_live` (`jvm_threads_current`, 27→29, varie). **Schéma v5.1 → v5.2.** ✅ vérifié.
3. **Drift bénin** — 3 scénarios `drift` (rolling_deploy, config_rollout, autoscale_up)
   via kubectl natif (rollout/scale), labellés regime=normal + drift_flag fenêtré.
   Sépare drift↔anomalie (unique en public, réhabilite H2a).
4. **Bug réel F5** — reproduit par config (`env` : `-Dserver.tomcat.max-threads=5`,
   épuisement pool de threads) sans rebuild, comme F3. **status: needs_pilot.**

## Second balayage 2026-07-07 (« t'es sûr qu'on ne peut rien améliorer ? »)

Trois trouvailles supplémentaires, dont une bloquante :

5. **BLOQUANT corrigé — `validate_v5` était figé sur v5.1** : le gate aurait rejeté
   100 % des épisodes v5.2 de la campagne (mismatch `signal_feature_names`) et
   `run_campaign` aurait bouclé en retries. Corrigé : validation contre le schéma
   déclaré dans `metadata.dataset_schema_version` (v5.1 ET v5.2 passent ; testé
   sur épisode existant + simulation 4 cas).
6. **4e régime θ_drift∩anomaly enfin échantillonné** : `drift_anomaly` était documenté
   (datasheet, validate, formalisation) mais JAMAIS produit. Ajout de 2 scénarios
   `overlap` (kind natif drift + chaos simultanés sur le même service) :
   `faulty_rollout_cpu` (rollout + StressChaos travel) et `faulty_scale_delay`
   (scale 1→3 + NetworkChaos delay seat). Labels : regime=drift_anomaly fenêtré +
   drift_flag + is_injection=True. En TRAINING (support H2b). → **27 scénarios**.
7. **Collision multi-runner corrigée** : les state files `/tmp/ewat_{drift,bug}_*.json`
   n'étaient pas namespacés — 2 runners (tt / tt-b) sur le même scénario se seraient
   écrasé la restauration. Chemins désormais `/tmp/ewat_*_{ns}_*.json`.

Vérifié aussi : gate OK pour épisodes drift all-normal (aucun check n'exige des steps
injection) ; Loki sain + logs tt frais (5 min) au 2026-07-07.

## Bloquants (must be green avant lancement)

- [ ] **jvm_threads_live vivant** sur pilotes : non-dégénéré (std>0), varie sous charge.
      *(Prometheus : `jvm_threads_current{namespace="tt"}` — déjà 27→29 hors charge.)*
- [ ] **Coverage traces ≤ 0.20** sur 1 pilote buildé (mask indices 10:14). Cible ~0.13.
- [x] **Drift : 3 pilotes** ✅ **2026-07-07 live** : `autoscale_up` scale 1→3 puis
      restauré à 1 (state file OK) ; `rolling_deploy` + `config_rollout` = nouveaux pods
      (hash changé), replicas inchangés, delete no-op. Reste à vérifier `labels.parquet`
      sur le premier épisode drift complet de la campagne (`regime` normal partout,
      `drift_flag` fenêtré, `is_injection=False`, `fault_type=drift`).
- [x] **F5 : pilote fait → EXCLU (not_reproducible)** ✅ **2026-07-07 A/B live** :
      la config se lie (threads 27→22, agent JMX préservé via `env_append`, restauration
      exacte) mais symptôme inattribuable — baseline p99 PIRE que F5-ON à charge
      identique, et rafale concurrente sur chemin sain déjà à p50=15s (saturation amont,
      CPU limits cluster partagé). Verdict figé dans `chaos/catalog.yaml`. Le dataset
      livre **2 bugs réels** : F3 (OOM, visible) + F1 (logique, négatif invisible documenté).
- [ ] **Overlap : 1 pilote** (`faulty_rollout_cpu`) — vérifier `labels.parquet` :
      `regime=drift_anomaly` dans la fenêtre, `drift_flag=True`, `is_injection=True`,
      recovery après ; et restauration complète (chaos supprimé, replicas intacts).
      Injecteur vérifié en dry-run, labels simulés corrects — ce pilote peut être le
      premier épisode overlap de la campagne, buildé + audité avant de dérouler.
- [ ] **Purger les ressources chaos périmées avant lancement** — fait le 2026-07-07
      (3 reliquats du 07-01 coincés en Terminating par finalizer `chaos-mesh/records`,
      pods cibles disparus → patch finalizers). **Re-vérifier au lancement** :
      `for k in stresschaos networkchaos podchaos dnschaos timechaos iochaos; do
      kubectl -n tt get $k; done` doit être vide.
- [ ] **18 features vivantes** sur pilotes (aucune all-zero/constante). Audit :
      per-feature std/zero% sur les épisodes buildés.
- [ ] **Loki + Prometheus scrapent tt** (le gate `_backends_scraping` doit passer).

## Non-bloquants (à faire pendant/après collecte)

- [ ] **n ≥ 25 par scénario** : `--reps 30` (déjà uniforme) ; le rejet chute avec la
      coverage corrigée → viser rejet < 15 %. Rééquilibrer par re-collecte ciblée si besoin.
- [ ] **Packaging** : régénérer `schema.json` (v5.2), MAJ **DATASHEET** :
      - schéma v5.1 → **v5.2** (M[9] = jvm_threads_live).
      - nouvelle catégorie **drift** (3 scénarios) + tâche drift↔anomalie.
      - **table coverage par service** (documenter les ~4 structurellement non-tracés :
        news, ticket-office, ui-dashboard, voucher — jamais dans Jaeger).
      - held-out bugs : F1 + F3 (+ F5 si pilote OK).
- [ ] **Audit fuite** (`audit_leak_v5`) + SHA256 après re-build.

## Commandes pilote

```bash
cd v5
# drift (un par kind) — épisode court via override phases
V5_PHASES="4,4,3,6,3" python -m collect.run_episode --scenario rolling_deploy \
    --out ../data/pilot/rolling_deploy --address http://172.16.203.12:32677
V5_PHASES="4,4,3,6,3" python -m collect.run_episode --scenario autoscale_up ...
# F5 (env bug) — passer par apply-bug manuel puis observer, OU un run_episode dédié
python -m chaos.inject apply-bug F5 ; # observer Grafana/Prom ; puis :
python -m chaos.inject delete-bug F5
# vérifier restauration :
kubectl --context observit-cluster1 get deploy -n tt ts-seat-service ts-basic-service \
    -o jsonpath='{range .items[*]}{.metadata.name}{" replicas="}{.spec.replicas}{" "}{.spec.template.spec.containers[0].env}{"\n"}{end}'
```

## État vérifié à ce jour

| Item | État |
|---|---|
| Loadgen coverage (0.146) | ✅ live |
| jvm_threads_live (schéma v5.2) | ✅ live |
| Injecteur drift/env (commandes) | ✅ dry-run |
| Labeling drift (regime/drift_flag) | ✅ simulé |
| Drift apply + restauration live (×3) | ✅ live 2026-07-07 |
| env_append préserve l'agent JMX | ✅ live 2026-07-07 |
| F5 manifestation | ❌ not_reproducible (A/B) — exclu |
| Reliquats chaos purgés (tt propre) | ✅ 2026-07-07 (re-vérifier au lancement) |

## Pilotes bout-en-bout 2026-07-08 (record → build v5.2 → gate → audit) — ✅ GO

3 épisodes (30 steps, V5_PHASES court) : `cpu_stress`, `autoscale_up`, `faulty_rollout_cpu`.

| Vérification | Résultat |
|---|---|
| Gate validate_v5 (schéma v5.2) | **3/3 PASS** (aurait été 0/3 sans le fix du gate) |
| Labels anomalie (cpu_stress) | injection 14 / normal 13 / recovery 3, fault_type=chaos ✓ |
| Labels drift (autoscale_up) | regime 100% normal, drift_flag=14 fenêtré, is_injection=0, fault_type=drift ✓ |
| **Labels overlap (faulty_rollout_cpu)** | **drift_anomaly=14** + drift_flag=14 + is_injection=14 ✓ — 4e régime produit pour la 1re fois |
| jvm_threads_live | vivant : 8–37, std 1.2–2.7 (vs BLOCKED=0 mort) ✓ |
| 18 features | toutes vivantes sur épisodes anomalie ; error/abnormal ~0 sur drift bénin = **propriété attendue** (pas de signature d'erreur sur drift pur) |
| Coverage épisode | 37/41 services tracés (4 structurels attendus) ✓ |
| Coverage par step | **0.25–0.35 imputé** (bénin 0.25, sous chaos 0.30–0.35) vs **0.53 avant → ÷2** ; au-dessus de la cible 0.20 qui était calibrée sur le proxy fenêtré (0.146), métrique par-bin plus stricte. Sous chaos, la chute de débit → moins de spans est une physique réelle, capturée par le mask. **Accepté + à documenter (les 2 chiffres dans la datasheet)** |
| Restauration cluster | replicas=1 partout, zéro chaos résiduel ✓ |

Knob optionnel si on veut grappiller du per-bin : réduire la pause loadgen
(0.3–1.2s → 0.2–0.8s) ou users 12→14 — à tester sur la VM, PAS requis pour le GO.

## Design d'échantillonnage v5.2 (3e balayage, 2026-07-08) — saturation de la topologie

4 dimensions du design figées à la collecte, ajoutées avant lancement :

1. **Jitter d'onset** : baseline ∈ [8,16], pre ∈ [10,18] steps, tirés PAR épisode
   (seed = nom d'épisode → déterministe à la reprise). Tue la fuite positionnelle
   (tous les épisodes injectaient au step 26/60). `V5_PHASES` (test) désactive.
   Onsets simulés : 19→33 steps, bien étalés.
2. **`normal_baseline`** (catégorie `normal`, kind `none`) : runs sains complets →
   FPR mesurable + calibration drift. regime=normal partout, intensity 0,
   fault_type=none. 30 reps auto via campagne.
3. **Rotation de cibles** : `target_pool` (4 services porteurs de trafic) sur les
   15 scénarios mono-cause ; round-robin déterministe `pool[rep % 4]` → 7-8 reps
   par couple type×cible. Décorrèle type↔service (fuite signature scénario v3) et
   débloque la tâche localisation. held-out/compo/drift/overlap : cibles fixes.
   `container_kill` : `containerNames: ["{{TARGET}}"]`. Validation stricte hors-pool.
4. **Pics d'intensité variés** : 60 % high / 30 % med / 10 % low par rep%10 —
   pannes subtiles pour le spectre de difficulté early-warning. `intensity_t`
   plafonne au pic réel (meta `peak_value`).
5. (bonus) **Charge variée** : users 12/10/14 par rep%3.

Tout est déterministe par rep → reprise de campagne idempotente inchangée.
Vérifié offline : dry-run injecteur (rotation, containerNames, no-op normal,
rejet hors-pool), simulation labels (normal all-normal ; peak 0.667/0.333 exacts ;
jitter déterministe), plan campagne (cibles 8/8/7/7, peaks 18/9/3 sur 30 reps).

- [x] **Pilotes v5.2 design** ✅ **2026-07-14 live, gate 2/2, audit 4/4** :
      - jitter : phases b=12/pre=13 (T=59) et b=8/pre=15 (T=56) → onset step 23 ≠ 26 ✓
      - normal_baseline : regime normal 59/59, intensity 0, fault_type=none ✓
      - rotation : target_service=ts-order-service (≠ historique travel) ✓
      - pic : intensity_t max = 0.667 exact (peak med) ✓
      - cluster restauré, zéro chaos résiduel ✓
      **GO FINAL — design d'échantillonnage saturé. Prochaine étape : commit puis VM.**

**Catalogue final : 28 scénarios** (15 mono rotatifs + 4 compo + 3 drift + 2 overlap +
1 normal + 3 held-out) + bugs F1/F3.

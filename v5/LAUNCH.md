# EWAT v5 — Runbook de lancement de la collecte (3 runners)

À exécuter **sur la VM**. Cible : ~720 épisodes (19 training × 30 reps + 5 held-out × ~28),
**3 runners** (`tt` reps 0-9, `tt-b` reps 10-19, `tt-c` reps 20-29). Tout est préparé et
validé ; ce runbook ne fait que **lancer**.

**Calendrier** : ~33 min/épisode (30 min phases + ~3 min collecte) → **~720 ép en ~7-9 jours**
à 3 runners. Mini-batch 3 runners validé : collecte parallèle sans collision (ports décalés
0/10/20), **RAM workers 87 % au pic** (sous le plafond 90 %), **0 éviction**, chaos localisé.

**Contrainte = RAM, pas CPU** : chaque runner ≈ ~20 GB (41 JVM + mongos). À 3 runners les
workers sont à ~80-87 %. Le gate de collecte a un **garde-fou RAM** (`--ram-ceiling`, défaut
90) qui met en pause avant un épisode si un worker dépasse le seuil → anti-éviction. Si
`health_monitor` signale `ram_max` répété > 90 % ou des évictions, **revenir à 2 runners**.

Pré-requis (faits) : `tt` + `tt-b` + `tt-c` déployés (64/64, JVM instrumenté), stack collecte
paramétrée par namespace + contexte épinglé, schéma v5.1, séparation collecte/build.

`NODE_IP=172.16.203.12` · `cd ~/repos/ewat/v5` · `export PYTHONPATH=../src`

---

## 0. Pré-vol (à vérifier avant de lancer)

```bash
# CONTEXTE kubectl — toute la collecte est épinglée dessus (défaut observit-cluster1).
# run_campaign fait un préflight bloquant ; si la VM utilise un autre nom de contexte,
# exporter V5_KUBE_CONTEXT=<nom> (vaut pour run_episode/inject/probe/run_campaign).
kubectl config current-context            # doit être observit-cluster1
# export V5_KUBE_CONTEXT=observit-cluster1   # si besoin de forcer

# les 3 runners sont sains
kubectl get pods -n tt    --no-headers | grep -c 1/1   # attendu 64
kubectl get pods -n tt-b  --no-headers | grep -c 1/1   # attendu 64
kubectl get pods -n tt-c  --no-headers | grep -c 1/1   # attendu 64
# RAM workers < ~85 % au repos (3 runners idle) — laisse la marge pour load+chaos
kubectl top nodes | awk '/workers/{print $1, $5}'
# JVM scrapé pour les 3 namespaces
#   (port-forward prometheus puis : count(jvm_threads_state{namespace="tt|tt-b|tt-c"}))
# login OK sur les 3 UIs
curl -s -m10 -XPOST http://$NODE_IP:32677/api/v1/users/login -H 'Content-Type: application/json' -d '{"username":"fdse_microservice","password":"111111"}' | head -c 80
curl -s -m10 -XPOST http://$NODE_IP:32679/api/v1/users/login -H 'Content-Type: application/json' -d '{"username":"fdse_microservice","password":"111111"}' | head -c 80
curl -s -m10 -XPOST http://$NODE_IP:32681/api/v1/users/login -H 'Content-Type: application/json' -d '{"username":"fdse_microservice","password":"111111"}' | head -c 80
mkdir -p ../data/raw_v5
```

## 1. COLLECTE — 3 runners en parallèle (Phase 1)

Chacun le catalogue complet, split par plage de **10 reps**, ports locaux décalés
(`--pf-offset` 0/10/20). Lancer dans 3 terminaux (ou tmux). `--ram-ceiling 90` (défaut)
met en pause avant un épisode si un worker dépasse 90 % de RAM.

```bash
# Runner A — namespace tt, reps 0-9, ports locaux 19090/16686/13100
PYTHONPATH=../src python -m collect.run_campaign \
  --namespace tt --address http://$NODE_IP:32677 \
  --rep-start 0 --rep-end 10 --reps 30 --pf-offset 0 \
  --out-root ../data/raw_v5 --users 12 --reset-every 10 --held-out-cap 28 --ram-ceiling 90 \
  2>&1 | tee ../data/raw_v5/_campaign_tt.log

# Runner B — namespace tt-b, reps 10-19, ports locaux 19100/16696/13110
PYTHONPATH=../src python -m collect.run_campaign \
  --namespace tt-b --address http://$NODE_IP:32679 \
  --rep-start 10 --rep-end 20 --reps 30 --pf-offset 10 \
  --out-root ../data/raw_v5 --users 12 --reset-every 10 --held-out-cap 28 --ram-ceiling 90 \
  2>&1 | tee ../data/raw_v5/_campaign_ttb.log

# Runner C — namespace tt-c, reps 20-29, ports locaux 19110/16706/13120
PYTHONPATH=../src python -m collect.run_campaign \
  --namespace tt-c --address http://$NODE_IP:32681 \
  --rep-start 20 --rep-end 30 --reps 30 --pf-offset 20 \
  --out-root ../data/raw_v5 --users 12 --reset-every 10 --held-out-cap 28 --ram-ceiling 90 \
  2>&1 | tee ../data/raw_v5/_campaign_ttc.log
```

Reprise après interruption : relancer la **même** commande (idempotent — saute les épisodes
déjà collectés via `episode_meta.json`).

Surveillance santé (optionnel, 1 par runner — affiche aussi `ram_max` des workers) :
```bash
python -m collect.health_monitor --namespace tt   --interval 60 &
python -m collect.health_monitor --namespace tt-b --interval 60 &
python -m collect.health_monitor --namespace tt-c --interval 60 &
```

## 2. BUILD offline (Phase 2) — en parallèle de la collecte

Rejouable, idempotent (saute si `signal.npz` existe). Lancer périodiquement (ou en boucle) :
```bash
PYTHONPATH=../src python -m collect.build_features_v5 --raw-root ../data/raw_v5 --workers 4
```
Si le schéma S(t) évolue : `--force` reconstruit depuis les dumps (pas de recollecte).

## 3. ASSEMBLAGE + VALIDATION (Phase 3) — quand la collecte est finie

```bash
cd ~/repos/ewat
# gate par épisode
PYTHONPATH=src python scripts/validate_v5.py --features-root data/raw_v5 --output data/raw_v5/_validate.json
# assemblage stratifié
PYTHONPATH=src python -m scripts.assemble_dataset --features-root data/raw_v5 \
  --output data/datasets/ewat_v5 --stratified --train-ratio 0.7 --val-ratio 0.15
# held-out → test only, puis check fuite
PYTHONPATH=src python scripts/enforce_heldout_v5.py --dataset data/datasets/ewat_v5
PYTHONPATH=src python scripts/validate_v5.py --dataset data/datasets/ewat_v5
```

## 4. Suivi quotidien

```bash
# nb d'épisodes collectés (meta présent, pas de raw_failed)
ls data/raw_v5 | wc -l
find data/raw_v5 -name .raw_failed | wc -l          # échecs collecte
find data/raw_v5 -name signal.npz   | wc -l          # épisodes buildés
du -sh data/raw_v5                                   # taille disque
```

## Garde-fous
- Les dumps `data/raw_v5/` sont **sacrés** (jamais modifier in-place ; le build écrit à côté).
- **3 runners : la contrainte est la RAM** (CPU large). Workers ~80-87 % mesuré, pic mini-batch
  87 % < plafond 90 %. Le gate met en pause avant un épisode si un worker > `--ram-ceiling`.
  Si `health_monitor` signale `ram_max` répété > 90 % ou des évictions (`kubectl get events -A
  | grep -i evict`) → **revenir à 2 runners** (arrêter le runner C, relancer A/B sur 0-15/15-30).
- Si `health_monitor` signale DEGRADED répété, réduire `--users` ou espacer.
- Reset deep auto tous les 10 épisodes (anti-dérive baseline).

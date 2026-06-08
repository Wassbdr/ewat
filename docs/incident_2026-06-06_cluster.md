# Incident Report — Cluster observit-cluster1 — 2026-06-06

_Rédigé à 19h08 (CEST) suite à la session de diagnostic en direct._

---

## Contexte

Campagne de collecte v5 (EWAT Train Ticket, 3 runners parallèles, cible ~720 épisodes).
Monitoring tmux `ewat` sur la VM `ewat-0` (172.16.203.12).

État observé à **15h51 CEST** :

```
réussis=186/710 (~26%)  échecs=156  buildés=85
[tt]   [pod_kill]           baseline 360s + pre 420s ...
[tt-b] [pod_kill]           injection high 600s ...
[tt-c] [network_duplicate]  recovery 240s ...
```

---

## Chronologie

| Heure (CEST) | Événement |
|---|---|
| ~13h20 UTC (15h20) | `jnk2v` bascule NotReady définitivement (après 8 jours de flapping) |
| ~15h20 | `loki-0` (monitoring-metrics) passe Pending — Loki mort |
| ~15h50 | Détection : 156 échecs, runners en boucle de pause |
| 18h29 | Lancement du diagnostic |
| ~18h50 | Analyse des `.raw_failed` et état cluster |
| ~19h00 | `jnk2v` revient Ready temporairement |
| 19h06 | Arrêt runner tt-c + nettoyage chaos orphelins |
| 19h08 | Rédaction de ce rapport |

---

## Causes racines identifiées

### 🔴 #1 — Disque plein sur `jnk2v` (cause primaire, ancienne)

**Nœud** : `observit-cluster1-workers-58w74-jnk2v` (172.16.203.91)

```
FreeDiskSpaceFailed  x809 over 8d  kubelet  Failed to garbage collect required amount
                                             of images. Attempted to free 4011841126 bytes,
                                             but only found 0 bytes eligible to free.
ImageGCFailed        x793 over 8d  kubelet  rpc error: code=DeadlineExceeded
NodeNotReady         x11  over 8d  kubelet  Node status is now: NodeNotReady
```

`/var/lib/containerd` plein → kubelet ne peut plus GC les images → cycles NodeNotReady depuis 8 jours → bascule définitive NotReady le 06/06 à 13h20 UTC.

**Conséquence** : taint `node.kubernetes.io/not-ready:NoExecute` → éviction de tous les pods du nœud → 119 pods Pending dans les 3 namespaces (tt, tt-b, tt-c).

---

### 🔴 #2 — Deadlock de scheduling

Répartition des 8 nœuds au moment du diagnostic :

| Nœud | IP | Status | Schedulable | Raison |
|---|---|---|---|---|
| `9bwls` | 172.16.203.12 | Ready | ❌ | Cordonné juin 3 — intentionnel (VM campagne) |
| `dkhfv` | 172.16.203.232 | Ready | ✅ | "Too many pods" (pod count limit) |
| `gvx5k` | 172.16.203.191 | Ready | ✅ | "Too many pods" (pod count limit) |
| `jnk2v` | 172.16.203.91 | NotReady→Ready | ✅ | Disque plein, flapping |
| `jz7mn` | 172.16.203.196 | Ready | ❌ | Cordonné depuis **juin 5** — 16% RAM, raison inconnue |
| masters ×3 | — | Ready | ❌ | Taint control-plane/etcd |

Message scheduler :
```
0/8 nodes available:
  3 → control-plane (taint)
  1 → jnk2v (not-ready taint)
  2 → "Too many pods" (dkhfv + gvx5k)
  2 → "unschedulable" (9bwls + jz7mn)
```

Seul `jnk2v` pouvait accueillir des pods — mais il est NotReady. **119 pods bloqués.**

---

### 🔴 #3 — Loki `monitoring-metrics/loki-0` Pending depuis 115 min

`loki-0` est un StatefulSet avec PVC probablement lié à `jnk2v`. Quand `jnk2v` est NotReady, le pod ne peut pas reschedule → Loki mort → tous les épisodes collectés depuis ~13h30 UTC ont `logs=0`.

```
monitoring-metrics  loki-0  0/1  Pending  115m
```

---

### 🟠 #4 — Pattern `prom=0` (cause historique, intermittente)

**~80 épisodes** sur la période juin 3–6 ont échoué à cause de `prom=0` alors que traces et logs étaient corrects. Cause : `jnk2v` flappait NotReady → lors de ses cycles d'instabilité, Prometheus ne scrapait plus les pods qui étaient en cours d'éviction → pulls vides.

---

## Statistiques d'échecs (162 `.raw_failed` au total)

### Par type d'erreur

| Pattern | Épisodes | Cause |
|---|---|---|
| `traces=✅ logs=✅ prom=0` | ~83 | Prometheus vide au moment du pull (jnk2v instable) |
| `traces=✅ logs=0 prom=✅` | ~22 | Loki down ou timeout |
| `traces=12 logs=N prom=0` | ~25 | Pods Pending → JVM ne génère que 12 traces internes |
| `traces=0 logs=0 prom=0` | ~15 | Collecte totale échouée |
| `exception: port-forward` | ~12 | Historique avant migration NodePort (juin 3–4) |
| `exception: Connection reset` | ~5 | Réseau instable lors du pull |

### Par scénario (top échecs)

```
10  pod_kill       (hard)
9   memory_stress  (contention)
9   cpu_starvation (contention)
8   memory_pressure(contention)
8   F3             (bug)
8   container_kill (hard)
7   pod_failure    (hard)
7   held_net_bandwidth (held_out)
7   held_kernel_fault  (held_out)
7   F1             (bug)
6   network_partition  (hard)
6   network_duplicate  (gray)
```

### Épisodes récupérables

| Catégorie | Nombre | Condition |
|---|---|---|
| Repull Prometheus | **83** | traces+logs OK, prom=0 — données dans rétention Prometheus (~15j) |
| Repull Loki | **22** | traces+prom OK, logs=0 — après recovery de `loki-0` |

---

## Actions effectuées

### ✅ Runner tt-c arrêté + chaos nettoyés

```bash
pkill -f "collect.run_campaign.*tt-c"
kubectl --context observit-cluster1 delete stresschaos,networkchaos,podchaos,dnschaos,timechaos,iochaos \
  -n tt-c --all --ignore-not-found
# Résultat : networkchaos "v5-network-partition" deleted
```

Raison : avec jnk2v NotReady et les pods tt-c massivement Pending, le runner C consommait
des ressources cluster sans produire d'épisodes valides.

---

## Actions en attente (admin cluster requis)

### 🔴 Priorité 1 — Nettoyer le disque de `jnk2v`

```bash
# Sur le nœud jnk2v (SSH ou console Rancher)
df -h /var/lib/containerd
crictl rmi --prune          # purge images inutilisées containerd
# ou
systemctl restart containerd
```

### 🔴 Priorité 2 — Uncordon `jz7mn`

```bash
kubectl --context observit-cluster1 uncordon observit-cluster1-workers-58w74-jz7mn
# jz7mn : 16% RAM, 9% CPU — nœud le plus sain, cordonné depuis juin 5 sans raison connue
# Son uncordon soulagerait dkhfv (76%) et gvx5k (78%) qui sont au pod limit
```

---

## Actions planifiées (post-stabilisation cluster)

### Script repull Prometheus (83 épisodes)

Principe : lire `episode_meta.json` pour récupérer `t_start` + durée → requêter Prometheus
(rétention 15j, données accessibles) → réécrire `prometheus.json.gz` → supprimer `.raw_failed`.

À implémenter dans `v5/collect/repull_prometheus.py`.

### Script repull Loki (22 épisodes)

Même principe mais via `loki-np` (NodePort 32701) — à lancer après recovery de `loki-0`.

---

## État du cluster au moment de la clôture du diagnostic (19h08 CEST)

```bash
kubectl top nodes:
  9bwls   14% CPU  31% RAM  (SchedulingDisabled — VM campagne)
  dkhfv   48% CPU  76% RAM  (schedulable, pod limit)
  gvx5k   64% CPU  77% RAM  (schedulable, pod limit)
  jnk2v   <unknown>          (revenu Ready à ~19h, flapping)
  jz7mn   10% CPU  16% RAM  (SchedulingDisabled — à uncordon)

Pods Pending : 119 (en cours de résorption si jnk2v tient)
Loki-0 (monitoring-metrics) : Pending (suit jnk2v)
Prometheus NodePort :32700 : ✅ Ready
Jaeger tt  :32688 : ✅ OK (~25 services)
Jaeger tt-b :32690 : ⚠️  Dégradé (2 services seulement — était sur jnk2v)
Jaeger tt-c :32692 : ✅ OK (~30 services)
```

---

## Pronostic

- **Si jnk2v tient Ready** + admin uncordon jz7mn → pods se reschedulen → runners tt + tt-b reprennent automatiquement (boucle de pauses s'arrête) → campagne continue à 2 runners.
- **Si jnk2v re-bascule NotReady** sans intervention admin → retour au deadlock → arrêter tt-b aussi et attendre.
- **Jaeger tt-b** dégradé (2 services) → les épisodes tt-b en cours de collecte auront `traces` très faibles → echecs raw-gate prévisibles → à surveiller.
- **Récupération nette estimée post-repull** : 83 + 22 = **105 épisodes supplémentaires** récupérables → porte le comptage OK de 186 à ~291 si les validate_v5 passent.

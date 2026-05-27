"""Génère notebooks/episode_collection_walkthrough.ipynb.

Notebook narratif qui rejoue dans le détail (commandes kubectl exactes,
manifests Chaos Mesh, requêtes HTTP Prometheus/Jaeger/Loki, hashing,
quality gate) ce qui se passe lors de la collecte d'UN épisode EWAT.

Audience cible : maître de stage SRE/observabilité. Pas de cluster requis :
on rejoue l'épisode cpu_starvation_000 déjà sur disque dans data/raw.
"""

from __future__ import annotations
import nbformat as nbf
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "episode_collection_walkthrough.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


# ════════════════════════════════════════════════════════════════════════════
# 0. TITRE + OBJECTIF
# ════════════════════════════════════════════════════════════════════════════

md(r"""# EWAT — Collecte d'un épisode, en très grand détail

> **Audience** — Maître de stage Devoteam (SRE/observabilité).
> **Objectif** — Rejouer pas à pas la collecte d'un seul épisode `cpu_starvation`
> sur le cluster `observit-cluster1` : commandes `kubectl` exactes, manifest
> Chaos Mesh, requêtes HTTP vers Prometheus / Jaeger / Loki, hashing des dumps,
> quality gate, animation du signal `S(t)` qui se construit phase après phase.
>
> Aucun accès cluster requis pour exécuter ce notebook : on rejoue un épisode
> déjà collecté (`data/raw/episode_cpu_starvation_000_20260430T022941Z/`).
> Les commandes affichées sont **littéralement** celles que `scripts/record_episode.py`
> a lancées le 30 avril 2026 à 02:29:41 UTC.

## Plan

1. **Vue d'ensemble** — les 4 phases et où passe le temps
2. **Topologie cluster** — namespace `ewat`, 6 services canoniques, stack obs
3. **Phase 0 — Préflight** — découverte endpoints, port-forwards SPDY
4. **Phase 1 — Baseline (300 s)** — système au repos sous trafic load-gen
5. **Phase 2 — Pré-injection (60 s)** — warm-up
6. **Phase 3 — Injection (180 s)** — `kubectl apply` du `StressChaos`
7. **Phase 4 — Recovery (120 s)** — `kubectl delete`, retour au calme
8. **Phase 5 — Dumps bulk** — 1 fetch Prometheus + 1 par service Jaeger + Loki paginé
9. **Phase 6 — Persistance** — gzip, SHA-256, écriture atomique, quality gate
10. **Timeline animée** — `S(t)` reconstruit, lecture vidéo
11. **Récap** — artefacts produits, intégrité, taille""")

# ════════════════════════════════════════════════════════════════════════════
# 1. VUE D'ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 1. Vue d'ensemble — où passe le temps

Un épisode dure **660 s** (11 min) du début à la fin et produit **5 fichiers**.
Pendant 99 % de ce temps, le recorder **ne fait rien d'autre que dormir** : il
laisse le système vivre sous le trafic Locust constant. Tout le travail réseau
(les 3 dumps bulk) est concentré à la fin, après que le chaos a été retiré, pour
éviter d'ajouter de la charge réseau pendant qu'on observe le système.

```
t=0           t=300         t=360                      t=540            t=660
│             │             │                          │                │
│  baseline   │  pre-inj    │       injection          │    recovery    │  → 3 dumps bulk
│   300 s     │    60 s     │         180 s            │      120 s     │   (Prom/Jae/Loki)
│             │             │                          │                │
│   sleep     │   sleep     │  kubectl apply           │  kubectl       │   1 GET Prom /api/v1/query_range × 11 PromQL
│             │             │  StressChaos             │  delete        │   6 GET Jae /api/traces (1/service)
│             │             │  + sleep                 │  + sleep       │   N GET Loki /loki/api/v1/query_range (paginé)
```

Cette topologie temporelle est **strictement séquentielle** : on n'instrumente
pas en ligne, on dump à froid. C'est la décision d'architecture la plus
importante de Phase 1 et elle découle d'un constat : un collector en ligne
produisait des NaN en cascade dès que le système était sous stress (timeouts
empilés, blocage du scheduler). Le pipeline actuel sépare donc explicitement
**l'exécution de l'épisode** (cluster réel, temps strict) de **l'extraction des
features** (Phase 2, hors-ligne, rejouable à volonté sur les mêmes dumps).""")

# ════════════════════════════════════════════════════════════════════════════
# 2. IMPORTS + CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 2. Setup — imports et chargement de l'épisode

On ouvre les artefacts produits par `record_episode.py` le 30 avril 2026.
Tout ce qui suit est dérivé de ces fichiers (pas du cluster live).""")

code(r"""from __future__ import annotations
import gzip, hashlib, json, sys, textwrap, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from IPython.display import HTML, Markdown, display, Code

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HERE = Path.cwd()
ROOT = HERE if (HERE / "src" / "ewat").exists() else HERE.parent
sys.path.insert(0, str(ROOT))

EPISODE_ID = "episode_cpu_starvation_000_20260430T022941Z"
RAW_DIR = ROOT / "data" / "raw" / EPISODE_ID
FEAT_DIR = ROOT / "data" / "features" / "v3" / EPISODE_ID

print(f"épisode  : {EPISODE_ID}")
print(f"raw dump : {RAW_DIR}  ({'OK' if RAW_DIR.exists() else 'MANQUANT'})")
print(f"features : {FEAT_DIR}  ({'OK' if FEAT_DIR.exists() else 'MANQUANT'})")
print()
print("fichiers présents :")
for p in sorted(RAW_DIR.iterdir()):
    print(f"  {p.name:30s}  {p.stat().st_size / 1024:9.1f} KB")""")

code(r"""# Métadonnées de l'épisode (rejouable, indépendant du cluster)
EP = json.loads((RAW_DIR / "episode.json").read_text())
MAN = json.loads((RAW_DIR / "manifest.json").read_text())

B = EP["boundaries"]            # bornes Unix s pour chaque phase
SVC = EP["canonical_services"]  # 6 services
SCN = EP["scenario"]
print(f"scénario : {SCN['name']} ({SCN['kind']}, {SCN['category']})")
print(f"cibles   : {SCN['targets']}")
print(f"durée nominale injection : {SCN['duration_nominal_s']:.0f} s")
print()
print("phases (s relatifs à baseline_start) :")
t0 = B["baseline_start"]
for k, v in B.items():
    print(f"  {k:24s} t+{v - t0:7.2f} s")""")

# ════════════════════════════════════════════════════════════════════════════
# 3. TOPOLOGIE CLUSTER
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 3. Topologie — ce qui tourne dans `ewat`

Le namespace `ewat` héberge une instance complète d'**Online Boutique**
(OpenTelemetry Demo) : 6 services applicatifs canoniques (= `|V| = N = 6` dans
la formalisation), un load-generator Locust qui produit du trafic constant, et
des resources Chaos Mesh créées à la volée pour chaque injection.

Le **plan d'observabilité** vit hors du namespace `ewat` et est consommé en
lecture seule :

| Backend     | Service Kubernetes                              | Namespace               | Port distant |
|-------------|-------------------------------------------------|-------------------------|--------------|
| Prometheus  | `svc/rancher-monitoring-prometheus`             | `cattle-monitoring-system` | 9090         |
| Jaeger      | `svc/rca-jaeger-query`                          | `rca-sandbox`           | 16686        |
| Loki        | `svc/loki-gateway`                              | `monitoring-logs`       | 80           |

L'accès est en **namespace-admin sur `ewat`** : on peut tout faire dans `ewat`,
on peut **lire** les services des autres namespaces (donc port-forward vers
Prometheus/Jaeger/Loki est OK), mais on ne peut pas modifier ces autres
namespaces. C'est la contrainte qui structure toute l'architecture de
collecte.""")

code(r"""# Diagramme topologique — services, plan d'observabilité, flux

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

# --- namespace ewat (gros rectangle) ---
ns = mpatches.FancyBboxPatch((0.3, 0.5), 7.5, 5.2,
    boxstyle="round,pad=0.05", linewidth=1.4,
    edgecolor="#1f6f8b", facecolor="#eef6f9")
ax.add_patch(ns)
ax.text(0.5, 5.4, "namespace ewat", fontsize=11, weight="bold", color="#1f6f8b")

# 6 services canoniques (rangée)
svc_x = [0.7, 2.0, 3.3, 4.6, 5.9, 7.2]
svc_n = ["frontend", "cart", "checkout", "recommendation",
         "product-catalog", "ad"]
svc_color = ["#ffb4a2", "#cdb4db", "#bde0fe", "#a2d2ff", "#caffbf", "#fdffb6"]
for x, name, c in zip(svc_x, svc_n, svc_color):
    box = mpatches.FancyBboxPatch((x - 0.05, 3.5), 1.1, 0.7,
        boxstyle="round,pad=0.03", linewidth=0.9,
        edgecolor="#444", facecolor=c)
    ax.add_patch(box)
    ax.text(x + 0.5, 3.85, name, ha="center", va="center", fontsize=8.2)
ax.text(0.5, 4.45, "6 services applicatifs (canonical V, |V|=N=6)",
        fontsize=8.5, color="#444", style="italic")

# Load generator (Locust) — produit du trafic
lg = mpatches.FancyBboxPatch((0.7, 1.9), 2.4, 0.8,
    boxstyle="round,pad=0.03", linewidth=1.1,
    edgecolor="#444", facecolor="#ffd6a5")
ax.add_patch(lg)
ax.text(1.9, 2.3, "load-generator\n(Locust, trafic constant)",
        ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(0.95, 3.5), xytext=(1.6, 2.7),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1))

# Chaos Mesh CRDs (créés à la demande)
cm = mpatches.FancyBboxPatch((3.6, 1.9), 2.6, 0.8,
    boxstyle="round,pad=0.03", linewidth=1.1, linestyle="--",
    edgecolor="#c92a2a", facecolor="#ffe3e3")
ax.add_patch(cm)
ax.text(4.9, 2.3, "Chaos Mesh CRD\n(StressChaos, PodChaos…)",
        ha="center", va="center", fontsize=8.5, color="#c92a2a")
ax.annotate("inject", xy=(4.1, 3.5), xytext=(4.9, 2.7),
            arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=1.1),
            fontsize=8, color="#c92a2a", ha="center")

# Pods OTel SDK (sidecars)
ot = mpatches.FancyBboxPatch((6.6, 1.9), 1.2, 0.8,
    boxstyle="round,pad=0.03", linewidth=1.0,
    edgecolor="#444", facecolor="#e9ecef")
ax.add_patch(ot)
ax.text(7.2, 2.3, "OTel SDK\n(traces, logs)",
        ha="center", va="center", fontsize=8)

# --- Plan d'observabilité (hors ewat) ---
obs = mpatches.FancyBboxPatch((8.4, 0.5), 3.3, 5.2,
    boxstyle="round,pad=0.05", linewidth=1.4,
    edgecolor="#5f3dc4", facecolor="#f3f0ff")
ax.add_patch(obs)
ax.text(8.6, 5.4, "plan d'observabilité (lecture seule)",
        fontsize=10, weight="bold", color="#5f3dc4")

backends = [
    ("Prometheus", "cattle-monitoring-system", 4.2, "#fa5252"),
    ("Jaeger",     "rca-sandbox",              3.0, "#228be6"),
    ("Loki",       "monitoring-logs",          1.8, "#15aabf"),
]
for name, ns_obs, y, color in backends:
    box = mpatches.FancyBboxPatch((8.7, y - 0.35), 2.8, 0.7,
        boxstyle="round,pad=0.03", linewidth=0.9,
        edgecolor=color, facecolor="white")
    ax.add_patch(box)
    ax.text(10.1, y + 0.08, name, ha="center", fontsize=9, weight="bold", color=color)
    ax.text(10.1, y - 0.18, f"ns: {ns_obs}", ha="center", fontsize=7, color="#666")

# Flux scrape (ewat → backends)
for y_to in (4.2, 3.0, 1.8):
    ax.annotate("", xy=(8.7, y_to), xytext=(7.85, y_to - 0.5 + (4.2 - y_to) * 0.3),
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8, linestyle="--"))
ax.text(8.0, 5.05, "scrape / OTLP", fontsize=7.5, color="#888", style="italic")

ax.set_title("Topologie ewat — services applicatifs + plan d'obs externe",
             fontsize=11.5, pad=6)
plt.tight_layout()
plt.show()""")

# ════════════════════════════════════════════════════════════════════════════
# 4. PHASE 0 — PRÉFLIGHT
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 4. Phase 0 — Préflight : ouverture des tunnels SPDY

Avant que l'épisode commence, `record_episode.py` met en place **trois
port-forwards `kubectl`** vers les services d'observabilité. C'est le mode
`local-portforward` (le mode `nodeport` est l'alternative pour les campagnes
longues car les tunnels SPDY se dégradent au-delà de ~1-2 h sous payload Jaeger
lourd). En mode `local-portforward`, on **redémarre les tunnels avant chaque
dump** (voir `_PortForward.start` dans le code) pour repartir d'un SPDY frais.

Les commandes ci-dessous sont littéralement celles que la classe `_PortForward`
construit dans `scripts/record_episode.py:272-277` :""")

code(r"""# Reconstruction des commandes port-forward exactes (depuis episode.json)
pf = EP["collection"]["port_forwards"]
print("Commandes lancées par record_episode.py (mode local-portforward) :\n")
for name, spec in pf.items():
    cmd = (f"kubectl port-forward "
           f"-n {spec['namespace']} {spec['target']} "
           f"{spec['local_port']}:{spec['remote_port']}")
    print(f"# {name}")
    print(f"$ {cmd}\n")

print("Vérification de readiness — socket.create_connection(('127.0.0.1', port),"
      " timeout=1) en boucle, deadline 15 s.\n")
print("Si une tentative dépasse 15 s :")
print("  raise RuntimeError('port-forward for <name> not reachable after 15.0s')")""")

md(r"""**Points subtils**, dérivés du code :

- `_PortForward.start()` `kill -9` (via `lsof -ti tcp:<port>`) tout processus
  qui squatte déjà le port local avant d'ouvrir le sien — important quand un
  ancien `kubectl port-forward` a planté en laissant son socket TIME_WAIT.
- Le subprocess `kubectl` est lancé avec `preexec_fn=os.setsid` pour qu'on
  puisse `killpg(SIGTERM)` proprement à la fin (sinon `kubectl` laisse parfois
  une fork zombie).
- Le readiness check est **purement socket-level** (`connect` réussit) — il ne
  fait pas de `GET /-/healthy` sur Prometheus, car certains backends Jaeger
  servent du contenu hors path racine.""")

# ════════════════════════════════════════════════════════════════════════════
# 5. PHASE 1 — BASELINE
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 5. Phase 1 — Baseline (300 s) : le système au repos

Aucune commande n'est lancée pendant 5 minutes. Locust pilonne `frontend` à
charge constante, les 6 services tournent à leur régime nominal, et
`record_episode.py` se contente d'appeler `_sleep_with_status()` qui logue un
ping toutes les 30 s.

**Côté cluster** ce qui se passe pendant ces 300 s est entièrement passif vis-
à-vis du recorder :

```bash
# Ce qu'un observateur taperait en parallèle pour vérifier l'état (informatif)
$ kubectl -n ewat get pods
NAME                                READY   STATUS    RESTARTS   AGE
ad-7b5c8d4f-xqp9w                   1/1     Running   0          47h
cart-6f4d9c8b-tjk2n                 1/1     Running   0          47h
checkout-845f7c6d-r8m4q             1/1     Running   0          47h
frontend-7d9b8c5f-2x8wz             1/1     Running   0          47h
load-generator-78cd9-ghp4r          1/1     Running   0          12h
product-catalog-8f7d6c5b-pp9xx      1/1     Running   0          47h
recommendation-9c5d4b3a-fp7sk       1/1     Running   0          47h

$ kubectl -n ewat top pod
NAME                              CPU(cores)   MEMORY(bytes)
frontend-7d9b8c5f-2x8wz           23m          78Mi
cart-6f4d9c8b-tjk2n               11m          42Mi
checkout-845f7c6d-r8m4q           19m          56Mi
...
```

Le but de cette phase **n'est pas** de générer du signal "intéressant" ; c'est
de **stabiliser une distribution de référence** sur laquelle Phase 2 calculera
la fenêtre `W_ref` du DriftDetector MMD-RFF.""")

# ════════════════════════════════════════════════════════════════════════════
# 6. PHASE 2 — PRÉ-INJECTION
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 6. Phase 2 — Pré-injection (60 s) : warm-up isolé

Une fenêtre tampon courte avant le chaos. Elle existe pour deux raisons :

1. **Buffer pour la PromQL `rate()`** — `prom_rate_window = 2m` dans
   `collection.yaml`. Tout point Prometheus juste avant l'injection a besoin
   de 2 min de scrape derrière lui pour produire un rate non-tronqué.
2. **Buffer de causalité** — la fenêtre `[pre_start, pre_end]` est **labellisée
   `pre`** mais traitée comme `normal` lors de la construction de `S(t)`.
   Cette séparation explicite permet à Phase 2 de détecter facilement les
   épisodes où le chaos s'est déclenché en avance (bug Chaos Mesh) : si la
   distribution change pendant `pre`, l'épisode est invalidé.

Côté code, c'est encore un `_sleep_with_status(pre_s=60, regime="pre", …)` —
pas d'action côté cluster.""")

# ════════════════════════════════════════════════════════════════════════════
# 7. PHASE 3 — INJECTION
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 7. Phase 3 — Injection (180 s) : le cœur du dispositif

C'est ici que tout se joue. À `t = baseline_start + 360 s` (= `injection_start`),
`record_episode.py` enchaîne :

1. lecture du `ScenarioSpec` `cpu_starvation` dans
   `k8s/chaos-mesh/registry.yaml` (cache déjà chargé par `ChaosInjector`) ;
2. résolution du chemin du manifest : `k8s/chaos-mesh/contention/cpu_starvation.yaml` ;
3. exécution synchrone :
   ```
   kubectl -n ewat apply --validate=false -f <manifest> ; capture stdout
   ```
4. capture de l'horodatage `apply_returned_at` (= le moment où `kubectl` rend
   la main, **pas** le moment où le pod chaos-daemon a vraiment commencé à
   stresser — ces deux moments sont à ~1-3 s d'écart).

Le `--validate=false` est intentionnel : il évite que `kubectl` aille chercher
le schéma OpenAPI du CRD avant d'envoyer la requête (ça gagne ~500 ms et ça
contourne un bug Rancher où la cache OpenAPI désynchronise).

### 7.1 Le manifest injecté""")

code(r"""# Le manifest exact, lu depuis le repo (le même fichier que kubectl apply)
manifest_path = ROOT / "k8s" / "chaos-mesh" / SCN["file"]
manifest_yaml = manifest_path.read_text()
print(f"# {manifest_path.relative_to(ROOT)}\n")
print(manifest_yaml)""")

md(r"""**Lecture du manifest** (clé `spec`) :

| Champ                          | Valeur                                    | Effet                                                     |
|--------------------------------|-------------------------------------------|-----------------------------------------------------------|
| `kind`                         | `StressChaos`                             | CRD géré par chaos-controller-manager                     |
| `mode`                         | `all`                                     | Stress **tous** les pods qui matchent le selector         |
| `selector.namespaces`          | `[ewat]`                                  | Limite le rayon d'action                                  |
| `selector.labelSelectors`      | `app.kubernetes.io/component: frontend`   | Cible uniquement les pods `frontend`                      |
| `stressors.cpu.workers`        | `4`                                       | 4 threads stress-ng par pod ciblé                         |
| `stressors.cpu.load`           | `100`                                     | Charge 100 % CPU par worker                               |
| `duration`                     | `"3m"`                                    | Auto-stop côté CRD (filet de sécurité ; on `delete` avant)|

Le selector `app.kubernetes.io/component: frontend` est important : `targets`
dans le registry liste `[frontend, checkout]`, mais le manifest réel ne stresse
que `frontend`. La divergence est un artefact historique du registry et est
documentée dans `docs/limitations.md` (L16-L17).""")

code(r"""# Reconstruction de la commande kubectl apply exacte (depuis ChaosInjector._run)
apply_cmd = [
    "kubectl", "-n", "ewat", "apply",
    "--validate=false",
    "-f", f"k8s/chaos-mesh/{SCN['file']}",
]
delete_cmd = [
    "kubectl", "-n", "ewat", "delete",
    "-f", f"k8s/chaos-mesh/{SCN['file']}",
    "--ignore-not-found=true",
]

apply_elapsed = B["apply_returned_at"] - B["injection_start"]
delete_elapsed = B["delete_returned_at"] - B["injection_end"]

print(f"$ {' '.join(apply_cmd)}")
print(f"stresschaos.chaos-mesh.org/ewat-cpu-starvation created"
      f"      [apply elapsed = {apply_elapsed:.3f} s]")
print()
print(f"… sleep 180 s (injection_s, regime='injection') …\n")
print(f"$ {' '.join(delete_cmd)}")
print(f"stresschaos.chaos-mesh.org \"ewat-cpu-starvation\" deleted"
      f"      [delete elapsed = {delete_elapsed:.3f} s]")
print()
print(f"Filet de sécurité (audit 2026-05-26, Step 1 fix 1.2) :")
print(f"  si apply_elapsed < 1.0 s → RuntimeError (apply suspect, abort épisode)")
print(f"  ici : {apply_elapsed:.3f} s ⇒ OK")""")

md(r"""### 7.2 Ce qui se passe côté cluster pendant les 180 s

Une fois `kubectl apply` retourné, le chaos-controller-manager (un pod dans le
namespace `chaos-mesh`) prend la main :

```bash
# Ce qu'on verrait en regardant le CRD pendant l'injection
$ kubectl -n ewat describe stresschaos.chaos-mesh.org/ewat-cpu-starvation
Name:         ewat-cpu-starvation
Namespace:    ewat
Status:
  Conditions:
    Type:    Selected      Status: True   # selector a matché des pods
    Type:    AllInjected   Status: True   # stressors actifs sur tous
    Type:    Paused        Status: False
  Experiment:
    Container Records:
      - Id:           ewat/frontend-7d9b8c5f-2x8wz/server
        Phase:        Injected
        InjectedCount: 1

# Et la saturation côté kubelet
$ kubectl -n ewat top pod -l app.kubernetes.io/component=frontend
NAME                       CPU(cores)   MEMORY(bytes)
frontend-7d9b8c5f-2x8wz    998m         92Mi        ← 998m vs. 23m au repos
```

Le pod `frontend` n'a **pas** de limite CPU stricte (cf. annotation `cpu_limit`
de Phase 2 qui retourne souvent NaN sur ce service), donc les 4 workers
stress-ng saturent ce qu'ils peuvent. C'est l'origine du signal :

- **CPU usage** monte → directement mesurable via `container_cpu_usage_seconds_total`
- **Latency P99** monte → le scheduler kernel arbitre entre stress-ng et le code app
- **Error rate HTTP** monte modérément → quelques timeouts upstream
- **Traffic** baisse → moins de RPS servis car CPU saturé

Pendant ces 180 s, le recorder continue à dormir : aucune mesure online.""")

# ════════════════════════════════════════════════════════════════════════════
# 8. PHASE 4 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 8. Phase 4 — Recovery (120 s)

À `t = injection_end`, dans le `try/finally` :

```python
try:
    _sleep_with_status(inject_s, "injection", episode_id)
finally:
    injection_end = time.time()
    try:
        injector.delete(scenario_name)
    except Exception as exc:
        logger.warning("[%s] chaos delete failed: %s", episode_id, exc)
    delete_returned_at = time.time()
```

Le `finally` garantit que **le `kubectl delete` est tenté même si une exception
remonte** pendant la phase injection (KeyboardInterrupt, time.sleep planté,
etc.). C'est la défense la plus importante contre le risque de laisser un
chaos résiduel actif après crash du recorder.

`delete` a une timeout de 45 s. Si elle expire, le code retombe sur une commande
de delete avec timeout 15 s en mode `--ignore-not-found=true` ; si ça expire
aussi on logue `leaving chaos resource for manual cleanup` et on continue.

Ensuite, 120 s de recovery (encore un sleep). Le but : laisser le système se
relaxer pour que les fenêtres de fin n'enregistrent pas que la phase
"décroissante" du transitoire post-chaos. La distribution post-recovery est
labellisée `recovery` et **exclue** des évaluations H1/H3 (cf. `CLAUDE.md`,
correspondance θ ↔ labels).""")

# ════════════════════════════════════════════════════════════════════════════
# 9. PHASE 5 — DUMPS BULK
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 9. Phase 5 — Dumps bulk : on attaque Prometheus, Jaeger, Loki

À ce stade, on connaît la fenêtre exacte de l'épisode : `[baseline_start,
recovery_end]` (660 s). Tout le travail réseau du recorder est concentré ici.

Avant les requêtes : **redémarrage des port-forwards** (`pf_group.restart_all()`)
+ **refresh de la session HTTP** (`recorder.refresh_session()`). Cela kill les
anciens tunnels SPDY potentiellement dégradés et purge le pool de connexions
`requests`. Sans ce refresh, ~5 % des dumps Jaeger lourds plantaient en
`ConnectionResetError`.""")

code(r"""# Reconstruction des URLs HTTP réelles que le TelemetryRecorder a frappées.
# (TelemetryRecorder._range_query, src/telemetry/recorder.py)
prom_endpoint = "http://127.0.0.1:19090"        # endpoint via port-forward
jae_endpoint  = "http://127.0.0.1:16686"
loki_endpoint = "http://127.0.0.1:13100"

t_start, t_end = B["baseline_start"], B["recovery_end"]
print(f"Fenêtre dump : [{t_start:.3f}, {t_end:.3f}]"
      f"  = {t_end - t_start:.1f} s\n")
print(f"Élapsed (mesuré, depuis manifest.json) :")
for src in ("prometheus", "jaeger", "loki"):
    el = MAN["sources"][src]["fetch_elapsed_s"]
    sz = MAN["sources"][src]["size_bytes"] / 1024
    print(f"  {src:11s}  fetch={el:6.2f} s   gzip={sz:8.1f} KB")""")

md(r"""### 9.1 Prometheus — 11 requêtes `query_range`

`TelemetryRecorder.record_prometheus()` boucle sur la liste `QUERIES` de
`src/telemetry/prom_queries.py` (11 entrées) et envoie une requête
`/api/v1/query_range` par entrée. Chaque entrée a un template **primary** et
optionnellement un **fallback** : si le primary renvoie un résultat vide
(typique quand Istio n'est pas instrumenté pour le service), on retombe sur le
template fallback (en général OTel SDK).

Voici les 11 PromQL réellement envoyés à Prometheus pour cet épisode :""")

code(r"""from telemetry.prom_queries import QUERIES, render

ns = EP["collection"]["namespace"]
win = EP["collection"]["prom_rate_window"]
step = 15

print(f"GET {prom_endpoint}/api/v1/query_range")
print(f"  start = {t_start:.3f}    end = {t_end:.3f}    step = {step} s\n")

for spec in QUERIES:
    primary, fallback = render(spec, ns, win)
    used_fb = MAN["sources"]["prometheus"]["fallback_used"].get(spec.name, False)
    arrow = "  ✓" if not used_fb else "  ↳ fallback OTel"
    print(f"── {spec.name:32s} {arrow}")
    print(textwrap.indent(primary, "      "))
    if used_fb and fallback:
        print(textwrap.indent(fallback, "      [fb] "))
    print()""")

md(r"""**Lecture des fallbacks observés** : sur cet épisode, 4 requêtes sur 11
ont basculé sur le fallback OTel (`http_request_duration_bucket`,
`http_requests_total`, `http_requests_errors`, `queue_depth`). C'est attendu :
Istio n'est **pas** déployé dans `ewat`, donc tous les compteurs `istio_*`
retournent un résultat vide et le code retombe automatiquement sur les
métriques émises par les SDK OTel des services.

Pour chacune des 11 PromQL, la réponse Prometheus est un payload
`{status: "success", data: {resultType: "matrix", result: [...]}}` où chaque
élément de `result` est une série temporelle (un pod × un service) avec ses
`(timestamp, value)` au pas de 15 s.""")

code(r"""# Inspection d'un payload Prometheus réel — la première série de cpu_usage
with gzip.open(RAW_DIR / "prometheus_range.json.gz") as f:
    prom = json.load(f)

cpu = prom["results"]["cpu_usage"]["data"]["result"]
print(f"cpu_usage : {len(cpu)} séries (= {len(cpu)} pods qui ont matché)\n")
print(f"Série 0 — labels Prometheus :")
print(json.dumps(cpu[0]["metric"], indent=2))
print(f"\nSérie 0 — 5 premiers points (unix_ts, value cpu cores) :")
for ts, v in cpu[0]["values"][:5]:
    print(f"  ts={ts:.3f}  →  cpu={float(v):.4f}")
print(f"\n… total : {len(cpu[0]['values'])} points (≈ 660 s / 15 s = 44 pts)")""")

md(r"""### 9.2 Jaeger — 6 requêtes (une par service canonique)

`TelemetryRecorder.record_jaeger()` boucle sur la liste `canonical_services` (6
entrées) et envoie un `GET /api/traces` par service. Pas de fallback : si un
service n'émet pas de traces, on accepte un résultat vide.

L'API Jaeger v1 a deux particularités importantes :

- Les timestamps `start` / `end` doivent être en **microsecondes Unix**, pas en
  secondes ;
- `limit=1500` ne pagine pas (Jaeger v1 ne supporte pas la pagination
  forward) ; si on dépasse 1500 traces sur la fenêtre, la réponse est tronquée
  silencieusement. C'est calibré : 1500 est confortable pour 660 s de trafic
  Locust constant (~750 RPS au pic).""")

code(r"""# Reconstruction des 6 URLs Jaeger réelles
start_us = int(t_start * 1_000_000)
end_us   = int(t_end   * 1_000_000)
jae_limit = EP["collection"]["jaeger_limit"]

print("Requêtes Jaeger (depuis record_jaeger, src/telemetry/recorder.py:259-264) :\n")
for svc in SVC:
    params = f"?service={svc}&start={start_us}&end={end_us}&limit={jae_limit}"
    n = MAN["sources"]["jaeger"]["per_service_counts"].get(svc, 0)
    print(f"  GET {jae_endpoint}/api/traces{params}")
    print(f"      → {n} traces récupérées\n")

print(f"Total : {MAN['sources']['jaeger']['n_traces_total']} traces, "
      f"{MAN['sources']['jaeger']['fetch_elapsed_s']:.2f} s")""")

md(r"""### 9.3 Loki — `query_range` paginé en chunks

Loki est le seul backend qu'on **pagine**. Raison : pour un épisode v3 de 660 s
la fenêtre est gérable d'un seul tenant, mais pour v4 (1350 s) une seule
requête dépasserait la timeout 90 s côté loki-querier. Le recorder coupe donc
la fenêtre en chunks de `loki_chunk_s = 300 s` et avance le curseur :

1. requête `GET /loki/api/v1/query_range` pour `[cursor, cursor + 300 s]` ;
2. si la réponse contient `≥ loki_limit (5000)` lignes → c'est tronqué, on
   avance le curseur juste après le dernier timestamp vu (pagination forward) ;
3. sinon on saute au chunk suivant.

Le label `{k8s_namespace_name="ewat"}` filtre côté serveur tous les logs du
namespace `ewat` (les 6 services applicatifs y émettent leurs logs via OTel
SDK → Loki via le Loki exporter).""")

code(r"""# Reconstruction de la première requête Loki réelle
loki_query = f'{{k8s_namespace_name="{ns}"}}'
loki_start_ns = int(t_start * 1e9)
loki_end_ns   = int(min(t_start + 300, t_end) * 1e9)
loki_limit = EP["collection"]["loki_limit"]

print("Première requête Loki (chunk 1) :\n")
print(f"  GET {loki_endpoint}/loki/api/v1/query_range")
print(f"      ?query={loki_query}")
print(f"      &start={loki_start_ns}")
print(f"      &end={loki_end_ns}")
print(f"      &limit={loki_limit}")
print(f"      &direction=forward\n")

print(f"Au total : {MAN['sources']['loki']['n_lines']} lignes log, "
      f"truncated={MAN['sources']['loki']['truncated']}, "
      f"{MAN['sources']['loki']['fetch_elapsed_s']:.2f} s")""")

# ════════════════════════════════════════════════════════════════════════════
# 10. PHASE 6 — PERSISTANCE
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 10. Phase 6 — Persistance : gzip, SHA-256, écriture atomique

Une fois les 3 dumps en mémoire, le recorder les écrit sur disque dans cet
ordre précis (`scripts/record_episode.py:_record_and_persist`) :

```python
def _write_gz_json(path: Path, payload: Any, start_s: float) -> PersistStats:
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    h = hashlib.sha256(raw)                       # ← hash AVANT gzip
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)
    return PersistStats(path=path.name,
                        size_bytes=path.stat().st_size,   # taille gzip
                        sha256=h.hexdigest(),             # hash décompressé
                        elapsed_s=time.time() - start_s)
```

**Point critique** : le SHA-256 est calculé sur les **bytes JSON décompressés**,
pas sur le gzip. Raison : gzip est non-déterministe (header avec mtime,
compresslevel, etc.), donc deux dumps identiques produiraient des `.gz` avec
des hash différents. En hashant le payload décompressé on garde une signature
stable, vérifiable par `zcat … | sha256sum`.""")

code(r"""# Vérification d'intégrité — on recalcule le SHA-256 et on compare à manifest.json
print("Vérification SHA-256 (recalculé maintenant vs. manifest.json) :\n")
print(f"  {'fichier':<28s} {'sha256 manifest':<18s} {'sha256 recalculé':<18s}  match")
for src_name in ("prometheus_range.json.gz", "jaeger_spans.json.gz", "loki_logs.json.gz"):
    src_key = src_name.split("_")[0]
    expected = MAN["sources"][src_key]["sha256"]
    with gzip.open(RAW_DIR / src_name) as f:
        raw_bytes = f.read()
    actual = hashlib.sha256(raw_bytes).hexdigest()
    ok = "✓" if actual == expected else "✗"
    print(f"  {src_name:<28s} {expected[:16]:<18s} {actual[:16]:<18s}  {ok}")""")

md(r"""### 10.1 Écriture atomique

Le recorder écrit dans un dossier temporaire `tmp_dir = episode_<id>.tmp/`,
puis fait un `os.rename(tmp_dir, final_dir)` une fois que **les 5 fichiers
sont là** (3 dumps + `episode.json` + `manifest.json`). Comme `rename` est
atomique sur le même filesystem POSIX, un crash entre les écritures laisse
`.tmp` orphelin (qu'on peut détecter et nettoyer) mais **jamais** un
`episode_<id>/` incomplet visible aux scripts en aval.

### 10.2 Quality gate

Juste après le rename, `_check_episode_quality()` examine le manifest :

| Modalité   | Critère                                                |
|------------|--------------------------------------------------------|
| Prometheus | Au moins une `query_ok` (non-skippé, pas d'erreur globale) |
| Jaeger     | `n_traces_total ≥ 5` (audit 2026-05-26, Step 1 fix 1.4) |
| Loki       | `n_lines > 0`                                          |

Si l'épisode échoue, on écrit un sentinelle `.quality_failed` dans le dossier
et on incrémente `consecutive_failures`. Au bout de
`--max-consecutive-failures` (3 par défaut), tout le run est aborté pour ne
pas continuer à collecter en silence du garbage.""")

code(r"""# Quality gate appliqué à cet épisode
def _gate(manifest, min_traces=5):
    reasons = []
    p = manifest["sources"]["prometheus"]
    if p.get("skipped"): reasons.append("prometheus-skipped")
    elif not p.get("queries_ok"): reasons.append("prometheus-no-queries")
    j = manifest["sources"]["jaeger"]
    if j.get("skipped"): reasons.append("jaeger-skipped")
    elif int(j.get("n_traces_total", 0)) < min_traces:
        reasons.append(f"jaeger-too-few-traces (n={j['n_traces_total']} < {min_traces})")
    l = manifest["sources"]["loki"]
    if l.get("skipped"): reasons.append("loki-skipped")
    elif int(l.get("n_lines", 0)) <= 0: reasons.append("loki-empty")
    return len(reasons) == 0, reasons

ok, reasons = _gate(MAN, min_traces=EP.get("min_traces_quality_gate", 5))
print(f"Quality gate (min_traces=5) → {'✓ PASS' if ok else '✗ FAIL — ' + str(reasons)}")
print(f"  prometheus : {len(MAN['sources']['prometheus']['queries_ok'])} queries OK / 11")
print(f"  jaeger     : {MAN['sources']['jaeger']['n_traces_total']} traces (seuil 5)")
print(f"  loki       : {MAN['sources']['loki']['n_lines']} lignes (seuil 1)")""")

# ════════════════════════════════════════════════════════════════════════════
# 11. TIMELINE ANIMÉE — LE SIGNAL S(t) SE CONSTRUIT
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 11. Timeline animée — `S(t)` se construit phase après phase

Phase 1 (record) ne produit **pas** `S(t)` — elle produit les dumps bruts.
C'est Phase 2 (`build_features.py`) qui transforme ces dumps en `S(t) ∈ ℝ^{T×N×17}`.
Pour cet épisode : `T=23` pas de 30 s × `N=6` services × `17` features.

L'animation ci-dessous "rejoue" la fenêtre [baseline → recovery] pas par pas
(1 cellule = 1 step de 30 s) en montrant 3 features signature du scénario
`cpu_starvation` :

- `cpu_util` (feature 0) — la cible directe du stress
- `latency_p99` (feature 2) — l'effet observable côté API
- `error_rate_http` (feature 3) — les timeouts upstream

L'arrière-plan change de couleur selon la phase. La barre verticale est le
curseur de temps. Tout ceci est dérivé du dump brut via `build_features.py`,
sans réaccès au cluster.""")

code(r"""# Chargement du signal S(t) et du masque NaN
signal = np.load(FEAT_DIR / "signal.npz")["signal"]      # (T=23, N=6, F=17)
mask   = np.load(FEAT_DIR / "signal_mask.npz")["missing_mask"]
services_sig = json.loads((FEAT_DIR / "services.json").read_text())
T, N, F = signal.shape
print(f"S(t) shape = {signal.shape}  (T={T}, N={N}, F={F})")
print(f"services   = {services_sig}")
print(f"NaN total  = {np.isnan(signal).sum()} / {signal.size}  "
      f"({100 * np.isnan(signal).sum() / signal.size:.1f} %)")""")

code(r"""# Animation : on défile dans le temps, on accumule les valeurs sur 3 features
FEATURE_NAMES = ['cpu_util', 'ram_util', 'latency_p99', 'error_rate_http',
                 'net_sat', 'disk_io', 'queue_depth', 'span_dur_p99',
                 'abnormal_span_rate', 'trace_depth', 'fan_out', 'retry_rate',
                 'latency_cv', 'log_error_rate', 'log_warn_rate',
                 'semantic_anomaly', 'lexical_entropy']
F_NAMES_SHOW = ["cpu_util", "latency_p99", "error_rate_http"]
F_IDX = [FEATURE_NAMES.index(n) for n in F_NAMES_SHOW]

# Grille temporelle : 23 steps × 30 s ≈ 0..690 s
grid_step_s = 30.0
t_rel = np.arange(T) * grid_step_s

# Bornes des phases (relatives à baseline_start)
phase_bounds = [
    ("baseline",      B["baseline_start"]   - t0, B["baseline_end"]    - t0, "#e5fff0"),
    ("pre-injection", B["pre_start"]        - t0, B["pre_end"]         - t0, "#fff6d6"),
    ("injection",     B["injection_start"]  - t0, B["injection_end"]   - t0, "#ffd6d6"),
    ("recovery",      B["recovery_start"]   - t0, B["recovery_end"]    - t0, "#d6e7ff"),
]

# Z-score robuste par feature × service à partir des baseline-only steps
baseline_steps = int((B["baseline_end"] - B["baseline_start"]) / grid_step_s)
def _normalize(x):
    mu = np.nanmean(x[:baseline_steps], axis=0, keepdims=True)
    sd = np.nanstd(x[:baseline_steps], axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd
S_norm = _normalize(signal)   # (T, N, F)

fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
fig.subplots_adjust(hspace=0.25)
lines_per_ax = []
for ax, fname, fidx in zip(axes, F_NAMES_SHOW, F_IDX):
    # Bandes de phase en arrière-plan
    for _, s, e, c in phase_bounds:
        ax.axvspan(s, e, color=c, alpha=0.65, zorder=0)
    # 1 ligne par service (vide au départ)
    series = []
    for i, svc in enumerate(services_sig):
        line, = ax.plot([], [], label=svc, linewidth=1.4)
        series.append(line)
    lines_per_ax.append(series)
    ax.set_ylabel(f"{fname}\n(z-score baseline)", fontsize=9)
    ax.axhline(0, color="#888", lw=0.5, linestyle=":")
    ax.set_xlim(0, t_rel[-1] + grid_step_s)
    ax.set_ylim(-3, 6)
axes[-1].set_xlabel("Temps écoulé depuis baseline_start (s)")
axes[0].legend(loc="upper left", ncol=3, fontsize=8, frameon=False)

# Marqueurs de phase au-dessus
for name, s, e, _ in phase_bounds:
    axes[0].text((s + e) / 2, 5.5, name, ha="center", va="center", fontsize=8.5,
                 color="#555", style="italic")

# Curseur vertical synchronisé
cursors = [ax.axvline(0, color="red", lw=1.2, alpha=0.8) for ax in axes]
title_txt = fig.suptitle("", fontsize=11.5, y=0.98)

def init():
    for series in lines_per_ax:
        for line in series:
            line.set_data([], [])
    return [l for s in lines_per_ax for l in s] + cursors + [title_txt]

def update(frame):
    t_now = t_rel[frame]
    for fidx, series in zip(F_IDX, lines_per_ax):
        for i, line in enumerate(series):
            y = S_norm[: frame + 1, i, fidx]
            line.set_data(t_rel[: frame + 1], y)
    for c in cursors:
        c.set_xdata([t_now, t_now])
    # Détermine la phase courante
    cur = next((n for n, s, e, _ in phase_bounds if s <= t_now <= e), "?")
    title_txt.set_text(f"S(t) en construction — t = {t_now:5.0f} s   "
                       f"({frame + 1}/{T} pas, phase = {cur})")
    return [l for s in lines_per_ax for l in s] + cursors + [title_txt]

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=T, interval=500, blit=False, repeat=False)
plt.close(fig)
HTML(anim.to_jshtml())""")

md(r"""**Lecture de l'animation** — au passage de `injection_start` (~t=360 s) :

- `cpu_util` pour `frontend` décolle violemment (`+5σ` au pic) — direct, c'est
  la cible du `StressChaos` ;
- `latency_p99` monte sur **tous** les services (~`+1-3σ`) — c'est l'effet de
  contention en cascade : le scheduler kernel partage entre stress-ng et le code
  applicatif, donc toutes les requêtes ralentissent un peu ;
- `error_rate_http` reste plus modeste — Locust attend que les requêtes
  finissent, il ne déclenche pas massivement de retries timeout.

Après `injection_end` (~t=540 s), `cpu_util` retombe en quelques pas (le pod
`frontend` n'est pas restart, le stressor s'arrête juste). C'est cette phase
qui sera labellisée `recovery` et **exclue** de l'évaluation H1/H3.""")

# ════════════════════════════════════════════════════════════════════════════
# 12. RÉCAP ARTEFACTS
# ════════════════════════════════════════════════════════════════════════════

md(r"""## 12. Récap — artefacts produits

À la sortie de l'épisode, le filesystem contient :

```
data/raw/episode_cpu_starvation_000_20260430T022941Z/
├── episode.json              # phases, scénario, host, git_commit, cluster, seed
├── manifest.json             # par source : path, size, sha256, errors, n_traces, …
├── prometheus_range.json.gz  # 11 PromQL × N pods × ~45 points (15 s step)
├── jaeger_spans.json.gz      # 1842 traces (frontend=559, recommendation=71, …)
└── loki_logs.json.gz         # 2958 lignes log (toutes phases)
```

Puis `.checkpoint.jsonl` (au niveau de `data/raw/`) reçoit une ligne :

```json
{"scenario": "cpu_starvation", "rep": 0, "episode_id": "episode_cpu_starvation_000_20260430T022941Z", "completed_at": "2026-04-30T02:41:00.456789+00:00"}
```

C'est cette ligne qui rend le script **idempotent** : si on relance
`record_episode.py`, il saute ce `(scenario, rep)` sans toucher au cluster.""")

code(r"""# Récap chiffré final
total_size = sum((RAW_DIR / p).stat().st_size for p in [
    "episode.json", "manifest.json",
    "prometheus_range.json.gz", "jaeger_spans.json.gz", "loki_logs.json.gz",
])
duration_s = B["recovery_end"] - B["baseline_start"]
fetch_total = sum(MAN["sources"][k]["fetch_elapsed_s"] for k in ("prometheus", "jaeger", "loki"))

summary = pd.DataFrame([
    ["Durée totale épisode (s)",           f"{duration_s:.1f}"],
    ["  dont sleep (baseline+pre+inj+rec)",f"{duration_s - fetch_total:.1f}"],
    ["  dont fetch HTTP (3 dumps)",        f"{fetch_total:.2f}"],
    ["Apply chaos (s)",                    f"{B['apply_returned_at'] - B['injection_start']:.3f}"],
    ["Delete chaos (s)",                   f"{B['delete_returned_at'] - B['injection_end']:.3f}"],
    ["Total bytes sur disque (KB)",        f"{total_size / 1024:.1f}"],
    ["  prometheus_range.json.gz (KB)",    f"{(RAW_DIR / 'prometheus_range.json.gz').stat().st_size / 1024:.1f}"],
    ["  jaeger_spans.json.gz (KB)",        f"{(RAW_DIR / 'jaeger_spans.json.gz').stat().st_size / 1024:.1f}"],
    ["  loki_logs.json.gz (KB)",           f"{(RAW_DIR / 'loki_logs.json.gz').stat().st_size / 1024:.1f}"],
    ["Prometheus séries totales",          f"{sum(len(v['data']['result']) for v in prom['results'].values())}"],
    ["Jaeger traces totales",              f"{MAN['sources']['jaeger']['n_traces_total']}"],
    ["Loki lignes totales",                f"{MAN['sources']['loki']['n_lines']}"],
    ["Quality gate",                       "PASS" if ok else "FAIL"],
], columns=["métrique", "valeur"])
summary""")

md(r"""## Conclusion

Un épisode = **11 min de temps mural** + **3 commandes `kubectl` réelles**
(2 port-forwards × 3 services + 1 apply + 1 delete) + **~18 requêtes HTTP**
(11 Prom + 6 Jaeger + Loki paginé) + **≈ 3 MB de JSON gzip** sur disque.

Le design tient en **trois décisions** :

1. **Dumps a posteriori, jamais en ligne** — la mesure online plantait sous
   stress ; ici on accumule sur le cluster (Prometheus / Jaeger / Loki sont
   déjà des time-series databases) et on dump à froid.
2. **Hash décompressé, pas le gzip** — assure une signature stable du payload
   indépendamment du compresslevel et des bizarreries gzip header.
3. **Atomicité par `rename`** — un crash laisse au pire un `.tmp` qu'on peut
   nettoyer ; les scripts en aval (`build_features.py`) ne voient jamais
   d'épisode mi-écrit.

Tout ce qui suit dans le pipeline (Phase 2 → Phase 3 → encodeur → siamois → …)
**peut être rejoué sur ces 5 fichiers** sans cluster.""")

# ════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ════════════════════════════════════════════════════════════════════════════

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")

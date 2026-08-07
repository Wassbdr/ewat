# Collecte de dataset EWAT — commencer ici

Point d'entrée unique pour produire un dataset EWAT. Tout ce qui suit est
opérationnel ; les documents détaillés sont référencés au fur et à mesure plutôt
que recopiés, pour qu'il n'existe qu'une seule source de vérité par sujet.

**État courant (2026-08).** La collecte de référence est **v5 / Train Ticket**,
41 services, schéma de features **v5.1 (18 features)**. Les campagnes v1 à v4
portaient sur une topologie à 6 services et un schéma à 17 features ; elles sont
historiques. Si un document vous parle de « 6 services canoniques » ou de « 17
features », il décrit v4 — vérifiez toujours contre
[`datasets.md`](datasets.md), qui fait foi sur *quelle version fait foi*.

---

## 1. Le pipeline en trois phases

La règle non négociable : **Record → Build → Assemble**, dans cet ordre, jamais
en boucle. La phase 1 écrit des dumps bruts, la phase 2 en dérive des features
hors ligne, la phase 3 consolide. Les dumps de `data/raw*/` sont **sacrés** :
jamais modifiés sur place, toujours réécrits ailleurs. C'est ce qui permet de
recalculer tout le dataset après un correctif de featurisation, sans retourner
sur le cluster.

| Phase | Ce qu'elle fait | Entrée → sortie | Où |
|---|---|---|---|
| **1. Record** | injecte le chaos, pompe les 3 sources | cluster → `data/raw_v5/<ep>/` (gzip) | sur la VM, en ligne |
| **2. Build** | dumps → `S(t)`, `G(t)`, masque, labels | `raw_v5/` → `data/features/v5/<ep>/` | hors ligne, parallélisable |
| **3. Assemble** | consolide et découpe | `features/v5/` → `data/datasets/ewat_v5` | hors ligne, instantané |

La phase 2 est **hors ligne et rejouable**. C'est délibéré : les collecteurs de
`src/telemetry/collectors/` portent la logique de features en ligne, les
extracteurs de `src/telemetry/extractors/` rejouent la même logique sur les
dumps. Un bug de featurisation ne coûte donc jamais une recollecte.

### Les commandes

Le runbook opérationnel complet — trois runners parallèles, décalage des ports,
garde-fou RAM, suivi quotidien — est dans **[`../v5/LAUNCH.md`](../v5/LAUNCH.md)**.
Il est à jour et testé ; ne le paraphrasez pas, suivez-le.

En version minimale, pour un épisode isolé :

```bash
cd v5 && export PYTHONPATH=../src

# Phase 1 — un épisode : charge + baseline → injection → recovery + collecte
python -m collect.run_episode --namespace tt --scenario cpu_stress \
    --intensity high --out ../data/raw_v5/<episode_id>

# Phase 2 — featurisation hors ligne (batch : --raw-root ... --workers N)
python -m collect.build_features_v5 --dump ../data/raw_v5/<ep> --out ../data/features/v5/<ep> \
    --episode-id <ep> --scenario cpu_stress --category contention --step 30

# Phase 3 — assemblage stratifié + held-out en test seulement
cd .. && python -m scripts.assemble_dataset \
    --features-root data/features/v5 --output data/datasets/ewat_v5 --stratified
```

### Les portes de qualité

`validate_dataset` s'exécute à trois granularités et **doit passer avant toute
expérience** :

```bash
python -m scripts.validate_dataset --episode data/features/v5/<ep>   # un épisode
python -m scripts.validate_dataset --features-root data/features/v5  # tous
python -m scripts.validate_dataset --dataset data/datasets/ewat_v5   # l'assemblé
```

Un épisode est écarté si plus de 50 % de NaN — sur v5, 202 épisodes sur 611 l'ont
été. Ce n'est pas anormal : c'est le prix d'une collecte sur cluster partagé, et
c'est pour ça que la cible de collecte est largement surdimensionnée par rapport
au besoin.

Depuis l'audit 2026-06, `assemble_dataset` est **stratifié par défaut**, route
les épisodes `held_out_flag=True` en test uniquement, et **refuse** un scénario
absent du test. Ces trois comportements viennent d'un incident réel : le dataset
`ewat_v4`, découpé temporellement, avait quatre scénarios absents du train, ce
qui donnait un AUROC de 0,500 trivial. `ewat_v4` est conservé comme pièce à
conviction — ne l'utilisez pas.

---

## 2. Ce que contient le signal — provenance feature par feature

Schéma **v5.1, 18 features**. Cette table est la référence ; elle n'existait
jusqu'ici que dans la docstring de `v5/collect/build_features_v5.py`.

| # | Feature | Source |
|---|---|---|
| M0 | `cpu_util` | cAdvisor (dump Prometheus) |
| M1 | `ram_util` | cAdvisor |
| M2 | `latency_p99` | `SpanLatencyIndex` — **traces**, pas Istio |
| M3 | `error_rate_http` | `SpanErrorRateIndex` — **traces** |
| M4 | `net_sat` | cAdvisor |
| M5 | `disk_io` | cAdvisor |
| M6 | `mem_limit_ratio` | cAdvisor — working set / limite (saturation) |
| M7 | `jvm_heap_ratio` | `jmx_prometheus_javaagent` (annotations) |
| M8 | `jvm_gc_util` | idem |
| M9 | `jvm_threads_live` | idem |
| T10 | `abnormal_span_rate` | `TraceCollector` |
| T11 | `trace_depth` | `TraceCollector` |
| T12 | `fan_out` | `TraceCollector` |
| T13 | `latency_cv` | `TraceCollector` |
| L14 | `log_error_rate` | Loki |
| L15 | `restart_count` | kube-state-metrics (dump Prometheus) |
| L16 | `semantic_anomaly` | SentenceBERT — `collect/semantic.py` |
| L17 | `lexical_entropy` | Loki |

Deux points qu'un repreneur doit savoir : **Train Ticket n'a ni Istio ni OTel
HTTP**, donc latence et taux d'erreur viennent des traces Jaeger, pas de
métriques de service mesh — c'est le changement de fond entre v4 et v5.1. Et le
schéma est **versionné par épisode** : `metadata.signal_feature_names` fait foi,
pas une constante globale. `src/telemetry/feature_names.py` porte encore le
schéma v4 à 17 features ; ne le prenez pas pour la référence v5.

`G(t) ∈ ℝ^{T×N×N×3}` est construit par `compute_graph_for_window` : volume,
latence médiane, taux d'erreur par arête.

Les définitions mathématiques (agrégations intra-service, régimes θ, budget de
latence) sont dans [`formalisation.md`](formalisation.md).

---

## 3. Les scénarios injectés

Le catalogue est **`v5/chaos/catalog.yaml`** — 28 scénarios plus 5 bugs réels.
Il est abondamment commenté : les choix d'intensité y sont justifiés, y compris
les corrections (par exemple le passage de `size: "80%"` à des tailles mémoire
absolues, parce que le pourcentage était ambigu — cgroup du conteneur ou mémoire
du nœud ?).

| Catégorie | Nombre | Nature |
|---|---:|---|
| `gray` | 6 | dégradation lente ou partielle |
| `hard` | 5 | panne franche (crash, kill) |
| `contention` | 4 | pression ressource |
| `compo` | 4 | composition — cascade ou concurrence |
| `drift` | 3 | dérive bénigne, sans panne |
| `held_out` | 3 | réservés au test, jamais en entraînement |
| `overlap` | 2 | régime θ(drift ∩ anomaly) |
| `normal` | 1 | référence sans injection |

```bash
cd v5 && python -m chaos.inject list
python -m chaos.inject apply cpu_stress --intensity high --duration 600s
python -m chaos.inject delete cpu_stress
```

Les intensités `low / med / high` servent au ramp-up. La catégorie `drift` est
essentielle et souvent oubliée : sans elle, impossible de calibrer le seuil
ε_drift du détecteur, ni de vérifier qu'un déploiement bénin ne déclenche pas
d'alerte.

---

## 4. Prérequis cluster

- **Contexte kubectl** épinglé — `observit-cluster1` par défaut ; sur une autre
  VM, exporter `V5_KUBE_CONTEXT`. Tous les outils v5 font un préflight bloquant.
- **Chaos Mesh en mode cluster-scoped** — l'installation initiale demande un
  cluster-admin, ce que le compte de stage n'est pas. À demander à l'admin.
- **NodePort Prometheus + Loki** créés une fois :
  `kubectl apply -f v5/deploy/monitoring_nodeports.yaml`. Depuis v5.2 la
  télémétrie passe en TCP direct, **sans port-forward** — les tunnels SPDY
  lâchaient sous la contention de trois runners et faisaient perdre des épisodes.
- **Droits** : namespace-admin sur `ewat` uniquement. Jamais de ressource
  cluster-wide, jamais de namespace système. Voir [`../agents.md`](../agents.md).
- **RAM, pas CPU, est la contrainte** : ~20 Go par runner (41 JVM + mongos).
  À trois runners les workers sont à 80-87 %. Le garde-fou `--ram-ceiling`
  (défaut 90) met en pause avant un épisode plutôt que de risquer une éviction.

La liste de pré-vol complète, avec les vérifications bloquantes, est dans
[`../v5/PREFLIGHT.md`](../v5/PREFLIGHT.md).

---

## 5. Pièges connus

Ils sont tous documentés parce qu'ils ont tous coûté des épisodes.

- **Un nœud `NotReady` fausse silencieusement une feature.** Sur v3, `disk_io`
  était à 16,7 % de NaN à cause d'un seul worker en panne. Vérifiez
  `kubectl top nodes` avant, pas après.
- **Une panne d'infra d'observabilité vide une modalité entière.** Sur v4, une
  indisponibilité de Loki (7-13 mai) a mis 32 épisodes à 100 % de NaN sur `L`,
  et une de Jaeger 4 épisodes sur `T`. Les épisodes ont dû être recollectés.
  Surveillez le taux de NaN par modalité **pendant** la campagne.
- **Des épisodes trop courts invalident le look-through.** Le détecteur de drift
  a besoin d'un warm-up (référence + post-drift) ; sur des épisodes de ~21 pas,
  il ne reste presque rien d'exploitable, et l'hypothèse H2a a échoué pour cette
  raison — pas par défaut de conception. Visez des épisodes longs.
- **Le split temporel est un piège sur peu de scénarios.** Voir plus haut :
  stratifié par défaut depuis l'audit 2026-06.

---

## 6. Lancer une nouvelle campagne (v6)

Ce qui est spécifique à v5 : la topologie Train Ticket, le catalogue de
scénarios, les endpoints NodePort. Ce qui est réutilisable tel quel : les trois
phases, le contrat par épisode, `assemble_dataset`, `validate_dataset`, et tout
`src/telemetry`.

Pour une nouvelle campagne :

1. Déclarer la version dans [`datasets.md`](datasets.md) **avant** de collecter —
   c'est la page qui tranche « quelle version fait foi ».
2. Adapter `v5/chaos/catalog.yaml` (cibles, intensités) et vérifier chaque type
   de chaos sur un épisode pilote avant la campagne complète.
3. Si le schéma de features change, l'inscrire dans `telemetry.feature_names` et
   vérifier que `metadata.signal_feature_names` est bien écrit par épisode —
   c'est ce qui permet à un dataset ancien de rester lisible.
4. Geler le protocole d'évaluation **avant** de regarder les résultats, comme
   [`evaluation_protocol_v5.md`](evaluation_protocol_v5.md) le fait pour v5.
5. Faire tourner des pilotes bout-en-bout (record → build → gate → audit de
   fuite) sur un scénario par catégorie. `PREFLIGHT.md` montre à quoi ressemble
   ce contrôle.

**Point de vigilance méthodologique, ajouté en août 2026.** L'analyse conduite
dans le dépôt `fusion` a montré que sur les jeux publics RCAEval, le service
ciblé est identifiable **avant** l'injection du chaos — la préparation de
l'expérience laisse une trace sur le conteneur visé. Toute nouvelle collecte
doit être auditée de la même façon : entraîner un modèle sur des fenêtres
strictement antérieures à l'injection et vérifier qu'il reste au niveau du
hasard. `fusion audit` fait exactement cela. Concrètement, pour la collecte :
évitez de toucher le pod cible avant l'injection (pas de `kubectl exec`, pas de
copie de fichier), et injectez via un opérateur qui agit à distance.

---

## 7. Carte des documents

| Document | Ce qu'il couvre | À jour |
|---|---|:--:|
| **ce fichier** | point d'entrée, provenance des features, pièges | ✅ |
| [`datasets.md`](datasets.md) | nomenclature, **quelle version fait foi** | ✅ |
| [`../v5/LAUNCH.md`](../v5/LAUNCH.md) | runbook opérationnel 3 runners | ✅ |
| [`../v5/PREFLIGHT.md`](../v5/PREFLIGHT.md) | vérifications avant campagne | ✅ |
| [`../v5/README.md`](../v5/README.md) | infrastructure v5, composants | ✅ |
| [`evaluation_protocol_v5.md`](evaluation_protocol_v5.md) | protocole gelé v5 | ✅ |
| [`formalisation.md`](formalisation.md) | définitions mathématiques, régimes θ | ✅ |
| [`runbook_v4.md`](runbook_v4.md) | collecte v4, 6 services | historique |
| [`limitations.md`](limitations.md) | limites connues, avec causes | ✅ |

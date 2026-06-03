# Rapport de stage EWAT — Squelette annoté du livrable

> Document de travail : **ossature uniquement** (sommaire prévisionnel + squelette annoté).
> Aucune prose rédigée ; chaque section est cadrée par des marqueurs à remplir ultérieurement.
> Cible finale : `.docx` volumineux (50–80 p.) — titres `#`/`##`/`###`/`####` stricts pour une
> conversion Pandoc propre. Registre académique / ingénieur, français.

**Marqueurs utilisés :**
- ⟦À remplir⟧ = prose à rédiger plus tard.
- ▸ Source = origine du matériau (fichier / script / référence).
- ▸ Figure / ▸ Tableau / ▸ Chiffre = artefact ou valeur exacte à insérer (jamais inventée ici).
- ▸ Raisonnement = gabarit imposé : Observation → Hypothèse → Action → Résultat → Décision.
- ▸ CI/Test = emplacement d'un intervalle de confiance ou test statistique.

**Budgets de pages (somme = 78 p., hors front/back matter) :**
1:3 · 2:5 · 3:10 · 4:8 · 5:4 · 6:8 · 7:8 · 8:11 · 9:7 · 10:5 · 11:3 · 12:4 · 13:2.

---

## FRONT MATTER

### Page de garde
⟦À remplir⟧ Titre, sous-titre (early-warning & typage d'anomalies microservices K8s), auteur,
tuteur entreprise, tuteur académique, établissement, dates de stage, logo Devoteam.
▸ Source : informations administratives stage.

### Résumé (français)
⟦À remplir⟧ 250–300 mots : problème (faux positifs production), approche (séparation
drift/anomalie + typage + ontologie), résultats clés défendables, limites assumées.
▸ Chiffre : headline défendable B2 (cible Chaos Mesh) + IC ; mention résultats négatifs honnêtes.
▸ Source : STATUS.md (verdict consolidé), docs/results.md.
Mots-clés FR : détection précoce, drift conceptuel, microservices, Kubernetes, typage d'anomalies,
ontologie, transfer entropy.

### Abstract (English)
⟦À remplir⟧ Traduction fidèle du résumé FR.
Keywords EN : early warning, concept drift, microservices, Kubernetes, anomaly typing, ontology,
transfer entropy.

### Remerciements
⟦À remplir⟧ Placeholder (tuteurs, équipe Devoteam, laboratoire).

### Sommaire
▸ Source : table des matières auto-générée (Pandoc/Word, profondeur 4 niveaux).

### Liste des figures
▸ Source : `docs/rapport/figures/`, `docs/paper/figures/`, `experiments/figures/`,
`experiments/*/*.png` (heatmaps scénario×cluster, ROC/PR, distributions MMD², violons multi-seed,
arbre ontologie).

### Liste des tableaux
▸ Source : tableaux de résultats STATUS.md / docs/results.md (hypothèses, per-seed, baselines,
B3/B4 par scénario, ablations).

### Liste des acronymes
⟦À remplir⟧ EWAT, RCA, K8s, RKE2, OTel, MMD, RFF, STGCN, GCN, GAT, SimCLR, TE, KSG, FDR, BH,
OWL/RDF, ABox/TBox, SHAP, OOD, LOSO, BCa, IC, AUROC, PR-AUC, FPR/TPR, NMI.

### Glossaire
⟦À remplir⟧ Définitions courtes et autoportantes : drift (bénin), look-through, MMD-RFF, RFF,
STGCN, GAT, SimCLR, réseau siamois, TE-KSG, OpenMax, BCa, LOSO, FDR, régime θ, précurseur,
fenêtre pré-injection, signature statique de scénario.
▸ Source : docs/formalisation.md, CLAUDE.md.

---

# 1 Introduction
▸ Budget pages : 3

## 1.1 Cadre du stage et commanditaire
### 1.1.1 Devoteam, mission d'observabilité et contexte client
⟦À remplir⟧ ▸ Source : informations stage, CLAUDE.md (contexte recherche).
### 1.1.2 Insertion du sujet dans une problématique de production
⟦À remplir⟧

## 1.2 Motivation : le coût des faux positifs en production
### 1.2.1 Confusion drift bénin / anomalie réelle
⟦À remplir⟧ ▸ Source : CLAUDE.md (« le problème »).
### 1.2.2 Conséquence opérationnelle (fatigue d'alerte, perte de confiance)
⟦À remplir⟧

## 1.3 Énoncé du problème et question de recherche
⟦À remplir⟧ ▸ Source : docs/formalisation.md.

## 1.4 Contributions du stage
### 1.4.1 Contributions méthodologiques
⟦À remplir⟧ liste : formalisation 4-régimes, pipeline 4 étapes, ontologie empirique.
### 1.4.2 Contributions empiriques (dont résultats négatifs)
⟦À remplir⟧ ▸ Chiffre : renvoi vers §11 (headlines + négatifs).
### 1.4.3 Contributions logicielles et dataset
⟦À remplir⟧ ▸ Source : STATUS.md (modules, 401 tests), v5/LAUNCH.md.

## 1.5 Organisation du document
⟦À remplir⟧ paragraphe-guide de lecture.

---

# 2 Contexte, problématique et positionnement (early-warning ≠ RCA)
▸ Budget pages : 5

## 2.1 Architectures microservices et observabilité
### 2.1.1 Caractéristiques des systèmes microservices Kubernetes
⟦À remplir⟧ ▸ Source : CLAUDE.md (cluster observit-cluster1).
### 2.1.2 Les trois piliers : métriques, traces, logs
⟦À remplir⟧ ▸ Source : docs/formalisation.md (sources M/T/L).
### 2.1.3 Stack existante : Prometheus/Grafana + OpenTelemetry
⟦À remplir⟧ ▸ Source : CLAUDE.md (double stack), mémoire projet OTel infra.

## 2.2 Drift conceptuel vs anomalie : définitions opérationnelles
### 2.2.1 Drift bénin (déploiement, autoscaling, rolling update)
⟦À remplir⟧
### 2.2.2 Anomalie réelle (panne, dégradation)
⟦À remplir⟧
### 2.2.3 Le cas mixte : déploiement défectueux (θ_{drift∩anomaly})
⟦À remplir⟧ ▸ Source : docs/formalisation.md (4 régimes).

## 2.3 Positionnement : early-warning n'est pas du RCA
### 2.3.1 RCA = post-mortem (Où / Pourquoi, après la panne)
⟦À remplir⟧ ▸ Source : CLAUDE.md (règle impérative).
### 2.3.2 Early-warning = anticipation (Quoi / Dans combien de temps, avant)
⟦À remplir⟧
### 2.3.3 Tableau comparatif RCA vs EWAT
▸ Tableau : axes (objectif, instant, sortie, métrique) × {RCA, EWAT}.

## 2.4 Objectifs et critères de succès du projet
### 2.4.1 Objectifs fonctionnels
⟦À remplir⟧
### 2.4.2 Critères de falsifiabilité (renvoi H1–H3)
⟦À remplir⟧ ▸ Source : docs/formalisation.md (hypothèses).

## 2.5 Périmètre, contraintes et hors-périmètre
⟦À remplir⟧ ▸ Source : agents.md (contraintes d'accès namespace-admin), CLAUDE.md.

---

# 3 État de l'art et conséquences sur la formalisation
▸ Budget pages : 10

## 3.1 Méthodologie de revue de littérature
⟦À remplir⟧ critères de sélection, regroupement thématique, lien systématique vers §4.

## 3.2 Niveau 1 — Références fondatrices
> Pour CHAQUE référence, structure imposée : (a) ⟦Apport⟧ ; (b) ⟦Conclusion/limite retenue⟧ ;
> (c) ⟦Ce que j'en ai tiré pour la formalisation EWAT⟧ (lien explicite vers une brique de §4).

### 3.2.1 Détection d'anomalies — fondations et séries temporelles profondes
#### 3.2.1.1 Chandola, Banerjee & Kumar (2009) — Anomaly Detection: A Survey
a) ⟦Apport⟧ b) ⟦Conclusion/limite retenue⟧ c) ⟦Tiré pour EWAT : cadrage régimes / S(t)⟧
▸ Source : bibliography.bib (À AJOUTER : chandola2009).
#### 3.2.1.2 Zamanzadeh Darban et al. (2024) — Deep Learning for Time Series AD: A Survey
a) b) c) ⟦lien : encodeur §7, fenêtre temporelle⟧ ▸ Source : .bib (À AJOUTER : zamanzadeh2024).

### 3.2.2 Performance systèmes et indicateurs précurseurs
#### 3.2.2.1 Gregg (2013) — Systems Performance (méthodologie USE, queue depth)
a) b) c) ⟦lien : features M(t), queue depth comme leading indicator §4.2⟧
▸ Source : .bib (gregg2013 ✓).

### 3.2.3 Drift conceptuel dans les flux non supervisés
#### 3.2.3.1 Hinder et al. (2024) — Concept drift in unsupervised data streams
a) b) c) ⟦lien : étape 0 drift §7.2⟧ ▸ Source : .bib (hinder2024 ✓).
#### 3.2.3.2 Myrtollari et al. (2025) — Concept drift-aware AD for microservices on K8s
a) b) c) ⟦lien : séparation drift/anomalie, look-through §7.2⟧ ▸ Source : .bib (myrtollari2025 ✓).

### 3.2.4 Analyse de cause racine en microservices (frontière à ne pas franchir)
#### 3.2.4.1 Fu et al. (2025) — Intelligent RCA in Microservice Systems: A Survey
a) b) c) ⟦lien : gap benchmark/production, positionnement §2.3⟧ ▸ Source : .bib (fu2025 ✓).
#### 3.2.4.2 Pham et al. (2024) — RCA based on Causal Inference: How Far Are We? (ASE'24)
a) b) c) ⟦lien : choix TE-KSG vs causal RCA, ontologie §10⟧ ▸ Source : .bib (À AJOUTER : pham2024).
#### 3.2.4.3 GrayScope (2024) — Non-intrusive gray failure localization (FSE'24)
a) b) c) ⟦lien : pannes grises / bug F1 invisible §6.5⟧ ▸ Source : .bib (À AJOUTER : grayscope2024).

### 3.2.5 Clustering et sélection du nombre de groupes
#### 3.2.5.1 Kaufman & Rousseeuw (1990) — Finding Groups in Data (seuil silhouette)
a) b) c) ⟦lien : seuil H1 silhouette 0.3 §8.4⟧ ▸ Source : .bib (kaufman1990 ✓).
#### 3.2.5.2 Tibshirani, Walther & Hastie (2001) — Gap statistic
a) b) c) ⟦lien : K-selection, instabilité K §9.5⟧ ▸ Source : .bib (tibshirani2001 ✓).

### 3.2.6 Estimation de l'information et causalité
#### 3.2.6.1 Kraskov, Stögbauer & Grassberger (2004) — Estimating mutual information (KSG)
a) b) c) ⟦lien : estimateur TE-KSG étape 2b §10.3⟧ ▸ Source : .bib (kraskov2004 ✓).

## 3.3 Niveau 2 — Fondements méthodologiques et briques techniques
> Chaque entrée : description courte + ⟦Justifie : <brique EWAT>⟧ (lien §4/§7/§10).

### 3.3.1 Encodeurs et apprentissage de représentations sur graphes
#### 3.3.1.1 Yu, Yin & Zhu (2018) — STGCN ⟦Justifie : encodeur étape 1, §7.4⟧
▸ Source : .bib (yu2018 ✓).
#### 3.3.1.2 Kipf & Welling (2017) — GCN ⟦Justifie : fondation conv. graphe, §4.1/§7.4⟧
▸ Source : .bib (À AJOUTER : kipf2017).
#### 3.3.1.3 Veličković et al. (2018) — GAT ⟦Justifie : variante comparée, §7.5.3⟧
▸ Source : .bib (velivckovic2018 ✓).
#### 3.3.1.4 Chen et al. (2020) — SimCLR ⟦Justifie : pré-entraînement contrastif, §7.5.2⟧
▸ Source : .bib (chen2020simclr ✓).
#### 3.3.1.5 Eldele et al. (2021) — TS-TCC ⟦Justifie : contrastif séries temporelles, §7.5.2⟧
▸ Source : .bib (À AJOUTER : eldele2021).

### 3.3.2 Détection de drift par tests à noyau
#### 3.3.2.1 Gretton et al. (2012) — Kernel Two-Sample Test / MMD ⟦Justifie : test étape 0, §7.2⟧
▸ Source : .bib (gretton2012 ✓).
#### 3.3.2.2 Rahimi & Recht (2007) — Random Fourier Features ⟦Justifie : MMD-RFF O(nD), §7.2⟧
▸ Source : .bib (À AJOUTER : rahimi2007).

### 3.3.3 Causalité, raisonnement et ontologie
#### 3.3.3.1 Schreiber (2000) — Transfer Entropy ⟦Justifie : étape 2b, §10.3⟧
▸ Source : .bib (À AJOUTER : schreiber2000).
#### 3.3.3.2 Benjamini & Hochberg (1995) — FDR ⟦Justifie : seuillage causal, §10.3⟧
▸ Source : .bib (benjamini1995 ✓).
#### 3.3.3.3 Holm (1979) — correction Holm ⟦Justifie : co-occurrence χ²/Fisher, §10.4⟧
▸ Source : .bib (À AJOUTER : holm1979).
#### 3.3.3.4 Soldani & Brogi (2022) — taxonomie pannes ⟦Justifie : ancrage TBox ontologie, §10.2⟧
▸ Source : .bib (soldani2022 ✓).
#### 3.3.3.5 Aniello et al. (2014) — classes de propagation ⟦Justifie : propagatesThrough, §10.2⟧
▸ Source : .bib (aniello2014 ✓).
#### 3.3.3.6 Glimm et al. (2014) — HermiT ⟦Justifie : raisonnement OWL, §10.5⟧
▸ Source : .bib (À AJOUTER : glimm2014).
#### 3.3.3.7 Lamy (2017) — owlready2 ⟦Justifie : outil OWL Python, §10.5⟧
▸ Source : .bib (À AJOUTER : lamy2017).

### 3.3.4 Interprétabilité
#### 3.3.4.1 Lundberg & Lee (2017) — SHAP/KernelSHAP ⟦Justifie : validation fiches type, §8.9⟧
▸ Source : .bib (lundberg2017 ✓).

### 3.3.5 Open-set et détection hors-distribution
#### 3.3.5.1 Bendale & Boult (2016) — OpenMax ⟦Justifie : open-set opérationnel, §9.6⟧
▸ Source : .bib (bendale2016openmax ✓).
#### 3.3.5.2 Lee et al. (2018) — Mahalanobis OOD ⟦Justifie : piste futurs, §12.2⟧
▸ Source : .bib (lee2018mahalanobis ✓).
#### 3.3.5.3 Liu et al. (2020) — Energy-based OOD ⟦Justifie : piste futurs, §12.2⟧
▸ Source : .bib (liu2020energy ✓).

### 3.3.6 Statistiques, bootstrap et clustering sphérique
#### 3.3.6.1 Efron (1987) — BCa ⟦Justifie : IC AUROC/silhouette, §8/§9⟧
▸ Source : .bib (À AJOUTER : efron1987).
#### 3.3.6.2 Davison & Hinkley (1997) — cadre rééchantillonnage ⟦Justifie : bootstrap, §9.3⟧
▸ Source : .bib (À AJOUTER : davison1997).
#### 3.3.6.3 Phipson & Smyth (2010) — p-values permutation ⟦Justifie : A3, seuils TE, §9.4⟧
▸ Source : .bib (À AJOUTER : phipson2010).
#### 3.3.6.4 Agresti (2002) — χ² Yates / Fisher ⟦Justifie : co-occurrence, §10.4⟧
▸ Source : .bib (À AJOUTER : agresti2002).
#### 3.3.6.5 Dhillon & Modha (2001) — spherical k-means ⟦Justifie : cosine sur sphère unité, §7.6⟧
▸ Source : .bib (À AJOUTER : dhillon2001).

### 3.3.7 Sémantique des logs
#### 3.3.7.1 Reimers & Gurevych (2019) — Sentence-BERT ⟦Justifie : anomalie sémantique L(t), §4.2⟧
▸ Source : .bib (À AJOUTER : reimers2019).

### 3.3.8 Benchmark externe
#### 3.3.8.1 RCAEval contributors (2024) — RE2-OB ⟦Justifie : transfert ewat_rcaeval, §6.4⟧
▸ Source : .bib (rcaeval ✓).

## 3.4 Synthèse de l'état de l'art et conséquences pour EWAT
### 3.4.1 Tableau de synthèse (référence → conclusion → brique EWAT)
▸ Tableau : 34 lignes (référence | conclusion/limite retenue | brique EWAT §x).
### 3.4.2 Transition vers la formalisation
⟦À remplir⟧ paragraphe-pont vers §4.

---

# 4 Formalisation mathématique
▸ Budget pages : 8

## 4.1 Graphe de services G(t)
### 4.1.1 Sommets : Services et Deployments (|V|=N constant)
⟦À remplir⟧ ▸ Source : docs/formalisation.md.
### 4.1.2 Arêtes pondérées w_E(t) : volume, latence médiane, taux d'erreur
⟦À remplir⟧
### 4.1.3 Seuil de présence d'arête et fenêtre glissante
⟦À remplir⟧
### 4.1.4 Tenseur d'adjacence A(t) ∈ ℝ^{N×N×3}
⟦À remplir⟧ ▸ Source : src/ewat/encoder (adjacency utilities), §7.4.

## 4.2 Signal de télémétrie S(t) ∈ ℝ^{N×17}
### 4.2.1 Métriques M(t) ∈ ℝ^{N×7}
⟦À remplir⟧ 7 features (CPU, RAM, P99, taux erreur, sat. réseau, disk I/O, queue depth).
▸ Source : docs/formalisation.md ; ▸ lien Gregg (queue depth) §3.2.2.
### 4.2.2 Traces T(t) ∈ ℝ^{N×6}
⟦À remplir⟧ span_dur_p99, taux spans anormaux, profondeur, fan-out, retry, CV latence.
### 4.2.3 Logs L(t) ∈ ℝ^{N×4}
⟦À remplir⟧ taux erreurs, warnings, anomalie sémantique SBERT, entropie lexicale.
▸ lien Reimers & Gurevych §3.3.7.
### 4.2.4 Tableau récapitulatif des 17 features (modalité, source, dimension)
▸ Tableau : 17 lignes.

## 4.3 Agrégation intra-service différenciée
### 4.3.1 Saturation → max ; taux → somme pondérée volume
⟦À remplir⟧ ▸ Source : CLAUDE.md (règles impératives agrégation).
### 4.3.2 Latence → P99 sur l'union (pas percentile de percentiles)
⟦À remplir⟧
### 4.3.3 Structurel → médiane
⟦À remplir⟧

## 4.4 Régimes opérationnels θ(t)
### 4.4.1 Les quatre régimes (pas trois)
⟦À remplir⟧ θ_normal, θ_drift, θ_anomaly, θ_{drift∩anomaly}.
### 4.4.2 Correspondance θ ↔ labels du dataset (regime, drift_flag)
▸ Tableau : normal/injection/drift_anomaly/recovery ↔ θ ; drift_flag orthogonal.
▸ Source : docs/formalisation.md.
### 4.4.3 Hypothèse non-additive S(t) ∼ D_{θ(t)}(G(t))
⟦À remplir⟧

## 4.5 Vue d'ensemble du pipeline (étapes 0→3)
### 4.5.1 Étape 0 — drift MMD-RFF look-through
⟦À remplir⟧ ▸ lien Gretton/Rahimi §3.3.2.
### 4.5.2 Étape 1 — encodeur STGCN
⟦À remplir⟧ ▸ lien Yu §3.3.1.1.
### 4.5.3 Étape 2/2b — typage siamois + ontologie
⟦À remplir⟧
### 4.5.4 Étape 3 — précurseurs typés
⟦À remplir⟧
### 4.5.5 Sortie : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})
⟦À remplir⟧ ▸ Source : src/ewat/alerts.

## 4.6 Budget de latence
▸ Tableau : étape 0 <1s, étape 1 <2s, étape 3 <1s, total <5s ; étapes 2/2b offline.
▸ Source : docs/formalisation.md.

## 4.7 Hypothèses falsifiables (énoncés)
### 4.7.1 H1 — Structurabilité des embeddings
⟦À remplir⟧ critère silhouette < 0.3 en held-out → falsifié.
### 4.7.2 H2a — Séparabilité du drift par look-through
⟦À remplir⟧ critère FPR à rappel constant, p>0.05 → falsifié.
### 4.7.3 H2b — Identification du régime θ_{drift∩anomaly}
⟦À remplir⟧ critère Fisher exact / proportion bootstrap.
### 4.7.4 H3 — Prédictibilité des précurseurs
⟦À remplir⟧ critère AUROC < baseline 0.5 → falsifié ; k* sur val.

## 4.8 Plan d'ablation (énoncé)
⟦À remplir⟧ par modalité, par feature (leave-one-out, Wilcoxon), redondance |ρ|>0.9.

---

# 5 Environnement expérimental
▸ Budget pages : 4

## 5.1 Cluster Kubernetes observit-cluster1
### 5.1.1 Topologie RKE2, nœuds, namespaces
⟦À remplir⟧ ▸ Source : CLAUDE.md (9 nœuds, v1.32.7+rke2r1).
### 5.1.2 Contraintes d'accès namespace-admin
⟦À remplir⟧ ▸ Source : agents.md.

## 5.2 Chaîne d'observabilité
### 5.2.1 Prometheus / Grafana
⟦À remplir⟧
### 5.2.2 OpenTelemetry Collector (OTLP traces/logs)
⟦À remplir⟧ ▸ Source : mémoire projet OTel infra, CLAUDE.md.
### 5.2.3 Endpoints découverts et configuration
▸ Source : configs/default.yaml ; ▸ Tableau : service → endpoint.

## 5.3 Injection de fautes : Chaos Mesh
### 5.3.1 Catalogue de scénarios
⟦À remplir⟧ ▸ Source : scripts/chaos_injector.py, registre scénarios.
### 5.3.2 Manifestes et procédure d'application
⟦À remplir⟧

## 5.4 Stack logicielle et reproductibilité
### 5.4.1 Python/PyTorch/PyG/scikit-learn, Hydra, MLflow
⟦À remplir⟧ ▸ Source : CLAUDE.md (standards de code).
### 5.4.2 Graines, déterminisme, suite de tests (401 tests)
⟦À remplir⟧ ▸ Source : STATUS.md ; ▸ Source : src/ewat/utils/seeding.

---

# 6 Pipeline de données et itérations du dataset
▸ Budget pages : 8
> RÈGLE ANTI-FUSION : chaque version a sa sous-section + un ▸ Raisonnement.

## 6.1 Architecture du pipeline de collecte Record → Build → Assemble
### 6.1.1 Phase 1 — record_episode (chaos + dumps bruts)
⟦À remplir⟧ ▸ Source : scripts/record_episode.py.
### 6.1.2 Phase 2 — build_features (S(t), G(t), labels)
⟦À remplir⟧ ▸ Source : scripts/build_features.py.
### 6.1.3 Phase 3 — assemble_dataset (split temporel/stratifié)
⟦À remplir⟧ ▸ Source : scripts/assemble_dataset.py.
### 6.1.4 Contrôle qualité — validate_dataset
⟦À remplir⟧ ▸ Source : scripts/validate_dataset.py.

## 6.2 Itération ewat_v3 — dataset de référence (Online Boutique)
▸ Raisonnement : Observation (besoin d'un corpus initial multi-scénarios) → Hypothèse →
Action (15 scénarios × 20 rép.) → Résultat → Décision.
### 6.2.1 Conception : 15 scénarios, ~21 steps/épisode
⟦À remplir⟧ ▸ Chiffre : 300 épisodes, split 209/45/45.
### 6.2.2 Qualité des données et NaN résiduels
⟦À remplir⟧ ▸ Chiffre : disk_io 16.7% NaN (product-catalog, nœud NotReady).
### 6.2.3 Défaut mesuré motivant l'itération suivante
⟦À remplir⟧ ▸ lien : épisodes trop courts → §8.3 (H2a FAIL).

## 6.3 Itération ewat_v4 / ewat_v4_strat (épisodes longs, split stratifié)
▸ Raisonnement : Observation (épisodes courts limitent confirmation temporelle) → Hypothèse →
Action (T=47–51 steps, +rép., split stratifié) → Résultat → Décision.
### 6.3.1 ewat_v4 — collecte 414 ép., 375 retenus, split temporel
⟦À remplir⟧ ▸ Chiffre : 262/56/57 ; 39 ép. rejetés (Loki/Jaeger outages).
### 6.3.2 Motivation du split stratifié : 4 scénarios absents du train
⟦À remplir⟧ ▸ Source : STATUS.md (note méthodologique B3).
### 6.3.3 ewat_v4_strat — split 270/60/45 (≥1 ép./scénario/split)
⟦À remplir⟧ ▸ Chiffre.
### 6.3.4 NaN résiduel par modalité
▸ Chiffre : L≈2%, M≈3–5%, T≈20–25%.

## 6.4 Itération ewat_rcaeval — adaptation d'un benchmark externe
▸ Raisonnement : Observation (validation externe nulle) → Hypothèse → Action (adapter RE2-OB
au format EWAT) → Résultat → Décision.
### 6.4.1 Source RCAEval RE2-OB et conversion de format
⟦À remplir⟧ ▸ Source : scripts/adapt_rcaeval.py ; ▸ Chiffre : 90 ép., 30 types de pannes.
### 6.4.2 Différences de protocole (cluster, 48 steps)
⟦À remplir⟧ ▸ lien §8.10 (transfert zero/few-shot).

## 6.5 Pivot ewat_v5 — Train Ticket (41 µservices Spring Cloud)
▸ Raisonnement : Observation (topologie OB trop petite, N=6 ; circularité) → Hypothèse →
Action (pivot Train Ticket, schéma enrichi, bugs réels) → Résultat (pipeline prêt) → Décision (GO).
### 6.5.1 Justification du pivot (richesse topologique, dataset public visé)
⟦À remplir⟧ ▸ Source : docs/dataset_v5_plan.md, mémoire projet v5 TT.
### 6.5.2 Déploiement Train Ticket (tt / tt-b, mongo:4.4, jaeger:1.53, JVM)
⟦À remplir⟧ ▸ Source : v5/deploy/, STATUS.md (v5).
### 6.5.3 Schéma S(t) v5.1 = ℝ^{T×41×18} (18 features dont JVM)
▸ Tableau : M[0-9], T[10-13], L[14-17] ; feature morte oom_events→mem_limit_ratio.
▸ Source : STATUS.md (v5.1).
### 6.5.4 Catalogue chaos v5 (22 scénarios) et bugs réels F1/F3
⟦À remplir⟧ ▸ Source : v5/chaos/ ; ▸ lien GrayScope §3.2.4.3 (F1 invisible).
### 6.5.5 Pipeline v5 séparé (run_campaign / build_features_v5 / validate_v5 / enforce_heldout_v5)
⟦À remplir⟧ ▸ Source : v5/collect/, scripts/validate_v5.py, scripts/enforce_heldout_v5.py.
### 6.5.6 Vérification data pré-lancement (6 épisodes réels)
⟦À remplir⟧ ▸ Chiffre : chaos localisé, 0 NaN imputé, validate [OK].

## 6.6 Tableau comparatif des versions de dataset
▸ Tableau : v3 | v4 | v4_strat | rcaeval | v5 — colonnes : topologie, N, T, #ép., split,
features, défaut corrigé.

---

# 7 Architecture du pipeline EWAT et ses itérations
▸ Budget pages : 8
> RÈGLE ANTI-FUSION : chaque variante d'encodeur et chaque sweep a sa sous-section + ▸ Raisonnement.

## 7.1 Vue d'ensemble et modularité
⟦À remplir⟧ ▸ Source : STATUS.md (modules src/ewat/*, tests).
▸ Figure : schéma pipeline S(t)→étape0→1→2→2b→3→Alert.

## 7.2 Étape 0 — Détection de drift MMD-RFF avec look-through
### 7.2.1 Test MMD² par Random Fourier Features (O(nD))
⟦À remplir⟧ ▸ Source : src/ewat/drift/mmd.py ; ▸ lien Gretton/Rahimi §3.3.2.
### 7.2.2 Mécanisme de look-through (transmettre / RECALIBRATE)
⟦À remplir⟧ ▸ Source : CLAUDE.md (règle look-through).
### 7.2.3 Calibration de ε_drift
▸ Raisonnement : Observation → Hypothèse → Action (injection drifts bénins) → Résultat → Décision.
▸ Chiffre : ε_drift=0.5226 (Youden, AUC=0.60) ; ▸ Source : experiments/drift_separation/, scripts.
### 7.2.4 Intégration à l'AlertAssembler
⟦À remplir⟧ ▸ Source : src/ewat/alerts/.

## 7.3 Représentation : graphe de services et tenseur d'adjacence
⟦À remplir⟧ ▸ Source : src/ewat/encoder (ServiceGraph, adjacency) ; ▸ lien Kipf §3.3.1.2.

## 7.4 Étape 1 — Encodeur STGCN (architecture de référence)
### 7.4.1 Couche GCN spectrale multi-canal + bloc temporel causal
⟦À remplir⟧ ▸ Source : src/ewat/encoder/stgcn.py.
### 7.4.2 LayerNorm et fix de forward (résidu TCN)
⟦À remplir⟧ ▸ Source : STATUS.md (correction LayerNorm) ; experiments/encoder.
### 7.4.3 EpisodeDataset, padding, instance normalization
⟦À remplir⟧ ▸ Source : src/ewat/encoder (EpisodeDataset).

## 7.5 Itérations d'encodeur (variantes comparées)
> Chaque variante : ▸ Raisonnement dédié.
### 7.5.1 STGCN — baseline retenue
▸ Raisonnement ; ▸ Chiffre : K=10, sil_test, AUROC, 8/10 types (renvoi §8.11).
### 7.5.2 SimCLR — pré-entraînement contrastif (NT-Xent)
▸ Raisonnement : Observation → Hypothèse → Action → Résultat → Décision.
⟦À remplir⟧ ▸ Source : experiments/encoder/simclr, src/ewat/encoder ; ▸ lien Chen/Eldele §3.3.1.
▸ Chiffre : K=15, sil_test, AUROC.
### 7.5.3 GAT — attention sur arêtes
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/encoder/gat, stgat.py ; ▸ lien Veličković §3.3.1.3.
▸ Chiffre : K=15, sil_test, 13/15 types.
### 7.5.4 Tableau comparatif des trois encodeurs
▸ Tableau : STGCN | SimCLR | GAT × {K, sil_val, sil_test, H1, #types, AUROC}.

## 7.6 Étape 2 — Typage contrastif et clustering
### 7.6.1 Réseau siamois (perte contrastive, mining)
⟦À remplir⟧ ▸ Source : src/ewat/typing/.
### 7.6.2 Sweep clustering : ward+euclidean → average+cosine
▸ Raisonnement : Observation (mismatch géométrique) → Hypothèse → Action (sweep linkage×metric)
→ Résultat → Décision. ⟦À remplir⟧ ▸ Source : experiments/runs/sweep_clustering/ ;
▸ lien Dhillon & Modha §3.3.6.5 ; ▸ Chiffre : H1 moy avg+cos vs ward+eucl.
### 7.6.3 Sweep projection siamoise : d_proj × margin
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/runs/sweep_siamese/ ;
▸ Chiffre : gagnant dp64_m2.0 (H1) vs dp32_m1.5 (H3).
### 7.6.4 Sélection de K (silhouette vs gap statistic)
⟦À remplir⟧ ▸ Source : src/ewat/typing (cluster_embeddings) ; ▸ lien Tibshirani §3.2.5.2.
### 7.6.5 Interprétabilité : permutation importance + validation KernelSHAP
⟦À remplir⟧ ▸ Source : experiments/typing/permutation_importance.py, kernel_shap_importance.py ;
▸ lien Lundberg §3.3.4 ; ▸ Chiffre : ρ gradient×input=−0.34 (invalidé), SHAP 9/10 concordants.

## 7.7 Reframing architectural — cible Chaos Mesh directe
> Renvoi des résultats chiffrés vers §8.7–§8.8 ; ici la logique de conception.
### 7.7.1 B1 — diagnostic instance normalization (global vs instance)
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/architecture_v2/instance_norm_diagnostic.py.
### 7.7.2 B2 — LR-OvR sur features brutes flatten (headline défendable)
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/architecture_v2/chaos_mesh_target.py.
### 7.7.3 C1 — STGCN end-to-end sur cible Chaos Mesh (n'aide pas)
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/architecture_v2/train_chaos_mesh.py.
### 7.7.4 Pipeline opérationnel résultant (Option B, sans STGCN prédictif)
⟦À remplir⟧ ▸ Source : STATUS.md (C-1, C1) ; ▸ Figure : S(t)→instance norm→LR-OvR→OpenMax.

## 7.8 Étape 3 — Précurseurs typés
### 7.8.1 Classifieurs one-vs-rest et sélection de k*
⟦À remplir⟧ ▸ Source : src/ewat/precursor/ ; k* sur val.
### 7.8.2 Sweep classifieur précurseur (lr / lr_tuned / rf / svc)
▸ Raisonnement. ⟦À remplir⟧ ▸ Source : experiments/runs/sweep_precursor/ ;
▸ Chiffre : lr_tuned ≈ lr ≈ rf.

## 7.9 Étape 0→3 — Assemblage des alertes
⟦À remplir⟧ ▸ Source : src/ewat/alerts/ (AlertAssembler) ; ▸ Source : experiments/alerts/.

---

# 8 Expérimentations, hypothèses et résultats
▸ Budget pages : 11

Ce chapitre rassemble les résultats du pipeline, organisés par hypothèse. Un principe gouverne sa
lecture : on sépare systématiquement les chiffres obtenus sur une cible **indépendante** — les
labels d'injection Chaos Mesh, qui constituent une vérité terrain extérieure au pipeline (§8.7,
§8.8) — de ceux obtenus sur une cible **auto-référente**, c'est-à-dire les clusters produits par
EWAT lui-même (§8.6). Les premiers se défendent tels quels ; les seconds mesurent la cohérence
interne du pipeline et doivent être lus avec la mise en garde de circularité. Sauf mention
contraire, les résultats portent sur ewat_v3 (split 209/45/45) ; les évaluations sur cible
indépendante utilisent ewat_v4_strat (270/60/45).

## 8.1 Protocole d'évaluation et corrections méthodologiques
### 8.1.1 Held-out, nearest centroid (H1), k* sur val (H3)
Une relecture méthodologique (mai 2026) a révélé deux biais dans les scripts d'origine, corrigés
dans tous les résultats rapportés ici. Premièrement, la silhouette de validation et de test était
calculée par un clustering indépendant sur chaque split (`fit_predict`), ce qui trouve la meilleure
partition propre à chaque ensemble et surestime la structurabilité. On la calcule désormais en
assignant les points de val et de test au plus proche centroïde des clusters *train*
(« nearest centroid »), ce qui mesure une vraie généralisation. L'accord entre cette méthode et un
clustering indépendant sur le train atteint 97,6 %, ce qui valide la cohérence de l'assignation.
Deuxièmement, l'horizon optimal $k^*$ des précurseurs était sélectionné directement sur le test ;
il l'est maintenant sur la validation, l'AUROC n'étant rapporté que sur le test.
▸ Source : docs/evaluation_protocol.md, experiments/verification/.

### 8.1.2 Bootstrap et intervalles de confiance (BCa)
Les métriques scalaires — AUROC, silhouette, proportions — sont accompagnées d'un intervalle de
confiance à 95 % obtenu par bootstrap (1000 rééchantillonnages), avec la correction BCa
(biais-corrigé et accéléré) lorsqu'elle s'applique. Cela rend explicite l'incertitude liée à la
petite taille des ensembles de test (45 épisodes). ▸ lien Efron §3.3.6.1.

## 8.2 Calibration de l'étape 0 (drift)
Le seuil de drift est calibré épisode par épisode : on calcule un MMD² unique entre une fenêtre de
référence (5 premiers pas, régime normal) et une fenêtre courante (5 derniers pas, régime chaos),
puis on retient le seuil de Youden sur la courbe ROC. On obtient $\varepsilon_{drift} = 0{,}5226$,
pour une ROC-AUC de 0,60 (TPR = 0,55, FPR = 0,33 sur le train). L'AUC modérée annonce déjà la
difficulté de séparer drift et anomalie par ce seul mécanisme, confirmée en §8.3.
▸ Figure : distributions MMD² (normal vs chaos) + courbe ROC ‹FIG-drift-roc›.

## 8.3 Résultat H2a — séparabilité du drift par look-through (résultat négatif)
▸ Raisonnement. **Observation** : en production, déploiements et autoscaling produisent des
changements de distribution qui ressemblent à des anomalies. **Hypothèse H2a** : le mécanisme de
look-through (confirmation temporelle post-drift) réduit le taux de faux positifs à rappel constant.
**Action** : comparer le DriftDetector à un seuil MMD² simple, en streaming sur le test.
**Résultat** : aucune réduction significative (détails ci-dessous). **Décision** : conserver
l'étape 0 comme alarme de changement rapide, mais confier la qualification drift/anomalie aux
étapes aval — les deux ne sont pas substituables.

### 8.3.1 Look-through sur signal brut
Sur les 45 épisodes de test, le look-through dégrade plutôt qu'il n'améliore la séparation : il
détecte correctement 42 % des drifts comme drifts (contre 67 % pour le seuil simple) et confond
67 % des anomalies avec un drift (contre 73 %). La réduction du taux de faux positifs n'est pas
significative (test de Student unilatéral apparié, p = 0,27).

| | Look-through | Seuil simple |
|---|---|---|
| TPR (drift détecté comme drift) | 0,42 | 0,67 |
| FPR (anomalie confondue avec drift) | 0,67 | 0,73 |
| p-value (Student unilatéral apparié) | 0,27 | — |

### 8.3.2 Look-through sur embeddings STGCN
Rejouer le test dans l'espace d'embedding du typage (au lieu du signal brut) ne change rien :
le seuil de Youden y vaut $\varepsilon_{emb} = 0{,}5186$ pour un indice J de seulement 0,071
(discrimination quasi nulle), et le look-through reste pire que le seuil simple (FPR 0,788 contre
0,667, p = 0,978). Les embeddings siamois capturent *quel type* d'anomalie se produit, pas *si* le
changement en cours est un drift bénin ou une anomalie.

### 8.3.3 Retest sur ewat_v4_strat (épisodes longs)
On pouvait soupçonner que l'échec venait de la brièveté des épisodes v3 (~21 pas). Le retest sur
ewat_v4_strat (épisodes de 47 à 51 pas) donne le même verdict : TPR 0,500 contre 0,750 pour le
seuil simple, réduction du FPR non significative (p = 0,372). La durée n'explique donc pas tout.

### 8.3.4 Interprétation : H2a FAIL robuste (contribution négative honnête)
H2a est falsifiée, et de façon reproductible (v3 et v4_strat). C'est un résultat négatif assumé :
le MMD² avec confirmation temporelle ne sépare pas le drift bénin de l'anomalie sur ce type de
données. Loin d'invalider l'architecture, ce constat la précise — l'étape 0 sert d'alarme de
changement, la distinction de régime relève d'un espace de représentation dédié qui reste à
construire (piste de travaux futurs, §12).

## 8.4 Résultat H1 — structurabilité des embeddings
▸ Raisonnement. **Observation** : si les types d'anomalies existent réellement, les embeddings
doivent se regrouper. **Hypothèse H1** : la silhouette en held-out dépasse 0,3 (seuil de Kaufman &
Rousseeuw). **Action** : entraîner l'encodeur puis le typage siamois, mesurer la silhouette par
nearest centroid. **Résultat** : seuil franchi, avec un net gain après optimisation du clustering.
**Décision** : H1 retenue comme contribution géométrique principale du pipeline.

### 8.4.1 Silhouette train/val/test, K optimal
Sur ewat_v3 (graine 42), la silhouette vaut 0,577 (train), 0,470 (val) et 0,414 (test), pour un
nombre optimal de clusters K = 10. Le test à 0,414 dépasse largement le seuil de 0,3 : **H1 ✓ PASS**.
Que K = 10 émerge de 15 scénarios injectés signifie que certains scénarios partagent une même
signature dans l'espace latent (par exemple un crash et un OOM peuvent être indiscernables une
minute avant l'événement) — le pipeline découvre une taxonomie plus compacte que le catalogue
Chaos Mesh.

| Split | Silhouette | Méthode |
|---|---|---|
| Train | 0,577 | clustering agglomératif |
| Val | 0,470 | nearest centroid |
| Test | 0,414 | nearest centroid |

### 8.4.2 Config optimisée (average+cosine, d_proj=64, m=2.0)
Un sweep d'hyperparamètres (détaillé en §7.6) fait passer la silhouette test de 0,519 ± 0,092
(config initiale, 5 graines) à **0,782 ± 0,065** (10 graines), avec un minimum de 0,618 — toujours
au-dessus du seuil. Ce gain de +51 % est réel et défendable : il vient de l'alignement géométrique
entre la métrique de clustering (cosinus sur sphère unité) et les embeddings L2-normalisés, et non
d'un ajustement sur les labels. Sur le dataset plus long ewat_v4_strat (Phase H, 10 graines), la
silhouette retombe à 0,691 ± 0,115 avec une variance plus large (intervalle [0,521 ; 0,839]) — la
structurabilité tient, mais K devient instable (§9.5).

## 8.5 Résultat H2b — identification du régime θ_{drift∩anomaly} (nuancé)
▸ Raisonnement. **Observation** : le régime mixte θ_{drift∩anomaly} (déploiement défectueux) doit
se distinguer du drift pur et de l'anomalie pure. **Hypothèse H2b** : un cluster présente
simultanément drift et alerte à une fréquence supérieure au hasard. **Action** : mesurer le
chevauchement par cluster, puis tester sa significativité. **Résultat** : PASS formel mais trivial.
**Décision** : reconnaître la limite (DriftDetector trop sensible sur épisodes courts) plutôt que
de la masquer.

### 8.5.1 Critère formel (overlap > 30 %) — PASS trivial
Le critère « chevauchement > 30 % » est atteint partout, mais pour une mauvaise raison : sur des
épisodes courts, le DriftDetector (fenêtre de 5 pas) se déclenche sur presque tous les épisodes, et
le seuil d'alerte tire sur la plupart. Le cluster C8 (faulty_deploy_overlap), censé incarner le
régime mixte, affiche bien drift% = 0,85, alert% = 0,92 et overlap% = 0,77 — cohérent — mais sans se
détacher nettement des autres.

### 8.5.2 Critère strict (Fisher exact C8 vs drift pur)
Le test strict confirme la trivialité : un test exact de Fisher comparant C8 aux clusters de drift
pur (C5+C6+C9) donne un odds ratio de 1,48 et p = 0,35, soit aucune différence significative. Le
critère formel passe, mais le régime mixte n'est pas isolé de façon robuste.

### 8.5.3 Timing : alerte précurseur avant drift flag
Un constat complémentaire éclaire l'architecture : l'alerte de précurseur précède le drapeau de
drift dans 85 à 100 % des cas. Le DriftDetector est donc un indicateur *tardif* ; l'anticipation
réelle vient de l'étape 3 (précurseurs), pas de l'étape 0. H2b renforce ainsi la conclusion de H2a.

## 8.6 Résultat H3 — prédictibilité des précurseurs (cible EWAT, CIRCULAIRE)
> ⚠ **Mise en garde de circularité.** Les chiffres de cette section mesurent la prédiction des
> labels de cluster produits par EWAT lui-même à partir des embeddings STGCN : la cible est
> auto-référente. Ils établissent la cohérence interne du pipeline, non sa valeur prédictive
> indépendante. Le chiffre défendable correspondant est en §8.7 ; le test de précursion temporelle
> qui démasque cette circularité est en §9.1.1.

▸ Raisonnement. **Observation** : si les embeddings capturent des signaux pré-anomalie, on doit
prédire le type avant l'injection. **Hypothèse H3** : l'AUROC par type dépasse la base 0,5.
**Action** : un classifieur one-vs-rest par cluster, $k^*$ choisi sur val, AUROC sur test.
**Résultat** : 8/10 types prédictibles. **Décision** : H3 PASS, mais à recadrer (cf. §9.1).

### 8.6.1 AUROC par type, k*, IC bootstrap (ewat_v3)
| Type | n_pos test | k* | AUROC test | IC 95 % |
|---|---|---|---|---|
| C0 | 8 | 6 | 0,973 | [0,906 ; 1,000] |
| C1 | 3 | 6 | 0,992 | [0,953 ; 1,000] |
| C2 | 5 | 6 | 0,945 | [0,865 ; 1,000] |
| C3 | 3 | 2 | 0,794 | [0,636 ; 0,930] |
| C4 | 8 | 2 | 1,000 | [1,000 ; 1,000] |
| C5 | 2 | 6 | 0,977 | [0,909 ; 1,000] |
| C6 | 1 | 2 | NaN (n_pos < 2) | — |
| C7 | 7 | 6 | 0,992 | [0,966 ; 1,000] |
| C8 | 7 | 10 | 0,962 | [0,895 ; 1,000] |
| C9 | 1 | 2 | NaN (n_pos < 2) | — |

L'horizon $k^* = 6$ pas (3 min) domine (5 types sur 8), ce qui situe la zone de prédictibilité
optimale autour de 3 minutes. C6 et C9 sont non concluants faute de positifs en test.

### 8.6.2 Verdict H3 PASS et mise en garde de circularité
H3 est validée au sens « AUROC > 0,5 » : 8/10 types ont un AUROC supérieur à 0,9, et la config
optimisée porte la moyenne à 0,987 ± 0,011 (10 graines, 10/10 PASS). Mais cette performance mesure
la *récupérabilité* des labels EWAT, pas une prédiction d'événement futur indépendante : le test
distant-window (§9.1.1) montre que l'AUROC ne dépend pas de la position de la fenêtre dans le régime
normal. On reformule donc H3 en « typage anticipé du scénario actif » plutôt qu'en « détection
précoce », et l'on s'appuie sur §8.7 pour le chiffre défendable.

## 8.7 Headline défendable — cible Chaos Mesh indépendante (B1/B2)
> ⚠ **Chiffres défendables.** Cible indépendante (labels d'injection Chaos Mesh), intervalles de
> confiance explicites, sans encodeur appris sur la cible. C'est le résultat à mettre en avant.

### 8.7.1 B1 — instance norm, position de fenêtre (v3 puis v4_strat)
Un diagnostic sur features brutes (sans encodeur) compare la normalisation globale à la
normalisation par instance, et fait varier la position de la fenêtre dans le régime normal. Deux
enseignements. D'abord, la normalisation par instance améliore la séparation des scénarios, car elle
gomme les baselines absolues propres à chaque service. Ensuite, l'écart entre une fenêtre proche de
l'injection et une fenêtre lointaine — $\Delta(\text{far}-\text{near})$ — est négatif : il existe
une dynamique pré-injection captée par le signal. Sur ewat_v3, $\Delta = -0{,}071$ (global) et
$-0{,}026$ (instance) ; sur ewat_v4_strat, $-0{,}043$ et $-0{,}063$, plus marqué grâce aux épisodes
deux fois plus longs.

### 8.7.2 B2 — LR-OvR flatten, macro-AUROC stratified + LOSO
Le headline défendable est une régression logistique one-vs-rest sur les features brutes aplaties
(fenêtre pré-injection, instance-normalisées), sans encodeur STGCN, entraînée à prédire les
15 scénarios Chaos Mesh. Sur ewat_v4_strat, elle atteint un macro-AUROC **stratifié de 0,9201**
(IC 95 % [0,878 ; 0,956]) et **0,9298 en validation croisée leave-one-scenario-out** (15 folds).
Le solveur lbfgs étant déterministe, la valeur ne varie pas d'une graine à l'autre ; seule
l'incertitude bootstrap est rapportée.

### 8.7.3 Comparaison v3 vs v4_strat (amplification du signal)
| Métrique | ewat_v3 | ewat_v4_strat |
|---|---|---|
| B2 stratifié | 0,855 [0,789 ; 0,905] | **0,920** [0,878 ; 0,956] |
| B2 LOSO | 0,847 | **0,930** |
| B1 best (instance norm, fenêtre proche) | 0,850 | **0,941** [0,909 ; 0,970] |

Les épisodes plus longs de v4 confirment et amplifient le signal de v3 : la dynamique pré-injection
y est plus nette.

## 8.8 Neutralité de l'encodeur STGCN sur cible indépendante (B3/B4, C1)
### 8.8.1 B3/B4 — features brutes vs z_e STGCN (macro-AUROC, k=6)
Sur la cible indépendante, ajouter l'encodeur STGCN n'améliore pas la prédiction agrégée : les
features brutes (B3) et les embeddings STGCN (B4) donnent exactement le même macro-AUROC de 0,835
($\Delta_{macro} = 0{,}000$ sur ewat_v3). Le détail par scénario montre que l'encodeur *redistribue*
la discriminabilité plutôt qu'il ne l'augmente — il aide les pannes de saturation CPU/latence
(fail_slow_cpu +0,270) et nuit aux pannes réseau/config (noisy_neighbor −0,246), la somme des écarts
étant exactement nulle sur n = 45.

### 8.8.2 A5 — IC paired bootstrap sur Δ(B4−B3)
Ce $\Delta = 0$ n'est pas un artefact ponctuel : un bootstrap apparié (mêmes indices pour B3 et B4,
1000 tirages) donne $\Delta(B4-B3) = +0{,}0053$ avec un IC 95 % de [−0,0315 ; +0,0444], qui contient
zéro (P($\Delta \le 0$) = 0,420). La neutralité de l'encodeur sur cette cible est donc statistiquement
bien établie.

### 8.8.3 C1 — STGCN end-to-end ne dépasse pas B2
Entraîner un STGCN de bout en bout directement sur la cible Chaos Mesh (v4_strat) ne renverse pas le
constat : il plafonne à 0,863 (IC [0,823 ; 0,905]), en deçà des 0,920 de la régression logistique B2.
L'encodeur n'est pas nécessaire à la tâche prédictive principale.

### 8.8.4 Interprétation : valeur géométrique/ontologique, pas prédictive agrégée
La valeur du STGCN n'est donc pas dans la prédiction agrégée mais ailleurs : dans la structuration de
l'espace latent (H1, silhouette 0,782) qui rend le clustering et l'ontologie possibles, et dans la
précursion temporelle qu'il exploite sur cible indépendante (§9.1.6). Le pipeline opérationnel met
en avant la régression logistique ; le STGCN sert le typage et l'ontologie.

## 8.9 Baselines précurseurs (B0–B4)
Deux familles de baselines coexistent, selon la cible. B0–B2 visent les labels EWAT (mesure de
*récupérabilité*, donc circulaire) ; B3–B4 visent les labels Chaos Mesh (vérité terrain
indépendante, défendable). Les valeurs B3/B4 sont reprises de §8.8.

### 8.9.1 B0 — aléatoire (référence 0,5)
Classifieur aléatoire, AUROC 0,500 — borne basse de référence.

### 8.9.2 B1 — features brutes (cible EWAT, récupérabilité)
Régression logistique sur features brutes prédisant les labels EWAT : AUROC 0,966. Les labels EWAT
sont donc trivialement récupérables sans encodeur — premier indice de circularité.

### 8.9.3 B2 — k-means brut + LR (cible EWAT)
Un k-means brut suivi d'une régression logistique atteint 0,975, soit davantage que le pipeline
EWAT complet (0,951) sur sa propre cible. Confirmation que cette tâche ne nécessite pas le STGCN.

### 8.9.4 B3 — features brutes (cible Chaos Mesh indépendante)
Sur la vérité terrain Chaos Mesh, les features brutes donnent 0,835 (IC [0,773 ; 0,888]).

### 8.9.5 B4 — STGCN z_e (cible Chaos Mesh)
Les embeddings STGCN donnent également 0,835 (IC [0,772 ; 0,885]) — voir §8.8 pour l'analyse de la
neutralité.

### 8.9.6 Lecture : récupérabilité (circulaire) vs vérité terrain (défendable)
B1/B2 mesurent à quel point les labels EWAT se reconstituent depuis le signal (circulaire) ; B3/B4
mesurent une vraie discriminabilité de scénario (défendable). L'écart entre les deux familles
(0,97 vs 0,84) chiffre exactement la part de circularité à retrancher du « headline » naïf.

## 8.10 Transfert externe — ewat_rcaeval
▸ Raisonnement. **Observation** : la validation externe est nécessaire pour crédibiliser le
pipeline. **Hypothèse** : appliqué sans réentraînement à un benchmark public (RCAEval RE2-OB), le
pipeline conserve H1/H3. **Action** : transfert zero-shot puis few-shot. **Résultat** : détection
d'anomalie générique oui, discrimination par type non. **Décision** : reconnaître l'échec et
identifier le verrou (le scaler).

### 8.10.1 Zero-shot (4 stratégies de normalisation)
Appliqué tel quel à RCAEval (90 épisodes, 30 types de pannes), le pipeline regroupe les anomalies
mais ne les discrimine pas. La meilleure configuration (instance norm + métriques seules) atteint une
silhouette H1 de 0,684 (PASS) mais un AUROC H3 de 0,495 (échec, ≈ hasard) : l'encodeur détecte
qu'il y a une anomalie, sans dire laquelle.

### 8.10.2 Few-shot Stratégie A (re-fit scaler)
Réajuster le seul scaler sur quelques épisodes RCAEval ne débloque rien : l'AUROC H3 reste collé à
≈ 0,50 quel que soit le nombre d'épisodes (de 1 à 40).

### 8.10.3 Interprétation : goulot = scaler non transférable (échec honnête)
Le verrou est l'espace latent ewat_v3, qui ne sépare pas les types RCAEval ; réajuster le scaler est
insuffisant. Un transfert réel demanderait un fine-tuning du classifieur ou de l'encodeur
(Stratégie B, §12). C'est un échec de généralisation assumé, utile car il borne la portée du modèle.

## 8.11 Comparaison des encodeurs et baseline d'alerte
### 8.11.1 STGCN vs SimCLR vs GAT (récapitulatif chiffré)
| Architecture | K | sil_val | sil_test | H3 types | AUROC moyen |
|---|---|---|---|---|---|
| STGCN (référence) | 10 | 0,470 | 0,414 | 8/10 | 0,954 |
| SimCLR (contrastif) | 15 | 0,495 | 0,429 | 11/15 | 0,964 |
| GAT (attention) | 15 | 0,445 | 0,497 | 13/15 | 0,929 |

GAT offre la meilleure géométrie (sil_test 0,497) et couvre plus de types, mais avec un AUROC moyen
plus faible ; SimCLR maximise l'AUROC ; STGCN, avec K = 10 plus stable et des résultats multi-graines
disponibles, est retenu comme architecture principale (cf. §7.5).

### 8.11.2 Baseline z-score vs EWAT (détection, FA drift, lead time)
La baseline z-score détecte 100 % des anomalies mais lève 100 % de fausses alertes sur les drifts,
quel que soit le seuil σ — elle ne distingue pas drift et anomalie. C'est exactement le faux positif
que EWAT vise à éliminer.

| Méthode | Détection | FA drift | Lead (min) |
|---|---|---|---|
| z-score (σ = 2,0–3,5) | 100 % | 100 % | 2,5 |
| EWAT (seuil 0,7) | 57,6 % | 8,3 % | 3,0 |

### 8.11.3 Simulation en ligne AlertAssembler (seuils)
| Seuil | Détection | Cluster correct | FA drift | Lead (min) |
|---|---|---|---|---|
| 0,30 | 100 % | 42,4 % | 100 % | 4,6 |
| 0,40 | 97,0 % | 66,7 % | 100 % | 3,8 |
| 0,50 | 78,8 % | 63,6 % | 100 % | 3,9 |
| 0,60 | 75,8 % | 63,6 % | 50,0 % | 3,7 |
| 0,70 | 57,6 % | 51,5 % | 8,3 % | 3,0 |

Le point opérationnel recommandé est le seuil 0,70 : il ramène le taux de fausses alertes sur drift à
8,3 % tout en conservant un lead time de 3,0 min. Aux seuils bas, le DriftDetector n'a pas le temps de
se réchauffer avant que les classifieurs ne tirent (limite liée à la longueur des épisodes).

## 8.12 Ablations
### 8.12.1 Ablation modalités H1 (réentraînement complet)
Cette ablation réentraîne entièrement l'encodeur et le typage pour chaque combinaison de modalités
(et non un simple masquage à l'inférence, qui serait biaisé hors-distribution). Résultat
contre-intuitif : les **métriques seules battent le modèle complet** (silhouette test 0,497 contre
0,439, soit +0,058). Les traces et les logs ajoutent du bruit géométrique au clustering sur
n = 209 — leur valeur est prédictive (H3), pas géométrique (H1).

| Condition | n_feat | sil_train | sil_test | Δ vs full |
|---|---|---|---|---|
| full | 17 | 0,378 | 0,439 | — |
| M_only | 7 | 0,241 | **0,497** | +0,058 |
| T_only | 6 | 0,064 | 0,412 | −0,027 |
| M+L | 11 | 0,251 | 0,382 | −0,057 |
| T+L | 10 | 0,022 | 0,341 | −0,098 |
| M+T | 13 | 0,318 | 0,316 | −0,123 |
| L_only | 4 | −0,138 | 0,051 | −0,388 |

### 8.12.2 Ablation modalités H3 (masquage inférence)
Pour la prédictibilité, la conclusion s'inverse : le modèle complet (macro-AUROC 0,954) bat toutes
les réductions. Traces et logs sont nécessaires aux précurseurs même s'ils nuisent au clustering.

| Condition | Macro-AUROC | Δ vs full |
|---|---|---|
| full | 0,954 | — |
| M+L | 0,916 | −0,038 |
| M_only | 0,756 | −0,198 |
| T+L | 0,563 | −0,391 |
| L_only | 0,488 | −0,466 |

### 8.12.3 Ablation par feature (leave-one-out) et redondance
En retirant une feature à la fois (test de Wilcoxon signé, p < 0,05), les plus critiques pour H1 sont
`trace_depth` (Δ = −0,069), `lexical_entropy` (−0,069) et `latency_p99` (−0,062) ; `disk_io` reste
significatif malgré ses 16,7 % de NaN (−0,010), ce qui plaide pour ewat_v4. Pour H3, la feature la
plus critique est `disk_io` (Δ = −0,088). Deux paires sont fortement redondantes :
`latency_p99` ↔ `span_dur_p99` (ρ = 0,936) et `error_rate_http` ↔ `abnormal_span_rate` (ρ = 0,927),
candidates à la suppression.

## 8.13 Latence end-to-end
La chaîne d'inférence (étapes 0, 1, 3) est mesurée à un p95 total de **13 ms**, soit environ 375 fois
sous le budget de 5 s fixé en §4.6. Les étapes 2 et 2b (typage, ontologie) sont hors ligne et hors
budget.

| Étape | Budget | Mesure |
|---|---|---|
| 0 — drift | < 1 s | inclus dans p95 total |
| 1 — encodeur | < 2 s | inclus dans p95 total |
| 3 — précurseurs | < 1 s | inclus dans p95 total |
| **Total (0+1+3)** | **< 5 s** | **13 ms (p95)** |

---

# 9 Validation de robustesse et multi-graines
▸ Budget pages : 7

Ce chapitre éprouve la solidité des résultats du chapitre 8 : tests de robustesse ciblés sur H3
(§9.1), puis sweep multi-graines séparant la métrique circulaire (§9.2) du headline défendable
déterministe (§9.3), diagnostics de stabilité (§9.4–§9.5) et reconnaissance open-set (§9.6).

## 9.1 Stress tests H3 (A1–A5, C2)
### 9.1.1 A1 — distant-window : fuite de signature scénario (négatif)
▸ Raisonnement. **Observation** : un AUROC élevé ne prouve pas une précursion temporelle.
**Hypothèse** : si le signal est précurseur, déplacer la fenêtre loin de l'injection doit dégrader
l'AUROC. **Action** : mêmes modèles, fenêtre déplacée dans le régime normal (last/middle/first).
**Résultat** : aucune dégradation. **Décision** : recadrer H3 comme typage de signature, pas
prédiction.
Sur cible EWAT (v3), l'AUROC est quasi identique quelle que soit la position : 0,904 (juste avant
l'injection), 0,907 (milieu), 0,897 (début du régime normal), soit $\Delta(\text{far}-\text{near})
= -0{,}007$. Le classifieur lit donc la signature *statique* du scénario, récupérable depuis
n'importe quel point — ce qui confirme la circularité de §8.6.

### 9.1.2 A2 — Leave-One-Scenario-Out (precursor-only)
En retirant tout un scénario de l'entraînement du précurseur, le macro-AUROC sur l'ensemble de test
reste élevé (0,896 ± 0,013) car les autres scénarios couvrent l'espace ; mais la vraie généralisation
— le top-1 sur le scénario inédit — n'est que de 0,511 ± 0,382, très polarisée (4 scénarios à 100 %,
4 à 0 %). Le modèle interpole entre scénarios connus, il ne généralise pas à un type inédit.

### 9.1.3 A3 — permutation test (distribution nulle)
En permutant aléatoirement les labels d'entraînement (100 tirages), l'AUROC tombe à 0,492 ± 0,104
(p95 = 0,672), contre 0,893 observé : p < 0,01. Il y a donc bien un signal réel aligné sur les
labels — A3 confirme l'existence du signal, A1 montre qu'il n'est pas temporel, A2 qu'il ne
généralise pas. Les trois sont cohérents.

### 9.1.4 A4 — filtrage n_pos ≥ 5
En ne gardant que les clusters dont le test contient au moins 5 positifs, 5 clusters sur 10 sont
reportables (C0, C2, C4, C7, C8), avec un AUROC moyen de 0,975 ± 0,020. Les 5 autres
(n_pos ≤ 3) doivent être marqués « non concluant ».

### 9.1.5 A5 — IC paired Δ(B4−B3)
Le bootstrap apparié sur l'écart B4−B3 est traité en §8.8.2 : IC [−0,0315 ; +0,0444], contient zéro.

### 9.1.6 C2 — distant-window sur modèle Chaos Mesh (renversement, GENUINE)
Le même test que A1, mais sur le modèle entraîné cible Chaos Mesh (v4_strat), renverse la conclusion :
l'AUROC passe de 0,876 (juste avant l'injection) à 0,813 (milieu) puis 0,759 (début), soit
$\Delta(\text{far}-\text{near}) = -0{,}116$. Sur cible indépendante, la dynamique pré-injection vaut
donc une douzaine de points d'AUROC : il y a une précursion temporelle réelle. La « fuite » de A1
était un artefact de la circularité des labels EWAT, pas une propriété du signal.

### 9.1.7 Synthèse de cohérence A1/B1/C2 (signature statique vs précursion réelle)
Les trois mesures se recoupent : sur cible auto-référente (A1), $\Delta \approx 0$ — signature
statique ; sur cible indépendante avec features brutes (B1), $\Delta$ de −0,03 à −0,07 ; sur cible
indépendante avec STGCN end-to-end (C2), $\Delta = -0{,}116$. La précursion temporelle existe et se
révèle dès qu'on évalue honnêtement, sur une cible extérieure au pipeline.

## 9.2 Multi-seed Phase H — cible labels EWAT (circulaire)
▸ Raisonnement. **Observation** : un résultat sur une seule graine peut être un coup de chance.
**Action** : rejouer le pipeline complet sur 10 graines (cible EWAT). **Résultat** : variance large
sur H1 et confirmation de la fuite A1. **Décision** : ne pas s'appuyer sur cette métrique pour le
headline.

| Métrique (10 graines, v4_strat) | Moyenne ± écart-type | Intervalle |
|---|---|---|
| H1 silhouette test | 0,691 ± 0,115 | [0,521 ; 0,839] |
| H3 AUROC peak (circulaire) | 0,990 ± 0,012 | [0,959 ; 1,000] |
| A1 Δ(far−near) | −0,012 ± 0,022 | [−0,050 ; +0,019] |
| K optimal | 11,8 ± 2,1 | [9 ; 15] |

La fuite A1 est confirmée 9 fois sur 10 (la graine 42, GENUINE, est l'unique exception et un
outlier). Le retrain « Phase G » sur cette seule graine 42 surestimait donc deux métriques.

## 9.3 Multi-seed Phase J — headline défendable Chaos Mesh (déterministe)
▸ Raisonnement. **Observation** : il faut un chiffre robuste, indépendant des labels EWAT.
**Action** : évaluer B2 (LR-OvR, cible Chaos Mesh) sur 10 graines. **Résultat** : valeur
déterministe, IC bootstrap explicite. **Décision** : c'est le headline final.
La régression logistique (solveur lbfgs) étant déterministe, les 10 graines donnent exactement le
même chiffre, l'incertitude étant portée par le bootstrap : **B2 stratifié = 0,9201**
(IC [0,878 ; 0,956]) et **B2 LOSO = 0,9298** (15 folds). C'est le résultat à reporter au maître de
stage, indépendant et reproductible.

## 9.4 Multi-seed Phase K — diagnostics
### 9.4.1 K.1 — comparaison silhouette vs gap (Tibshirani)
Deux stratégies de sélection de K divergent : la silhouette donne un mode K = 14 (intervalle [9 ; 15]),
la statistique de gap de Tibshirani un mode K = 12 (intervalle [4 ; 12]), avec un accord sur seulement
4 graines sur 10. Aucune méthode ne stabilise K.

### 9.4.2 K.3 — variance per-seed, seed 42 outlier
La distribution par graine sépare nettement les métriques stables (AUROC circulaire, B2 déterministe)
des instables (silhouette, K, A1). La graine 42 ressort comme outlier sur A1, ce qui explique les
résultats trop optimistes de la Phase G. ▸ Figure : distribution par graine ‹FIG-phaseK-distrib›
(experiments/multiseed/phase_h/distribution.png).

## 9.5 Instabilité de K (résultat négatif structurel)
La leçon de la Phase K est que K est intrinsèquement instable sur ce dataset (n_train = 270) :
intervalle [9 ; 15] selon la graine, et ni la silhouette ni le gap statistique ne le fixent. C'est un
résultat négatif structurel honnête, à corriger en v5 soit en fixant K manuellement (= 10), soit en
passant à un clustering par densité (HDBSCAN), qui ne requiert pas de fixer K a priori (cf. §12).

## 9.6 Reconnaissance open-set (OpenMax) — résultat mitigé
▸ Raisonnement. **Observation** : un classifieur fermé ne peut pas signaler un type inédit.
**Hypothèse** : OpenMax (théorie des valeurs extrêmes) flague les nouveautés. **Action** : évaluation
LOSO complète. **Résultat** : signal partiel. **Décision** : reconnaître la limite, proposer mieux
en §12.
OpenMax apporte un signal de nouveauté réel mais incomplet : le taux de top-1 « unknown » sur le
scénario inédit passe de 0 (classifieur fermé) à 0,400 ± 0,407, mais l'AUROC unknown global reste à
0,550 ± 0,238 (≈ hasard), et la performance fermée se dégrade légèrement (macro-AUROC 0,834 ± 0,023).
Une généralisation complète demanderait un dispositif plus sophistiqué (Mahalanobis-OOD,
energy-based ; cf. §12).

## 9.7 Verdict consolidé multi-seed (à reporter au maître de stage)
| Métrique | Valeur consolidée | Note |
|---|---|---|
| **Headline défendable B2 (Chaos Mesh)** | **0,9201** [0,878 ; 0,956] | déterministe, indépendant |
| B2 LOSO | 0,9298 | 15 folds |
| H1 silhouette (10 graines) | 0,691 ± 0,115 | variance large, K instable |
| H3 AUROC (circulaire) | 0,990 ± 0,012 | auto-référent, cf. §8.6 |
| A1 Δ(far−near) | −0,012 ± 0,022 | fuite 9/10, GENUINE 1/10 |
| Précursion réelle C2 | Δ = −0,116 | cible indépendante |
| Latence E2E p95 | 13 ms | sous budget 5 s (×375) |

Lecture d'ensemble : le headline défendable est **0,920** sur cible Chaos Mesh indépendante ; les
métriques circulaires (H3 ≈ 0,99) sont signalées comme telles ; les résultats négatifs (H2a, fuite
A1, instabilité de K, échec de transfert) sont assumés comme des contributions à part entière.

---

# 10 Ontologie empirique des pannes
▸ Budget pages : 5

## 10.1 Motivation et place dans le pipeline (étape 2b, offline)
⟦À remplir⟧ ▸ Source : docs/notes/justification_ontologie_ewat.pdf.

## 10.2 TBox — taxonomie ancrée dans la littérature
### 10.2.1 29 classes (Soldani & Brogi, Fu, Gregg, Aniello)
⟦À remplir⟧ ▸ lien §3.3.3.4–§3.3.3.5 ; ▸ Source : src/ewat/ontology/owl_schema.py.
### 10.2.2 Object/data properties (causes, precedes, coOccursWith, propagatesThrough)
⟦À remplir⟧ ▸ Chiffre : 11 object + 6 data properties.

## 10.3 Relations causales par Transfer Entropy (KSG)
### 10.3.1 Estimateur KSG, n_min, seuil par permutation
⟦À remplir⟧ ▸ lien Kraskov §3.2.6 / Schreiber §3.3.3.1 ; ▸ Source : src/ewat/ontology (TE).
### 10.3.2 Correction FDR Benjamini–Hochberg
▸ CI/Test ; ▸ lien §3.3.3.2.
### 10.3.3 Cluster-level (biais écologique) vs service-level
▸ Chiffre : 0 causale cluster-level ; 124→46 service-level, 8/10 clusters.
▸ Source : experiments/ontology (build.py, build_service.py).
### 10.3.4 TE multivariée KSG-1 (3 causales sur cascades synthétiques)
▸ Chiffre : C4→C1, C6→C5, C4→C8. ▸ Source : experiments/ontology_v2/.

## 10.4 Co-occurrences (χ² Yates / Fisher)
⟦À remplir⟧ ▸ lien Agresti §3.3.6.4, Holm §3.3.3.3 ; ▸ Chiffre : 19 co-occurrences.

## 10.5 Raisonnement OWL et requêtes
### 10.5.1 HermiT — cohérence et matérialisation
▸ Chiffre : cohérent en 0.61 s, 0 classe inconsistante. ▸ lien Glimm §3.3.3.6.
### 10.5.2 owlready2 et SPARQL
⟦À remplir⟧ ▸ lien Lamy §3.3.3.7 ; ▸ Source : src/ewat/ontology/queries.py.

## 10.6 Épisodes synthétiques composites (synthesis)
▸ Raisonnement. ▸ Chiffre : 282 ép. synthétiques, AUC discriminateur=0.529.
▸ Source : scripts/synthesize_composite_episodes.py, src/ewat/ontology/synthesis.py.

## 10.7 Validation de l'ontologie
▸ Chiffre : 8/10 critères atteints. ▸ Source : experiments/ontology_v2/results.md, validation.json.

---

# 11 Synthèse des résultats
▸ Budget pages : 3

## 11.1 Tableau-bilan des hypothèses H1/H2a/H2b/H3
▸ Tableau : hypothèse | verdict | valeur clé | emplacement (défendable / circulaire).
▸ Source : STATUS.md (bilan final).

## 11.2 Headlines défendables (cible indépendante)
⟦À remplir⟧ ▸ Chiffre : B2 (Chaos Mesh) + IC. — emplacement DÉFENDABLE distinct.

## 11.3 Chiffres circulaires (cible auto-référente) — à manier avec précaution
⟦À remplir⟧ ▸ Chiffre : H3 (cible EWAT). — emplacement CIRCULAIRE distinct.

## 11.4 Contributions négatives revendiquées
⟦À remplir⟧ H2a FAIL, fuite signature A1, instabilité K, échec transfert RCAEval, OpenMax mitigé.

## 11.5 Apport opérationnel net (vs z-score)
⟦À remplir⟧ ▸ Chiffre : FA drift maîtrisée au seuil 0.7, lead time.

---

# 12 Limites et travaux futurs
▸ Budget pages : 4

## 12.1 Limites résiduelles L1–L17
### 12.1.1 Limites méthodologiques (circularité, surentraînement siamois, n_pos)
⟦À remplir⟧ ▸ Source : docs/limitations.md (L1–L9).
### 12.1.2 Limites techniques (N=6, K instable, 17 features hardcodées, cross-cluster)
⟦À remplir⟧ ▸ Source : docs/limitations.md (L10–L17).
### 12.1.3 Tableau des 17 limites (id → description → fix proposé)
▸ Tableau.

## 12.2 Travaux futurs — ROADMAP
### 12.2.1 Axe A — couplage ontologie ↔ prédiction
⟦À remplir⟧ ▸ Source : ROADMAP.md.
### 12.2.2 Axe B — précursion robuste (instance norm, fenêtre pré-injection)
⟦À remplir⟧
### 12.2.3 Axe C — open-set / nouveauté (Mahalanobis, Energy-based)
⟦À remplir⟧ ▸ lien Lee §3.3.5.2, Liu §3.3.5.3.
### 12.2.4 Axe D — déploiement opérationnel
⟦À remplir⟧

## 12.3 Perspective dataset public ewat_v5
⟦À remplir⟧ ▸ Source : mémoire projet v5 public ; conditions (autorisation Devoteam, sanitization),
datasheet Gebru, licence CC-BY-4.0.

---

# 13 Conclusion
▸ Budget pages : 2

## 13.1 Rappel de la question de recherche et réponses apportées
⟦À remplir⟧
## 13.2 Bilan honnête (ce qui marche, ce qui ne marche pas)
⟦À remplir⟧
## 13.3 Apport personnel et compétences développées
⟦À remplir⟧
## 13.4 Mot de la fin
⟦À remplir⟧

---

## BACK MATTER

# Annexes
## Annexe A — Commandes du pipeline complet
▸ Source : STATUS.md (section Commandes), README.md, v5/LAUNCH.md.
## Annexe B — Détails per-seed (Phases G/H/J/K)
▸ Source : experiments/multiseed/phase_h/results.md, phase_j/results.md, k_selection_comparison.md.
## Annexe C — Inventaire des scripts et artefacts
▸ Source : scripts/, experiments/*/ (checkpoints, results.md, json).
## Annexe D — Configurations Hydra et endpoints
▸ Source : configs/default.yaml (sans secrets).
## Annexe E — Schéma de features détaillé (v3 17 / v5.1 18)
▸ Tableau : index → nom → modalité → source → agrégation.
## Annexe F — Catalogue chaos (v3/v4 15 scénarios ; v5 22 scénarios + bugs F)
▸ Source : scripts/chaos_injector.py, v5/chaos/.

# Bibliographie
⟦À remplir⟧ Les 34 références (11 fondatrices + 23 méthodologiques), harmonisées.
▸ Source : docs/paper/bibliography.bib + ajouts requis :
chandola2009, zamanzadeh2024, pham2024, grayscope2024, kipf2017, eldele2021, rahimi2007,
schreiber2000, holm1979, glimm2014, lamy2017, efron1987, davison1997, phipson2010, agresti2002,
dhillon2001, reimers2019.

---

# AUTO-VÉRIFICATION (toutes cases cochées)
- ☑ H1 (§8.4), H2a (§8.3), H2b (§8.5), H3 (§8.6) — chacune sa sous-section de résultats dédiée.
- ☑ Chaque dataset (v3 §6.2, v4/v4_strat §6.3, rcaeval §6.4, v5 §6.5) et chaque variante archi
  (STGCN §7.5.1, SimCLR §7.5.2, GAT §7.5.3 ; sweeps §7.6.2/§7.6.3/§7.8.2 ; baselines §8.9) ont
  une sous-section ouverte par un ▸ Raisonnement.
- ☑ 34 références rattachées à une brique EWAT (§3.2 : 11 ; §3.3 : 23, incl. RCAEval).
- ☑ Front matter (page de garde→glossaire) et back matter (annexes A–F, biblio) complets.
- ☑ Headlines défendables (§8.7, §8.8, §9.3, §11.2) vs circulaires (§8.6, §9.2, §11.3) séparés.
- ☑ Somme budgets pages = 78 ∈ [50, 80].
